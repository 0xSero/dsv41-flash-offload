// DSV41 CPU tier + B70 expert tier (BT), GPU side. Based on cand10 (route ring) / cand9.
//  * CUDA-graph safe: job sequence numbers live in device counters (seqc for the CPU job, seqc2 for the B70 job)
//  * routing weights fp32 in and out; per-layer routing counts for the VRAM mirror cache (EC)
//  * B70 tier (pol[9] = 1): routed experts that the B70 worker holds (bt_res[li*E+e] != 0) plus the hottest misses chosen
//    by the 3-engine cost model are published to a second mailbox (shared with the B70 worker process); ft_combine2 waits
//    for the CPU and B70 partials and adds both into apply_exl3_fused_moe's fp32 accumulator
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// policy p[]: 0 t_zc (ms per zero-copy miss), 1 t_hit (ms per GPU-processed expert), 2 cpu_a (ms per CPU job),
// 3 cpu_b (ms per CPU expert), 4 cpu_tok (extra fraction per extra token of an expert), 5 max_cpu, 6 force_n (-1 = cost
// model), 7 handshake (1 = publish even empty jobs), 8 x already in hx (skip the in-kernel host copy),
// 9 bt_on, 10 bt_a (ms per B70 job), 11 bt_hit (ms per B70-resident expert), 12 bt_stage (ms per staged expert),
// 13 bt_max_bsz, 14 bt_max_stage (max staged experts per job)
// stats[]: 0 EC hits, 1 misses (non-EC), 2 CPU experts, 3 calls, 4 CPU wait ns, 5 CPU waits, 6 CPU timeouts,
//          8 B70 resident experts, 9 B70 staged experts, 10 B70 wait ns, 11 B70 waits, 12 B70 timeouts, 13 B70 jobs
__global__ void ft_split_k(const int64_t* __restrict__ sel, const float* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int topk, int H, int E, const int* __restrict__ slotof, const float* __restrict__ score,
    const float* __restrict__ p, volatile long long* ctrl, __half* hx, int* hpicks, long long* seqc, int li,
    int64_t* sel_out, float* w_out, long long* dflag, long long* stats, int* counts,
    int* ring, long long ring_cap, long long* ringpos, volatile long long* ringpos_host,
    const signed char* __restrict__ bt_res, volatile long long* ctrl2, __half* hx2, int* hpicks2, long long* seqc2,
    long long* dflag2)
{
    __shared__ int cnt[1024];
    __shared__ unsigned char cls[1024];          // 0 GPU (EC hit or zero-copy), 1 CPU, 2 B70
    __shared__ int npk, publish, npk2, publish2;
    __shared__ long long s_seq, s_seq2;
    __shared__ int s_miss[1024];
    const int t = threadIdx.x;
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; cls[e] = 0; }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        if (w[s] != 0.0f && e >= 0 && e < E) { atomicAdd(&cnt[e], 1); if (counts) atomicAdd(&counts[e], 1); }
    }
    __syncthreads();
    if (t == 0)
    {
        const bool bt = bt_res != nullptr && p[9] > 0.5f && bsz <= (int) p[13];
        int nm = 0, nh = 0, nb = 0;
        int* miss = s_miss;
        for (int e = 0; e < E; ++e)
        {
            if (!cnt[e]) continue;
            if (slotof[e] >= 0) { ++nh; continue; }
            if (bt && bt_res[(long long) li * E + e]) { cls[e] = 2; ++nb; continue; }
            int b = nm++;
            while (b > 0 && score[miss[b - 1]] > score[e]) { miss[b] = miss[b - 1]; --b; }
            miss[b] = e;
        }
        int best = 0, bestz = 0;
        const int force = (int) p[6], maxc = min((int) p[5], nm);
        if (force >= 0) best = min(force, nm);
        else
        {
            const int maxz = bt ? min((int) p[14], nm) : 0;
            float bt_t = 1e30f;
            for (int zb = 0; zb <= maxz; ++zb)
            {
                const float b70 = (nb + zb) ? p[10] + p[11] * nb + p[12] * zb : 0.0f;
                float c = 0.0f;
                for (int k = 0; k <= min(maxc, nm - zb); ++k)
                {
                    if (k) c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1));
                    const int zg = nm - zb - k;
                    const float gpu = p[1] * (nh + zg) + p[0] * zg;
                    const float tt = fmaxf(fmaxf(gpu, k ? p[2] + c : 0.0f), b70);
                    if (tt < bt_t - 1e-6f) { bt_t = tt; best = k; bestz = zb; }
                }
            }
        }
        for (int k = 0; k < best; ++k) cls[miss[k]] = 1;                  // coldest -> CPU
        for (int k = 0; k < bestz; ++k) cls[miss[nm - 1 - k]] = 2;         // hottest -> B70 (staged + admitted)
        publish = best > 0 || p[7] > 0.5f;
        publish2 = (nb + bestz) > 0;
        stats[0] += nh; stats[1] += nm + nb; stats[2] += best; stats[3] += 1; stats[8] += nb; stats[9] += bestz;
        if (ring && bsz <= 8)                       // DSV41_ROUTE_RING: decode routing capture (original picks)
        {
            const long long pp = ringpos[0]; ringpos[0] = pp + 1;
            int* r = ring + (pp % ring_cap) * 52;
            const int nn = n < 48 ? n : 48;
            r[0] = li; r[1] = bsz; r[2] = nn; r[3] = (int) (stats[3] & 0x7fffffff);
            for (int s2 = 0; s2 < nn; ++s2) r[4 + s2] = (w[s2] != 0.0f) ? (int) sel[s2] : -1;
            __threadfence_system();
            ringpos_host[0] = pp + 1;
        }
        npk = 0; npk2 = 0;
        if (publish)
        {
            s_seq = ++seqc[0];
            for (int s = 0; s < n; ++s)
            {
                const long long e = sel[s];
                if (e < 0 || e >= E || cls[e] != 1 || w[s] == 0.0f) continue;
                hpicks[npk * 3] = s / topk; hpicks[npk * 3 + 1] = (int) e; hpicks[npk * 3 + 2] = __float_as_int(w[s]);
                ++npk;
            }
        }
        if (publish2)
        {
            stats[13] += 1;
            s_seq2 = ++seqc2[0];
            for (int s = 0; s < n; ++s)
            {
                const long long e = sel[s];
                if (e < 0 || e >= E || cls[e] != 2 || w[s] == 0.0f) continue;
                hpicks2[npk2 * 3] = s / topk; hpicks2[npk2 * 3 + 1] = (int) e; hpicks2[npk2 * 3 + 2] = __float_as_int(w[s]);
                ++npk2;
            }
        }
    }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        const bool c = e >= 0 && e < E && cls[e] != 0;
        sel_out[s] = c ? -1 : e;
        w_out[s] = c ? 0.0f : w[s];
    }
    if (publish)
    {
        const int nx = p[8] > 0.5f ? 0 : bsz * H / 8;   // p[8] = 1: x already in hx (copy-engine DtoH before this kernel)
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
    }
    if (publish2)
    {
        const int nx = bsz * H / 8;
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx2)[i] = reinterpret_cast<const int4*>(z)[i];
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0)
    {
        if (publish)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = s_seq;
            __threadfence_system();
            dflag[0] = s_seq;
        }
        else dflag[0] = 0;
        if (dflag2)
        {
            if (publish2)
            {
                ctrl2[2] = li; ctrl2[3] = bsz; ctrl2[4] = npk2;
                __threadfence_system();
                ctrl2[0] = s_seq2;
                __threadfence_system();
                dflag2[0] = s_seq2;
            }
            else dflag2[0] = 0;
        }
    }
}

__device__ __forceinline__ bool ft_wait(const long long s, volatile long long* ctrl, long long timeout_ns, long long* stats,
                                        int i_ns, int i_n, int i_to)
{
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    t1 = t0;
    bool ok = true;
    while (ctrl[1] < s)
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if ((long long) (t1 - t0) > timeout_ns) { ok = false; if (blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[i_to], 1ull); break; }
    }
    if (blockIdx.x == 0) { atomicAdd((unsigned long long*) &stats[i_ns], (unsigned long long) (t1 - t0)); atomicAdd((unsigned long long*) &stats[i_n], 1ull); }
    return ok;
}

__global__ void ft_combine_k(float* out, int n, const long long* __restrict__ dflag, volatile long long* ctrl,
    const float* hout, long long* stats, long long timeout_ns, const long long* __restrict__ dflag2,
    volatile long long* ctrl2, const float* hout2)
{
    __shared__ int go, go2;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0 ? ft_wait(s, ctrl, timeout_ns, stats, 4, 5, 6) : 0;
        const long long s2 = dflag2 ? dflag2[0] : 0;
        go2 = s2 != 0 ? ft_wait(s2, ctrl2, timeout_ns, stats, 10, 11, 12) : 0;
    }
    __syncthreads();
    if (!go && !go2) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        float v = out[i];
        if (go) v += __ldcv(hout + i);
        if (go2) v += __ldcv(hout2 + i);
        out[i] = v;
    }
}

void ft_split(torch::Tensor sel, torch::Tensor w, torch::Tensor z, int64_t E, torch::Tensor slotof, torch::Tensor score,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, torch::Tensor seqc, int64_t li,
              torch::Tensor sel_out, torch::Tensor w_out, torch::Tensor dflag, torch::Tensor stats, torch::Tensor counts,
              int64_t ring, int64_t ring_cap, torch::Tensor ringpos, int64_t ringpos_host,
              torch::Tensor bt_res, int64_t ctrl2, int64_t hx2, int64_t hpicks2, torch::Tensor seqc2, torch::Tensor dflag2)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kFloat && z.dtype() == at::kHalf && z.is_contiguous(), "dtypes");
    TORCH_CHECK(pol.numel() >= 15 && stats.numel() >= 16, "ft_split: pol >= 15, stats >= 16");
    const int bsz = (int) z.size(0), H = (int) z.size(1), n = (int) sel.numel(), topk = n / bsz;
    TORCH_CHECK(E <= 1024 && H % 8 == 0, "ft_split: E <= 1024");
    const bool bt = bt_res.numel() > 0 && ctrl2 != 0;
    ft_split_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), w.data_ptr<float>(), (const __half*) z.data_ptr(), n, bsz, topk, H,
        (int) E, slotof.data_ptr<int>(), score.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl,
        (__half*) hx, (int*) hpicks, (long long*) seqc.data_ptr<int64_t>(), (int) li, sel_out.data_ptr<int64_t>(),
        w_out.data_ptr<float>(), (long long*) dflag.data_ptr<int64_t>(), (long long*) stats.data_ptr<int64_t>(),
        counts.numel() ? counts.data_ptr<int>() : nullptr, (int*) ring, (long long) ring_cap,
        ringpos.numel() ? (long long*) ringpos.data_ptr<int64_t>() : nullptr, (volatile long long*) ringpos_host,
        bt ? (const signed char*) bt_res.data_ptr() : nullptr, (volatile long long*) ctrl2, (__half*) hx2, (int*) hpicks2,
        bt ? (long long*) seqc2.data_ptr<int64_t>() : nullptr, bt ? (long long*) dflag2.data_ptr<int64_t>() : nullptr);
}

void ft_combine(torch::Tensor out, torch::Tensor dflag, int64_t ctrl, int64_t hout, torch::Tensor stats, int64_t timeout_ns,
                torch::Tensor dflag2, int64_t ctrl2, int64_t hout2)
{
    c10::cuda::CUDAGuard g(out.device());
    auto st = at::cuda::getCurrentCUDAStream(out.device().index());
    TORCH_CHECK(out.is_contiguous() && out.dtype() == at::kFloat, "out fp32 contiguous");
    const int n = (int) out.numel();
    const bool bt = dflag2.numel() > 0 && ctrl2 != 0;
    ft_combine_k<<<16, 256, 0, st>>>(out.data_ptr<float>(), n, (const long long*) dflag.data_ptr<int64_t>(),
        (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns,
        bt ? (const long long*) dflag2.data_ptr<int64_t>() : nullptr, (volatile long long*) ctrl2, (const float*) hout2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ft_split", &ft_split);
    m.def("ft_combine", &ft_combine);
}
