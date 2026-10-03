// DSV41 CPU tier, GPU side (vLLM port of kernels/cpu_avx2/ft_tier_cu.cu). Differences from the GLM version:
//  * CUDA-graph safe: the job sequence number lives in a device counter (seqc) instead of a host argument,
//    so a captured graph publishes a fresh seq on every replay (vLLM FULL_DECODE_ONLY graphs)
//  * routing weights are fp32 in and out (vLLM topk_weights), so the GPU share keeps its exact weights and the
//    CPU share gets the fp32 weight bits
//  * per-layer routing counts (int32 [E]) for the VRAM mirror cache (ct_vllm.py EC)
//  * ft_combine adds into the fp32 routed accumulator of apply_exl3_fused_moe
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// policy p[]: 0 t_zc (ms per zero-copy miss), 1 t_hit (ms per GPU-resident expert), 2 cpu_a (ms fixed per CPU job),
// 3 cpu_b (ms per CPU expert), 4 cpu_tok (extra fraction per extra token of an expert), 5 max_cpu, 6 force_n (-1 = cost
// model), 7 handshake (1 = publish even empty jobs), 8 x already in hx (skip the in-kernel host copy)
__global__ void ft_split_k(const int64_t* __restrict__ sel, const float* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int topk, int H, int E, const int* __restrict__ slotof, const float* __restrict__ score,
    const float* __restrict__ p, volatile long long* ctrl, __half* hx, int* hpicks, long long* seqc, int li,
    int64_t* sel_out, float* w_out, long long* dflag, long long* stats, int* counts)
{
    __shared__ int cnt[1024];
    __shared__ unsigned char cpu[1024];
    __shared__ int npk, publish;
    __shared__ long long s_seq;
    __shared__ int s_miss[1024];
    const int t = threadIdx.x;
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; cpu[e] = 0; }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        if (w[s] != 0.0f && e >= 0 && e < E) { atomicAdd(&cnt[e], 1); if (counts) atomicAdd(&counts[e], 1); }
    }
    __syncthreads();
    if (t == 0)
    {
        int nm = 0, nh = 0;
        int* miss = s_miss;
        for (int e = 0; e < E; ++e)
        {
            if (!cnt[e]) continue;
            if (slotof[e] >= 0) { ++nh; continue; }
            int b = nm++;
            while (b > 0 && score[miss[b - 1]] > score[e]) { miss[b] = miss[b - 1]; --b; }
            miss[b] = e;
        }
        int best = 0;
        const int force = (int) p[6], maxc = min((int) p[5], nm);
        if (force >= 0) best = min(force, nm);
        else
        {
            float bt = 1e30f, c = 0.0f;
            for (int k = 0; k <= maxc; ++k)
            {
                if (k) c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1));
                const float gpu = p[1] * (nh + nm - k) + p[0] * (nm - k);
                const float tt = fmaxf(gpu, k ? p[2] + c : 0.0f);
                if (tt < bt - 1e-6f) { bt = tt; best = k; }
            }
        }
        for (int k = 0; k < best; ++k) cpu[miss[k]] = 1;
        publish = best > 0 || p[7] > 0.5f;
        stats[0] += nh; stats[1] += nm; stats[2] += best; stats[3] += 1;
        npk = 0;
        if (publish)
        {
            s_seq = ++seqc[0];
            for (int s = 0; s < n; ++s)
            {
                const long long e = sel[s];
                if (e < 0 || e >= E || !cpu[e] || w[s] == 0.0f) continue;
                hpicks[npk * 3] = s / topk; hpicks[npk * 3 + 1] = (int) e; hpicks[npk * 3 + 2] = __float_as_int(w[s]);
                ++npk;
            }
        }
    }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        const bool c = e >= 0 && e < E && cpu[e];
        sel_out[s] = c ? -1 : e;
        w_out[s] = c ? 0.0f : w[s];
    }
    if (publish)
    {
        const int nx = p[8] > 0.5f ? 0 : bsz * H / 8;   // p[8] = 1: x already in hx (copy-engine DtoH before this kernel)
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
        __threadfence_system();
        __syncthreads();
        if (t == 0)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = s_seq;
            __threadfence_system();
            dflag[0] = s_seq;
        }
    }
    else if (t == 0) dflag[0] = 0;
}

__global__ void ft_combine_k(float* out, int n, const long long* __restrict__ dflag, volatile long long* ctrl,
    const float* hout, long long* stats, long long timeout_ns)
{
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0;
        if (go)
        {
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            t1 = t0;
            while (ctrl[1] < s)
            {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
                if ((long long) (t1 - t0) > timeout_ns) { go = 0; if (blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[6], 1ull); break; }
            }
            if (blockIdx.x == 0) { atomicAdd((unsigned long long*) &stats[4], (unsigned long long) (t1 - t0)); atomicAdd((unsigned long long*) &stats[5], 1ull); }
        }
    }
    __syncthreads();
    if (!go) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] += __ldcv(hout + i);
}

void ft_split(torch::Tensor sel, torch::Tensor w, torch::Tensor z, int64_t E, torch::Tensor slotof, torch::Tensor score,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, torch::Tensor seqc, int64_t li,
              torch::Tensor sel_out, torch::Tensor w_out, torch::Tensor dflag, torch::Tensor stats, torch::Tensor counts)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kFloat && z.dtype() == at::kHalf && z.is_contiguous(), "dtypes");
    const int bsz = (int) z.size(0), H = (int) z.size(1), n = (int) sel.numel(), topk = n / bsz;
    TORCH_CHECK(E <= 1024 && H % 8 == 0, "ft_split: E <= 1024");
    ft_split_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), w.data_ptr<float>(), (const __half*) z.data_ptr(), n, bsz, topk, H,
        (int) E, slotof.data_ptr<int>(), score.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl,
        (__half*) hx, (int*) hpicks, (long long*) seqc.data_ptr<int64_t>(), (int) li, sel_out.data_ptr<int64_t>(),
        w_out.data_ptr<float>(), (long long*) dflag.data_ptr<int64_t>(), (long long*) stats.data_ptr<int64_t>(),
        counts.numel() ? counts.data_ptr<int>() : nullptr);
}

void ft_combine(torch::Tensor out, torch::Tensor dflag, int64_t ctrl, int64_t hout, torch::Tensor stats, int64_t timeout_ns)
{
    c10::cuda::CUDAGuard g(out.device());
    auto st = at::cuda::getCurrentCUDAStream(out.device().index());
    TORCH_CHECK(out.is_contiguous() && out.dtype() == at::kFloat, "out fp32 contiguous");
    const int n = (int) out.numel();
    ft_combine_k<<<16, 256, 0, st>>>(out.data_ptr<float>(), n, (const long long*) dflag.data_ptr<int64_t>(),
        (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ft_split", &ft_split);
    m.def("ft_combine", &ft_combine);
}
