// FreeToken CPU tier: host side (torch CPU extension). Owns the ft_mul1 pool and a worker thread that
// serves one MoE job at a time through pinned, GPU-mapped buffers written by the GPU split kernel
// (ft_tier_cu.cu):
//   ctrl[0] job seq (GPU writes last)   ctrl[1] done seq (worker writes last)
//   ctrl[2] layer   ctrl[3] ntok   ctrl[4] npicks
//   hx    [ntok, H] fp16 expert input      picks [npicks][3] = (token, expert, weight f32 bits)
//   hout  [ntok, H] fp32 weighted routed partial of the CPU picks (the GPU adds it into the MoE output)
// Weights are read in place from the native pinned trellis (the same home copy the GPU cache /
// zero-copy / staging paths use); suh/svh are CPU copies of the GPU's tensors.
#include <torch/extension.h>
#include "ft_core.h"

namespace ft {

struct Stats { std::atomic<long> jobs{0}, experts{0}, tokens{0}, empty{0}; std::atomic<long long> busy_ns{0}, lat_ns{0}; };

struct Tier
{
    Pool pool;
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    volatile int64_t* ctrl = nullptr;
    const uint16_t* hx = nullptr;
    const int32_t* picks = nullptr;
    float* hout = nullptr;
    int mode = -1, H = 4096, I = 2048, swz = 1, band = 0;
    size_t swz_bytes = 0;
    std::thread th;
    std::atomic<bool> quit{false}, running{false};
    Stats st;
    std::vector<float> xf;
};

Tier* T = nullptr;

void tier_init(int64_t threads, std::vector<int64_t> cpus, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout,
               int64_t mode, int64_t H, int64_t I, int64_t nlayers, int64_t swz)
{
    TORCH_CHECK(T == nullptr, "tier already initialised");
    init_perm(); init_tables();
    T = new Tier();
    std::vector<int> c(cpus.begin(), cpus.end());
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    T->pool.start(int(threads), c);          // pins the calling thread to c[0]; the worker thread takes that role
    sched_setaffinity(0, sizeof(saved), &saved);   // give the Python thread its affinity back
    T->ctrl = reinterpret_cast<volatile int64_t*>(ctrl.data_ptr<int64_t>());
    T->hx = reinterpret_cast<const uint16_t*>(hx.data_ptr());
    T->picks = picks.data_ptr<int32_t>();
    T->hout = hout.data_ptr<float>();
    T->keep = { ctrl, hx, picks, hout };
    T->mode = int(mode); T->H = int(H); T->I = int(I); T->swz = int(swz);
    T->layers.resize(nlayers);
}

// ptrs: int64 [E, 3] host addresses of gate/up/down trellis (native layout); scales fp16 CPU [E, k] / [E, n]
void tier_add_layer(int64_t li, torch::Tensor ptrs, torch::Tensor suh_g, torch::Tensor svh_g, torch::Tensor suh_u, torch::Tensor svh_u,
                    torch::Tensor suh_d, torch::Tensor svh_d)
{
    TORCH_CHECK(T, "init first");
    const int E = int(ptrs.size(0)), H = T->H, I = T->I;
    for (auto& t : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d })
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kHalf && t.is_contiguous() && t.size(0) == E, "scales: CPU fp16 contiguous [E, .]");
    TORCH_CHECK(suh_g.size(1) == H && svh_g.size(1) == I && suh_d.size(1) == I && svh_d.size(1) == H, "scale shapes");
    auto p = ptrs.accessor<int64_t, 2>();
    Layer& L = T->layers[li];
    L.H = H; L.I = I; L.ex.resize(E);
    auto u16 = [](const torch::Tensor& t, int e) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(e) * t.size(1); };
    for (int e = 0; e < E; ++e)
    {
        L.ex[e].g = { reinterpret_cast<const uint8_t*>(p[e][0]), u16(suh_g, e), u16(svh_g, e), H, I };
        L.ex[e].u = { reinterpret_cast<const uint8_t*>(p[e][1]), u16(suh_u, e), u16(svh_u, e), H, I };
        L.ex[e].d = { reinterpret_cast<const uint8_t*>(p[e][2]), u16(suh_d, e), u16(svh_d, e), I, H };
    }
    for (auto& t : { ptrs, suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) T->keep.push_back(t);
    if (T->swz)
    {
        // CPU-side block-contiguous copy ([n/128][k/16][8 tiles]): every 128-output work unit streams one
        // contiguous run (C004: 0.114-0.120 vs 0.165-0.172 ms/expert on the native order). Plain (unpinned)
        // huge-page memory; the pinned native home copy stays the GPU's.
        const size_t mb = size_t(H / 16) * (I / 16) * TILE_BYTES;           // bytes per matrix (all three equal)
        uint8_t* buf = static_cast<uint8_t*>(big_alloc(mb * 3 * E));
        T->swz_bytes += mb * 3 * E;
        std::vector<std::thread> ts;
        const int nt = 16;
        for (int w = 0; w < nt; ++w)
            ts.emplace_back([&, w] {
                for (int e = w; e < E; e += nt)
                    for (int k = 0; k < 3; ++k)
                    {
                        Mat& m = k == 0 ? L.ex[e].g : k == 1 ? L.ex[e].u : L.ex[e].d;
                        uint8_t* dst = buf + (size_t(e) * 3 + k) * mb;
                        const int tk_n = m.k / 16, tn = m.n / 16;
                        for (int b = 0; b < tn / 8; ++b)
                            for (int tk = 0; tk < tk_n; ++tk)
                                std::memcpy(dst + (size_t(b) * tk_n + tk) * 768, m.tr + (size_t(tk) * tn + b * 8) * TILE_BYTES, 768);
                        m.tr = dst; m.swz = 1;
                    }
            });
        for (auto& t : ts) t.join();
    }
}

void run_job(int li, int ntok, int np)
{
    const int H = T->H;
    T->xf.resize(size_t(std::max(ntok, 1)) * H);
    const size_t nx = size_t(ntok) * H;
    if (nx < (size_t(1) << 15)) { for (size_t i = 0; i < nx; ++i) T->xf[i] = h2f(T->hx[i]); }
    else
    {
        // dsv41 hybrid: thousands of tokens per job -> convert on the whole pool (8 halves per F16C op)
        struct Cv { const uint16_t* h; float* f; size_t n; } cv{ T->hx, T->xf.data(), nx };
        T->pool.run([](void* c, int w, int nw) {
            auto* a = static_cast<Cv*>(c);
            const size_t per = ((a->n + size_t(nw) - 1) / size_t(nw) + 7) & ~size_t(7);
            const size_t lo = std::min(a->n, per * size_t(w)), hi = std::min(a->n, lo + per);
            size_t i = lo;
            for (; i + 8 <= hi; i += 8)
                _mm256_storeu_ps(a->f + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a->h + i))));
            for (; i < hi; ++i) a->f[i] = h2f(a->h[i]);
        }, &cv);
    }
    std::vector<std::vector<std::pair<int, float>>> route(ntok);
    for (int k = 0; k < np; ++k)
    {
        const int t = T->picks[k * 3], e = T->picks[k * 3 + 1]; float w; std::memcpy(&w, &T->picks[k * 3 + 2], 4);
        route[t].push_back({ e, w });
    }
    if (np == 0) { std::memset(T->hout, 0, size_t(ntok) * H * 4); T->st.empty++; return; }
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int mode = T->mode >= 0 ? T->mode : (maxm <= 2 ? 2 : 0);
    if (T->band) moe_forward_band(T->pool, T->layers[li], T->xf.data(), ntok, route, T->hout, mode);
    else moe_forward(T->pool, T->layers[li], T->xf.data(), ntok, route, T->hout, mode);
    T->st.experts += np; T->st.tokens += ntok;
}

void worker_main(int cpu0)
{
    Pool::pin(cpu0);
    int64_t last = T->ctrl[0];
    long idle = 0;
    while (!T->quit.load(std::memory_order_relaxed))
    {
        const int64_t s = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
        if (s == last)
        {
            if (++idle < 400000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
        }
        idle = 0;
        const double t0 = now();
        run_job(int(T->ctrl[2]), int(T->ctrl[3]), int(T->ctrl[4]));
        T->st.busy_ns += (long long) ((now() - t0) * 1e9);
        T->st.jobs++;
        last = s;
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 1), s, __ATOMIC_RELEASE);
    }
}

void tier_start(int64_t cpu0)
{
    TORCH_CHECK(T && !T->running, "init first / already running");
    T->running = true;
    T->th = std::thread(worker_main, int(cpu0));
}

void tier_stop()
{
    if (!T) return;
    T->quit = true;
    if (T->th.joinable()) T->th.join();
    T->pool.stop();
}

// Synchronous forward for tests (worker must not be running): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
torch::Tensor tier_forward(int64_t li, torch::Tensor x, torch::Tensor sel, torch::Tensor w, int64_t mode)
{
    TORCH_CHECK(T && !T->running, "tier_forward: worker running");
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, T->H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j) if (sa[t][j] >= 0) route[t].push_back({ sa[t][j], wa[t][j] });
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int md = mode >= 0 ? int(mode) : (maxm <= 2 ? 2 : 0);
    if (T->band) moe_forward_band(T->pool, T->layers[li], x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    else moe_forward(T->pool, T->layers[li], x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    return out;
}

// C062: repack the pinned home copy IN PLACE from native [tk][tn][96 B] to block-contiguous [tn/8][tk][768 B]
// (ptrs: int64 [N] matrix base addresses, all [k/16, n/16, 48]); 16 threads, one 3 MB scratch each
void tier_swizzle_home(torch::Tensor ptrs, int64_t k, int64_t n, int64_t threads)
{
    auto p = ptrs.accessor<int64_t, 1>();
    const int N = int(ptrs.size(0)), tk_n = int(k / 16), tn = int(n / 16);
    const size_t mb = size_t(tk_n) * tn * TILE_BYTES;
    std::vector<std::thread> ts;
    for (int w = 0; w < threads; ++w)
        ts.emplace_back([&, w] {
            std::vector<uint8_t> tmp(mb);
            for (int i = w; i < N; i += int(threads))
            {
                uint8_t* m = reinterpret_cast<uint8_t*>(p[i]);
                for (int b = 0; b < tn / 8; ++b)
                    for (int t = 0; t < tk_n; ++t)
                        std::memcpy(tmp.data() + (size_t(b) * tk_n + t) * 768, m + (size_t(t) * tn + b * 8) * TILE_BYTES, 768);
                std::memcpy(m, tmp.data(), mb);
            }
        });
    for (auto& t : ts) t.join();
}

// C062: the registered layers already point at the (now swizzled) home copy: mark them block-contiguous
void tier_mark_swz()
{
    for (auto& L : T->layers)
        for (auto& E : L.ex) { E.g.swz = 1; E.u.swz = 1; E.d.swz = 1; }
}

void tier_set_mode(int64_t mode) { if (T) T->mode = int(mode); }
void tier_set_band(int64_t band) { if (T) T->band = int(band); }   // D109: split-K row-band forward on the native layout
void tier_set_limit(double limit) { g_act_limit = float(limit); }   // B004

std::vector<int64_t> tier_stats()
{
    if (!T) return {};
    return { T->st.jobs.load(), T->st.experts.load(), T->st.tokens.load(), T->st.empty.load(), (int64_t) T->st.busy_ns.load(), (int64_t) T->swz_bytes };
}

}  // namespace ft

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tier_init", &ft::tier_init);
    m.def("tier_add_layer", &ft::tier_add_layer);
    m.def("tier_start", &ft::tier_start);
    m.def("tier_stop", &ft::tier_stop);
    m.def("tier_forward", &ft::tier_forward);
    m.def("tier_stats", &ft::tier_stats);
    m.def("tier_set_mode", &ft::tier_set_mode);
    m.def("tier_set_band", &ft::tier_set_band);
    m.def("tier_set_limit", &ft::tier_set_limit);
    m.def("tier_swizzle_home", &ft::tier_swizzle_home);
    m.def("tier_mark_swz", &ft::tier_mark_swz);
}
