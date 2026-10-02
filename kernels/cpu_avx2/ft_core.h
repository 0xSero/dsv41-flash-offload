// FreeToken CPU tier core (AVX2 mul1 K=3 expert kernels, pool, reference). See ft_mul1_avx2.cpp for notes.
#pragma once
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t MUL1 = 0x83DCD12Du;
constexpr float HAD = 0.088388347648f;
constexpr int BITS = 3;
constexpr int TILE_BYTES = 16 * BITS * 2;   // 96
constexpr int MAXM = 8;

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }
inline float r16(float f) { return h2f(f2h(f)); }

const float KINV = h2f(0x1eee);
const float KBIAS = h2f(0xc931);
const float CAFF = 1024.0f * h2f(0x1eee) + h2f(0xc931);   // exact in fp32 (multiple of 2^-8, |.| < 4)

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ------------------------------------------------------------------------------------------
// Format: scalar reference decode (mirrors decode_state_scalar + make_tc_perm of exllamav3)
// ------------------------------------------------------------------------------------------

int PERM[256];   // stream index -> r*16 + c

void init_perm()
{
    for (int t = 0; t < 32; ++t)
    {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        PERM[t * 8 + 0] = r0 * 16 + c0; PERM[t * 8 + 1] = r1 * 16 + c0;
        PERM[t * 8 + 2] = r2 * 16 + c0; PERM[t * 8 + 3] = r3 * 16 + c0;
        PERM[t * 8 + 4] = r0 * 16 + c1; PERM[t * 8 + 5] = r1 * 16 + c1;
        PERM[t * 8 + 6] = r2 * 16 + c1; PERM[t * 8 + 7] = r3 * 16 + c1;
    }
}

inline uint32_t ld32(const uint16_t* p, int i) { uint32_t v; std::memcpy(&v, p + i * 2, 4); return v; }

uint16_t state_scalar(const uint16_t* packed, int t)
{
    constexpr int words32 = BITS * 256 / 32;
    const int b0 = t * BITS + BITS - 16 + 256 * BITS;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const uint64_t merged = (uint64_t(ld32(packed, (b0 / 32) % words32)) << 32) | ld32(packed, ((b1 - 1) / 32) % words32);
    return uint16_t(merged >> shift);
}

inline int bytesum(uint16_t st)
{
    const uint32_t x = uint32_t(st) * MUL1;
    return (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
}

// GPU decode_mul1_product_2: hfma(half(1024 + sum), k_inv, k_bias), one fp16 rounding
inline float w_gpu(uint16_t st) { return r16(std::fma(float(1024 + bytesum(st)), KINV, KBIAS)); }
inline float w_gpu_s(int s) { return r16(std::fma(float(1024 + s), KINV, KBIAS)); }

// ------------------------------------------------------------------------------------------
// Transforms (fp32 math, fp16 rounding at the GPU's storage points)
// ------------------------------------------------------------------------------------------

void had128(float* v)
{
    for (int w = 1; w < 128; w *= 2)
        for (int b = 0; b < 128; b += 2 * w)
            for (int i = 0; i < w; ++i) { const float a = v[b + i], c = v[b + w + i]; v[b + i] = a + c; v[b + w + i] = a - c; }
}

// x (fp16-valued fp32) -> round16(had(x * suh) * HAD)
void prep_block(const float* x, const uint16_t* suh, float* xt)
{
    for (int i = 0; i < 128; ++i) xt[i] = x[i] * h2f(suh[i]);
    had128(xt);
    for (int i = 0; i < 128; ++i) xt[i] = r16(xt[i] * HAD);
}

// GEMM output (fp32) -> round16 -> had -> * HAD * svh -> round16
void out_block(float* y, const uint16_t* svh)
{
    for (int i = 0; i < 128; ++i) y[i] = r16(y[i]);
    had128(y);
    for (int i = 0; i < 128; ++i) y[i] = r16(y[i] * HAD * h2f(svh[i]));
}

// permuted activation layout for the fast kernel: xp[tk][q][lane] = xt[tk*16 + row(q, lane)]
inline void permute_rows(const float* xt16, float* xp32)   // one tile row: 16 in, 32 out
{
    for (int q = 0; q < 4; ++q)
    {
        const int r0 = 2 * q;
        const int rows[8] = { r0, r0 + 1, r0 + 8, r0 + 9, r0, r0 + 1, r0 + 8, r0 + 9 };
        for (int l = 0; l < 8; ++l) xp32[q * 8 + l] = xt16[rows[l]];
    }
}

inline float silu(float g) { return g / (1.0f + std::exp(-g)); }
// B004: swiglu clamp with exllamav3 act_gate semantics (exl3_moe_coop_kernel.cuh: u clamped to [-limit, limit],
// silu(g) clamped from above). GLM-5.3-Flash has swiglu_limit 10; 0 = off (pre-B004 behaviour). Set by tier_set_limit.
inline float g_act_limit = 0.0f;
inline float act_gu(float g, float u)
{
    float x = silu(g);
    const float L = g_act_limit;
    if (L != 0.0f) { u = std::fmin(std::fmax(u, -L), L); x = std::fmin(x, L); }
    return x * u;
}

// ------------------------------------------------------------------------------------------
// Fast AVX2 kernel
// ------------------------------------------------------------------------------------------

alignas(32) uint8_t CTRL[32][32];
int WOFF[32];
alignas(32) int32_t SHV[8];

inline int mem_of_stream(int q) { q = ((q % 96) + 96) % 96; return 4 * (q / 4) + 3 - q % 4; }

void init_tables()
{
    for (int j = 0; j < 8; ++j) SHV[j] = 16 - (3 + 3 * j) % 8;
    for (int g = 0; g < 32; ++g)
    {
        const int q0 = 3 * g - 2;
        int win[16];                 // window byte -> mem byte
        if (g == 0) { for (int b = 0; b < 4; ++b) win[b] = 92 + b; for (int b = 4; b < 16; ++b) win[b] = b - 4; WOFF[g] = -1; }
        else { const int L = 4 * (q0 / 4); for (int b = 0; b < 16; ++b) win[b] = L + b; WOFF[g] = L; }
        auto wpos = [&](int q) -> int {
            if (q >= 96) return -1;  // only ever the unused third byte of g=31 lane 7
            const int m = mem_of_stream(q);
            for (int b = 0; b < 16; ++b) if (win[b] == m) return b;
            return -2;
        };
        for (int j = 0; j < 8; ++j)
        {
            const int fb = (3 + 3 * j) / 8;
            const int src[4] = { -1, wpos(q0 + fb + 2), wpos(q0 + fb + 1), wpos(q0 + fb) };   // lane bytes 0..3
            for (int b = 0; b < 4; ++b)
            {
                if (src[b] == -2) { fprintf(stderr, "table error g=%d j=%d\n", g, j); exit(1); }
                CTRL[g][(j < 4 ? 0 : 16) + (j % 4) * 4 + b] = src[b] < 0 ? 0x80 : uint8_t(src[b]);
            }
        }
    }
}

struct Consts
{
    __m256i m16, mul, one8, one16, sh, expm, magic;
    __m256 kinv, caff;
};

inline Consts consts()
{
    return { _mm256_set1_epi32(0xffff), _mm256_set1_epi32(int(MUL1)), _mm256_set1_epi8(1), _mm256_set1_epi16(1),
             _mm256_load_si256(reinterpret_cast<const __m256i*>(SHV)), _mm256_set1_epi32(0x7F800000), _mm256_set1_epi32(0x06C00000),
             _mm256_set1_ps(KINV), _mm256_set1_ps(CAFF) };
}

template <int g>
__attribute__((always_inline)) inline __m256 decode8(const uint8_t* tile, const Consts& K)
{
    __m256i win;
    if constexpr (g == 0)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + 80));
        win = _mm256_broadcastsi128_si256(_mm_alignr_epi8(a, b, 12));
    }
    else
    {
        constexpr int L = 4 * ((3 * g - 2) / 4);
        win = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + L)));
    }
    const __m256i st = _mm256_and_si256(_mm256_srlv_epi32(
        _mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRL[g]))), K.sh), K.m16);
    const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_mullo_epi32(st, K.mul), K.one8), K.one16);
    return _mm256_cvtepi32_ps(s);
}

template <int g>
__attribute__((always_inline)) inline __m256i decode8p(const uint8_t* tile, const Consts& K)
{
    __m256i win;
    if constexpr (g == 0)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + 80));
        win = _mm256_broadcastsi128_si256(_mm_alignr_epi8(a, b, 12));
    }
    else
    {
        constexpr int L = 4 * ((3 * g - 2) / 4);
        win = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + L)));
    }
    const __m256i st = _mm256_and_si256(_mm256_srlv_epi32(
        _mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRL[g]))), K.sh), K.m16);
    return _mm256_maddubs_epi16(_mm256_mullo_epi32(st, K.mul), K.one8);
}

// acc layout per tile: [c 8][m][8 lanes]; X per token: [q 4][8] for this tile row
template <int M, bool EXACT, int C = 0>
__attribute__((always_inline)) inline void tile_accum(const uint8_t* tile, const float* const* X, float* acc, const Consts& K)
{
    if constexpr (C < 8)
    {
        __m256 a[M];
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i) a[i] = _mm256_loadu_ps(acc + (C * M + i) * 8);
        auto step = [&](__m256 s, int q) {
            __m256 w = s;
            if constexpr (EXACT)
            {
                // round-to-nearest-even to fp16 precision without the (slow on Zen 3) cvtps2ph/cvtph2ps
                // pair: c = 1.5 * 2^(e+13) from x's own exponent e, (x + c) - c rounds x to 2^(e-10) =
                // the fp16 ulp of x. Exact for this codebook: |w| < 3.5, values are multiples of 2^-18,
                // so fp16 subnormals (|w| < 2^-14) are representable and the finer grid is a no-op.
                const __m256 v = _mm256_fmadd_ps(s, K.kinv, K.caff);
                const __m256 c = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_and_si256(_mm256_castps_si256(v), K.expm), K.magic));
                w = _mm256_sub_ps(_mm256_add_ps(v, c), c);
            }
            #pragma GCC unroll 8
            for (int i = 0; i < M; ++i) a[i] = _mm256_fmadd_ps(w, _mm256_loadu_ps(X[i] + q * 8), a[i]);
        };
        step(decode8<4 * C + 0>(tile, K), 0);
        step(decode8<4 * C + 1>(tile, K), 1);
        step(decode8<4 * C + 2>(tile, K), 2);
        step(decode8<4 * C + 3>(tile, K), 3);
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i) _mm256_storeu_ps(acc + (C * M + i) * 8, a[i]);
        tile_accum<M, EXACT, C + 1>(tile, X, acc, K);
    }
}

// swz = 1: block-contiguous copy ([n/128][k/16][8 tiles]) so each work unit streams one contiguous run
struct Mat { const uint8_t* tr; const uint16_t* suh; const uint16_t* svh; int k, n; int swz = 0; };

int PF_DIST = 6;

// One 128-output block (8 tiles) of `mat` for M tokens. xp[i]: permuted activations of token i
// ([k/16][32]); sumx[i]: sum of token i's transformed activations (AFFINE fixup). out[i*128 + ..]
template <int M, bool EXACT>
void gemv_block(const Mat& mat, int blk, const float* const* xp, const float* sumx, float* out)
{
    alignas(64) float acc[8][8 * M * 8];
    std::memset(acc, 0, sizeof(acc));
    const Consts K = consts();
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t row_stride = mat.swz ? size_t(8) * TILE_BYTES : size_t(tiles_n) * TILE_BYTES;
    const uint8_t* p = mat.swz ? mat.tr + size_t(blk) * tiles_k * 8 * TILE_BYTES : mat.tr + size_t(blk) * 8 * TILE_BYTES;
    const float* X[M];
    for (int tk = 0; tk < tiles_k; ++tk, p += row_stride)
    {
        const uint8_t* pf = p + PF_DIST * row_stride;
        #pragma GCC unroll 12
        for (int l = 0; l < 12; ++l) _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);
        for (int i = 0; i < M; ++i) X[i] = xp[i] + size_t(tk) * 32;
        #pragma GCC unroll 1
        for (int t = 0; t < 8; ++t) tile_accum<M, EXACT>(p + t * TILE_BYTES, X, acc[t], K);
    }
    for (int i = 0; i < M; ++i)
        for (int t = 0; t < 8; ++t)
            for (int c = 0; c < 8; ++c)
            {
                const float* v = acc[t] + (c * M + i) * 8;
                float lo = (v[0] + v[1]) + (v[2] + v[3]), hi = (v[4] + v[5]) + (v[6] + v[7]);
                if constexpr (!EXACT) { lo = KINV * lo + CAFF * sumx[i]; hi = KINV * hi + CAFF * sumx[i]; }
                out[i * 128 + t * 16 + c] = lo;
                out[i * 128 + t * 16 + c + 8] = hi;
            }
}

template <bool EXACT>
void gemv_block_m(int m, const Mat& mat, int blk, const float* const* xp, const float* sumx, float* out)
{
    switch (m)
    {
        case 1: gemv_block<1, EXACT>(mat, blk, xp, sumx, out); break;
        case 2: gemv_block<2, EXACT>(mat, blk, xp, sumx, out); break;
        case 3: gemv_block<3, EXACT>(mat, blk, xp, sumx, out); break;
        case 4: gemv_block<4, EXACT>(mat, blk, xp, sumx, out); break;
        case 5: gemv_block<5, EXACT>(mat, blk, xp, sumx, out); break;
        case 6: gemv_block<6, EXACT>(mat, blk, xp, sumx, out); break;
        case 7: gemv_block<7, EXACT>(mat, blk, xp, sumx, out); break;
        default: gemv_block<8, EXACT>(mat, blk, xp, sumx, out); break;
    }
}

// I16 mode: activations as int16 (one scale per 128-row block, |x| <= 16383 so an i32 block sum of
// bytesum(<=1020) * x over 128 rows cannot overflow), accumulate = vpmaddwd of the product-byte pair
// sums against x duplicated in both 16-bit halves (no cvt / fma per weight); flushed to fp32 with the
// block scale every 8 tile rows. Weights as AFFINE (unrounded codebook value, one fixup per output).
template <int M, int C = 0>
__attribute__((always_inline)) inline void tile_accum_i16(const uint8_t* tile, const int32_t* const* X, int32_t* acc, const Consts& K)
{
    if constexpr (C < 8)
    {
        __m256i a[M];
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i) a[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + (C * M + i) * 8));
        __m256i p[4];
        p[0] = decode8p<4 * C + 0>(tile, K); p[1] = decode8p<4 * C + 1>(tile, K);
        p[2] = decode8p<4 * C + 2>(tile, K); p[3] = decode8p<4 * C + 3>(tile, K);
        #pragma GCC unroll 4
        for (int q = 0; q < 4; ++q)
            #pragma GCC unroll 8
            for (int i = 0; i < M; ++i)
                a[i] = _mm256_add_epi32(a[i], _mm256_madd_epi16(p[q], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + q * 8))));
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i) _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + (C * M + i) * 8), a[i]);
        tile_accum_i16<M, C + 1>(tile, X, acc, K);
    }
}

template <int M>
void gemv_block_i16(const Mat& mat, int blk, const int32_t* const* xq, const float* const* qs, const float* sumx, float* out)
{
    alignas(64) int32_t acc[8][8 * M * 8];
    alignas(64) float accf[8][8 * M * 8];
    std::memset(acc, 0, sizeof(acc));
    std::memset(accf, 0, sizeof(accf));
    const Consts K = consts();
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t row_stride = mat.swz ? size_t(8) * TILE_BYTES : size_t(tiles_n) * TILE_BYTES;
    const uint8_t* p = mat.swz ? mat.tr + size_t(blk) * tiles_k * 8 * TILE_BYTES : mat.tr + size_t(blk) * 8 * TILE_BYTES;
    const int32_t* X[M];
    for (int tk = 0; tk < tiles_k; ++tk, p += row_stride)
    {
        const uint8_t* pf = p + PF_DIST * row_stride;
        #pragma GCC unroll 12
        for (int l = 0; l < 12; ++l) _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);
        for (int i = 0; i < M; ++i) X[i] = xq[i] + size_t(tk) * 32;
        #pragma GCC unroll 1
        for (int t = 0; t < 8; ++t) tile_accum_i16<M>(p + t * TILE_BYTES, X, acc[t], K);
        if ((tk & 7) == 7)
        {
            for (int t = 0; t < 8; ++t)
                for (int c = 0; c < 8; ++c)
                    for (int i = 0; i < M; ++i)
                    {
                        const __m256 sc = _mm256_set1_ps(qs[i][tk >> 3]);
                        int32_t* ai = acc[t] + (c * M + i) * 8; float* af = accf[t] + (c * M + i) * 8;
                        _mm256_store_ps(af, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(ai))), sc, _mm256_load_ps(af)));
                        _mm256_store_si256(reinterpret_cast<__m256i*>(ai), _mm256_setzero_si256());
                    }
        }
    }
    for (int i = 0; i < M; ++i)
        for (int t = 0; t < 8; ++t)
            for (int c = 0; c < 8; ++c)
            {
                const float* v = accf[t] + (c * M + i) * 8;
                const float lo = (v[0] + v[1]) + (v[2] + v[3]), hi = (v[4] + v[5]) + (v[6] + v[7]);
                out[i * 128 + t * 16 + c] = KINV * lo + CAFF * sumx[i];
                out[i * 128 + t * 16 + c + 8] = KINV * hi + CAFF * sumx[i];
            }
}

void gemv_block_i16_m(int m, const Mat& mat, int blk, const int32_t* const* xq, const float* const* qs, const float* sumx, float* out)
{
    switch (m)
    {
        case 1: gemv_block_i16<1>(mat, blk, xq, qs, sumx, out); break;
        case 2: gemv_block_i16<2>(mat, blk, xq, qs, sumx, out); break;
        case 3: gemv_block_i16<3>(mat, blk, xq, qs, sumx, out); break;
        case 4: gemv_block_i16<4>(mat, blk, xq, qs, sumx, out); break;
        case 5: gemv_block_i16<5>(mat, blk, xq, qs, sumx, out); break;
        case 6: gemv_block_i16<6>(mat, blk, xq, qs, sumx, out); break;
        case 7: gemv_block_i16<7>(mat, blk, xq, qs, sumx, out); break;
        default: gemv_block_i16<8>(mat, blk, xq, qs, sumx, out); break;
    }
}

// quantise one transformed 128-block to int16 (dup layout, permuted) -> returns scale; adds sum(x_deq) to *sum
inline float quant_block_i16(const float* xt, int32_t* xq_blk /* 8 tile rows x 32 */, float* sum)
{
    float amax = 0; for (int r = 0; r < 128; ++r) amax = std::max(amax, std::fabs(xt[r]));
    const float sc = amax > 0 ? amax / 16383.0f : 1.0f, rs = 1.0f / sc;
    int xi[128]; long si = 0;
    for (int r = 0; r < 128; ++r) { xi[r] = int(std::nearbyint(xt[r] * rs)); si += xi[r]; }
    for (int tt = 0; tt < 8; ++tt)
        for (int q = 0; q < 4; ++q)
        {
            const int r0 = 2 * q; const int rows[8] = { r0, r0 + 1, r0 + 8, r0 + 9, r0, r0 + 1, r0 + 8, r0 + 9 };
            for (int l = 0; l < 8; ++l) { const uint32_t v = uint16_t(int16_t(xi[tt * 16 + rows[l]])); xq_blk[tt * 32 + q * 8 + l] = int32_t(v | (v << 16)); }
        }
    *sum += float(si) * sc;
    return sc;
}

// ------------------------------------------------------------------------------------------
// Thread pool (pinned, spinning; master = worker 0)
// ------------------------------------------------------------------------------------------

struct Pool
{
    int n = 1;
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done{0};
    void (*fn)(void*, int, int) = nullptr;
    void* ctx = nullptr;
    std::atomic<bool> quit{false};

    static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }

    void start(int threads, const std::vector<int>& cpus)
    {
        n = threads;
        pin(cpus[0]);
        for (int w = 1; w < n; ++w)
            th.emplace_back([this, w, cpu = cpus[w % cpus.size()]] {
                pin(cpu);
                uint64_t seen = 0;
                while (true)
                {
                    uint64_t g;
                    long idle = 0;
                    while ((g = gen.load(std::memory_order_acquire)) == seen)
                    {
                        if (quit.load(std::memory_order_relaxed)) return;
                        // spin ~2 ms (covers the gap between MoE layers in decode), then nap
                        if (++idle < 200000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20));
                    }
                    seen = g;
                    fn(ctx, w, n);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    void run(void (*f)(void*, int, int), void* c)
    {
        fn = f; ctx = c; done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_acq_rel);
        f(c, 0, n);
        while (done.load(std::memory_order_acquire) != n - 1) _mm_pause();
    }
    void stop() { quit = true; for (auto& t : th) t.join(); }
};

std::vector<int> cpu_order()   // physical cores first, SMT siblings after
{
    std::vector<int> first, sib; std::vector<std::pair<int, int>> seen;
    const int ncpu = int(std::thread::hardware_concurrency());
    for (int c = 0; c < ncpu; ++c)
    {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_id"); int core = -1; f >> core;
        std::ifstream f2("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id"); int pk = 0; f2 >> pk;
        auto key = std::make_pair(pk, core);
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) { seen.push_back(key); first.push_back(c); } else sib.push_back(c);
    }
    first.insert(first.end(), sib.begin(), sib.end());
    return first;
}

// ------------------------------------------------------------------------------------------
// Layer storage + MoE forward (m tokens, per-token list of (expert, weight))
// ------------------------------------------------------------------------------------------

struct Expert { Mat g, u, d; };

struct Layer
{
    std::vector<Expert> ex;
    std::vector<uint8_t*> blobs;
    int H = 4096, I = 2048;
};

void* big_alloc(size_t bytes)
{
    void* p = nullptr;
    const size_t al = size_t(2) << 20;
    if (posix_memalign(&p, al, (bytes + al - 1) / al * al)) { perror("alloc"); exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

void swizzle_layer(Layer& L);
Layer load_layer(const std::string& manifest, int max_experts)
{
    struct Row { int e; char proj; std::string kind, file; size_t off, nb; };
    std::vector<Row> rows;
    std::ifstream f(manifest);
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream ss(line); Row r; std::string pj; int d0, d1, d2;
        ss >> r.e >> pj >> r.kind >> r.file >> r.off >> r.nb >> d0 >> d1 >> d2; r.proj = pj[0];
        if (r.e < max_experts) rows.push_back(r);
    }
    int E = 0; for (auto& r : rows) E = std::max(E, r.e + 1);
    Layer L; L.ex.resize(E);
    // per expert one blob: g,u,d trellis (3,145,728 B each) + 64 B pad, then scales
    const size_t tb = 3145728, pad = 64;
    const size_t per = 3 * (tb + pad) + (4096 + 2048) * 2 * 3;
    uint8_t* all = static_cast<uint8_t*>(big_alloc(per * E + 4096));
    for (int e = 0; e < E; ++e) L.blobs.push_back(all + per * e);
    for (auto& r : rows)
    {
        uint8_t* base = L.blobs[r.e];
        const int pi = r.proj == 'g' ? 0 : r.proj == 'u' ? 1 : 2;
        uint8_t* dst;
        if (r.kind == "trellis") dst = base + pi * (tb + pad);
        else
        {
            uint8_t* sc = base + 3 * (tb + pad) + pi * (4096 + 2048) * 2;
            dst = r.kind == "suh" ? sc : sc + (pi == 2 ? 2048 : 4096) * 2;
        }
        int fd = open(r.file.c_str(), O_RDONLY);
        size_t got = 0;
        while (got < r.nb) { ssize_t k = pread(fd, dst + got, r.nb - got, r.off + got); if (k <= 0) { perror("pread"); exit(1); } got += k; }
        close(fd);
    }
    for (int e = 0; e < E; ++e)
    {
        uint8_t* base = L.blobs[e];
        auto sc = [&](int pi) { return reinterpret_cast<const uint16_t*>(base + 3 * (tb + pad) + pi * (4096 + 2048) * 2); };
        L.ex[e].g = { base, sc(0), sc(0) + 4096, 4096, 2048 };
        L.ex[e].u = { base + (tb + pad), sc(1), sc(1) + 4096, 4096, 2048 };
        L.ex[e].d = { base + 2 * (tb + pad), sc(2), sc(2) + 2048, 2048, 4096 };
    }
    if (getenv("FT_SWZ") && atoi(getenv("FT_SWZ")) && !getenv("FT_NO_LOAD_SWZ")) swizzle_layer(L);
    return L;
}

void swizzle_layer(Layer& L)
{
    const size_t tb = 3145728; const int E = int(L.ex.size());
    {
        std::vector<uint8_t> tmp(tb);
        for (int e = 0; e < E; ++e)
            for (Mat* m : { &L.ex[e].g, &L.ex[e].u, &L.ex[e].d })
            {
                const int tk_n = m->k / 16, tn = m->n / 16;
                uint8_t* src = const_cast<uint8_t*>(m->tr);
                for (int b = 0; b < tn / 8; ++b)
                    for (int tk = 0; tk < tk_n; ++tk)
                        std::memcpy(tmp.data() + (size_t(b) * tk_n + tk) * 768, src + (size_t(tk) * tn + b * 8) * TILE_BYTES, 768);
                std::memcpy(src, tmp.data(), tb);
                m->swz = 1;
            }
    }
}

struct Job { int e, m; int tok[MAXM]; float w[MAXM]; };

struct Fwd
{
    const Layer* L;
    const float* x;          // [ntok][H], fp16-valued
    float* out;              // [ntok][H]
    int ntok;
    int mode;   // 0 AFFINE fp32, 1 EXACT fp32, 2 I16
    std::vector<Job> jobs;
    // workspaces
    std::vector<float> xg, xu, xd, sg, su, sd, yd, qg, qu, qd;
    std::atomic<int> next{0};
    int phase = 0, units = 0;
};

void fwd_phase(void* vc, int w, int nw)
{
    Fwd& F = *static_cast<Fwd*>(vc);
    const Layer& L = *F.L;
    const int H = L.H, I = L.I;
    const size_t XH = size_t(H) * 2, XI = size_t(I) * 2;    // permuted activation sizes
    while (true)
    {
        const int u = F.next.fetch_add(1, std::memory_order_relaxed);
        if (u >= F.units) break;
        if (F.phase == 0)
        {
            // prep gate/up inputs: unit = (job, token slot, g|u), whole 4096 vector
            const int j = u / (2 * MAXM), rem = u % (2 * MAXM), i = rem / 2, which = rem % 2;
            const Job& J = F.jobs[j];
            if (i >= J.m) continue;
            const Mat& M = which ? L.ex[J.e].u : L.ex[J.e].g;
            float* xp = (which ? F.xu : F.xg).data() + (size_t(j) * MAXM + i) * XH;
            float s = 0; alignas(32) float xt[128];
            float* qs = (which ? F.qu : F.qg).data() + (size_t(j) * MAXM + i) * (H / 128);
            for (int b = 0; b < H / 128; ++b)
            {
                prep_block(F.x + size_t(J.tok[i]) * H + b * 128, M.suh + b * 128, xt);
                if (F.mode == 2) { qs[b] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(b) * 256, &s); continue; }
                for (int r = 0; r < 128; ++r) s += xt[r];
                for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(b) * 8 + tt) * 32);
            }
            (which ? F.su : F.sg)[size_t(j) * MAXM + i] = s;
        }
        else if (F.phase == 1)
        {
            // gate+up block -> act -> down-input block. unit = (job, blk of I/128)
            const int nb = I / 128, j = u / nb, blk = u % nb;
            const Job& J = F.jobs[j];
            const Expert& E = L.ex[J.e];
            const float* xpg[MAXM]; const float* xpu[MAXM];
            for (int i = 0; i < J.m; ++i) { xpg[i] = F.xg.data() + (size_t(j) * MAXM + i) * XH; xpu[i] = F.xu.data() + (size_t(j) * MAXM + i) * XH; }
            alignas(32) float yg[MAXM * 128], yu[MAXM * 128];
            const float* sg = F.sg.data() + size_t(j) * MAXM; const float* su = F.su.data() + size_t(j) * MAXM;
            if (F.mode == 1) { gemv_block_m<true>(J.m, E.g, blk, xpg, sg, yg); gemv_block_m<true>(J.m, E.u, blk, xpu, su, yu); }
            else if (F.mode == 0) { gemv_block_m<false>(J.m, E.g, blk, xpg, sg, yg); gemv_block_m<false>(J.m, E.u, blk, xpu, su, yu); }
            else
            {
                const int32_t* qg_[MAXM]; const int32_t* qu_[MAXM]; const float* sg_[MAXM]; const float* su_[MAXM];
                for (int i = 0; i < J.m; ++i)
                {
                    qg_[i] = reinterpret_cast<const int32_t*>(xpg[i]); qu_[i] = reinterpret_cast<const int32_t*>(xpu[i]);
                    sg_[i] = F.qg.data() + (size_t(j) * MAXM + i) * (H / 128); su_[i] = F.qu.data() + (size_t(j) * MAXM + i) * (H / 128);
                }
                gemv_block_i16_m(J.m, E.g, blk, qg_, sg_, sg, yg); gemv_block_i16_m(J.m, E.u, blk, qu_, su_, su, yu);
            }
            for (int i = 0; i < J.m; ++i)
            {
                float* g = yg + i * 128; float* up = yu + i * 128;
                out_block(g, E.g.svh + blk * 128);
                out_block(up, E.u.svh + blk * 128);
                alignas(32) float a[128], xt[128];
                for (int r = 0; r < 128; ++r) a[r] = r16(act_gu(g[r], up[r]));
                prep_block(a, E.d.suh + blk * 128, xt);
                float* xp = F.xd.data() + (size_t(j) * MAXM + i) * XI;
                float s = 0;
                if (F.mode == 2) F.qd[(size_t(j) * MAXM + i) * (I / 128) + blk] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(blk) * 256, &s);
                else
                {
                    for (int r = 0; r < 128; ++r) s += xt[r];
                    for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(blk) * 8 + tt) * 32);
                }
                F.sd[(size_t(j) * MAXM + i) * (I / 128) + blk] = s;
            }
        }
        else if (F.phase == 2)
        {
            // down block. unit = (job, blk of H/128)
            const int nb = H / 128, j = u / nb, blk = u % nb;
            const Job& J = F.jobs[j];
            const Expert& E = L.ex[J.e];
            const float* xpd[MAXM]; float sd[MAXM];
            for (int i = 0; i < J.m; ++i)
            {
                xpd[i] = F.xd.data() + (size_t(j) * MAXM + i) * XI;
                float s = 0; for (int b = 0; b < I / 128; ++b) s += F.sd[(size_t(j) * MAXM + i) * (I / 128) + b]; sd[i] = s;
            }
            alignas(32) float y[MAXM * 128];
            if (F.mode == 1) gemv_block_m<true>(J.m, E.d, blk, xpd, sd, y);
            else if (F.mode == 0) gemv_block_m<false>(J.m, E.d, blk, xpd, sd, y);
            else
            {
                const int32_t* q_[MAXM]; const float* s_[MAXM];
                for (int i = 0; i < J.m; ++i) { q_[i] = reinterpret_cast<const int32_t*>(xpd[i]); s_[i] = F.qd.data() + (size_t(j) * MAXM + i) * (I / 128); }
                gemv_block_i16_m(J.m, E.d, blk, q_, s_, sd, y);
            }
            for (int i = 0; i < J.m; ++i)
            {
                out_block(y + i * 128, E.d.svh + blk * 128);
                std::memcpy(F.yd.data() + (size_t(j) * MAXM + i) * H + blk * 128, y + i * 128, 128 * 4);
            }
        }
        else
        {
            // weighted accumulate: unit = (token, 512-col chunk)
            const int nc = H / 512, t = u / nc, c0 = (u % nc) * 512;
            float* o = F.out + size_t(t) * H + c0;
            std::memset(o, 0, 512 * 4);
            for (size_t j = 0; j < F.jobs.size(); ++j)
                for (int i = 0; i < F.jobs[j].m; ++i)
                    if (F.jobs[j].tok[i] == t)
                    {
                        const float* y = F.yd.data() + (j * MAXM + i) * H + c0; const float wt = F.jobs[j].w[i];
                        for (int c = 0; c < 512; ++c) o[c] += wt * y[c];
                    }
        }
    }
}

struct PhaseTimes { double t[4] = {0, 0, 0, 0}; };

void moe_forward(Pool& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route,
                 float* out, int mode, PhaseTimes* pt = nullptr)
{
    static Fwd F;
    F.L = &L; F.x = x; F.out = out; F.ntok = ntok; F.mode = mode; F.jobs.clear();
    std::vector<int> jid(L.ex.size(), -1);
    for (int t = 0; t < ntok; ++t)
        for (auto& [e, wt] : route[t])
        {
            if (jid[e] < 0 || F.jobs[jid[e]].m == MAXM) { jid[e] = int(F.jobs.size()); F.jobs.push_back({ e, 0, {}, {} }); }
            Job& J = F.jobs[jid[e]]; J.tok[J.m] = t; J.w[J.m] = wt; ++J.m;
        }
    const size_t nj = F.jobs.size();
    auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(F.xg, nj * MAXM * L.H * 2); grow(F.xu, nj * MAXM * L.H * 2); grow(F.xd, nj * MAXM * L.I * 2);
    grow(F.sg, nj * MAXM); grow(F.su, nj * MAXM); grow(F.qg, nj * MAXM * (L.H / 128)); grow(F.qu, nj * MAXM * (L.H / 128)); grow(F.qd, nj * MAXM * (L.I / 128)); grow(F.sd, nj * MAXM * (L.I / 128)); grow(F.yd, nj * MAXM * L.H);
    const int units[4] = { int(nj) * 2 * MAXM, int(nj) * (L.I / 128), int(nj) * (L.H / 128), ntok * (L.H / 512) };
    for (int ph = 0; ph < 4; ++ph)
    {
        const double t0 = pt ? now() : 0;
        F.phase = ph; F.units = units[ph]; F.next.store(0);
        pool.run(&fwd_phase, &F);
        if (pt) pt->t[ph] += now() - t0;
    }
}

// ------------------------------------------------------------------------------------------
// Row-band forward on the NATIVE layout (split-K): a work unit = 8 tile rows (128 k) x all tiles_n
// of one matrix = one contiguous 98 KB (gate/up) / 196 KB (down) run, so the single native pinned
// copy that the GPU cache / zero-copy / staging paths use streams as fast as a re-laid-out copy.
// Partial sums per band are reduced in the next phase (fused with the output transforms).
// ------------------------------------------------------------------------------------------

// Rows [tk0, tk0+8) for all n, row-sequential (each row = one contiguous tiles_n*96 B run). The
// 8-lane accumulators go straight to the caller's partial buffer, layout [tile][c][M][8] (fp32; int32
// for I16), un-reduced: bands are summed as vectors and lane-reduced once, in the next phase.
template <int M, bool EXACT>
void gemv_band(const Mat& mat, int tk0, const float* const* xp, float* acc)
{
    const Consts K = consts();
    const int tiles_n = mat.n / 16;
    const size_t row_stride = size_t(tiles_n) * TILE_BYTES;
    std::memset(acc, 0, size_t(tiles_n) * 64 * M * 4);
    const float* X[M];
    const uint8_t* p = mat.tr + size_t(tk0) * row_stride;
    for (int r = 0; r < 8; ++r, p += row_stride)
    {
        for (int i = 0; i < M; ++i) X[i] = xp[i] + size_t(tk0 + r) * 32;
        #pragma GCC unroll 1
        for (int t = 0; t < tiles_n; ++t)
        {
            const char* pf = reinterpret_cast<const char*>(p + t * TILE_BYTES + 2 * row_stride);
            _mm_prefetch(pf, _MM_HINT_T0); _mm_prefetch(pf + 64, _MM_HINT_T0);
            tile_accum<M, EXACT>(p + t * TILE_BYTES, X, acc + size_t(t) * 64 * M, K);
        }
    }
}

template <int M>
void gemv_band_i16(const Mat& mat, int tk0, const int32_t* const* xq, int32_t* acc)
{
    const Consts K = consts();
    const int tiles_n = mat.n / 16;
    const size_t row_stride = size_t(tiles_n) * TILE_BYTES;
    std::memset(acc, 0, size_t(tiles_n) * 64 * M * 4);
    const int32_t* X[M];
    const uint8_t* p = mat.tr + size_t(tk0) * row_stride;
    for (int r = 0; r < 8; ++r, p += row_stride)
    {
        for (int i = 0; i < M; ++i) X[i] = xq[i] + size_t(tk0 + r) * 32;
        #pragma GCC unroll 1
        for (int t = 0; t < tiles_n; ++t)
        {
            const char* pf = reinterpret_cast<const char*>(p + t * TILE_BYTES + 2 * row_stride);
            _mm_prefetch(pf, _MM_HINT_T0); _mm_prefetch(pf + 64, _MM_HINT_T0);
            tile_accum_i16<M>(p + t * TILE_BYTES, X, acc + size_t(t) * 64 * M, K);
        }
    }
}

void gemv_band_m(int mode, int m, const Mat& mat, int tk0, const float* const* xp, float* acc)
{
#define FT_CASE(MM) case MM: \
    if (mode == 1) gemv_band<MM, true>(mat, tk0, xp, acc); \
    else if (mode == 0) gemv_band<MM, false>(mat, tk0, xp, acc); \
    else gemv_band_i16<MM>(mat, tk0, reinterpret_cast<const int32_t* const*>(xp), reinterpret_cast<int32_t*>(acc)); break;
    switch (m) { FT_CASE(1) FT_CASE(2) FT_CASE(3) FT_CASE(4) FT_CASE(5) FT_CASE(6) FT_CASE(7) default: FT_CASE(8) }
#undef FT_CASE
}

// Sum `nb` band partials (stride band_stride floats) for one 128-output block (tiles t0..t0+7) of token
// slot i (of m), scale per band (I16: int32 partials x scale; else fp32 x 1), lane-reduce -> y[128]
inline void reduce_bands(const float* base, size_t band_stride, int nb, int m, int i, int t0, bool i16, const float* scales, float* y)
{
    for (int t = 0; t < 8; ++t)
        for (int c = 0; c < 8; ++c)
        {
            __m256 v = _mm256_setzero_ps();
            const size_t off = (size_t(t0 + t) * 8 + c) * m * 8 + size_t(i) * 8;
            for (int b = 0; b < nb; ++b)
            {
                const float* q = base + size_t(b) * band_stride + off;
                if (i16) v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q))), _mm256_set1_ps(scales[b]), v);
                else v = _mm256_add_ps(v, _mm256_loadu_ps(q));
            }
            alignas(32) float l[8]; _mm256_store_ps(l, v);
            y[t * 16 + c] = (l[0] + l[1]) + (l[2] + l[3]);
            y[t * 16 + c + 8] = (l[4] + l[5]) + (l[6] + l[7]);
        }
}

struct Fwd2
{
    const Layer* L;
    const float* x; float* out; int ntok; int mode;
    std::vector<Job> jobs;
    std::vector<size_t> joff;             // per job: token-slot base (sum of m of previous jobs)
    size_t nslot = 0;
    // per token-slot: permuted activations (fp32 or i16-dup), block scales, sums; per slot partials
    std::vector<float> xg, xu, xd, qg, qu, qd, sg, su, sd, pg, pu, pd;
    std::atomic<int> next{0};
    int phase = 0, units = 0;
};

void fwd2_phase(void* vc, int, int)
{
    Fwd2& F = *static_cast<Fwd2*>(vc);
    const Layer& L = *F.L;
    const int H = L.H, I = L.I, BH = H / 128, BI = I / 128;
    const size_t XH = size_t(H) * 2, XI = size_t(I) * 2;
    const int nj = int(F.jobs.size());
    while (true)
    {
        const int u = F.next.fetch_add(1, std::memory_order_relaxed);
        if (u >= F.units) break;
        if (F.phase == 0)
        {
            // prep: unit = (slot, g|u)
            const size_t s = size_t(u / 2); const int which = u % 2;
            int j = 0; while (j + 1 < nj && F.joff[j + 1] <= s) ++j;
            const Job& J = F.jobs[j]; const int i = int(s - F.joff[j]);
            const Mat& M = which ? L.ex[J.e].u : L.ex[J.e].g;
            float* xp = (which ? F.xu : F.xg).data() + s * XH;
            float* qs = (which ? F.qu : F.qg).data() + s * BH;
            float sum = 0; alignas(32) float xt[128];
            for (int b = 0; b < BH; ++b)
            {
                prep_block(F.x + size_t(J.tok[i]) * H + b * 128, M.suh + b * 128, xt);
                if (F.mode == 2) { qs[b] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(b) * 256, &sum); continue; }
                for (int r = 0; r < 128; ++r) sum += xt[r];
                for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(b) * 8 + tt) * 32);
            }
            (which ? F.su : F.sg)[s] = sum;
        }
        else if (F.phase == 1)
        {
            // gate / up bands: unit = (job, g|u, band)
            const int j = u / (2 * BH), rem = u % (2 * BH), which = rem / BH, band = rem % BH;
            const Job& J = F.jobs[j]; const size_t s0 = F.joff[j];
            const Mat& M = which ? L.ex[J.e].u : L.ex[J.e].g;
            const float* xp[MAXM];
            for (int i = 0; i < J.m; ++i) xp[i] = (which ? F.xu : F.xg).data() + (s0 + i) * XH;
            // partials per job: [band][tiles_n * 64 * m] (= band_stride I * 4 * m floats)
            float* out = (which ? F.pu : F.pg).data() + s0 * BH * I * 4 + size_t(band) * I * 4 * J.m;
            gemv_band_m(F.mode, J.m, M, band * 8, xp, out);
        }
        else if (F.phase == 2)
        {
            // reduce gate/up, transform, act, prep down input: unit = (job, 128-block of I)
            const int j = u / BI, blk = u % BI;
            const Job& J = F.jobs[j]; const size_t s0 = F.joff[j];
            const Expert& E = L.ex[J.e];
            for (int i = 0; i < J.m; ++i)
            {
                alignas(32) float g[128], up[128], a[128], xt[128];
                reduce_bands(F.pg.data() + s0 * BH * I * 4, size_t(I) * 4 * J.m, BH, J.m, i, blk * 8, F.mode == 2, F.qg.data() + (s0 + i) * BH, g);
                reduce_bands(F.pu.data() + s0 * BH * I * 4, size_t(I) * 4 * J.m, BH, J.m, i, blk * 8, F.mode == 2, F.qu.data() + (s0 + i) * BH, up);
                if (F.mode != 1)
                {
                    const float cg = CAFF * F.sg[s0 + i], cu = CAFF * F.su[s0 + i];
                    for (int c = 0; c < 128; ++c) { g[c] = KINV * g[c] + cg; up[c] = KINV * up[c] + cu; }
                }
                out_block(g, E.g.svh + blk * 128);
                out_block(up, E.u.svh + blk * 128);
                for (int r = 0; r < 128; ++r) a[r] = r16(act_gu(g[r], up[r]));
                prep_block(a, E.d.suh + blk * 128, xt);
                float* xp = F.xd.data() + (s0 + i) * XI;
                float sum = 0;
                if (F.mode == 2) F.qd[(s0 + i) * BI + blk] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(blk) * 256, &sum);
                else
                {
                    for (int r = 0; r < 128; ++r) sum += xt[r];
                    for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(blk) * 8 + tt) * 32);
                }
                F.sd[(s0 + i) * BI + blk] = sum;
            }
        }
        else if (F.phase == 3)
        {
            // down bands: unit = (job, band of I)
            const int j = u / BI, band = u % BI;
            const Job& J = F.jobs[j]; const size_t s0 = F.joff[j];
            const float* xp[MAXM];
            for (int i = 0; i < J.m; ++i) xp[i] = F.xd.data() + (s0 + i) * XI;
            gemv_band_m(F.mode, J.m, L.ex[J.e].d, band * 8, xp, F.pd.data() + s0 * BI * H * 4 + size_t(band) * H * 4 * J.m);
        }
        else
        {
            // reduce down, transform, weighted accumulate: unit = 128-col block of H (race-free)
            const int blk = u;
            for (int t = 0; t < F.ntok; ++t) std::memset(F.out + size_t(t) * H + blk * 128, 0, 512);
            for (int j = 0; j < nj; ++j)
            {
                const Job& J = F.jobs[j]; const size_t s0 = F.joff[j];
                const Expert& E = L.ex[J.e];
                for (int i = 0; i < J.m; ++i)
                {
                    alignas(32) float y[128];
                    reduce_bands(F.pd.data() + s0 * BI * H * 4, size_t(H) * 4 * J.m, BI, J.m, i, blk * 8, F.mode == 2, F.qd.data() + (s0 + i) * BI, y);
                    if (F.mode != 1)
                    {
                        float sd = 0; for (int b = 0; b < BI; ++b) sd += F.sd[(s0 + i) * BI + b];
                        const float cd = CAFF * sd;
                        for (int c = 0; c < 128; ++c) y[c] = KINV * y[c] + cd;
                    }
                    out_block(y, E.d.svh + blk * 128);
                    float* o = F.out + size_t(J.tok[i]) * H + blk * 128; const float wt = J.w[i];
                    for (int c = 0; c < 128; ++c) o[c] += wt * y[c];
                }
            }
        }
    }
}

// mode: 0 AFFINE, 1 EXACT, 2 I16, -1 auto (I16 if every expert has <= 2 tokens, else AFFINE)
template <typename PoolT>
void moe_forward_band(PoolT& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route,
                      float* out, int mode, PhaseTimes* pt = nullptr)
{
    static Fwd2 F;
    F.L = &L; F.x = x; F.out = out; F.ntok = ntok; F.jobs.clear(); F.joff.clear();
    std::vector<int> jid(L.ex.size(), -1);
    for (int t = 0; t < ntok; ++t)
        for (auto& [e, wt] : route[t])
        {
            if (jid[e] < 0 || F.jobs[jid[e]].m == MAXM) { jid[e] = int(F.jobs.size()); F.jobs.push_back({ e, 0, {}, {} }); }
            Job& J = F.jobs[jid[e]]; J.tok[J.m] = t; J.w[J.m] = wt; ++J.m;
        }
    int maxm = 0; size_t ns = 0;
    for (auto& J : F.jobs) { F.joff.push_back(ns); ns += J.m; maxm = std::max(maxm, J.m); }
    F.nslot = ns;
    F.mode = mode >= 0 ? mode : (maxm <= 2 ? 2 : 0);
    const int H = L.H, I = L.I, BH = H / 128, BI = I / 128;
    auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(F.xg, ns * H * 2); grow(F.xu, ns * H * 2); grow(F.xd, ns * I * 2);
    grow(F.qg, ns * BH); grow(F.qu, ns * BH); grow(F.qd, ns * BI);
    grow(F.sg, ns); grow(F.su, ns); grow(F.sd, ns * BI);
    grow(F.pg, ns * BH * I * 4); grow(F.pu, ns * BH * I * 4); grow(F.pd, ns * BI * H * 4);
    const int nj = int(F.jobs.size());
    if (!nj) { std::memset(out, 0, size_t(ntok) * H * 4); return; }
    const int units[5] = { int(ns) * 2, nj * 2 * BH, nj * BI, nj * BI, BH };
    for (int ph = 0; ph < 5; ++ph)
    {
        const double t0 = pt ? now() : 0;
        F.phase = ph; F.units = units[ph]; F.next.store(0);
        pool.run(&fwd2_phase, &F);
        if (pt) pt->t[std::min(ph, 3)] += now() - t0;
    }
}

// ------------------------------------------------------------------------------------------
// Reference (double accumulation, GPU-exact weights, same fp16 rounding points)
// ------------------------------------------------------------------------------------------

std::vector<float> dense_weights(const Mat& m)    // [k][n]
{
    std::vector<float> W(size_t(m.k) * m.n);
    const int tn = m.n / 16;
    for (int tk = 0; tk < m.k / 16; ++tk)
        for (int t = 0; t < tn; ++t)
        {
            const uint16_t* packed = reinterpret_cast<const uint16_t*>(m.tr + (size_t(tk) * tn + t) * TILE_BYTES);
            for (int i = 0; i < 256; ++i)
            {
                const int rc = PERM[i];
                W[size_t(tk * 16 + rc / 16) * m.n + t * 16 + rc % 16] = w_gpu(state_scalar(packed, i));
            }
        }
    return W;
}

void ref_linear(const Mat& m, const std::vector<float>& W, const float* x, float* y)
{
    std::vector<float> xt(m.k);
    for (int b = 0; b < m.k / 128; ++b) prep_block(x + b * 128, m.suh + b * 128, xt.data() + b * 128);
    std::vector<double> acc(m.n, 0.0);
    for (int k = 0; k < m.k; ++k) { const double xv = xt[k]; const float* wr = W.data() + size_t(k) * m.n; for (int n = 0; n < m.n; ++n) acc[n] += xv * wr[n]; }
    for (int n = 0; n < m.n; ++n) y[n] = float(acc[n]);
    for (int b = 0; b < m.n / 128; ++b) out_block(y + b * 128, m.svh + b * 128);
}

void ref_expert(const Expert& E, const std::vector<float>* W, const float* x, float* y)
{
    std::vector<float> g(2048), u(2048), a(2048);
    ref_linear(E.g, W[0], x, g.data());
    ref_linear(E.u, W[1], x, u.data());
    for (int i = 0; i < 2048; ++i) a[i] = r16(act_gu(g[i], u[i]));
    ref_linear(E.d, W[2], a.data(), y);
}

// int8-activation emulation of the exllamav3 AVX2 tier (one symmetric scale per GEMV input row,
// codebook value (sum-510)*k_inv) to quantify its error with the same harness
void int8_linear(const Mat& m, const float* x, float* y)
{
    std::vector<float> xt(m.k);
    for (int b = 0; b < m.k / 128; ++b)
    {
        for (int i = 0; i < 128; ++i) xt[b * 128 + i] = x[b * 128 + i] * h2f(m.suh[b * 128 + i]);
        had128(xt.data() + b * 128);
        for (int i = 0; i < 128; ++i) xt[b * 128 + i] *= HAD;
    }
    float amax = 0; for (float v : xt) amax = std::max(amax, std::fabs(v));
    const float q = amax > 0 ? amax / 127.0f : 1.0f;
    std::vector<int> x8(m.k); long sx = 0;
    for (int k = 0; k < m.k; ++k) { x8[k] = std::clamp(int(std::nearbyint(xt[k] / q)), -127, 127); sx += x8[k]; }
    std::vector<int64_t> acc(m.n, 0);
    const int tn = m.n / 16;
    for (int tk = 0; tk < m.k / 16; ++tk)
        for (int t = 0; t < tn; ++t)
        {
            const uint16_t* packed = reinterpret_cast<const uint16_t*>(m.tr + (size_t(tk) * tn + t) * TILE_BYTES);
            for (int i = 0; i < 256; ++i) { const int rc = PERM[i]; acc[t * 16 + rc % 16] += int64_t(bytesum(state_scalar(packed, i))) * x8[tk * 16 + rc / 16]; }
        }
    for (int n = 0; n < m.n; ++n) y[n] = float(KINV * q * (double(acc[n]) - 510.0 * sx));
    for (int b = 0; b < m.n / 128; ++b)
    {
        float* v = y + b * 128;
        had128(v);
        for (int i = 0; i < 128; ++i) v[i] *= HAD * h2f(m.svh[b * 128 + i]);
    }
}

void int8_expert(const Expert& E, const float* x, float* y)
{
    std::vector<float> g(2048), u(2048), a(2048);
    int8_linear(E.g, x, g.data()); int8_linear(E.u, x, u.data());
    for (int i = 0; i < 2048; ++i) a[i] = silu(g[i]) * u[i];
    int8_linear(E.d, a.data(), y);
}

struct Err { double rel_rms, max_abs, cos; };
Err compare(const float* a, const float* ref, size_t n)
{
    double se = 0, sr = 0, mx = 0, dot = 0, na = 0;
    for (size_t i = 0; i < n; ++i) { const double d = double(a[i]) - ref[i]; se += d * d; sr += double(ref[i]) * ref[i]; mx = std::max(mx, std::fabs(d)); dot += double(a[i]) * ref[i]; na += double(a[i]) * a[i]; }
    return { std::sqrt(se / std::max(sr, 1e-30)), mx, dot / std::sqrt(std::max(na * sr, 1e-30)) };
}

std::vector<float> make_x(int ntok, int H, uint64_t seed)
{
    // hidden-state-like input: gaussian with a few large outlier channels, fp16-valued
    std::mt19937_64 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(size_t(ntok) * H);
    for (auto& v : x) v = r16(nd(rng) * 0.35f);
    for (int t = 0; t < ntok; ++t) for (int k = 0; k < 8; ++k) x[size_t(t) * H + (k * 509 + 17) % H] = r16(nd(rng) * 6.0f);
    return x;
}

}  // namespace
