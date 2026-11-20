// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Rectangular KDA, K=64 V=128. MFMA operand layout follows AMD CDNA wave64.
#include "kda_rectangular.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <type_traits>
#define KDA_CHECK_LAUNCH()                                    \
    do                                                        \
    {                                                         \
        auto err = hipGetLastError();                         \
        if(err != hipSuccess)                                 \
            throw std::runtime_error(hipGetErrorString(err)); \
    } while(0)

// A fat binary can also contain empty device images for other architectures.
// Only the gfx950 device pass instantiates these CDNA4 MFMA instructions.
#if !__HIP_DEVICE_COMPILE__ || defined(__gfx950__)
namespace aiter::kda_rectangular {

using T         = TensorDesc;
using Opt       = NativeOpt;
using B         = __bf16;
using F4        = float __attribute__((ext_vector_type(4)));
using B4        = __bf16 __attribute__((ext_vector_type(4)));
constexpr int K = 64, V = 128, C = 16;

__device__ __forceinline__ float f(B x) { return float(x); }
__device__ __forceinline__ B b(float x) { return B(x); }
__device__ __forceinline__ float sig(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sig(x); }
__device__ __forceinline__ float sum64(float x)
{
#pragma unroll
    for(int d = 32; d; d >>= 1)
        x += __shfl_xor(x, d, 64);
    return x;
}
__device__ __forceinline__ int64_t ix(const void* p, int64_t i, bool wide)
{
    return wide ? static_cast<const int64_t*>(p)[i] : static_cast<const int32_t*>(p)[i];
}
__device__ __forceinline__ float load_float(const void* p, int64_t i, bool fp32)
{
    return fp32 ? static_cast<const float*>(p)[i] : f(static_cast<const B*>(p)[i]);
}

// The scalar path can read noncontiguous weights and commits cache separately.
template <bool CACHE>
__global__ void conv_kernel(const B* x,
                            const B* weight,
                            const B* cache,
                            const void* cu,
                            bool wide,
                            const bool* has,
                            B* out,
                            int H,
                            int unused_channels,
                            int64_t sx,
                            int64_t sc0,
                            int64_t sc1,
                            int64_t sc2,
                            int64_t sw0,
                            int64_t sw1)
{
    constexpr int channels = 16384;
    int seq                = blockIdx.y;
    int64_t bos = ix(cu, seq, wide), eos = ix(cu, seq + 1, wide);
    int64_t flat = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n    = (eos - bos) * channels;
    for(; flat < n; flat += int64_t(gridDim.x) * blockDim.x)
    {
        int ch    = flat % channels;
        int64_t t = bos + flat / channels;
        float z   = 0;
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            int64_t src = t + j - 3;
            float a     = 0;
            if(src >= bos)
                a = f(x[src * sx + ch]);
            else if constexpr(CACHE)
            {
                if(has && has[seq])
                    a = f(cache[seq * sc0 + ch * sc1 + (src - bos + 3) * sc2]);
            }
            z = fmaf(a, f(weight[ch * sw0 + j * sw1]), z);
        }
        out[t * channels + ch] = b(silu(z));
    }
}
// One lane owns a channel vector across a temporal tile. Only the first
// tile reads the old history, so it can also commit that sequence's tail.
template <int ChannelsPerLane>
using ConvB = B __attribute__((ext_vector_type(ChannelsPerLane)));
template <int ChannelsPerLane>
using ConvF = float __attribute__((ext_vector_type(ChannelsPerLane)));

// Private mixer storage packs 16 consecutive rows of one 128-channel block.
template <bool Blocked>
__device__ __forceinline__ int64_t mixer_offset(int64_t t, int ch, int64_t stride)
{
    if constexpr(Blocked)
        return (t >> 4) * (16 * 16384) + (ch / 128) * (16 * 128) + (t & 15) * 128 + (ch & 127);
    return t * stride + ch;
}

template <int Rows, int ChannelsPerLane, int Prefetch, bool FullTile, bool Blocked>
__device__ __forceinline__ void conv_rows(const B* __restrict__ x,
                                          B* __restrict__ out,
                                          int ch,
                                          int64_t sx,
                                          int64_t t,
                                          int64_t eos,
                                          ConvB<ChannelsPerLane> (&hist)[3],
                                          ConvF<ChannelsPerLane> (&weights)[4])
{
    constexpr int channels = 16384;
    static_assert(Rows % Prefetch == 0);
#pragma unroll
    for(int group = 0; group < Rows; group += Prefetch)
    {
        ConvB<ChannelsPerLane> values[Prefetch];
#pragma unroll
        for(int p = 0; p < Prefetch; ++p)
        {
            const int64_t tr = FullTile ? t + group + p : min(t + group + p, eos - 1);
            const auto* ptr  = reinterpret_cast<const ConvB<ChannelsPerLane>*>(x + tr * sx + ch);
            if constexpr(Blocked)
                values[p] = __builtin_nontemporal_load(ptr);
            else
                values[p] = *ptr;
        }
#pragma unroll
        for(int p = 0; p < Prefetch; ++p)
        {
            ConvF<ChannelsPerLane> acc =
                __builtin_convertvector(hist[0], ConvF<ChannelsPerLane>) * weights[0];
            acc += __builtin_convertvector(hist[1], ConvF<ChannelsPerLane>) * weights[1];
            acc += __builtin_convertvector(hist[2], ConvF<ChannelsPerLane>) * weights[2];
            acc += __builtin_convertvector(values[p], ConvF<ChannelsPerLane>) * weights[3];
#pragma unroll
            for(int i = 0; i < ChannelsPerLane; ++i)
                acc[i] = silu(acc[i]);
            if(FullTile || t + group + p < eos)
            {
                auto* dst = reinterpret_cast<ConvB<ChannelsPerLane>*>(
                    out + mixer_offset<Blocked>(t + group + p, ch, channels));
                auto result = __builtin_convertvector(acc, ConvB<ChannelsPerLane>);
                if constexpr(Blocked)
                    __builtin_nontemporal_store(result, dst);
                else
                    *dst = result;
            }
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = values[p];
        }
    }
}

template <int Rows,
          int ChannelsPerLane,
          int Threads,
          int Prefetch,
          bool SplitFull,
          bool Blocked = false,
          int Half     = -1>
__global__ __launch_bounds__(Threads) void conv_tiled_kernel(const B* __restrict__ x,
                                                             const B* __restrict__ weight,
                                                             B* cache,
                                                             const void* cu,
                                                             bool wide,
                                                             const bool* has,
                                                             B* __restrict__ out,
                                                             int64_t sx,
                                                             int64_t sc0,
                                                             int64_t sc1,
                                                             int64_t sc2)
{
    const int seq     = blockIdx.z;
    const int ch      = (blockIdx.x * Threads + threadIdx.x) * ChannelsPerLane;
    const int64_t bos = ix(cu, seq, wide), eos = ix(cu, seq + 1, wide);
    int64_t t    = bos + int64_t(blockIdx.y) * Rows;
    int64_t stop = eos;
    if constexpr(Half >= 0)
    {
        t += Half * 16384;
        stop = min(eos, bos + (Half + 1) * int64_t(16384));
    }
    if(t >= stop)
        return;
    ConvF<ChannelsPerLane> weights[4];
#pragma unroll
    for(int i = 0; i < ChannelsPerLane; ++i)
    {
        B4 wc = *reinterpret_cast<const B4*>(weight + (ch + i) * 4);
#pragma unroll
        for(int j = 0; j < 4; ++j)
            weights[j][i] = f(wc[j]);
    }
    for(; t < stop; t += int64_t(gridDim.y) * Rows)
    {
        ConvB<ChannelsPerLane> hist[3];
#pragma unroll
        for(int j = 0; j < 3; ++j)
        {
            if(t > bos)
                hist[j] =
                    *reinterpret_cast<const ConvB<ChannelsPerLane>*>(x + (t + j - 3) * sx + ch);
            else
            {
                ConvB<ChannelsPerLane> val = {};
                if(cache && has && has[seq])
                {
#pragma unroll
                    for(int i = 0; i < ChannelsPerLane; ++i)
                        val[i] = cache[seq * sc0 + (ch + i) * sc1 + j * sc2];
                }
                hist[j] = val;
            }
        }
        if(cache && t == bos)
        {
#pragma unroll
            for(int j = 0; j < 3; ++j)
            {
                const int64_t tail = eos - 3 + j;
                ConvB<ChannelsPerLane> val;
                if(tail >= bos)
                    val = *reinterpret_cast<const ConvB<ChannelsPerLane>*>(x + tail * sx + ch);
                else // A one- or two-token sequence retains a suffix of the old window.
                    val = (tail + 3 - bos == 1) ? hist[1] : hist[2];
#pragma unroll
                for(int i = 0; i < ChannelsPerLane; ++i)
                    cache[seq * sc0 + (ch + i) * sc1 + j * sc2] = val[i];
            }
        }
        if constexpr(SplitFull)
        {
            if(t + Rows <= eos)
            {
                conv_rows<Rows, ChannelsPerLane, Prefetch, true, Blocked>(
                    x, out, ch, sx, t, eos, hist, weights);
                continue;
            }
        }
        conv_rows<Rows, ChannelsPerLane, Prefetch, false, Blocked>(
            x, out, ch, sx, t, eos, hist, weights);
    }
}
__global__ void conv_commit_kernel(const B* x,
                                   B* cache,
                                   const void* cu,
                                   bool wide,
                                   const bool* has,
                                   int unused_channels,
                                   int64_t sx,
                                   int64_t sc0,
                                   int64_t sc1,
                                   int64_t sc2)
{
    constexpr int channels = 16384;
    int seq = blockIdx.y, ch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ch >= channels)
        return;
    int64_t bos = ix(cu, seq, wide), eos = ix(cu, seq + 1, wide), len = eos - bos;
    if(!len)
        return;
    B old[3];
#pragma unroll
    for(int j = 0; j < 3; j++)
        old[j] = (has && has[seq]) ? cache[seq * sc0 + ch * sc1 + j * sc2] : b(0);
#pragma unroll
    for(int j = 0; j < 3; j++)
    {
        int64_t src                           = eos - 3 + j;
        cache[seq * sc0 + ch * sc1 + j * sc2] = src >= bos ? x[src * sx + ch] : old[src - bos + 3];
    }
}

// One C16 tile per workgroup. A lane starts with four contiguous K channels
// of one token. The four rows in a wave scan their gates in registers.
using PrepB8 = __bf16 __attribute__((ext_vector_type(8)));

template <typename Vec>
__device__ __forceinline__ Vec prep_load(const B* p)
{
    // A public input can be a strided view starting at an odd BF16 address.
    if((reinterpret_cast<uintptr_t>(p) & (sizeof(Vec) - 1)) == 0)
        return *reinterpret_cast<const Vec*>(p);
    Vec x;
    __builtin_memcpy(&x, p, sizeof(x));
    return x;
}

__device__ __forceinline__ float prep_sum16(float x)
{
#pragma unroll
    for(int d = 8; d; d >>= 1)
        x += __shfl_xor(x, d, 16);
    return x;
}

__device__ __forceinline__ B4 prep_pack(F4 x) { return __builtin_convertvector(x, B4); }

template <bool SpxLong>
struct alignas(16) PrepareScratch
{
    B kd[C * (K + 8)], ki[C * (K + 8)], qd[C * (K + 8)], carry_key[C * (K + 8)],
        vv[C * (V + (SpxLong ? 16 : 8))], inverse[C * C];
    float band_total[4][K], betas[C];
};
template <bool SpxLong>
__device__ __forceinline__ PrepareScratch<SpxLong>& prepare_storage()
{
    __shared__ PrepareScratch<SpxLong> mem;
    return mem;
}

__device__ __forceinline__ B4 prep_transposed(const B* source)
{
#if __has_builtin(__builtin_amdgcn_ds_read_tr16_b64_v4i16)
    using S4  = short __attribute__((ext_vector_type(4)));
    using Lds = S4 __attribute__((address_space(3)));
    return __builtin_bit_cast(B4, __builtin_amdgcn_ds_read_tr16_b64_v4i16((Lds*)(uintptr_t)source));
#else
    long long value;
    const unsigned addr = static_cast<unsigned>(reinterpret_cast<uintptr_t>(source));
    asm volatile("ds_read_b64_tr_b16 %0, %1\n\ts_waitcnt lgkmcnt(0)"
                 : "=v"(value)
                 : "v"(addr)
                 : "memory");
    return __builtin_bit_cast(B4, value);
#endif
}

template <bool Streaming, class Vec>
__device__ __forceinline__ Vec prepare_read(const Vec* ptr)
{
    if constexpr(Streaming)
        return __builtin_nontemporal_load(ptr);
    return *ptr;
}

template <bool Streaming, class Vec>
__device__ __forceinline__ void prepare_write(Vec* ptr, Vec value)
{
    if constexpr(Streaming)
        __builtin_nontemporal_store(value, ptr);
    else
        *ptr = value;
}

template <bool FULL_ALIGNED, bool SpxLong, bool Blocked>
__device__ __forceinline__ void prepare_tile(const B* __restrict__ q,
                                             const B* __restrict__ k,
                                             const B* __restrict__ v,
                                             const B* __restrict__ g,
                                             const void* beta,
                                             bool beta32,
                                             const float* alog,
                                             const float* dt,
                                             B* __restrict__ qg,
                                             B* __restrict__ w,
                                             B* __restrict__ u,
                                             B* __restrict__ kg,
                                             B* __restrict__ aq,
                                             float* __restrict__ decay,
                                             int H,
                                             int64_t sq,
                                             int64_t sk,
                                             int64_t sv,
                                             int64_t sg,
                                             int64_t sb,
                                             float lb,
                                             float scale,
                                             int64_t start,
                                             int64_t end)
{
    constexpr int pitch_k = K + 8, pitch_v = V + (SpxLong ? 16 : 8);
    const int64_t group = int64_t(blockIdx.y) * gridDim.x + blockIdx.x;
    int id = group / 64, h = group % 64;
    int tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
    int row = tid >> 4, col = 4 * (lane & 15);
    int64_t token = start + row;
    bool valid    = FULL_ALIGNED || token < end;
    int64_t tile  = int64_t(id) * H + h;
    auto& mem     = prepare_storage<SpxLong>();
    auto &kd = mem.kd, &ki = mem.ki, &qd = mem.qd, &carry_key = mem.carry_key;
    auto& vv         = mem.vv;
    auto& inverse    = mem.inverse;
    auto& band_total = mem.band_total;
    auto& betas      = mem.betas;
    B4 qr = {}, kr = {}, zr = {};
    PrepB8 vr                = {};
    float beta_value         = 0.f;
    constexpr bool late_beta = Blocked && FULL_ALIGNED;
    if(valid && !late_beta)
        beta_value = load_float(beta, token * sb + h, beta32);
    if constexpr(FULL_ALIGNED)
    {
        qr = prepare_read<SpxLong>(reinterpret_cast<const B4*>(
            q + (mixer_offset<Blocked>(token, 0 + h * K + col, sq) - 0)));
        kr = prepare_read<SpxLong>(reinterpret_cast<const B4*>(
            k + (mixer_offset<Blocked>(token, 4096 + h * K + col, sk) - 4096)));
        zr = prepare_read<SpxLong>(reinterpret_cast<const B4*>(g + token * sg + h * K + col));
        vr = prepare_read<SpxLong>(reinterpret_cast<const PrepB8*>(
            v + (mixer_offset<Blocked>(token, 8192 + h * V + 2 * col, sv) - 8192)));
    }
    else if(valid)
    {
        qr = prep_load<B4>(q + (mixer_offset<Blocked>(token, 0 + h * K + col, sq) - 0));
        kr = prep_load<B4>(k + (mixer_offset<Blocked>(token, 4096 + h * K + col, sk) - 4096));
        zr = prep_load<B4>(g + token * sg + h * K + col);
        vr = prep_load<PrepB8>(v +
                               (mixer_offset<Blocked>(token, 8192 + h * V + 2 * col, sv) - 8192));
    }
    if constexpr(late_beta)
    {
        if(col == 0)
            betas[row] = valid ? sig(load_float(beta, token * sb + h, beta32)) : 0.f;
    }
    float asum = 0.f, ksum = 0.f;
    F4 prefix;
    float a = expf(alog[h]);
#pragma unroll
    for(int d = 0; d < 4; ++d)
    {
        asum       = fmaf(f(qr[d]), f(qr[d]), asum);
        ksum       = fmaf(f(kr[d]), f(kr[d]), ksum);
        float z    = valid ? lb * sig(a * (f(zr[d]) + dt[h * K + col + d])) : 0.f;
        float prev = __shfl_up(z, 16, 64);
        if((row & 3) >= 1)
            z += prev;
        prev = __shfl_up(z, 32, 64);
        if((row & 3) >= 2)
            z += prev;
        prefix[d] = z;
    }
    float qnorm = rsqrtf(prep_sum16(asum) + 1.e-6f);
    float knorm = rsqrtf(prep_sum16(ksum) + 1.e-6f);
    if(!late_beta && col == 0)
        betas[row] = valid ? sig(beta_value) : 0.f;
    if((row & 3) == 3)
        *reinterpret_cast<F4*>(&band_total[warp][col]) = prefix;
    __syncthreads();
    F4 kdec, kinc, qdec, klast, dvec;
#pragma unroll
    for(int d = 0; d < 4; ++d)
    {
        float g0 = band_total[0][col + d], g1 = band_total[1][col + d];
        float g2 = band_total[2][col + d], g3 = band_total[3][col + d];
        float last = (g0 + g1) + (g2 + g3);
        float run =
            prefix[d] + ((warp >= 1 ? g0 : 0.f) + (warp >= 2 ? g1 : 0.f) + (warp >= 3 ? g2 : 0.f));
        float eg = expf(run), kn = f(kr[d]) * knorm;
        kdec[d]  = kn * eg;
        kinc[d]  = kn * expf(-run);
        qdec[d]  = (f(qr[d]) * qnorm) * eg * scale;
        klast[d] = kn * expf(last - run);
        dvec[d]  = expf(last);
    }
    if(row == 0)
        *reinterpret_cast<F4*>(decay + tile * K + col) = dvec;
    *reinterpret_cast<PrepB8*>(vv + row * pitch_v + 2 * col) = vr;
    auto qvec                                                = prep_pack(qdec);
    *reinterpret_cast<B4*>(kd + row * pitch_k + col)         = prep_pack(kdec);
    *reinterpret_cast<B4*>(ki + row * pitch_k + col)         = prep_pack(kinc);
    *reinterpret_cast<B4*>(qd + row * pitch_k + col)         = qvec;
    // Pack two adjacent K16 contractions in the native gfx950 x32 operand.
    auto qw_offset = [](int t, int k) {
        return ((k / 32) * 64 + ((k % 16) / 4) * 16 + t) * 8 + (k % 32) / 16 * 4;
    };
    prepare_write<SpxLong>(reinterpret_cast<B4*>(qg + tile * C * K + qw_offset(row, col)), qvec);
    *reinterpret_cast<B4*>(carry_key + row * pitch_k + col) = prep_pack(klast);
    __syncthreads();
    int rc = lane & 15, r4 = (lane >> 4) * 4;
    if(warp < 2)
    {
        const B* left = warp == 0 ? kd : qd;
        F4 gram       = {};
#pragma unroll
        for(int off = 0; off < K; off += 32)
        {
            int src = rc * pitch_k + off + (lane >> 4) * 8;
            gram    = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                *reinterpret_cast<const PrepB8*>(left + src),
                *reinterpret_cast<const PrepB8*>(ki + src),
                gram,
                0,
                0,
                0);
        }
        if(warp == 0)
        {
            // Keep only the strict lower triangle during elimination. At pivot p,
            // lower[r,p] vanishes for r <= p and x[p,c] vanishes for c >= p.
            // Restore the unit diagonal when writing the inverse.
            F4 beta_rows = *reinterpret_cast<const F4*>(betas + r4);
            F4 x;
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                int r = r4 + i;
                x[i]  = rc < r ? -gram[i] * beta_rows[i] : 0.f;
            }
            const F4 lower = x;
#pragma unroll
            for(int p = 1; p < C - 1; ++p)
            {
                float pivot = __shfl(x[p & 3], rc + (p & ~3) * 4, 64);
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    float mul = __shfl(lower[i], p, 16);
                    x[i]      = fmaf(mul, pivot, x[i]);
                }
            }
#pragma unroll
            for(int i = 0; i < 4; ++i)
                inverse[(r4 + i) * C + rc] = b((r4 + i == rc ? 1.f : x[i]) * betas[rc]);
        }
        else
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                int r                       = r4 + i;
                aq[(tile * C + r) * C + rc] = b(rc <= r ? gram[i] : 0.f);
            }
        }
    }
    __syncthreads();
    // Emit W and U in their respective MFMA operand layouts for the walk.
    B4 key_fragment =
        prep_transposed(carry_key + (lane >> 2) * pitch_k + warp * 16 + (lane & 3) * 4);
    prepare_write<SpxLong>(reinterpret_cast<B4*>(kg + tile * C * K + tid * 4), key_fragment);
    B4 inv = *reinterpret_cast<const B4*>(inverse + rc * C + r4);
#pragma unroll
    for(int part = 0; part < 3; ++part)
    {
        bool key     = part == 0;
        int pitch    = key ? pitch_k : pitch_v;
        int vcol     = warp * 16 + (part == 2 ? 64 : 0);
        const B* rhs = (key ? kd : vv) + (lane >> 2) * pitch + vcol + (lane & 3) * 4;
        B4 rhs4      = prep_transposed(rhs);
        if(key)
        {
            F4 product = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(rhs4, inv, F4{}, 0, 0, 0);
            prepare_write<SpxLong>(
                reinterpret_cast<B4*>(w + tile * C * K + qw_offset(rc, vcol + r4)),
                prep_pack(product));
        }
        else
        {
            // U is [time, value] directly in the next MFMA's accumulator layout.
            F4 product = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inv, rhs4, F4{}, 0, 0, 0);
            prepare_write<SpxLong>(
                reinterpret_cast<B4*>(u + tile * C * V + (vcol / 16 * 64 + lane) * 4),
                prep_pack(product));
        }
    }
}

template <bool FULL_ALIGNED, bool Single = false, bool SpxLong = false, bool Blocked = false>
__global__ __launch_bounds__(256, (SpxLong ? 2 : 1)) void prepare_kernel(const B* __restrict__ q,
                                                                         const B* __restrict__ k,
                                                                         const B* __restrict__ v,
                                                                         const B* __restrict__ g,
                                                                         const void* beta,
                                                                         bool beta32,
                                                                         const float* alog,
                                                                         const float* dt,
                                                                         const void* cu,
                                                                         bool cu64,
                                                                         const int32_t* ci,
                                                                         B* __restrict__ qg,
                                                                         B* __restrict__ w,
                                                                         B* __restrict__ u,
                                                                         B* __restrict__ kg,
                                                                         B* __restrict__ aq,
                                                                         float* __restrict__ decay,
                                                                         int H,
                                                                         int64_t sq,
                                                                         int64_t sk,
                                                                         int64_t sv,
                                                                         int64_t sg,
                                                                         int64_t sb,
                                                                         float lb,
                                                                         float scale,
                                                                         bool live_only)
{
    const int64_t group = int64_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const int id        = group / 64;
    // A one-sequence buffer uses sequence slot zero. Its start, end, and
    // chunk map are still read from device memory on every graph replay.
    const int seq       = Single ? 0 : ci[2 * id];
    const int64_t start = ix(cu, seq, cu64) + int64_t(ci[2 * id + 1]) * C;
    const int64_t end   = ix(cu, seq + 1, cu64);
    if constexpr(FULL_ALIGNED)
    {
        if(start + C <= end)
        {
            prepare_tile<true, SpxLong, Blocked>(q,
                                                 k,
                                                 v,
                                                 g,
                                                 beta,
                                                 beta32,
                                                 alog,
                                                 dt,
                                                 qg,
                                                 w,
                                                 u,
                                                 kg,
                                                 aq,
                                                 decay,
                                                 H,
                                                 sq,
                                                 sk,
                                                 sv,
                                                 sg,
                                                 sb,
                                                 lb,
                                                 scale,
                                                 start,
                                                 end);
            return;
        }
    }
    // The prefetched walker skips tiles beyond the live device length.
    // Raw prepare still materializes their identity transform.
    if(live_only && start >= end)
        return;
    prepare_tile<false, SpxLong, Blocked>(q,
                                          k,
                                          v,
                                          g,
                                          beta,
                                          beta32,
                                          alog,
                                          dt,
                                          qg,
                                          w,
                                          u,
                                          kg,
                                          aq,
                                          decay,
                                          H,
                                          sq,
                                          sk,
                                          sv,
                                          sg,
                                          sb,
                                          lb,
                                          scale,
                                          start,
                                          end);
}

// A wave shares one set of prepared Q/K operands across four V16 tiles.
constexpr int walk_value_width = 64, walk_waves = 4;
template <bool PrefetchChunks,
          int Width      = 64,
          int Waves      = 4,
          int Depth      = 7,
          int HeadGroup  = 16,
          bool Transpose = false,
          bool Rolling   = false,
          class... Segment>
__global__ __launch_bounds__(Waves * 64) void walk_kernel(const B* __restrict__ qg,
                                                          const B* __restrict__ w,
                                                          const B* __restrict__ u,
                                                          const B* __restrict__ kg,
                                                          const B* __restrict__ aq,
                                                          const float* __restrict__ decay,
                                                          const void* cu,
                                                          bool cu64,
                                                          const int64_t* offsets,
                                                          const float* initial,
                                                          float* final,
                                                          B* out,
                                                          int H,
                                                          int64_t ss0,
                                                          int64_t sf0,
                                                          int64_t so,
                                                          Segment... section)
{
    using B8                      = B __attribute__((ext_vector_type(8)));
    constexpr int value_fragments = Width / 16, nkt = K / 16, nv = V / Width;
    const int block = blockIdx.x * Waves + threadIdx.x / 64;
    const int seq = block / (H * nv), item = block % (H * nv);
    // Interleave independent heads before advancing to the next value tile.
    const int h      = PrefetchChunks ? (item / (HeadGroup * nv)) * HeadGroup + item % HeadGroup
                                      : (block / nv) % H;
    const int v_tile = PrefetchChunks ? (item / HeadGroup) % nv : block % nv;
    const int v0 = v_tile * value_fragments * 16, lane = threadIdx.x & 63;
    const int col = lane & 15, row = (lane >> 4) * 4;
    int64_t bos = ix(cu, seq, cu64), eos = ix(cu, seq + 1, cu64);
    int64_t c0 = offsets[seq], nc = offsets[seq + 1] - c0;
    if constexpr(sizeof...(Segment))
    {
        const int limits[]  = {section...};
        const int64_t begin = limits[0], chunks = limits[1];
        bos = min(eos, bos + begin * C);
        eos = min(eos, bos + chunks * C);
        nc  = max(int64_t(0), min(chunks, nc - begin));
        c0  = 0; // Scratch pointers start at this segment's first chunk.
    }
    const int64_t state_index = (int64_t(h) * V + v0 + col) * K + row;
    F4 state[value_fragments][nkt];
#pragma unroll
    for(int fragment = 0; fragment < value_fragments; fragment++)
    {
#pragma unroll
        for(int kt = 0; kt < nkt; kt++)
        {
#pragma unroll
            for(int i = 0; i < 4; i++)
                state[fragment][kt][i] =
                    initial ? initial[seq * ss0 + state_index + fragment * 16 * K + 16 * kt + i]
                            : 0;
        }
    }
    struct ChunkInputs
    {
        B8 wt[2], qt[2];
        B4 key[nkt], value[value_fragments], aq;
        F4 d[nkt];
    };
    auto load_chunk = [&](int64_t head) {
        const auto tile = head * C;
        ChunkInputs input;
        input.aq = *reinterpret_cast<const B4*>(aq + tile * C + col * C + row);
#pragma unroll
        for(int fragment = 0; fragment < value_fragments; fragment++)
            input.value[fragment] =
                *reinterpret_cast<const B4*>(u + tile * V + ((v0 / 16 + fragment) * 64 + lane) * 4);
#pragma unroll
        for(int kt = 0; kt < nkt; kt++)
        {
            input.key[kt] = *reinterpret_cast<const B4*>(kg + tile * K + (kt * 64 + lane) * 4);
            input.d[kt]   = *reinterpret_cast<const F4*>(decay + head * K + kt * 16 + row);
        }
#pragma unroll
        for(int p = 0; p < 2; p++)
        {
            input.wt[p] = *reinterpret_cast<const B8*>(w + tile * K + (p * 64 + lane) * 8);
            input.qt[p] = *reinterpret_cast<const B8*>(qg + tile * K + (p * 64 + lane) * 8);
        }
        return input;
    };
    auto step = [&](const ChunkInputs& input, int64_t t, auto full_chunk) {
#pragma unroll
        for(int fragment = 0; fragment < value_fragments; fragment++)
        {
            F4 prediction = {}, from_state = {};
#pragma unroll
            for(int p = 0; p < 2; p++)
            {
                B8 state_bf16;
#pragma unroll
                for(int half = 0; half < 2; half++)
                {
#pragma unroll
                    for(int i = 0; i < 4; i++)
                        state_bf16[4 * half + i] = b(state[fragment][2 * p + half][i]);
                }
                prediction = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    input.wt[p], state_bf16, prediction, 0, 0, 0);
                from_state =
                    __builtin_amdgcn_mfma_f32_16x16x32_bf16(Transpose ? state_bf16 : input.qt[p],
                                                            Transpose ? input.qt[p] : state_bf16,
                                                            from_state,
                                                            0,
                                                            0,
                                                            0);
            }
            B4 residual;
#pragma unroll
            for(int i = 0; i < 4; i++)
                residual[i] = b(f(input.value[fragment][i]) - prediction[i]);
            F4 local = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                Transpose ? residual : input.aq, Transpose ? input.aq : residual, F4{}, 0, 0, 0);
            if constexpr(Transpose)
            {
                // This MFMA orientation maps the four results to adjacent value
                // channels.
                const int64_t token = t - row + col;
                if(decltype(full_chunk)::value || token < eos)
                {
                    B* dst = out + token * so + h * V + v0 + 16 * fragment + row;
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        dst[i] = b(from_state[i] + local[i]);
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < 4; i++)
                    if(decltype(full_chunk)::value || t + i < eos)
                        out[(t + i) * so + h * V + v0 + 16 * fragment + col] =
                            b(from_state[i] + local[i]);
            }
#pragma unroll
            for(int kt = 0; kt < nkt; kt++)
            {
                F4 delta = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    input.key[kt], residual, F4{}, 0, 0, 0);
#pragma unroll
                for(int i = 0; i < 4; i++)
                    state[fragment][kt][i] = fmaf(state[fragment][kt][i], input.d[kt][i], delta[i]);
            }
        }
    };
    if constexpr(PrefetchChunks)
    {
        constexpr int prefetch_chunks = Depth;
        auto walk_complete            = [&](int64_t count) {
            if(count)
            {
                int64_t head = c0 * H + h, t = bos + row, c = 0;
                if constexpr(Rolling)
                {
                    // Load a replacement while the current tile is still available in registers.
                    // head and t advance when every slot in the window has been consumed.
                    if(count >= Depth)
                    {
                        ChunkInputs queue[Depth];
#pragma unroll
                        for(int i = 0; i < Depth; ++i)
                            queue[i] = load_chunk(head + i * H);
                        for(; c + 2 * Depth <= count; c += Depth, head += Depth * H, t += Depth * C)
                        {
#pragma unroll
                            for(int i = 0; i < Depth; ++i)
                            {
                                auto next = load_chunk(head + (Depth + i) * H);
                                step(queue[i], t + i * C, std::true_type{});
                                queue[i] = next;
                            }
                        }
#pragma unroll
                        for(int i = 0; i < Depth; ++i)
                            step(queue[i], t + i * C, std::true_type{});
                        c += Depth;
                        head += Depth * H;
                        t += Depth * C;
                    }
                }
                else
                {
                    for(; c + prefetch_chunks <= count;
                        c += prefetch_chunks, head += prefetch_chunks * H, t += prefetch_chunks * C)
                    {
                        ChunkInputs inputs[prefetch_chunks];
#pragma unroll
                        for(int i = 0; i < prefetch_chunks; ++i)
                            inputs[i] = load_chunk(head + i * H);
#pragma unroll
                        for(int i = 0; i < prefetch_chunks; ++i)
                            step(inputs[i], t + i * C, std::true_type{});
                    }
                }
                for(; c < count; c++, head += H, t += C)
                    step(load_chunk(head), t, std::true_type{});
            }
        };
        // Process only live chunks in a padded graph. A final partial chunk needs
        // predicated output stores; later prepared tiles represent the identity.
        const int64_t token_count = eos - bos;
        const int64_t complete    = min(nc, token_count / C);
        const int64_t active      = min(nc, (token_count + C - 1) / C);
        if(token_count >= nc * C)
            walk_complete(nc);
        else
        {
            walk_complete(complete);
            if(complete < active)
                step(load_chunk((c0 + complete) * H + h),
                     bos + complete * C + row,
                     std::false_type{});
        }
    }
    else if(nc)
    {
        int64_t head = c0 * H + h, t = bos + row;
        ChunkInputs even = load_chunk(head);
        for(int c = 0; c + 1 < nc; c += 2)
        {
            ChunkInputs odd = load_chunk(head + H);
            step(even, t, std::false_type{});
            if(c + 2 < nc)
                even = load_chunk(head + 2 * H);
            step(odd, t + C, std::false_type{});
            head += 2 * H;
            t += 2 * C;
        }
        if(nc & 1)
            step(even, t, std::false_type{});
    }
    if(final)
    {
#pragma unroll
        for(int fragment = 0; fragment < value_fragments; fragment++)
        {
#pragma unroll
            for(int kt = 0; kt < nkt; kt++)
            {
#pragma unroll
                for(int i = 0; i < 4; i++)
                    final[seq * sf0 + state_index + fragment * 16 * K + kt * 16 + i] =
                        state[fragment][kt][i];
            }
        }
    }
}

using WB8 = B __attribute__((ext_vector_type(8)));

struct UpdateOperands
{
    WB8 w[2];
    B4 key[4], value;
    F4 decay[4];
};

struct OutputOperands
{
    WB8 q[2];
    B4 aq;
};

struct OutputContext
{
    WB8 old[2];
    B4 residual;
};

struct WaveExchange
{
    WB8 old[2][2][64];
    B4 residual[2][64];
};

// One CTA owns head h and 16 value channels throughout a segment.
// The state wave sends rounded operands to the output wave through two LDS slots.
__global__ __launch_bounds__(128) void fresh_walk_part_kernel(const B* __restrict__ qg,
                                                              const B* __restrict__ w,
                                                              const B* __restrict__ u,
                                                              const B* __restrict__ kg,
                                                              const B* __restrict__ aq,
                                                              const float* __restrict__ d,
                                                              const void* cu,
                                                              bool wide,
                                                              const int64_t* offsets,
                                                              const float*,
                                                              float* final,
                                                              B* out,
                                                              int H,
                                                              int64_t,
                                                              int64_t,
                                                              int64_t so,
                                                              int begin,
                                                              int count)
{
    const int h = blockIdx.x % H, v = blockIdx.x / H;
    const int lane = threadIdx.x & 63, row = lane / 16 * 4, col = lane % 16;
    const int64_t seq_end = ix(cu, 1, wide);
    const int64_t bos     = min(seq_end, ix(cu, 0, wide) + int64_t(begin) * C);
    const int64_t eos     = min(seq_end, bos + int64_t(count) * C);
    const int64_t chunks  = max(int64_t(0), min(int64_t(count), offsets[1] - offsets[0] - begin));
    const int active      = min(chunks, (eos - bos + C - 1) / C);
    const int complete    = min(chunks, (eos - bos) / C);
    const int offset      = ((h * V + v * 16 + col) * K + row);
    constexpr int Depth   = 6;
    F4 state[4]{};
    const bool producer = threadIdx.x < 64;
    auto load_update    = [&](int c) {
        const int tile = c * H + h;
        UpdateOperands a;
        a.value = *reinterpret_cast<const B4*>(u + tile * C * V + (v * 64 + lane) * 4);
#pragma unroll
        for(int p = 0; p < 2; ++p)
            a.w[p] = *reinterpret_cast<const WB8*>(w + tile * C * K + (p * 64 + lane) * 8);
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            a.key[k]   = *reinterpret_cast<const B4*>(kg + tile * C * K + (k * 64 + lane) * 4);
            a.decay[k] = *reinterpret_cast<const F4*>(d + tile * K + k * 16 + row);
        }
        return a;
    };
    auto load_output = [&](int c) {
        const int tile = c * H + h;
        OutputOperands a;
        a.aq = *reinterpret_cast<const B4*>(aq + tile * C * C + col * C + row);
#pragma unroll
        for(int p = 0; p < 2; ++p)
            a.q[p] = *reinterpret_cast<const WB8*>(qg + tile * C * K + (p * 64 + lane) * 8);
        return a;
    };
    auto update = [&](const UpdateOperands& a) {
        OutputContext ctx;
        F4 pred{};
#pragma unroll
        for(int p = 0; p < 2; ++p)
        {
#pragma unroll
            for(int j = 0; j < 8; ++j)
                ctx.old[p][j] = b(state[2 * p + j / 4][j % 4]);
            pred = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a.w[p], ctx.old[p], pred, 0, 0, 0);
        }
#pragma unroll
        for(int j = 0; j < 4; ++j)
            ctx.residual[j] = b(f(a.value[j]) - pred[j]);
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            F4 delta =
                __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a.key[k], ctx.residual, F4{}, 0, 0, 0);
#pragma unroll
            for(int j = 0; j < 4; ++j)
                state[k][j] = fmaf(state[k][j], a.decay[k][j], delta[j]);
        }
        return ctx;
    };
    auto emit = [&](int c, const OutputOperands& a, const OutputContext& ctx, auto full) {
        F4 base{};
#pragma unroll
        for(int p = 0; p < 2; ++p)
            base = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ctx.old[p], a.q[p], base, 0, 0, 0);
        F4 local = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ctx.residual, a.aq, F4{}, 0, 0, 0);
        const int64_t t = bos + c * C + col;
        if(decltype(full)::value || t < eos)
        {
#pragma unroll
            for(int j = 0; j < 4; ++j)
                out[t * so + h * V + v * 16 + row + j] = b(base[j] + local[j]);
        }
    };
    __shared__ WaveExchange slots;
    auto exchange = [&](int c, auto full, auto depth) {
        constexpr int n = decltype(depth)::value;
        UpdateOperands a[n];
        OutputOperands o[n];
        if(producer)
        {
#pragma unroll
            for(int i = 0; i < n; ++i)
                a[i] = load_update(c + i);
        }
        else
        {
#pragma unroll
            for(int i = 0; i < n; ++i)
                o[i] = load_output(c + i);
        }
#pragma unroll
        for(int i = 0; i < n; ++i)
        {
            const int slot = (c + i) & 1;
            if(producer)
            {
                auto ctx                   = update(a[i]);
                slots.old[slot][0][lane]   = ctx.old[0];
                slots.old[slot][1][lane]   = ctx.old[1];
                slots.residual[slot][lane] = ctx.residual;
            }
            __syncthreads();
            if(!producer)
            {
                OutputContext ctx;
                ctx.old[0]   = slots.old[slot][0][lane];
                ctx.old[1]   = slots.old[slot][1][lane];
                ctx.residual = slots.residual[slot][lane];
                emit(c + i, o[i], ctx, full);
            }
        }
    };
    int c = 0;
    for(; c + Depth <= complete; c += Depth)
        exchange(c, std::true_type{}, std::integral_constant<int, Depth>{});
    for(; c < active; ++c)
        exchange(c, std::false_type{}, std::integral_constant<int, 1>{});
    if(producer && final)
    {
#pragma unroll
        for(int k = 0; k < 4; ++k)
#pragma unroll
            for(int j = 0; j < 4; ++j)
                final[offset + k * 16 + j] = state[k][j];
    }
}

// Packed offsets cover one contiguous interval, including empty interior sequences.
struct NormRange
{
    const void* cu;
    int64_t nseq;
    bool wide;
};

__device__ __forceinline__ bool norm_token_live(int64_t) { return true; }
__device__ __forceinline__ bool norm_token_live(int64_t t, NormRange span)
{
    return t >= ix(span.cu, 0, span.wide) && t < ix(span.cu, span.nseq, span.wide);
}

// Each wave normalizes one V128 row; four independent rows share a CTA.
template <class... Bounds>
__global__ void norm_kernel(const B* x,
                            const B* gate,
                            const void* weight,
                            bool w32,
                            B* out,
                            int T,
                            int64_t sx,
                            int64_t sg,
                            int64_t so,
                            float eps,
                            Bounds... bounds)
{
    int64_t row = int64_t(blockIdx.x) * 4 + (threadIdx.x >> 6);
    if(row >= int64_t(T) * 64)
        return;
    int64_t t = row >> 6;
    if(!norm_token_live(t, bounds...))
        return;
    int h = row & 63, lane = threadIdx.x & 63;
    float z0 = f(x[t * sx + h * V + lane]), z1 = f(x[t * sx + h * V + lane + 64]);
    float inv = rsqrtf(sum64(z0 * z0 + z1 * z1) / V + eps);
    out[t * so + h * V + lane] =
        b(z0 * inv * load_float(weight, lane, w32) * sig(f(gate[t * sg + h * V + lane])));
    out[t * so + h * V + lane + 64] =
        b(z1 * inv * load_float(weight, lane + 64, w32) * sig(f(gate[t * sg + h * V + lane + 64])));
}

// A 16-lane group owns one V128 row. Contiguous 16-byte loads/stores
// cover eight BF16 values per lane and four rows per wavefront.
using B8 = __bf16 __attribute__((ext_vector_type(8)));
template <bool W32, class... Bounds>
__global__ __launch_bounds__(256) void norm_vector_kernel(const B* x,
                                                          const B* gate,
                                                          const void* weight,
                                                          B* out,
                                                          int64_t sx,
                                                          int64_t sg,
                                                          int64_t so,
                                                          float eps,
                                                          Bounds... bounds)
{
    const int64_t row = int64_t(blockIdx.x) * 16 + (threadIdx.x >> 4);
    const int64_t t   = row >> 6;
    if(!norm_token_live(t, bounds...))
        return;
    const int col    = 8 * (threadIdx.x & 15);
    const int offset = (row & 63) * V + col;
    const B8 v       = *reinterpret_cast<const B8*>(x + t * sx + offset);
    const B8 g       = *reinterpret_cast<const B8*>(gate + t * sg + offset);
    float sum        = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i)
        sum = fmaf(f(v[i]), f(v[i]), sum);
#pragma unroll
    for(int d = 8; d; d >>= 1)
        sum += __shfl_xor(sum, d, 16);
    const float inv = rsqrtf(sum / V + eps);
    B8 result;
#pragma unroll
    for(int i = 0; i < 8; ++i)
        result[i] = b(f(v[i]) * inv * load_float(weight, col + i, W32) * sig(f(g[i])));
    *reinterpret_cast<B8*>(out + t * so + offset) = result;
}

// Each wave handles two consecutive heads at short sequence lengths.
template <bool W32, class... Bounds>
__global__ __launch_bounds__(64) void short_norm_kernel(
    const B* x, const B* gate, const void* weight, B* out, float eps, Bounds... bounds)
{
    using B2 = B __attribute__((ext_vector_type(2)));
    if(!norm_token_live(int64_t(blockIdx.x) >> 5, bounds...))
        return;
    const int col      = 2 * threadIdx.x;
    const int64_t base = int64_t(blockIdx.x) * 2 * V + col;
    B2 values[2], gates[2];
#pragma unroll
    for(int row = 0; row < 2; ++row)
    {
        values[row] = *reinterpret_cast<const B2*>(x + base + row * V);
        gates[row]  = *reinterpret_cast<const B2*>(gate + base + row * V);
    }
#pragma unroll
    for(int row = 0; row < 2; ++row)
    {
        float sum = fmaf(f(values[row][0]), f(values[row][0]), 0.f);
        sum       = fmaf(f(values[row][1]), f(values[row][1]), sum);
#pragma unroll
        for(int offset = 32; offset; offset >>= 1)
            sum += __shfl_xor(sum, offset, 64);
        const float inv = rsqrtf(sum / V + eps);
        B2 result;
#pragma unroll
        for(int j = 0; j < 2; ++j)
            result[j] = b(f(values[row][j]) * inv * load_float(weight, col + j, W32) *
                          sig(f(gates[row][j])));
        *reinterpret_cast<B2*>(out + base + row * V) = result;
    }
}

// Waves share this head's 128 value rows.
// Keep the original descending XOR reduction tree while each subgroup
// owns consecutive K columns for one V row.
template <int Pack, int FirstMask = 32 / Pack>
__device__ __forceinline__ float sum_state_columns(float (&x)[Pack])
{
#pragma unroll
    for(int mask = FirstMask; mask; mask >>= 1)
#pragma unroll
        for(int j = 0; j < Pack; ++j)
            x[j] += __shfl_xor(x[j], mask, 64 / Pack);
#pragma unroll
    for(int mask = Pack / 2; mask; mask >>= 1)
#pragma unroll
        for(int j = 0; j < mask; ++j)
            if constexpr(Pack > 2)
            {
                // Preserve the register reduction tree under fast-math.
                asm("v_add_f32 %0, %1, %2" : "=v"(x[j]) : "v"(x[j]), "v"(x[j + mask]));
            }
            else
            {
                x[j] += x[j + mask];
            }
    return x[0];
}

// Match the scalar dot product at XOR distance 32. The local product is
// fused into the addition; its partner has already rounded to FP32.
template <int Pack, class Columns>
__device__ __forceinline__ float
checkpoint_dot(const Columns& s, const float (&x)[Pack], float (&product)[Pack])
{
    if constexpr(Pack >= 8)
    {
#pragma unroll
        for(int j = 0; j < Pack; ++j)
        {
            const float peer = __shfl_xor(product[j], 32 / Pack, 64 / Pack);
            asm("v_fma_f32 %0, %1, %2, %3" : "=v"(product[j]) : "v"(s[j]), "v"(x[j]), "v"(peer));
        }
        return sum_state_columns<Pack, 16 / Pack>(product);
    }
    else
    {
        return sum_state_columns(product);
    }
}

template <bool SPEC,
          int Waves,
          bool SparseTokens = false,
          int Pack          = 1,
          bool HeadFirst    = false,
          bool StreamState  = false>
__global__ __launch_bounds__(64 * Waves) void decode_kernel(const B* x,
                                                            B* conv,
                                                            const B* weight,
                                                            const B* gate,
                                                            const void* beta,
                                                            bool beta32,
                                                            const B* og,
                                                            const float* alog,
                                                            const float* dt,
                                                            float* state,
                                                            const void* indices,
                                                            bool idx64,
                                                            const void* cu,
                                                            bool cu64,
                                                            const void* accepted,
                                                            bool acc64,
                                                            const void* cindices,
                                                            bool cidx64,
                                                            const void* nw,
                                                            bool nw32,
                                                            B* out,
                                                            int H,
                                                            int sequences,
                                                            int steps,
                                                            int state_len,
                                                            int64_t sx,
                                                            int64_t sc0,
                                                            int64_t sc1,
                                                            int64_t sc2,
                                                            int64_t sw0,
                                                            int64_t sw1,
                                                            int64_t sb,
                                                            int64_t sog,
                                                            int64_t ss0,
                                                            int64_t si0,
                                                            int64_t si1,
                                                            int64_t sci,
                                                            float lb,
                                                            float eps)
{
    int seq = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    if constexpr(HeadFirst)
    {
        seq = blockIdx.y;
        h   = blockIdx.x;
    }
    if constexpr(SparseTokens)
    {
        // A block handles a sequence only at its first live token. Multiple
        // leading/trailing empty slots and graph padding may share one offset.
        int lower = 0, upper = sequences;
        const int token = blockIdx.x;
        while(lower < upper)
        {
            const int mid = lower + (upper - lower) / 2;
            if(ix(cu, mid + 1, cu64) <= token)
                lower = mid + 1;
            else
                upper = mid;
        }
        if(lower == sequences || ix(cu, lower, cu64) != token)
            return;
        seq = lower;
    }
    int64_t bos = ix(cu, seq, cu64), eos = ix(cu, seq + 1, cu64);
    if(bos == eos)
        return;
    int cp = 0;
    if constexpr(SPEC)
    {
        int a = int(ix(accepted, seq, acc64));
        if(a > steps)
            return;
        cp = max(a - 1, 0);
    }
    // In head-first speculative decode each wave loads its checkpoint slots once.
    int64_t checkpoint = 0, slot;
    if constexpr(HeadFirst && SPEC)
    {
        if(lane < steps)
            checkpoint = ix(indices, seq * si0 + lane * si1, idx64);
        slot = __shfl(checkpoint, cp, 64);
    }
    else
    {
        slot = ix(indices, seq * si0 + cp * si1, idx64);
    }
    int64_t cslot = SPEC ? ix(cindices, seq * sci, cidx64) : slot;
    if(slot < (SPEC ? 1 : 0) || cslot < (SPEC ? 1 : 0))
        return;
    int n = int(eos - bos);
    if constexpr(SPEC)
    {
        n = min(n, min(steps, state_len - 2));
        if constexpr(HeadFirst)
        {
            const unsigned long long invalid = __ballot(lane < steps && checkpoint <= 0);
            if(invalid)
                n = min(n, __ffsll(invalid) - 1);
        }
        else
        {
            for(int j = 0; j < steps; j++)
                if(ix(indices, seq * si0 + j * si1, idx64) <= 0)
                {
                    n = min(n, j);
                    break;
                }
        }
        if(n <= 0)
            return;
    }
    int qc = h * K + lane, kc = H * K + qc, vc = 2 * H * K + h * V + (wave & 1) * 64 + lane;
    B* cache = conv + cslot * sc0;
    float qhist[3], khist[3], vhist[3], qw[4], kw[4], vw[4];
#pragma unroll
    for(int j = 0; j < 3; j++)
    {
        int pos  = SPEC ? cp + j : state_len - 3 + j;
        qhist[j] = f(cache[qc * sc1 + pos * sc2]);
        khist[j] = f(cache[kc * sc1 + pos * sc2]);
        vhist[j] = f(cache[vc * sc1 + pos * sc2]);
    }
#pragma unroll
    for(int j = 0; j < 4; j++)
    {
        qw[j] = f(weight[qc * sw0 + j * sw1]);
        kw[j] = f(weight[kc * sw0 + j * sw1]);
        vw[j] = f(weight[vc * sw0 + j * sw1]);
    }
    constexpr int group = K / Pack;
    using Columns       = float __attribute__((ext_vector_type(Pack)));
    const int column    = (lane % group) * Pack;
    const int subrow    = lane / group;
    float s[V / Waves];
    Columns state_columns[V / (Waves * Pack)];
    if constexpr(Pack == 1)
    {
#pragma unroll
        for(int r = 0; r < V / Waves; r++)
            s[r] = state[slot * ss0 + (int64_t(h) * V + wave + Waves * r) * K + lane];
    }
    else
    {
#pragma unroll
        for(int r = 0; r < V / (Waves * Pack); r++)
        {
            int row          = (wave + Waves * r) * Pack + subrow;
            state_columns[r] = *reinterpret_cast<const Columns*>(
                state + slot * ss0 + (int64_t(h) * V + row) * K + column);
        }
    }
    __shared__ float vv[V], oo[V], partials[Waves == 4 ? 4 : 2], rstd;
    __syncthreads();
    if constexpr(SPEC)
    {
        if(wave == 0)
            for(int j = 0; j < 2; j++)
            {
                cache[qc * sc1 + j * sc2] = b(qhist[j + 1]);
                cache[kc * sc1 + j * sc2] = b(khist[j + 1]);
            }
        if(wave < 2)
            for(int j = 0; j < 2; j++)
                cache[vc * sc1 + j * sc2] = b(vhist[j + 1]);
    }
    for(int it = 0; it < n; it++)
    {
        int64_t t = bos + it;
        float qx = f(x[t * sx + qc]), kx = f(x[t * sx + kc]), vx = f(x[t * sx + vc]);
        float qv = qx * qw[3], kv = kx * kw[3], val = vx * vw[3];
#pragma unroll
        for(int j = 0; j < 3; j++)
        {
            qv  = fmaf(qhist[j], qw[j], qv);
            kv  = fmaf(khist[j], kw[j], kv);
            val = fmaf(vhist[j], vw[j], val);
        }
        qv  = silu(qv);
        kv  = silu(kv);
        val = silu(val);
        qv *= rsqrtf(sum64(qv * qv) + 1e-6f) * 0.125f;
        kv *= rsqrtf(sum64(kv * kv) + 1e-6f);
        float dec =
            expf(lb * sig(expf(alog[h]) * (f(gate[(t * H + h) * K + lane]) + dt[h * K + lane])));
        float be = sig(load_float(beta, t * sb + h, beta32));
        if(wave < 2)
            vv[(wave & 1) * 64 + lane] = val;
        int64_t dest;
        if constexpr(HeadFirst && SPEC)
            dest = __shfl(checkpoint, it, 64);
        else
            dest = SPEC ? ix(indices, seq * si0 + it * si1, idx64) : slot;
        __syncthreads();
        if constexpr(Pack == 1)
        {
#pragma unroll
            for(int r = 0; r < V / Waves; r++)
            {
                int row = wave + Waves * r;
                s[r] *= dec;
                float dot = sum64(s[r] * kv);
                float dv  = (vv[row] - dot) * be;
                s[r]      = fmaf(dv, kv, s[r]);
                float z   = sum64(s[r] * qv);
                if(lane == 0)
                    oo[row] = f(b(z));
                state[dest * ss0 + (int64_t(h) * V + row) * K + lane] = s[r];
            }
        }
        else
        {
            float qcols[Pack], kcols[Pack], dcols[Pack];
#pragma unroll
            for(int j = 0; j < Pack; ++j)
            {
                qcols[j] = __shfl(qv, column + j, 64);
                kcols[j] = __shfl(kv, column + j, 64);
                dcols[j] = __shfl(dec, column + j, 64);
            }
#pragma unroll
            for(int r = 0; r < V / (Waves * Pack); r++)
            {
                int row = (wave + Waves * r) * Pack + subrow;
                float product[Pack];
#pragma unroll
                for(int j = 0; j < Pack; ++j)
                {
                    state_columns[r][j] *= dcols[j];
                    product[j] = state_columns[r][j] * kcols[j];
                }
                float dot = checkpoint_dot(state_columns[r], kcols, product);
                float dv  = (vv[row] - dot) * be;
#pragma unroll
                for(int j = 0; j < Pack; ++j)
                {
                    state_columns[r][j] = fmaf(dv, kcols[j], state_columns[r][j]);
                    product[j]          = state_columns[r][j] * qcols[j];
                }
                float z = checkpoint_dot(state_columns[r], qcols, product);
                if(lane % group == 0)
                    oo[row] = f(b(z));
                auto* checkpoint_row = reinterpret_cast<Columns*>(
                    state + dest * ss0 + (int64_t(h) * V + row) * K + column);
                if constexpr(StreamState)
                    __builtin_nontemporal_store(state_columns[r], checkpoint_row);
                else
                    *checkpoint_row = state_columns[r];
            }
        }
        int pos = SPEC ? 2 + it : 2;
#pragma unroll
        for(int j = 0; j < 2; j++)
        {
            qhist[j] = qhist[j + 1];
            khist[j] = khist[j + 1];
            vhist[j] = vhist[j + 1];
        }
        qhist[2] = qx;
        khist[2] = kx;
        vhist[2] = vx;
        if(wave == 0)
        {
            if constexpr(!SPEC)
                for(int j = 0; j < 2; j++)
                {
                    cache[qc * sc1 + j * sc2] = b(qhist[j]);
                    cache[kc * sc1 + j * sc2] = b(khist[j]);
                }
            cache[qc * sc1 + pos * sc2] = b(qx);
            cache[kc * sc1 + pos * sc2] = b(kx);
        }
        if(wave < 2)
        {
            if constexpr(!SPEC)
                for(int j = 0; j < 2; j++)
                    cache[vc * sc1 + j * sc2] = b(vhist[j]);
            cache[vc * sc1 + pos * sc2] = b(vx);
        }
        __syncthreads();
        if constexpr(!SparseTokens && (HeadFirst || (SPEC && Waves > 4)))
        {
            float z = tid < V ? oo[tid] : 0;
            if(wave < 2)
            {
                const float partial = sum64(z * z);
                if(lane == 0)
                    partials[wave] = partial;
            }
            __syncthreads();
            if(tid < V)
            {
                const float inv_rms = rsqrtf((partials[0] + partials[1]) / V + eps);
                out[(t * H + h) * V + tid] =
                    b(z * inv_rms * load_float(nw, tid, nw32) * sig(f(og[t * sog + h * V + tid])));
            }
        }
        else
        {
            float z = tid < V ? oo[tid] : 0, ps = sum64(z * z);
            if(lane == 0 && (Waves == 4 || wave < 2))
                partials[wave] = ps;
            __syncthreads();
            if(tid == 0)
            {
                if constexpr(Waves == 4)
                    rstd =
                        rsqrtf((partials[0] + partials[1] + partials[2] + partials[3]) / V + eps);
                else
                    rstd = rsqrtf((partials[0] + partials[1]) / V + eps);
            }
            __syncthreads();
            if(tid < V)
                out[(t * H + h) * V + tid] =
                    b(z * rstd * load_float(nw, tid, nw32) * sig(f(og[t * sog + h * V + tid])));
            __syncthreads();
        }
    }
}

// The binding guards the input tensor's device before entering native code.
// Cache the CU count per host thread; graph replay retains the selected kernel.
static bool full_spx_device()
{
    int device;
    hipError_t error = hipGetDevice(&device);
    if(error != hipSuccess)
        throw std::runtime_error(hipGetErrorString(error));
    static thread_local int last_device = -1;
    static thread_local bool full_card  = false;
    if(device != last_device)
    {
        int cus;
        error = hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device);
        if(error != hipSuccess)
            throw std::runtime_error(hipGetErrorString(error));
        full_card   = cus == 256;
        last_device = device;
    }
    return full_card;
}

static thread_local hipStream_t current_stream;
static hipStream_t stream() { return current_stream; }
static const B* bp(const T& x) { return reinterpret_cast<const B*>(x.data_ptr()); }
static B* bmp(const T& x) { return reinterpret_cast<B*>(x.data_ptr()); }
static bool wide(const T& x) { return x.dtype == NATIVE_INT64; }
static bool fp32(const T& x) { return x.dtype == NATIVE_FLOAT; }
static bool aligned_for(const T& x, int bytes)
{
    return reinterpret_cast<uintptr_t>(x.data_ptr()) % bytes == 0;
}

// A cache view can share storage with the inputs. Defer the commit in
// that case, so every convolution reads the caller's original input.
bool overlaps_bf16(const T& a, const T& b)
{
    if(!a.numel() || !b.numel())
        return false;
    auto extent = [](const T& tensor) {
        int64_t elems = 1;
        for(int i = 0; i < tensor.ndim; ++i)
            elems += (tensor.size(i) - 1) * tensor.stride(i);
        return uintptr_t(elems) * sizeof(B);
    };
    auto pa = reinterpret_cast<uintptr_t>(a.data_ptr());
    auto pb = reinterpret_cast<uintptr_t>(b.data_ptr());
    return pa < pb + extent(b) && pb < pa + extent(a);
}

template <int Rows,
          int ChannelsPerLane,
          int Threads,
          int Prefetch,
          bool SplitFull,
          bool Blocked = false>
static void launch_tiled_conv(T x,
                              T weight,
                              Opt cache,
                              T cu,
                              Opt has,
                              T out,
                              int64_t sc0,
                              int64_t sc1,
                              int64_t sc2,
                              hipStream_t s)
{
    int64_t seqs   = cu.numel() - 1;
    int64_t groups = (x.size(0) + seqs * Rows - 1) / (seqs * Rows);
    dim3 grid(16384 / (ChannelsPerLane * Threads), min(groups, int64_t(65535)), seqs);
    hipLaunchKernelGGL(
        (conv_tiled_kernel<Rows, ChannelsPerLane, Threads, Prefetch, SplitFull, Blocked>),
        grid,
        Threads,
        0,
        s,
        bp(x),
        bp(weight),
        cache ? bmp(*cache) : nullptr,
        cu.data_ptr(),
        wide(cu),
        has ? has->data_ptr<bool>() : nullptr,
        bmp(out),
        x.stride(0),
        sc0,
        sc1,
        sc2);
}
// Keep absolute packed token coordinates and the original live sequence bounds.
// The block that owns BOS commits the final three inputs just as in the unsplit kernel.
bool native_conv_half(T x, T weight, Opt cache, T cu, Opt has, T out, int half, hipStream_t s)
{
    if(x.size(0) != 32768 || x.size(1) != 16384 || cu.numel() != 2 || weight.stride(0) != 4 ||
       weight.stride(1) != 1 || x.stride(0) % 4 || !aligned_for(x, 8) || !aligned_for(weight, 8) ||
       !aligned_for(out, 8) || !full_spx_device() ||
       (cache && (overlaps_bf16(*cache, x) || overlaps_bf16(*cache, weight))))
        return false;
    auto kernel = half ? conv_tiled_kernel<16, 4, 128, 8, true, true, 1>
                       : conv_tiled_kernel<16, 4, 128, 8, true, true, 0>;
    hipLaunchKernelGGL(kernel,
                       dim3(32, 1024),
                       128,
                       0,
                       s,
                       bp(x),
                       bp(weight),
                       cache ? bmp(*cache) : nullptr,
                       cu.data_ptr(),
                       wide(cu),
                       has ? has->data_ptr<bool>() : nullptr,
                       bmp(out),
                       x.stride(0),
                       cache ? cache->stride(0) : 0,
                       cache ? cache->stride(1) : 0,
                       cache ? cache->stride(2) : 0);
    KDA_CHECK_LAUNCH();
    return true;
}
bool native_conv(
    T x, T weight, Opt cache, T cu, Opt has, T out, bool for_mixer, bool allow_tiled, hipStream_t s)
{
    current_stream = s;
    int channels = x.size(1), H = channels / 256;
    auto sc0 = cache ? cache->stride(0) : 0, sc1 = cache ? cache->stride(1) : 0,
         sc2 = cache ? cache->stride(2) : 0;
    dim3 grid(min(int64_t(4096), (x.size(0) * channels + 255) / 256), cu.numel() - 1);
    if(x.numel() && cu.numel() > 1)
    {
        if(weight.stride(0) == 4 && weight.stride(1) == 1 && x.stride(0) % 4 == 0 &&
           aligned_for(x, 8) && aligned_for(weight, 8) && aligned_for(out, 8) && allow_tiled)
        {
            const int64_t sequences      = cu.numel() - 1;
            const int64_t average_length = x.size(0) / sequences;
            const bool spx               = full_spx_device();
            const bool spx_long          = spx && sequences == 1 && average_length >= 8192;
            const bool blocked           = for_mixer && spx_long && average_length == 32768;
            if(blocked)
                launch_tiled_conv<16, 4, 128, 8, true, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(spx_long && average_length >= 32768)
                launch_tiled_conv<16, 4, 128, 8, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(spx_long)
                launch_tiled_conv<8, 4, 64, 8, false>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(spx && sequences == 1 && average_length >= 2048)
                launch_tiled_conv<8, 4, 128, 8, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(spx && sequences > 1 &&
                    (average_length >= 1024 || (sequences >= 8 && average_length >= 512)))
                launch_tiled_conv<8, 4, 128, 8, false>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(average_length >= 8192)
                launch_tiled_conv<256, 4, 128, 16, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(average_length >= 2048 || (sequences > 1 && average_length >= 1024))
                launch_tiled_conv<64, 4, 128, 8, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(sequences > 1 && average_length >= 512)
                launch_tiled_conv<16, 4, 128, 8, true>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else if(average_length <= (spx ? 128 : 48))
                launch_tiled_conv<4, 2, 256, 4, false>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            else
                launch_tiled_conv<8, 4, 64, 8, false>(
                    x, weight, cache, cu, has, out, sc0, sc1, sc2, s);
            KDA_CHECK_LAUNCH();
            return blocked;
        }
        if(cache)
            hipLaunchKernelGGL((conv_kernel<true>),
                               grid,
                               256,
                               0,
                               stream(),
                               bp(x),
                               bp(weight),
                               bp(*cache),
                               cu.data_ptr(),
                               wide(cu),
                               has ? has->data_ptr<bool>() : nullptr,
                               bmp(out),
                               H,
                               channels,
                               x.stride(0),
                               sc0,
                               sc1,
                               sc2,
                               weight.stride(0),
                               weight.stride(1));
        else
            hipLaunchKernelGGL((conv_kernel<false>),
                               grid,
                               256,
                               0,
                               stream(),
                               bp(x),
                               bp(weight),
                               nullptr,
                               cu.data_ptr(),
                               wide(cu),
                               nullptr,
                               bmp(out),
                               H,
                               channels,
                               x.stride(0),
                               sc0,
                               sc1,
                               sc2,
                               weight.stride(0),
                               weight.stride(1));
        if(cache)
            hipLaunchKernelGGL(conv_commit_kernel,
                               dim3((channels + 255) / 256, cu.numel() - 1),
                               256,
                               0,
                               stream(),
                               bp(x),
                               bmp(*cache),
                               cu.data_ptr(),
                               wide(cu),
                               has ? has->data_ptr<bool>() : nullptr,
                               channels,
                               x.stride(0),
                               sc0,
                               sc1,
                               sc2);
        KDA_CHECK_LAUNCH();
    }
    return false;
}
// Prepare can leave inactive scratch untouched only when walk also skips it.
static bool use_prefetched_walk(int64_t tokens, int64_t sequences, int heads)
{
    return sequences > 0 && heads == 64 && tokens / sequences >= 512;
}

void native_prepare(T q,
                    T k,
                    T v,
                    T g,
                    T beta,
                    T a,
                    T dt,
                    T cu,
                    T ci,
                    T qg,
                    T w,
                    T u,
                    T kg,
                    T aq,
                    T decay,
                    double lb,
                    double scale,
                    bool for_walk,
                    bool allow_streaming,
                    bool blocked,
                    hipStream_t s)
{
    current_stream = s;
    int H = q.size(2), nt = ci.size(0);
    // Live device metadata determines whether each aligned tile takes the fast
    // path.
    const int64_t sequences = cu.numel() - 1;
    const bool aligned = q.stride(1) % 4 == 0 && k.stride(1) % 4 == 0 && g.stride(1) % 4 == 0 &&
                         v.stride(1) % 8 == 0 && aligned_for(q, 8) && aligned_for(k, 8) &&
                         aligned_for(g, 8) && aligned_for(v, 16);
    // Segmented blocked prepare sees the whole input and a slice of the chunk
    // map. Individual tiles check the live device range before aligned access.
    const bool dense_single    = sequences == 1 && (q.size(1) == int64_t(nt) * C ||
                                                 (blocked && q.size(1) >= int64_t(nt) * C));
    const bool prefetch_chunks = use_prefetched_walk(q.size(1), sequences, H);
    auto kernel                = aligned && dense_single ? prepare_kernel<true, true>
                                 : aligned && sequences > 1 && prefetch_chunks ? prepare_kernel<true, false>
                                                                               : prepare_kernel<false, false>;
    // The mixer supplies adjacent Q/K/V views of its convolution output.
    // Gate streaming on this producer layout; standalone dense inputs reuse
    // cache.
    const int64_t packed_stride = int64_t(H) * (2 * K + V);
    const uintptr_t q_address   = reinterpret_cast<uintptr_t>(q.data_ptr());
    const uintptr_t key_bytes   = uintptr_t(H) * K * sizeof(B);
    const bool packed_qkv       = q.stride(1) == packed_stride && k.stride(1) == packed_stride &&
                            v.stride(1) == packed_stride &&
                            reinterpret_cast<uintptr_t>(k.data_ptr()) == q_address + key_bytes &&
                            reinterpret_cast<uintptr_t>(v.data_ptr()) == q_address + 2 * key_bytes;
    if(for_walk && allow_streaming && q.size(1) >= 16384 && packed_qkv && full_spx_device())
    {
        kernel = aligned && dense_single                       ? prepare_kernel<true, true, true>
                 : aligned && sequences > 1 && prefetch_chunks ? prepare_kernel<true, false, true>
                                                               : prepare_kernel<false, false, true>;
    }
    if(blocked)
    {
        kernel = aligned && dense_single ? prepare_kernel<true, true, true, true>
                                         : prepare_kernel<false, false, true, true>;
    }
    if(nt)
        hipLaunchKernelGGL(kernel,
                           dim3(nt, H),
                           256,
                           0,
                           stream(),
                           bp(q),
                           bp(k),
                           bp(v),
                           bp(g),
                           beta.data_ptr(),
                           fp32(beta),
                           a.data_ptr<float>(),
                           dt.data_ptr<float>(),
                           cu.data_ptr(),
                           wide(cu),
                           ci.data_ptr<int32_t>(),
                           bmp(qg),
                           bmp(w),
                           bmp(u),
                           bmp(kg),
                           bmp(aq),
                           decay.data_ptr<float>(),
                           H,
                           q.stride(1),
                           k.stride(1),
                           v.stride(1),
                           g.stride(1),
                           beta.stride(1),
                           float(lb),
                           float(scale),
                           for_walk && prefetch_chunks);
    KDA_CHECK_LAUNCH();
}
void native_walk(T qg,
                 T w,
                 T u,
                 T kg,
                 T aq,
                 T decay,
                 T cu,
                 T offsets,
                 Opt initial,
                 Opt final,
                 T out,
                 hipStream_t s)
{
    current_stream = s;
    int H = out.size(2), n = cu.numel() - 1;
    const bool prefetch_chunks = use_prefetched_walk(out.size(1), n, H);
    auto kernel                = prefetch_chunks ? walk_kernel<true> : walk_kernel<false>;
    int width = walk_value_width, waves = walk_waves;
    if(prefetch_chunks && n < 5 && full_spx_device())
    {
        if(n > 1)
        {
            kernel = walk_kernel<true, 32, 1, 2, 32>;
            width  = 32;
            waves  = 1;
        }
        else
        {
            kernel = walk_kernel<true, 16, 1, 2, 64, true>;
            if(out.size(1) == 32768)
            {
                kernel = walk_kernel<true, 16, 1, 3, 64, true, true>;
            }
            width = 16;
            waves = 1;
        }
    }
    if(n)
        hipLaunchKernelGGL(kernel,
                           dim3(n * H * (V / width) / waves),
                           64 * waves,
                           0,
                           stream(),
                           bp(qg),
                           bp(w),
                           bp(u),
                           bp(kg),
                           bp(aq),
                           decay.data_ptr<float>(),
                           cu.data_ptr(),
                           wide(cu),
                           offsets.data_ptr<int64_t>(),
                           initial ? initial->data_ptr<float>() : nullptr,
                           final ? final->data_ptr<float>() : nullptr,
                           bmp(out),
                           H,
                           initial ? initial->stride(0) : 0,
                           final ? final->stride(0) : 0,
                           out.stride(1));
    KDA_CHECK_LAUNCH();
}
void native_walk_part(T qg,
                      T w,
                      T u,
                      T kg,
                      T aq,
                      T decay,
                      T cu,
                      T offsets,
                      Opt initial,
                      Opt final,
                      T out,
                      int begin,
                      hipStream_t s)
{
    current_stream  = s;
    constexpr int H = 64;
    auto kernel     = walk_kernel<true, 16, 1, 2, 64, true, false, int, int>;
    int threads     = 64;
    if(out.size(1) == 32768 && cu.numel() == 2 && full_spx_device())
    {
        if(begin == 0 && !initial)
        {
            kernel  = fresh_walk_part_kernel;
            threads = 128;
        }
        else
        {
            kernel = walk_kernel<true, 16, 1, 3, 64, true, true, int, int>;
        }
    }
    hipLaunchKernelGGL(kernel,
                       dim3(H * (V / 16)),
                       threads,
                       0,
                       s,
                       bp(qg),
                       bp(w),
                       bp(u),
                       bp(kg),
                       bp(aq),
                       decay.data_ptr<float>(),
                       cu.data_ptr(),
                       wide(cu),
                       offsets.data_ptr<int64_t>(),
                       initial ? initial->data_ptr<float>() : nullptr,
                       final ? final->data_ptr<float>() : nullptr,
                       bmp(out),
                       H,
                       initial ? initial->stride(0) : 0,
                       final ? final->stride(0) : 0,
                       out.stride(1),
                       begin,
                       int(qg.size(0)));
    KDA_CHECK_LAUNCH();
}
template <class... Bounds>
static void dispatch_norm(T x, T gate, T weight, T out, double eps, hipStream_t s, Bounds... bounds)
{
    current_stream = s;
    if(!x.size(1))
        return;
    if(x.stride(1) % 8 == 0 && gate.stride(1) % 8 == 0 && out.stride(1) % 8 == 0 &&
       aligned_for(x, 16) && aligned_for(gate, 16) && aligned_for(out, 16))
    {
        if(x.size(1) <= 16 && x.stride(1) == 64 * V && gate.stride(1) == 64 * V &&
           out.stride(1) == 64 * V && full_spx_device())
        {
            auto kernel = fp32(weight) ? short_norm_kernel<true, Bounds...>
                                       : short_norm_kernel<false, Bounds...>;
            hipLaunchKernelGGL(kernel,
                               dim3(x.size(1) * 32),
                               64,
                               0,
                               stream(),
                               bp(x),
                               bp(gate),
                               weight.data_ptr(),
                               bmp(out),
                               float(eps),
                               bounds...);
            KDA_CHECK_LAUNCH();
            return;
        }
        dim3 grid(x.size(1) * 4);
        if(fp32(weight))
            hipLaunchKernelGGL((norm_vector_kernel<true, Bounds...>),
                               grid,
                               256,
                               0,
                               stream(),
                               bp(x),
                               bp(gate),
                               weight.data_ptr(),
                               bmp(out),
                               x.stride(1),
                               gate.stride(1),
                               out.stride(1),
                               float(eps),
                               bounds...);
        else
            hipLaunchKernelGGL((norm_vector_kernel<false, Bounds...>),
                               grid,
                               256,
                               0,
                               stream(),
                               bp(x),
                               bp(gate),
                               weight.data_ptr(),
                               bmp(out),
                               x.stride(1),
                               gate.stride(1),
                               out.stride(1),
                               float(eps),
                               bounds...);
    }
    else
        hipLaunchKernelGGL((norm_kernel<Bounds...>),
                           dim3((x.size(1) * 64 + 3) / 4),
                           256,
                           0,
                           stream(),
                           bp(x),
                           bp(gate),
                           weight.data_ptr(),
                           fp32(weight),
                           bmp(out),
                           int(x.size(1)),
                           x.stride(1),
                           gate.stride(1),
                           out.stride(1),
                           float(eps),
                           bounds...);
    KDA_CHECK_LAUNCH();
}
void native_norm(T x, T gate, T weight, T out, double eps, hipStream_t s)
{
    dispatch_norm(x, gate, weight, out, eps, s);
}
void native_norm_live(T x, T gate, T weight, T out, T cu, double eps, hipStream_t s)
{
    dispatch_norm(x, gate, weight, out, eps, s, NormRange{cu.data_ptr(), cu.numel() - 1, wide(cu)});
}
void native_decode(T x,
                   T conv,
                   T weight,
                   T gate,
                   T beta,
                   T og,
                   T a,
                   T dt,
                   T state,
                   T indices,
                   T cu,
                   T nw,
                   Opt accepted,
                   Opt cindices,
                   T out,
                   double lb,
                   double eps,
                   hipStream_t s)
{
    current_stream = s;
    int H = state.size(1), n = cu.numel() - 1;
    bool spec = accepted.has_value();
    int steps = spec ? indices.size(1) : 1;
    if(n)
    {
#define ARGS                                                                                    \
    bp(x), bmp(conv), bp(weight), bp(gate), beta.data_ptr(), fp32(beta), bp(og),                \
        a.data_ptr<float>(), dt.data_ptr<float>(), state.data_ptr<float>(), indices.data_ptr(), \
        wide(indices), cu.data_ptr(), wide(cu), spec ? accepted->data_ptr() : nullptr,          \
        spec ? wide(*accepted) : false, spec ? cindices->data_ptr() : nullptr,                  \
        spec ? wide(*cindices) : false, nw.data_ptr(), fp32(nw), bmp(out), H, n, steps,         \
        int(conv.size(2)), x.stride(0), conv.stride(0), conv.stride(1), conv.stride(2),         \
        weight.stride(0), weight.stride(1), beta.stride(1), og.stride(0), state.stride(0),      \
        indices.stride(0), spec ? indices.stride(1) : 1, spec ? cindices->stride(0) : 1,        \
        float(lb), float(eps)
        // The physical token capacity bounds the number of nonempty sequences.
        // Device offsets still select live tokens on each graph replay.
        const int64_t tokens      = x.size(0);
        const bool spx            = (!spec || n <= 256) && full_spx_device();
        const bool sparse         = spx && spec && n >= 9 && tokens > 0 && tokens <= 8;
        const int groups          = sparse ? int(tokens) : n;
        const bool use_scalar_spx = spx && (!spec || n <= 8 || sparse);
        int waves                 = use_scalar_spx ? (groups <= 4 ? 16 : 8) : 4;
        auto kernel               = spec ? (waves == 16  ? decode_kernel<true, 16>
                                            : waves == 8 ? decode_kernel<true, 8>
                                                         : decode_kernel<true, 4>)
                                         : (waves == 16  ? decode_kernel<false, 16>
                                            : waves == 8 ? decode_kernel<false, 8>
                                                         : decode_kernel<false, 4>);
        if(sparse)
            kernel = waves == 16 ? decode_kernel<true, 16, true> : decode_kernel<true, 8, true>;
        dim3 grid(groups, H);
        const int64_t pitch = state.stride(0);
        const bool pack2    = aligned_for(state, 8) && pitch % 2 == 0;
        const bool pack4    = aligned_for(state, 16) && pitch % 4 == 0;
        // At small batch only the 16- and 32-byte paths consistently beat scalar.
        if(spx && n <= 256 && tokens > 0 && !sparse && (n <= 8 ? pack4 : pack2) &&
           (!spec || (steps >= 2 && steps <= 8)))
        {
            grid  = dim3(H, n);
            waves = 4;
            if(aligned_for(state, 32) && pitch % 8 == 0)
            {
                // Each live speculative token writes a 2 MiB FP32 checkpoint.
                const bool stream_state = spec && tokens >= 128 && tokens <= n * steps;
                const bool tiny         = n <= 2 && tokens >= 4;
                const bool near16       = (n == 15 || n == 16) && steps == 5 && tokens >= 3 * n - 2;
                if(spec && (tiny || near16) && tokens <= n * steps && aligned_for(state, 64) &&
                   pitch % 16 == 0)
                {
                    waves  = tiny ? 4 : 2;
                    kernel = tiny ? decode_kernel<true, 4, false, 16, true>
                                  : decode_kernel<true, 2, false, 16, true>;
                }
                else if(spec && n > 44 && (n < 64 || tokens > 48))
                {
                    waves  = 2;
                    kernel = stream_state ? decode_kernel<true, 2, false, 8, true, true>
                                          : decode_kernel<true, 2, false, 8, true>;
                }
                else if(stream_state)
                {
                    kernel = decode_kernel<true, 4, false, 8, true, true>;
                }
                else
                {
                    kernel = spec ? decode_kernel<true, 4, false, 8, true>
                                  : decode_kernel<false, 4, false, 8, true>;
                }
            }
            else if(pack4)
            {
                kernel = spec ? decode_kernel<true, 4, false, 4, true>
                              : decode_kernel<false, 4, false, 4, true>;
            }
            else
            {
                kernel = spec ? decode_kernel<true, 4, false, 2, true>
                              : decode_kernel<false, 4, false, 2, true>;
            }
        }
        hipLaunchKernelGGL(kernel, grid, 64 * waves, 0, stream(), ARGS);
#undef ARGS
    }
    KDA_CHECK_LAUNCH();
}

} // namespace aiter::kda_rectangular
#endif
