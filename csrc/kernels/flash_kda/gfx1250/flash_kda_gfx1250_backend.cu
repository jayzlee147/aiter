// SPDX-License-Identifier: MIT
// Copyright (c) 2026 MoonshotAI
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Native gfx1250 FlashKDA backend.
//
// The public/workspace ABI is shared with the gfx942/gfx950 implementation,
// but every device primitive below is wave32/WMMA native.  K1 prepares one
// C16 tile per workgroup.  K2 assigns one V16 state slice to each wave and
// dynamically packs one, two, or four waves into a BV16/BV32/BV64 workgroup.
// Small shapes retain up to eight independent recurrence workgroups per K3
// head; larger shapes share every V-independent operand across waves to reduce
// HBM traffic while preserving enough grid parallelism for a 256-CU MI455.

#include "flash_kda.h"
#include "../hip_common.hpp"
#include "k2_kda_context_gfx1250_policy.hpp"
#include "opus/opus.hpp"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flashkda_hip::gfx1250 {
namespace {

constexpr int C = 16;
constexpr int D = 128;
constexpr int BV = 16;
constexpr int V_TILES = D / BV;
constexpr int WAVE = 32;
constexpr int KD_STRIDE = D + 8;
constexpr int FACTOR_STRIDE = C + 8;
constexpr float LOG2E = 1.4426950408889634074f;

using bf16 = opus::bf16_t;
using Wmma = opus::wmma<bf16, bf16, opus::fp32_t, 16, 16, 32, 32>;
using FragA = Wmma::vtype_a;
using FragB = Wmma::vtype_b;
using Acc = Wmma::vtype_c;

__device__ __forceinline__ float bf16_to_f32(bf16 x)
{
    return static_cast<float>(x);
}

__device__ __forceinline__ bf16 f32_to_bf16(float x)
{
    return static_cast<bf16>(x);
}

__device__ __forceinline__ uint32_t pack_f32x2_to_bf16x2_prepare(
    float lo,
    float hi)
{
    uint32_t packed;
    asm volatile(
        "v_cvt_pk_bf16_f32 %0, %1, %2"
        : "=v"(packed)
        : "v"(lo), "v"(hi));
    return packed;
}

__device__ __forceinline__ float ex2(float x)
{
    return __builtin_amdgcn_exp2f(x);
}

__device__ __forceinline__ float sigmoid_tanh(float x)
{
    return tanhf(x * 0.5f) * 0.5f + 0.5f;
}

__device__ __forceinline__ float wave_reduce_sum(float value)
{
    #pragma unroll
    for(int offset = 16; offset >= 1; offset >>= 1)
        value += __shfl_xor(value, offset, WAVE);
    return value;
}

using U32x2 = uint32_t __attribute__((ext_vector_type(2)));

// Preserve Prepare's established wave32 XOR-16/8/4/2/1 reduction tree while
// keeping the lane exchange off the LDS crossbar. The permlane operation is
// the exact XOR-16 exchange. After XOR-8, lanes four apart contain the same
// partials as the XOR-4 partners, so row_ror:4 retains the same binary-add
// DAG; the final two quad permutations are exact XOR-2 and XOR-1 exchanges.
//
// q/k are finite BF16 inference inputs. Their squared partial sums are
// non-negative, so a source-operand swap in a lane-exchange instruction is
// bit-identical to the established local+remote additions. NaN payload
// propagation remains outside this inference contract.
__device__ __forceinline__ float wave_reduce_sum_prepare_dpp(float value)
{
    const U32x2 half = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<uint32_t>(value),
        std::bit_cast<uint32_t>(value),
        false,
        false);
    value = std::bit_cast<float>(half[0]) + std::bit_cast<float>(half[1]);

    float next;
    asm volatile(
        "v_add_f32 %0, %1, %2 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(next) : "v"(value), "v"(value));
    value = next;
    asm volatile(
        "v_add_f32 %0, %1, %2 row_ror:4 row_mask:0xf bank_mask:0xf"
        : "=v"(next) : "v"(value), "v"(value));
    value = next;
    asm volatile(
        "v_add_f32 %0, %1, %2 quad_perm:[2,3,0,1] "
        "row_mask:0xf bank_mask:0xf"
        : "=v"(next) : "v"(value), "v"(value));
    value = next;
    asm volatile(
        "v_add_f32 %0, %1, %2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf"
        : "=v"(next) : "v"(value), "v"(value));
    value = next;
    return value;
}

__device__ __forceinline__ Acc wmma(FragA a, FragB b, Acc c)
{
    return Wmma{}(a, b, c);
}

struct AccPair
{
    Acc first;
    Acc second;
};

// gfx1250's 16x16x32 BF16 WMMA operand layout is
//   fragment[(repeat * 8) + i] = matrix[row][(repeat*2+lane_group)*8+i].
// The accumulator layout is column distributed: each lane owns one column and
// eight rows, row=(lane>>4)*8+i, col=lane&15.
template <typename T, int Ld>
__device__ __forceinline__ FragA load_row_fragment(
    const T* matrix, int row, int k0, int lane)
{
    FragA out{};
    const int lane_group = lane >> 4;
    #pragma unroll
    for(int repeat = 0; repeat < 2; ++repeat)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int k = k0 + (repeat * 2 + lane_group) * 8 + i;
            out[repeat * 8 + i] = f32_to_bf16(static_cast<float>(matrix[row * Ld + k]));
        }
    }
    return out;
}

template <int Lda, int Ldb, typename TB>
__device__ __forceinline__ Acc contract_last_128(
    const bf16* a, const TB* b, int lane)
{
    const int row = lane & 15;
    Acc acc{};
    #pragma unroll
    for(int k0 = 0; k0 < D; k0 += 32)
    {
        FragA af = load_row_fragment<bf16, Lda>(a, row, k0, lane);
        FragB bf = load_row_fragment<TB, Ldb>(b, row, k0, lane);
        acc = wmma(af, bf, acc);
    }
    return acc;
}

// Compute two M16xN16 contractions that share the same K128 right operand.
// Keeping both accumulators in one loop halves the right-hand LDS traffic and
// leaves the two WMMA calls adjacent for the gfx1250 scheduler.
template <int Lda0, int Lda1, int Ldb, typename TB>
__device__ __forceinline__ AccPair contract_pair_last_128(
    const bf16* a0, const bf16* a1, const TB* b, int lane)
{
    const int row = lane & 15;
    Acc acc0{};
    Acc acc1{};
    #pragma unroll
    for(int k0 = 0; k0 < D; k0 += 32)
    {
        const FragB bf = load_row_fragment<TB, Ldb>(b, row, k0, lane);
        const FragA af0 = load_row_fragment<bf16, Lda0>(a0, row, k0, lane);
        const FragA af1 = load_row_fragment<bf16, Lda1>(a1, row, k0, lane);
        acc0 = wmma(af0, bf, acc0);
        acc1 = wmma(af1, bf, acc1);
    }
    return {acc0, acc1};
}

// The recurrent state is kept transposed in raw accumulator layout:
//   state[t][i] = S[v=lane&15, k=16*t + 8*(lane>>4) + i].
// That layout is already the native B operand layout for K32: two adjacent
// K16 accumulator tiles concatenate lane-locally, with no LDS exchange or
// cross-lane permutation.  The returned matrices are A0 @ S^T and A1 @ S^T.
template <int Lda0, int Lda1>
__device__ __forceinline__ AccPair contract_pair_with_state(
    const bf16* a0,
    const bf16* a1,
    const Acc (&state)[D / C],
    int lane)
{
    const int row = lane & 15;
    Acc acc0{};
    Acc acc1{};
    #pragma unroll
    for(int p = 0; p < D / 32; ++p)
    {
        FragB bf{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            bf[i] = f32_to_bf16(state[2 * p][i]);
            bf[8 + i] = f32_to_bf16(state[2 * p + 1][i]);
        }
        const int k0 = p * 32;
        const FragA af0 = load_row_fragment<bf16, Lda0>(a0, row, k0, lane);
        const FragA af1 = load_row_fragment<bf16, Lda1>(a1, row, k0, lane);
        acc0 = wmma(af0, bf, acc0);
        acc1 = wmma(af1, bf, acc1);
    }
    return {acc0, acc1};
}

// A is [16,16], B is [16,N], output is [16,N].  WMMA consumes B as
// logical rows N with the contraction dimension last, so transpose B while
// forming the fragment and zero-pad K16 to the native K32 instruction.
template <int Lda, int Ldb>
__device__ __forceinline__ Acc standard_k16(
    const bf16* a, const bf16* b, int lane)
{
    const int a_row = lane & 15;
    const int b_row = lane & 15;
    const int lane_group = lane >> 4;
    FragA af{};
    FragB bf{};
    #pragma unroll
    for(int repeat = 0; repeat < 2; ++repeat)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int k = (repeat * 2 + lane_group) * 8 + i;
            const int e = repeat * 8 + i;
            if(k < C)
            {
                af[e] = a[a_row * Lda + k];
                bf[e] = b[k * Ldb + b_row];
            }
            else
            {
                af[e] = f32_to_bf16(0.0f);
                bf[e] = f32_to_bf16(0.0f);
            }
        }
    }
    Acc acc{};
    return wmma(af, bf, acc);
}

// Form kr^T[K16,C16] @ U[C16,V16], padding the C16 contraction to K32.
// The KxV result is exactly the transposed accumulator layout used by state.
template <int Ldkr>
__device__ __forceinline__ Acc state_update_k16(
    const bf16* kr, FragB u, int k_tile, int lane)
{
    const int k_row = lane & 15;
    const int lane_group = lane >> 4;
    FragA af{};
    #pragma unroll
    for(int repeat = 0; repeat < 2; ++repeat)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int c = (repeat * 2 + lane_group) * 8 + i;
            const int e = repeat * 8 + i;
            if(c < C)
            {
                af[e] = kr[c * Ldkr + k_tile * C + k_row];
            }
            else
            {
                af[e] = f32_to_bf16(0.0f);
            }
        }
    }
    Acc acc{};
    return wmma(af, u, acc);
}

// The production gfx1250 context route is compiled into this AOT translation
// unit. C16 below remains the only rollback implementation.
#include "k2_kda_context_gfx1250.hpp"

__global__ void build_tile_prefix(
    const int32_t* __restrict__ cu_seqlens,
    int nseq,
    int* __restrict__ tile_prefix)
{
    if(threadIdx.x != 0)
        return;
    int total_tiles = 0;
    tile_prefix[0] = 0;
    for(int seq = 0; seq < nseq; ++seq)
    {
        const int64_t length = static_cast<int64_t>(cu_seqlens[seq + 1]) -
                               static_cast<int64_t>(cu_seqlens[seq]);
        total_tiles += static_cast<int>((length + C - 1) / C);
        tile_prefix[seq + 1] = total_tiles;
    }
}

__global__ void build_context_prefix(
    const int32_t* __restrict__ cu_seqlens,
    int nseq,
    int* __restrict__ tile_prefix,
    int* __restrict__ context_prefix,
    int context_group_chunks)
{
    if(threadIdx.x != 0)
        return;
    int total_tiles = 0;
    int total_contexts = 0;
    tile_prefix[0] = 0;
    context_prefix[0] = 0;
    for(int seq = 0; seq < nseq; ++seq)
    {
        const int64_t length = static_cast<int64_t>(cu_seqlens[seq + 1]) -
                               static_cast<int64_t>(cu_seqlens[seq]);
        const int sequence_tiles = static_cast<int>((length + C - 1) / C);
        total_tiles += sequence_tiles;
        total_contexts +=
            (sequence_tiles + context_group_chunks - 1) /
            context_group_chunks;
        tile_prefix[seq + 1] = total_tiles;
        context_prefix[seq + 1] = total_contexts;
    }
}

template <bool Varlen>
__global__ __launch_bounds__(WAVE)
void prepare_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float scale,
    float gate_scale,
    int t_seq,
    int h_q_count,
    int h_count,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t g_token_stride,
    bf16* __restrict__ ws_kd,
    bf16* __restrict__ ws_qd,
    bf16* __restrict__ ws_kr,
    float* __restrict__ ws_gt,
    bf16* __restrict__ ws_inv,
    bf16* __restrict__ ws_mqk,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    int nseq,
    int total_tiles)
{
    constexpr bool ContextOperands = false;
    float* const ws_beta = nullptr;
#include "prepare_kda_gfx1250.inc"
}

template <bool Varlen>
__global__ __launch_bounds__(2 * WAVE)
void prepare_context_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float scale,
    float gate_scale,
    int t_seq,
    int h_q_count,
    int h_count,
    bf16* __restrict__ ws_kd,
    bf16* __restrict__ ws_qd,
    bf16* __restrict__ ws_kr,
    float* __restrict__ ws_gt,
    bf16* __restrict__ ws_inv,
    bf16* __restrict__ ws_mqk,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    int nseq,
    int total_tiles,
    float* __restrict__ ws_beta)
{
    constexpr bool ContextOperands = true;
    constexpr bool UseDppReduction = false;
    constexpr bool TotalExpHoist = false;
    constexpr bool PrepareVectorDrain = false;
    constexpr bool PrepareQkAsyncCopy = false;
    constexpr bool PrepareFactorVectorDrain = false;
    constexpr bool PrepareGatePairPrefix = false;
#include "prepare_context_2wave_gfx1250.inc"
}

template <bool HasStateIn, bool HasStateOut, bool StateFp32, bool Varlen>
__global__ __launch_bounds__(WAVE)
void walk_v16_kernel(
    const bf16* __restrict__ value,
    const float* __restrict__ beta,
    bf16* __restrict__ output,
    const bf16* __restrict__ ws_kd,
    const bf16* __restrict__ ws_qd,
    const bf16* __restrict__ ws_kr,
    const float* __restrict__ ws_gt,
    const bf16* __restrict__ ws_inv,
    const bf16* __restrict__ ws_mqk,
    const void* __restrict__ initial_state,
    void* __restrict__ final_state,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    int total_tiles,
    int t_seq,
    int h_count,
    int64_t value_token_stride,
    int nseq,
    int nt_dense)
{
    const int lane = static_cast<int>(threadIdx.x);
    const int program = static_cast<int>(blockIdx.x);
    const int v_tile = program % V_TILES;
    const int seq_head = program / V_TILES;
    const int head = seq_head % h_count;
    const int seq = seq_head / h_count;

    int bos = 0;
    int eos = 0;
    int tile0 = 0;
    int chunks = 0;
    if constexpr(Varlen)
    {
        bos = cu_seqlens[seq];
        eos = cu_seqlens[seq + 1];
        tile0 = tile_prefix[seq];
        chunks = tile_prefix[seq + 1] - tile0;
    }
    else
    {
        bos = seq * t_seq;
        eos = bos + t_seq;
        tile0 = 0;
        chunks = nt_dense;
    }

    __shared__ bf16 kd[C * KD_STRIDE];
    __shared__ bf16 qd[C * KD_STRIDE];
    __shared__ bf16 kr[C * KD_STRIDE];
    __shared__ bf16 inv[C * FACTOR_STRIDE];
    __shared__ bf16 mqk[C * FACTOR_STRIDE];
    __shared__ float g_total[D];
    __shared__ float beta_act[C];

    // Keep S^T as eight K16xV16 FP32 accumulator tiles.  This is both the
    // update result layout and, after lane-local BF16 conversion, the WMMA B
    // operand layout consumed by Kd/Qd.  No recurrent-state LDS is needed.
    Acc state[D / C];
    #pragma unroll
    for(int k_tile = 0; k_tile < D / C; ++k_tile)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int v_local = lane & 15;
            const int k_col = k_tile * C + (lane >> 4) * 8 + i;
            float x = 0.0f;
            if constexpr(HasStateIn)
            {
                const int64_t global =
                    ((static_cast<int64_t>(seq) * h_count + head) * D +
                     v_tile * BV + v_local) * D + k_col;
                if constexpr(StateFp32)
                    x = static_cast<const float*>(initial_state)[global];
                else
                    x = bf16_to_f32(static_cast<const bf16*>(initial_state)[global]);
            }
            state[k_tile][i] = x;
        }
    }

    for(int chunk = 0; chunk < chunks; ++chunk)
    {
        const int global_tile = Varlen ? tile0 + chunk : chunk;
        const int ht = Varlen
            ? head * total_tiles + global_tile
            : seq_head * nt_dense + chunk;
        const int token0 = bos + chunk * C;
        int active = eos - token0;
        active = active < C ? active : C;

        for(int idx = lane; idx < C * D; idx += WAVE)
        {
            const int64_t off = static_cast<int64_t>(ht) * C * D + idx;
            const int row = idx / D;
            const int col = idx - row * D;
            const int sidx = row * KD_STRIDE + col;
            kd[sidx] = ws_kd[off];
            qd[sidx] = ws_qd[off];
            kr[sidx] = ws_kr[off];
        }

        for(int idx = lane; idx < C * C; idx += WAVE)
        {
            const int64_t off = static_cast<int64_t>(ht) * C * C + idx;
            const int row = idx / C;
            const int col = idx - row * C;
            const int sidx = row * FACTOR_STRIDE + col;
            inv[sidx] = ws_inv[off];
            mqk[sidx] = ws_mqk[off];
        }
        for(int k_col = lane; k_col < D; k_col += WAVE)
            g_total[k_col] = ws_gt[static_cast<int64_t>(ht) * D + k_col];
        if(lane < C)
        {
            beta_act[lane] = lane < active
                ? sigmoid_tanh(beta[
                    static_cast<int64_t>(token0 + lane) * h_count + head])
                : 0.0f;
        }
        __syncthreads();

        // Compute both old-state projections while each K32 BF16 state
        // fragment is live.  The value slice is then consumed entirely into a
        // register B fragment before any output store, preserving v/out alias
        // safety without a shared residual tile.
        const AccPair state_products =
            contract_pair_with_state<KD_STRIDE, KD_STRIDE>(kd, qd, state, lane);
        const Acc prediction = state_products.first;
        const Acc state_out = state_products.second;
        FragB residual_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = (lane >> 4) * 8 + i;
            const int v_local = lane & 15;
            const float v = row < active
                ? bf16_to_f32(value[
                    static_cast<int64_t>(token0 + row) * value_token_stride +
                    static_cast<int64_t>(head) * D +
                    v_tile * BV + v_local])
                : 0.0f;
            residual_fragment[i] =
                f32_to_bf16((v - prediction[i]) * beta_act[row]);
            residual_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        // U = inverse @ residual.  C16 occupies the low half of the native
        // K32 operand; both U and its B-operand rounding remain register-only.
        FragA inv_fragment{};
        const int factor_row = lane & 15;
        const int lane_group = lane >> 4;
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int c = lane_group * 8 + i;
            inv_fragment[i] = inv[factor_row * FACTOR_STRIDE + c];
            inv_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        Acc zero{};
        const Acc u_acc = wmma(inv_fragment, residual_fragment, zero);
        FragB u_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            u_fragment[i] = f32_to_bf16(u_acc[i]);
            u_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        // out = qd @ state^T + mqk @ U.  Match the imported backend's
        // operand-wise BF16 rounding before the final addition.
        FragA mqk_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int c = lane_group * 8 + i;
            mqk_fragment[i] = mqk[factor_row * FACTOR_STRIDE + c];
            mqk_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        const Acc local_out = wmma(mqk_fragment, u_fragment, zero);
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = (lane >> 4) * 8 + i;
            const int v_local = lane & 15;
            if(row < active)
            {
                const bf16 a = f32_to_bf16(state_out[i]);
                const bf16 b = f32_to_bf16(local_out[i]);
                output[(static_cast<int64_t>(token0 + row) * h_count + head) * D +
                       v_tile * BV + v_local] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        // state = state * exp2(g_total) + U^T @ kr
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            const Acc update =
                state_update_k16<KD_STRIDE>(kr, u_fragment, k_tile, lane);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int k_col = k_tile * C + lane_group * 8 + i;
                state[k_tile][i] =
                    state[k_tile][i] * ex2(g_total[k_col]) + update[i];
            }
        }
    }

    if constexpr(HasStateOut)
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int v_local = lane & 15;
                const int k_col = k_tile * C + (lane >> 4) * 8 + i;
                const int64_t global =
                    ((static_cast<int64_t>(seq) * h_count + head) * D +
                     v_tile * BV + v_local) * D + k_col;
                if constexpr(StateFp32)
                    static_cast<float*>(final_state)[global] = state[k_tile][i];
                else
                    static_cast<bf16*>(final_state)[global] =
                        f32_to_bf16(state[k_tile][i]);
            }
        }
    }
}

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
        throw std::runtime_error(
            std::string("FlashKDA gfx1250: ") + operation + " failed: " +
            hipGetErrorString(status));
}

hipDeviceProp_t require_gfx1250()
{
    int device = 0;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
    const bool arch_ok = std::strncmp(props.gcnArchName, "gfx1250", 7) == 0 &&
        (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':');
    if(!arch_ok || props.warpSize != WAVE)
        throw std::runtime_error(
            "FlashKDA gfx1250 backend requires a wave32 gfx1250 device; active device is '" +
            std::string(props.gcnArchName) + "'");
    return props;
}

} // namespace
} // namespace flashkda_hip::gfx1250

namespace flashkda_hip {

void launch_fwd_hip(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* g_ptr,
    const void* beta_ptr,
    float scale,
    void* out_ptr,
    void* workspace_ptr,
    int64_t workspace_bytes,
    const float* A_log_ptr,
    const float* dt_bias_ptr,
    float gate_scale,
    int total_tiles,
    int T_total,
    int H_q,
    int H,
    int N,
    const void* init_state,
    void* final_state,
    bool has_state_in,
    bool has_state_out,
    bool state_fp32,
    const int32_t* cu_seqlens,
    hipStream_t stream,
    int max_seqlen_upper_bound)
{
    const hipDeviceProp_t device_props = gfx1250::require_gfx1250();

    const FwdParams params{
        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, scale, out_ptr, workspace_ptr,
        A_log_ptr, dt_bias_ptr, gate_scale, total_tiles, T_total, H_q, H, N,
        init_state, final_state, has_state_in, has_state_out, state_fp32,
        cu_seqlens, stream, max_seqlen_upper_bound};
    const LaunchShape shape = make_launch_shape(params);
    const WorkspaceLayout ws = carve_workspace(workspace_ptr, shape, H, N);
    const bool varlen = cu_seqlens != nullptr;
    const int t_seq = varlen ? 0 : T_total / N;
    const int nt = varlen ? 0 : total_tiles / N;
    const int64_t q_token_stride = static_cast<int64_t>(H_q) * gfx1250::D;
    const int64_t k_token_stride = q_token_stride;
    const int64_t v_token_stride = static_cast<int64_t>(H) * gfx1250::D;
    const int64_t g_token_stride = v_token_stride;
    const char* const route = std::getenv("FLASH_KDA_GFX1250_ROUTE");
    const bool request_auto =
        route == nullptr || std::strcmp(route, "auto") == 0;
    const bool use_c16 =
        route != nullptr && std::strcmp(route, "c16") == 0;
    if(!request_auto && !use_c16)
        throw std::runtime_error(
            "FLASH_KDA_GFX1250_ROUTE must be auto or c16");

    const bool auto_use_g16 = request_auto &&
        gfx1250::context_policy::should_use_auto_g16(
            static_cast<uint64_t>(T_total),
            static_cast<uint32_t>(H_q),
            static_cast<uint32_t>(H),
            static_cast<uint32_t>(N),
            static_cast<uint32_t>(device_props.multiProcessorCount),
            varlen,
            has_state_out,
            state_fp32);
    const int auto_group_chunks = request_auto
        ? (auto_use_g16
               ? gfx1250::K2_CONTEXT_GFX1250_GROUP16
               : static_cast<int>(
                     gfx1250::context_policy::choose_auto_group_chunks(
                         static_cast<uint64_t>(T_total),
                         static_cast<uint32_t>(H),
                         static_cast<uint32_t>(
                             device_props.multiProcessorCount))))
        : 0;
    const bool use_context = request_auto;
    const int context_group_chunks = auto_group_chunks;

    int total_context_groups = 0;
    float* context_affine_b = nullptr;
    if(use_context)
    {
        const int64_t context_upper = varlen
            ? (static_cast<int64_t>(T_total) +
               context_group_chunks * gfx1250::C - 1) /
                  (context_group_chunks * gfx1250::C) + N
            : static_cast<int64_t>(N) *
                  ((nt + context_group_chunks - 1) / context_group_chunks);
        if(context_upper <= 0 ||
           context_upper > std::numeric_limits<int>::max())
            throw std::runtime_error(
                "FlashKDA gfx1250: invalid context-group count");
        total_context_groups = static_cast<int>(context_upper);

        using wide = __int128_t;
        const wide required_a = static_cast<wide>(context_upper) * H *
            gfx1250::D * gfx1250::D * sizeof(gfx1250::bf16);
        const wide available_a = static_cast<wide>(H) *
            shape.total_segments * WorkspaceSizes::kCsplitSin;
        if(required_a > available_a)
            throw std::runtime_error(
                "FlashKDA gfx1250: internal context affine-A arena is too "
                "small for this shape");

        const wide required_b = static_cast<wide>(context_upper) * H *
            gfx1250::D * gfx1250::D * sizeof(float);
        const wide available_b = static_cast<wide>(shape.n_ht) *
            WorkspaceSizes::kCsplitU;
        if(required_b <= available_b)
        {
            context_affine_b = reinterpret_cast<float*>(ws.cs_u);
        }
        else
        {
            if(required_b > std::numeric_limits<int64_t>::max())
                throw std::runtime_error(
                    "FlashKDA gfx1250: context affine-B workspace overflows "
                    "int64_t");
            const int64_t base_workspace =
                get_workspace_size_hip(T_total, H, N);
            const wide required_workspace =
                static_cast<wide>(base_workspace) + required_b;
            if(base_workspace < 0 ||
               required_workspace > std::numeric_limits<int64_t>::max())
                throw std::runtime_error(
                    "FlashKDA gfx1250: context workspace size overflows "
                    "int64_t");
            if(workspace_bytes < static_cast<int64_t>(required_workspace))
                throw std::runtime_error(
                    "FlashKDA gfx1250: context route workspace is too small; "
                    "need " +
                    std::to_string(static_cast<int64_t>(required_workspace)) +
                    " bytes, got " + std::to_string(workspace_bytes));
            context_affine_b = reinterpret_cast<float*>(
                static_cast<char*>(workspace_ptr) + base_workspace);
        }
    }

    if(varlen && use_context)
    {
        gfx1250::build_context_prefix<<<1, 64, 0, stream>>>(
            cu_seqlens, N, ws.tile_prefix, ws.pair_prefix,
            context_group_chunks);
    }
    else if(varlen)
    {
        gfx1250::build_tile_prefix<<<1, 64, 0, stream>>>(
            cu_seqlens, N, ws.tile_prefix);
    }

    if(use_context)
    {
        if(varlen)
        {
            gfx1250::prepare_context_kernel<true>
                <<<dim3(total_tiles, H), 2 * gfx1250::WAVE, 0, stream>>>(
                static_cast<const gfx1250::bf16*>(q_ptr),
                static_cast<const gfx1250::bf16*>(k_ptr),
                static_cast<const gfx1250::bf16*>(g_ptr),
                static_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr, scale, gate_scale, 0, H_q, H,
                ws.kd, ws.qd, ws.kr, ws.gt, ws.inv, ws.mqk,
                cu_seqlens, ws.tile_prefix, N, total_tiles, ws.cs_beta);
        }
        else
        {
            gfx1250::prepare_context_kernel<false>
                <<<dim3(nt, N * H), 2 * gfx1250::WAVE, 0, stream>>>(
                static_cast<const gfx1250::bf16*>(q_ptr),
                static_cast<const gfx1250::bf16*>(k_ptr),
                static_cast<const gfx1250::bf16*>(g_ptr),
                static_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr, scale, gate_scale, t_seq, H_q, H,
                ws.kd, ws.qd, ws.kr, ws.gt, ws.inv, ws.mqk,
                nullptr, nullptr, N, total_tiles, ws.cs_beta);
        }
        gfx1250::check_hip(hipGetLastError(), "context K1 launch");

        gfx1250::launch_k2_kda_context_gfx1250(
            context_group_chunks,
            static_cast<const gfx1250::bf16*>(v_ptr),
            static_cast<gfx1250::bf16*>(out_ptr),
            ws.kd,
            ws.qd,
            ws.kr,
            ws.gt,
            ws.inv,
            ws.mqk,
            ws.cs_beta,
            ws.cs_sin,
            context_affine_b,
            init_state,
            final_state,
            cu_seqlens,
            ws.tile_prefix,
            ws.pair_prefix,
            total_context_groups,
            total_tiles,
            t_seq,
            H,
            N,
            nt,
            has_state_in,
            has_state_out,
            state_fp32,
            stream);
        gfx1250::check_hip(hipGetLastError(), "context K2 launch");
        return;
    }

    if(varlen)
    {
        gfx1250::prepare_kernel<true>
            <<<dim3(total_tiles, H), gfx1250::WAVE, 0, stream>>>(
            static_cast<const gfx1250::bf16*>(q_ptr),
            static_cast<const gfx1250::bf16*>(k_ptr),
            static_cast<const gfx1250::bf16*>(g_ptr),
            static_cast<const float*>(beta_ptr), A_log_ptr, dt_bias_ptr,
            scale, gate_scale, 0, H_q, H, q_token_stride, k_token_stride,
            g_token_stride, ws.kd, ws.qd, ws.kr, ws.gt,
            ws.inv, ws.mqk, cu_seqlens, ws.tile_prefix, N, total_tiles);
    }
    else
    {
        gfx1250::prepare_kernel<false>
            <<<dim3(nt, N * H), gfx1250::WAVE, 0, stream>>>(
            static_cast<const gfx1250::bf16*>(q_ptr),
            static_cast<const gfx1250::bf16*>(k_ptr),
            static_cast<const gfx1250::bf16*>(g_ptr),
            static_cast<const float*>(beta_ptr), A_log_ptr, dt_bias_ptr,
            scale, gate_scale, t_seq, H_q, H, q_token_stride, k_token_stride,
            g_token_stride, ws.kd, ws.qd, ws.kr, ws.gt,
            ws.inv, ws.mqk, nullptr, nullptr, N, total_tiles);
    }

    const dim3 walk_grid(N * H * gfx1250::V_TILES);
    auto launch_walk = [&]<bool HI, bool HO, bool FP, bool VL>() {
        gfx1250::walk_v16_kernel<HI, HO, FP, VL><<<walk_grid, gfx1250::WAVE, 0, stream>>>(
            static_cast<const gfx1250::bf16*>(v_ptr),
            static_cast<const float*>(beta_ptr),
            static_cast<gfx1250::bf16*>(out_ptr), ws.kd, ws.qd, ws.kr,
            ws.gt, ws.inv, ws.mqk, init_state, final_state,
            VL ? cu_seqlens : nullptr, VL ? ws.tile_prefix : nullptr,
            total_tiles, t_seq, H, v_token_stride, N, nt);
    };
    dispatch_state_mode(
        varlen, has_state_in, has_state_out, state_fp32, launch_walk);
    gfx1250::check_hip(hipGetLastError(), "kernel launch");
}

} // namespace flashkda_hip
