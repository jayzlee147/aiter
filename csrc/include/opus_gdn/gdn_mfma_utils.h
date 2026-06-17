// MFMA bf16 16×16×16 utility functions for GDN prefill kernels
// Shared across all K1/K2 kernel templates — gfx942 (MI300X)
#pragma once

#include <hip/hip_runtime.h>

namespace gdn_mfma {

using v4bf16_t = __bf16 __attribute__((ext_vector_type(4)));
using v4f32_t  = float  __attribute__((ext_vector_type(4)));

__device__ inline v4f32_t mfma_f32_16x16x16_bf16(v4bf16_t a, v4bf16_t b, v4f32_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ inline v4bf16_t load_mfma_tile(
        const __bf16* __restrict__ lds, int row_base, int col_base,
        int stride, int lane_id) {
    int addr = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return *reinterpret_cast<const v4bf16_t*>(&lds[addr]);
}

template<int E_M, int E_N, int E_K>
__device__ void tiled_gemm_mfma(
        v4f32_t* __restrict__ c,
        const __bf16* __restrict__ lds_a, int m_base, int stride_a,
        const __bf16* __restrict__ lds_b, int n_base, int stride_b,
        int lane_id) {
    for (int ek = 0; ek < E_K; ek++) {
        v4bf16_t a_tiles[E_M];
        for (int em = 0; em < E_M; em++)
            a_tiles[em] = load_mfma_tile(lds_a, m_base + em * 16, ek * 16, stride_a, lane_id);
        v4bf16_t b_tiles[E_N];
        for (int en = 0; en < E_N; en++)
            b_tiles[en] = load_mfma_tile(lds_b, n_base + en * 16, ek * 16, stride_b, lane_id);
        for (int em = 0; em < E_M; em++)
            for (int en = 0; en < E_N; en++)
                c[em * E_N + en] = mfma_f32_16x16x16_bf16(
                    a_tiles[em], b_tiles[en], c[em * E_N + en]);
    }
}

template<int N>
__device__ inline void clear_v4f32(v4f32_t* c) {
    for (int i = 0; i < N; i++) c[i] = v4f32_t{0.f, 0.f, 0.f, 0.f};
}

// exp(x) via single-cycle v_exp_f32: exp(x) = 2^(x * log2(e))
__device__ inline float fast_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.442695041f);
}

// Load fp32 tile from LDS, convert to bf16 for MFMA source operand
__device__ inline v4bf16_t load_fp32_tile(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int base = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return v4bf16_t{
        static_cast<__bf16>(s[base]),
        static_cast<__bf16>(s[base + 1]),
        static_cast<__bf16>(s[base + 2]),
        static_cast<__bf16>(s[base + 3])};
}

// Load fp32 tile transposed (row becomes column), convert to bf16
__device__ inline v4bf16_t load_fp32_tile_T(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int n = lane_id & 15;
    int col = col_base + n;
    int kb4 = (lane_id >> 4) << 2;
    return v4bf16_t{
        static_cast<__bf16>(s[(row_base + kb4) * stride + col]),
        static_cast<__bf16>(s[(row_base + kb4 + 1) * stride + col]),
        static_cast<__bf16>(s[(row_base + kb4 + 2) * stride + col]),
        static_cast<__bf16>(s[(row_base + kb4 + 3) * stride + col])};
}

// Convert MFMA fp32 accumulator to bf16 source operand (register chaining)
__device__ inline v4bf16_t accum_to_src(v4f32_t d) {
    return v4bf16_t{
        static_cast<__bf16>(d[0]), static_cast<__bf16>(d[1]),
        static_cast<__bf16>(d[2]), static_cast<__bf16>(d[3])};
}

// Store fp32 MFMA accumulator tile to LDS (transposed: rows → columns)
__device__ inline void store_fp32_tile(
        float* __restrict__ s, int row_base, int col_base,
        int stride, v4f32_t d, int lane_id) {
    int n = lane_id & 15;
    int mb4 = (lane_id >> 4) << 2;
    s[(row_base + mb4) * stride + col_base + n] = d[0];
    s[(row_base + mb4 + 1) * stride + col_base + n] = d[1];
    s[(row_base + mb4 + 2) * stride + col_base + n] = d[2];
    s[(row_base + mb4 + 3) * stride + col_base + n] = d[3];
}

} // namespace gdn_mfma
