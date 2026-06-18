// MFMA bf16 16×16×16 utility functions for GDN prefill kernels
// Shared across all K1/K2 kernel templates
// Uses opus template library for MFMA dispatch and LDS access
#pragma once

#include <hip/hip_runtime.h>
#include "opus.hpp"

namespace gdn_mfma {

using v4bf16_t = opus::bf16x4_t;
using v4f32_t  = opus::fp32x4_t;

__device__ inline v4f32_t mfma_f32_16x16x16_bf16(v4bf16_t a, v4bf16_t b, v4f32_t c) {
    return opus::mfma<opus::bf16_t, opus::bf16_t, opus::fp32_t, 16, 16, 16>{}(a, b, c);
}

__device__ inline v4bf16_t load_mfma_tile(
        const opus::bf16_t* __restrict__ lds, int row_base, int col_base,
        int stride, int lane_id) {
    int addr = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return opus::make_smem<opus::bf16_t>(const_cast<opus::bf16_t*>(lds)).template load<4>(addr);
}

template<int E_M, int E_N, int E_K>
__device__ void tiled_gemm_mfma(
        v4f32_t* __restrict__ c,
        const opus::bf16_t* __restrict__ lds_a, int m_base, int stride_a,
        const opus::bf16_t* __restrict__ lds_b, int n_base, int stride_b,
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
    for (int i = 0; i < N; i++) opus::clear(c[i]);
}

__device__ inline float fast_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.442695041f);
}

__device__ inline v4bf16_t load_fp32_tile(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int base = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return v4bf16_t{
        static_cast<opus::bf16_t>(s[base]),
        static_cast<opus::bf16_t>(s[base + 1]),
        static_cast<opus::bf16_t>(s[base + 2]),
        static_cast<opus::bf16_t>(s[base + 3])};
}

__device__ inline v4bf16_t load_fp32_tile_T(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int n = lane_id & 15;
    int col = col_base + n;
    int kb4 = (lane_id >> 4) << 2;
    return v4bf16_t{
        static_cast<opus::bf16_t>(s[(row_base + kb4) * stride + col]),
        static_cast<opus::bf16_t>(s[(row_base + kb4 + 1) * stride + col]),
        static_cast<opus::bf16_t>(s[(row_base + kb4 + 2) * stride + col]),
        static_cast<opus::bf16_t>(s[(row_base + kb4 + 3) * stride + col])};
}

__device__ inline v4bf16_t accum_to_src(v4f32_t d) {
    return v4bf16_t{
        static_cast<opus::bf16_t>(d[0]), static_cast<opus::bf16_t>(d[1]),
        static_cast<opus::bf16_t>(d[2]), static_cast<opus::bf16_t>(d[3])};
}

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
