// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based GDN prefill kernel — host launcher for aiter JIT.
// Fuses all 4 steps of Gated DeltaNet chunkwise recurrence into 2 HIP kernels:
//   K1 = Steps 1+2 (cumsum, KKT, trisol, WY factors)  — token-parallel
//   K2 = Steps 3+4 (h update, output)                  — head-parallel, chunk-serial

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>

#include "opus_gdn/gdn_defs.h"

// Forward declarations — definitions live in separate TUs to avoid ODR conflicts
template<typename Traits>
__global__ void gdn_k1_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_neumann_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_bt32_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_kernel(gdn_k2_kargs kargs);

enum class K1Algo { BASIC, NEUMANN, BT32 };

template<K1Algo algo, typename K1Traits, typename K2Traits>
void launch_gdn_prefill_impl(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state)
{
    const int B = q.size(0);
    const int T = q.size(1);
    const int H = q.size(2);
    const int K = q.size(3);
    const int V = v.size(3);
    constexpr int BT = K1Traits::BT;
    constexpr int BV = K2Traits::BV;
    const int NT = ceil_div(T, BT);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    auto w_bar    = torch::empty({B, T, H, K}, opts_bf16);
    auto u_bar    = torch::empty({B, T, H, V}, opts_bf16);
    auto g_cumsum = torch::empty({B, T, H},    opts_fp32);
    gdn_k1_kargs k1args{};
    k1args.ptr_k       = k.data_ptr();
    k1args.ptr_v       = v.data_ptr();
    k1args.ptr_beta    = beta.data_ptr();
    k1args.ptr_g       = g.data_ptr();
    k1args.ptr_w_bar   = w_bar.data_ptr();
    k1args.ptr_u_bar   = u_bar.data_ptr();
    k1args.ptr_g_cumsum = g_cumsum.data_ptr();
    k1args.B = B; k1args.T = T; k1args.H = H; k1args.K = K; k1args.V = V;

    gdn_k2_kargs k2args{};
    k2args.ptr_q       = q.data_ptr();
    k2args.ptr_k       = k.data_ptr();
    k2args.ptr_w_bar   = w_bar.data_ptr();
    k2args.ptr_u_bar   = u_bar.data_ptr();
    k2args.ptr_g_cumsum = g_cumsum.data_ptr();
    k2args.ptr_h0      = has_initial_state ? initial_state.data_ptr() : nullptr;
    k2args.ptr_o       = o.data_ptr();
    k2args.ptr_ht      = output_final_state ? final_state.data_ptr() : nullptr;
    k2args.ptr_h_snap  = nullptr;
    k2args.ptr_v_new   = nullptr;
    k2args.B = B; k2args.T = T; k2args.H = H; k2args.K = K; k2args.V = V;
    k2args.NT = NT;
    k2args.scale = scale;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 k1_grid(NT, B * H);
    dim3 k1_block(K1Traits::BLOCK_SIZE);
    size_t k1_smem = K1Traits::smem_size_bytes();

    dim3 k2_grid(ceil_div(V, BV), B * H);
    dim3 k2_block(K2Traits::BLOCK_SIZE);
    size_t k2_smem = K2Traits::smem_size_bytes();

    if constexpr (algo == K1Algo::BASIC)
        gdn_k1_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::NEUMANN)
        gdn_k1_neumann_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else
        gdn_k1_bt32_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);

    gdn_k2_kernel<K2Traits><<<k2_grid, k2_block, k2_smem, stream>>>(k2args);
}

void opus_gdn_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int BT,
    int BV,
    int num_warps,
    int k1_algo)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, T, H, K]");
    TORCH_CHECK(q.size(3) == 128, "K must be 128");
    TORCH_CHECK(v.size(3) == 128, "V must be 128");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(g.scalar_type() == at::ScalarType::Float, "g must be fp32");
    TORCH_CHECK(beta.scalar_type() == at::ScalarType::Float, "beta must be fp32");
    TORCH_CHECK(q.size(1) % BT == 0, "T must be a multiple of BT");
    TORCH_CHECK(v.size(3) % BV == 0, "V must be a multiple of BV");

    TORCH_CHECK(BV == 64 || BV == 32, "Unsupported BV=", BV, ". Supported: 64, 32");
    TORCH_CHECK(num_warps == 2 || num_warps == 4 || num_warps == 8, "Unsupported num_warps=", num_warps, ". Supported: 2, 4, 8");

#define DISPATCH(algo, bt, bv, nw) \
    launch_gdn_prefill_impl<K1Algo::algo, \
        gdn_k1_traits<bt, 128, 128, 4>, \
        gdn_k2_traits<bt, 128, 128, bv, nw>>( \
        q, k, v, g, beta, o, scale, \
        initial_state, final_state, has_initial_state, output_final_state)

    if (BT == 64 && k1_algo == 1) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(NEUMANN, 64, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(NEUMANN, 64, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(NEUMANN, 64, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(NEUMANN, 64, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 64) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BASIC, 64, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(BASIC, 64, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(BASIC, 64, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(BASIC, 64, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 32) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BT32, 32, 64, 4);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=32, BV=", BV,
                        " num_warps=", num_warps, ". Use BV=64 num_warps=4");
        }
    } else if (BT == 16 && num_warps == 4) {
        DISPATCH(BASIC, 16, 64, 4);
    } else {
        TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                    " num_warps=", num_warps);
    }
#undef DISPATCH
}
