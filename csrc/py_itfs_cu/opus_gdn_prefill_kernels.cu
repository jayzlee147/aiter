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

// hipcc two-pass: host pass gets empty stubs, device pass compiles real kernels
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void gdn_k1_kernel(gdn_k1_kargs kargs) {}
template<typename Traits>
__global__ void gdn_k2_kernel(gdn_k2_kargs kargs) {}
#else
#include "opus_gdn/gdn_k1_bt64_kernel_template.hpp"
#include "opus_gdn/gdn_k1_bt16_kernel_template.hpp"
#include "opus_gdn/gdn_k2_kernel_template.hpp"
#endif

// Explicit instantiations (both passes need these)
template __global__ void gdn_k1_kernel<gdn_k1_traits<64, 128, 128, 4>>(gdn_k1_kargs);
template __global__ void gdn_k1_kernel<gdn_k1_traits<16, 128, 128, 4>>(gdn_k1_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<16, 128, 128, 64, 4>>(gdn_k2_kargs);

template<typename K1Traits, typename K2Traits>
void launch_gdn_prefill(
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
    auto h_snap   = torch::empty({B, NT, H, K, V}, opts_fp32);
    auto v_new    = torch::empty({B, T, H, V}, opts_bf16);

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
    k2args.ptr_h_snap  = h_snap.data_ptr();
    k2args.ptr_v_new   = v_new.data_ptr();
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

    gdn_k1_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
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
    int BT)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, T, H, K]");
    TORCH_CHECK(q.size(3) == 128, "K must be 128");
    TORCH_CHECK(v.size(3) == 128, "V must be 128");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(g.scalar_type() == at::ScalarType::Float, "g must be fp32");
    TORCH_CHECK(beta.scalar_type() == at::ScalarType::Float, "beta must be fp32");
    TORCH_CHECK(q.size(1) % BT == 0, "T must be a multiple of BT");

    if (BT == 64) {
        launch_gdn_prefill<gdn_k1_traits<64, 128, 128, 4>,
                           gdn_k2_traits<64, 128, 128, 64, 4>>(
            q, k, v, g, beta, o, scale,
            initial_state, final_state, has_initial_state, output_final_state);
    } else if (BT == 16) {
        launch_gdn_prefill<gdn_k1_traits<16, 128, 128, 4>,
                           gdn_k2_traits<16, 128, 128, 64, 4>>(
            q, k, v, g, beta, o, scale,
            initial_state, final_state, has_initial_state, output_final_state);
    } else {
        TORCH_CHECK(false, "BT must be 16 or 64, got ", BT);
    }
}
