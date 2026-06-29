// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based GDN prefill kernel — host launcher for aiter JIT.
// Target: gfx942 (MI300X) / gfx950 (MI350)
// Fuses all 4 steps of Gated DeltaNet chunkwise recurrence into 2 HIP kernels:
//   K1 = Steps 1+2 (cumsum, KKT, trisol, WY factors)  — token-parallel
//   K2 = Steps 3+4 (h update, output)                  — head-parallel, chunk-serial

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>

#include "opus_gdn/gdn_defs.h"

namespace {
hipStream_t g_pipeline_stream = nullptr;
hipEvent_t  g_pipeline_event  = nullptr;

hipStream_t get_pipeline_stream() {
    if (!g_pipeline_stream)
        hipStreamCreateWithFlags(&g_pipeline_stream, hipStreamNonBlocking);
    return g_pipeline_stream;
}
hipEvent_t get_pipeline_event() {
    if (!g_pipeline_event)
        hipEventCreateWithFlags(&g_pipeline_event, hipEventDisableTiming);
    return g_pipeline_event;
}
} // namespace

// Forward declarations — definitions live in separate TUs to avoid ODR conflicts
template<typename Traits>
__global__ void gdn_k1_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_neumann_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_bt32_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_bt128_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_kernel(gdn_k2_kargs kargs);
__global__ void gdn_k2_kernel_occ2(gdn_k2_kargs kargs);
template<typename Traits>
__global__ void gdn_wf_h_kernel(gdn_wf_h_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_scan_kernel(gdn_k2_kargs kargs);

template<typename Traits>
__global__ void gdn_k2_output_kernel(gdn_k2_output_kargs kargs);

enum class K1Algo { BASIC, NEUMANN, BT32, BT128 };

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
    bool output_final_state,
    bool pipeline = false)
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

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Pipeline overlap: K1 and K2 run concurrently on separate streams
    torch::Tensor k1_done_t;
    uint32_t* k1_done_ptr = nullptr;
    hipStream_t k2_stream = stream;
    hipEvent_t pipeline_event = nullptr;

    if (pipeline) {
        auto opts_u32 = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
        k1_done_t = torch::zeros({NT * B * H}, opts_u32);
        k1_done_ptr = reinterpret_cast<uint32_t*>(k1_done_t.data_ptr());

        k2_stream = get_pipeline_stream();
        pipeline_event = get_pipeline_event();
        hipEventRecord(pipeline_event, stream);
        hipStreamWaitEvent(k2_stream, pipeline_event, 0);
    }

    gdn_k1_kargs k1args{};
    k1args.ptr_k       = k.data_ptr();
    k1args.ptr_v       = v.data_ptr();
    k1args.ptr_beta    = beta.data_ptr();
    k1args.ptr_g       = g.data_ptr();
    k1args.ptr_w_bar   = w_bar.data_ptr();
    k1args.ptr_u_bar   = u_bar.data_ptr();
    k1args.ptr_g_cumsum = g_cumsum.data_ptr();
    k1args.ptr_k1_done = k1_done_ptr;
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
    k2args.ptr_k1_done = k1_done_ptr;
    k2args.B = B; k2args.T = T; k2args.H = H; k2args.K = K; k2args.V = V;
    k2args.NT = NT;
    k2args.scale = scale;

    dim3 k1_grid(NT, B * H);
    dim3 k1_block(K1Traits::BLOCK_SIZE);
    size_t k1_smem = K1Traits::smem_size_bytes();

    dim3 k2_grid(ceil_div(V, BV), B * H);
    dim3 k2_block(K2Traits::BLOCK_SIZE);
    size_t k2_smem = K2Traits::smem_size_bytes();

    // Helper lambda to launch the right K2 kernel
    auto launch_k2 = [&](hipStream_t s) {
        if constexpr (K2Traits::SERIALIZE_BC) {
            gdn_k2_kernel_occ2<<<k2_grid, k2_block, k2_smem, s>>>(k2args);
        } else {
            gdn_k2_kernel<K2Traits><<<k2_grid, k2_block, k2_smem, s>>>(k2args);
        }
    };

    if (pipeline) {
        launch_k2(k2_stream);
    }

    // Launch K1 on main stream
    if constexpr (algo == K1Algo::BASIC)
        gdn_k1_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::NEUMANN)
        gdn_k1_neumann_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::BT128)
        gdn_k1_bt128_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else
        gdn_k1_bt32_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);

    if (pipeline) {
        hipEventRecord(pipeline_event, k2_stream);
        hipStreamWaitEvent(stream, pipeline_event, 0);
    } else {
        launch_k2(stream);
    }
}

// =========================================================================
// Split K2 launcher: scan-only (serial) + output (parallel)
// =========================================================================
template<K1Algo algo, typename K1Traits, typename K2Traits>
void launch_gdn_prefill_split_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor o,
    float scale,
    torch::Tensor initial_state, torch::Tensor final_state,
    bool has_initial_state, bool output_final_state)
{
    const int B = q.size(0), T = q.size(1), H = q.size(2), K = q.size(3);
    const int V = v.size(3);
    constexpr int BT = K1Traits::BT;
    constexpr int BV = K2Traits::BV;
    constexpr int PAD = K2Traits::SMEM_PAD;
    constexpr int BK_SUB = K2Traits::BK_SUB;
    const int NT = ceil_div(T, BT);
    const int NV = ceil_div(V, BV);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    auto w_bar    = torch::empty({B, T, H, K}, opts_bf16);
    auto u_bar    = torch::empty({B, T, H, V}, opts_bf16);
    auto g_cumsum = torch::empty({B, T, H},    opts_fp32);

    // Intermediates for split K2
    auto h_snap   = torch::empty({B, NT, H, K, V}, opts_fp32);
    auto v_new    = torch::empty({B, T, H, V}, opts_bf16);

    // --- K1 ---
    gdn_k1_kargs k1args{};
    k1args.ptr_k       = k.data_ptr();
    k1args.ptr_v       = v.data_ptr();
    k1args.ptr_beta    = beta.data_ptr();
    k1args.ptr_g       = g.data_ptr();
    k1args.ptr_w_bar   = w_bar.data_ptr();
    k1args.ptr_u_bar   = u_bar.data_ptr();
    k1args.ptr_g_cumsum = g_cumsum.data_ptr();
    k1args.B = B; k1args.T = T; k1args.H = H; k1args.K = K; k1args.V = V;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 k1_grid(NT, B * H);
    dim3 k1_block(K1Traits::BLOCK_SIZE);
    size_t k1_smem = K1Traits::smem_size_bytes();

    if constexpr (algo == K1Algo::NEUMANN)
        gdn_k1_neumann_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::BASIC)
        gdn_k1_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::BT128)
        gdn_k1_bt128_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else
        gdn_k1_bt32_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);

    // --- K2 scan-only ---
    gdn_k2_kargs scan_args{};
    scan_args.ptr_q       = nullptr;
    scan_args.ptr_k       = k.data_ptr();
    scan_args.ptr_w_bar   = w_bar.data_ptr();
    scan_args.ptr_u_bar   = u_bar.data_ptr();
    scan_args.ptr_g_cumsum = g_cumsum.data_ptr();
    scan_args.ptr_h0      = has_initial_state ? initial_state.data_ptr() : nullptr;
    scan_args.ptr_o       = nullptr;
    scan_args.ptr_ht      = output_final_state ? final_state.data_ptr() : nullptr;
    scan_args.ptr_h_snap  = h_snap.data_ptr();
    scan_args.ptr_v_new   = v_new.data_ptr();
    scan_args.B = B; scan_args.T = T; scan_args.H = H; scan_args.K = K; scan_args.V = V;
    scan_args.NT = NT;
    scan_args.scale = scale;

    dim3 scan_grid(NV, B * H);
    dim3 scan_block(K2Traits::BLOCK_SIZE);
    // Scan LDS: s_g + s_v_T + pool (no s_q)
    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;
    constexpr int scan_smem_g  = BT * (int)sizeof(float);
    constexpr int scan_smem_vT = BV * STRIDE_BT * (int)sizeof(bf16_t);
    constexpr int scan_pool_bc = BV * STRIDE_BK * (int)sizeof(bf16_t)
                               + BT * STRIDE_BK * (int)sizeof(bf16_t);
    constexpr int scan_pool_d  = BK_SUB * STRIDE_BT * (int)sizeof(bf16_t);
    constexpr int scan_pool    = (scan_pool_bc > scan_pool_d) ? scan_pool_bc : scan_pool_d;
    constexpr size_t scan_smem = scan_smem_g + scan_smem_vT + scan_pool;

    gdn_k2_scan_kernel<K2Traits><<<scan_grid, scan_block, scan_smem, stream>>>(scan_args);

    // --- K2 output (parallel) ---
    gdn_k2_output_kargs out_args{};
    out_args.ptr_q       = q.data_ptr();
    out_args.ptr_k       = k.data_ptr();
    out_args.ptr_v_new   = v_new.data_ptr();
    out_args.ptr_h_snap  = h_snap.data_ptr();
    out_args.ptr_g_cumsum = g_cumsum.data_ptr();
    out_args.ptr_o       = o.data_ptr();
    out_args.B = B; out_args.T = T; out_args.H = H; out_args.K = K; out_args.V = V;
    out_args.NT = NT; out_args.NV = NV;
    out_args.scale = scale;

    dim3 out_grid(NT * NV, B * H);
    dim3 out_block(K2Traits::BLOCK_SIZE);
    // Output LDS: s_g + s_v_T + pool
    // pool needs: phase c (s_h_T + s_sub) or phase e (s_q_e + s_k_e)
    constexpr int out_pool_c  = BV * STRIDE_BK * (int)sizeof(bf16_t)
                              + BT * STRIDE_BK * (int)sizeof(bf16_t);
    constexpr int out_pool_e  = 2 * BT * STRIDE_BK * (int)sizeof(bf16_t);
    constexpr int out_pool_av = BT * STRIDE_BT * (int)sizeof(bf16_t);
    constexpr int out_pool    = (out_pool_c > out_pool_e) ? out_pool_c : out_pool_e;
    constexpr int out_pool_final = (out_pool > out_pool_av) ? out_pool : out_pool_av;
    constexpr size_t out_smem = scan_smem_g + scan_smem_vT + out_pool_final;

    gdn_k2_output_kernel<K2Traits><<<out_grid, out_block, out_smem, stream>>>(out_args);
}

void opus_gdn_prefill_split_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor o,
    float scale,
    torch::Tensor initial_state, torch::Tensor final_state,
    bool has_initial_state, bool output_final_state,
    int BT, int BV, int num_warps)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, T, H, K]");
    TORCH_CHECK(q.size(3) == 128 && v.size(3) == 128);
    TORCH_CHECK(BT == 64, "Split mode currently only supports BT=64");
    TORCH_CHECK(BV == 64, "Split mode currently only supports BV=64");

    if (num_warps == 8) {
        launch_gdn_prefill_split_impl<K1Algo::NEUMANN,
            gdn_k1_traits<64, 128, 128, 4>,
            gdn_k2_traits<64, 128, 128, 64, 8>>(
            q, k, v, g, beta, o, scale,
            initial_state, final_state, has_initial_state, output_final_state);
    } else {
        launch_gdn_prefill_split_impl<K1Algo::NEUMANN,
            gdn_k1_traits<64, 128, 128, 4>,
            gdn_k2_traits<64, 128, 128, 64, 4>>(
            q, k, v, g, beta, o, scale,
            initial_state, final_state, has_initial_state, output_final_state);
    }
}

// =========================================================================
// Wavefront H-scan launcher
// =========================================================================
template<typename K2Traits>
void launch_gdn_wavefront_h_impl(
    torch::Tensor k,
    torch::Tensor w_bar,
    torch::Tensor u_bar,
    torch::Tensor g_cumsum,
    torch::Tensor h_out,
    torch::Tensor v_new_out,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int S,
    torch::Tensor q,
    torch::Tensor o,
    float scale)
{
    const int B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3);
    const int V = u_bar.size(3);
    constexpr int BT = K2Traits::BT;
    constexpr int BV = K2Traits::BV;
    const int NT = T / BT;
    const int N_super = NT / S;
    const int N_flat = ceil_div(V, BV) * B * H;

    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(k.device());
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32).device(k.device());

    auto h_pass = torch::zeros({N_flat, N_super, K2Traits::N_K, BV, K2Traits::BK_SUB}, opts_fp32);
    auto flags  = torch::zeros({N_flat * N_super}, opts_i32);

    gdn_wf_h_kargs args{};
    args.ptr_k         = k.data_ptr();
    args.ptr_w_bar     = w_bar.data_ptr();
    args.ptr_u_bar     = u_bar.data_ptr();
    args.ptr_g_cumsum  = g_cumsum.data_ptr();
    args.ptr_h0        = has_initial_state ? initial_state.data_ptr() : nullptr;
    args.ptr_h         = h_out.data_ptr();
    args.ptr_v_new     = v_new_out.defined() ? v_new_out.data_ptr() : nullptr;
    args.ptr_ht        = output_final_state ? final_state.data_ptr() : nullptr;
    args.ptr_h_pass    = h_pass.template data_ptr<float>();
    args.ptr_flags     = reinterpret_cast<uint32_t*>(flags.template data_ptr<int>());
    args.B = B; args.T = T; args.H = H; args.K = K; args.V = V;
    args.NT = NT; args.S = S; args.N_super = N_super;
    args.ptr_q     = q.defined() ? q.data_ptr() : nullptr;
    args.ptr_o     = o.defined() ? o.data_ptr() : nullptr;
    args.scale     = scale;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(k));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 grid(N_flat, N_super);
    dim3 block(K2Traits::BLOCK_SIZE);

    constexpr int PAD = K2Traits::SMEM_PAD;
    constexpr int STRIDE_BK = K2Traits::BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;
    constexpr int smem_g  = BT * (int)sizeof(float);
    constexpr int smem_vT = BV * STRIDE_BT * (int)sizeof(bf16_t);
    constexpr int pool_bc = BT * STRIDE_BK * (int)sizeof(bf16_t);
    constexpr int pool_d  = K2Traits::BK_SUB * STRIDE_BT * (int)sizeof(bf16_t);
    constexpr int pool_h  = K2Traits::BK_SUB * BV * (int)sizeof(bf16_t); // h LDS staging (bf16)
    constexpr int pool_max = (pool_bc > pool_d) ? pool_bc : pool_d;
    constexpr int pool    = (pool_max > pool_h) ? pool_max : pool_h;
    constexpr size_t smem = smem_g + smem_vT + pool;

    gdn_wf_h_kernel<K2Traits><<<grid, block, smem, stream>>>(args);
}

void opus_gdn_wavefront_h_fwd(
    torch::Tensor k,
    torch::Tensor w_bar,
    torch::Tensor u_bar,
    torch::Tensor g_cumsum,
    torch::Tensor h_out,
    torch::Tensor v_new_out,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int S,
    int BT,
    torch::Tensor q,
    torch::Tensor o,
    float scale)
{
    TORCH_CHECK(k.dim() == 4, "k must be 4D [B, T, H, K]");
    TORCH_CHECK(k.size(3) == 128, "K must be 128");
    TORCH_CHECK(u_bar.size(3) == 128, "V must be 128");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(k.size(1) % BT == 0, "T must be a multiple of BT");
    TORCH_CHECK((k.size(1) / BT) % S == 0, "NT must be a multiple of S");
    TORCH_CHECK(BT == 64 || BT == 128, "BT must be 64 or 128");

    if (BT == 128) {
        launch_gdn_wavefront_h_impl<gdn_k2_traits<128, 128, 128, 64, 4>>(
            k, w_bar, u_bar, g_cumsum, h_out, v_new_out,
            initial_state, final_state, has_initial_state, output_final_state, S,
            q, o, scale);
    } else if (q.defined() && q.numel() > 0) {
        // Fused mode: use nw=8 for better latency hiding
        launch_gdn_wavefront_h_impl<gdn_k2_traits<64, 128, 128, 64, 8>>(
            k, w_bar, u_bar, g_cumsum, h_out, v_new_out,
            initial_state, final_state, has_initial_state, output_final_state, S,
            q, o, scale);
    } else {
        launch_gdn_wavefront_h_impl<gdn_k2_traits<64, 128, 128, 64, 4>>(
            k, w_bar, u_bar, g_cumsum, h_out, v_new_out,
            initial_state, final_state, has_initial_state, output_final_state, S,
            q, o, scale);
    }
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
    int k1_algo,
    bool pipeline,
    int occ_hint)
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

    // OCC=2 override: serialized b/c GEMMs + reduced LDS
    if (occ_hint == 2 && BT == 64 && BV == 64 && num_warps == 8) {
        launch_gdn_prefill_impl<K1Algo::NEUMANN,
            gdn_k1_traits<64, 128, 128, 4>,
            gdn_k2_traits<64, 128, 128, 64, 8, 2>>(
            q, k, v, g, beta, o, scale,
            initial_state, final_state, has_initial_state, output_final_state, pipeline);
        return;
    }

#define DISPATCH(algo, bt, bv, nw) \
    launch_gdn_prefill_impl<K1Algo::algo, \
        gdn_k1_traits<bt, 128, 128, 4>, \
        gdn_k2_traits<bt, 128, 128, bv, nw>>( \
        q, k, v, g, beta, o, scale, \
        initial_state, final_state, has_initial_state, output_final_state, pipeline)

    if (BT == 128) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BT128, 128, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(BT128, 128, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(BT128, 128, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(BT128, 128, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=128, BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 64 && k1_algo == 1) {
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
