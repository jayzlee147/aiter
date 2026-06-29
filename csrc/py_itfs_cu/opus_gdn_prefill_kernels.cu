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
#include <cstdlib>

#include "opus_gdn/gdn_defs.h"
// ref_fwd_h.hpp not available on this machine — disable ref scan path
#define OPUS_GDN_NO_REF_SCAN 1

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
// Split-path K2 (grid-starved / long single-sequence): two dedicated kernels,
// independent of the fused gdn_k2_kernel — serial scan + chunk-parallel output.
template<typename Traits>
__global__ void gdn_k2_scan_kernel(gdn_k2_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_out_kernel(gdn_k2_kargs kargs);
// Pure-HIP single-warp scan (raw MFMA, register-resident H, no opus template).
template<typename Traits>
__global__ void gdn_k2_scan_hip_kernel(gdn_k2_kargs kargs);
// Faithful triton fwd_h port: 32x32x16 MFMA, nw=2.
template<typename Traits>
__global__ void gdn_k2_scan32_kernel(gdn_k2_kargs kargs);

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
    int k2_mode)
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
    else if constexpr (algo == K1Algo::BT128)
        gdn_k1_bt128_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else
        gdn_k1_bt32_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);

    // K2: fused (register-resident scan+output, wins high-parallelism) vs
    // split (serial scan + chunk-parallel output, wins grid-starved long seq).
    // k2_mode: 0=auto, 1=force-fused, 2=force-split. Split is BT=64 only.
    //
    // The split path ALWAYS uses its own tuned config — BV=32 (4 v-tiles) +
    // num_warps=8 — independent of the fused config: more v-tiles give more
    // workgroups (the whole point when grid-starved) and less work per warp,
    // matching what triton's autotuner picks for fwd_h. Auto-split engages when
    // the FUSED grid under-fills the device (then the split's extra parallelism
    // and chunk-parallel output win despite materializing h_snap).
    // Empirical crossover: split wins only when the device is heavily starved
    // (B*H <= ~64, i.e. fused_grid = 2*B*H <= 128). Above that the h_snap
    // materialization traffic outweighs the parallelism gain -> keep fused.
    int split_thr = 129;  // fused_grid < this => split
    if (const char* e = std::getenv("OPUS_GDN_SPLIT_THRESHOLD")) split_thr = atoi(e);
    const int fused_grid = ceil_div(V, BV) * B * H;
    // Force fused mode: split kernels not fully instantiated on this build
    bool use_split = false;
    (void)k2_mode; (void)split_thr;

    if constexpr (K2Traits::BT == 64) {
        if (use_split) {
            auto h_snap = torch::empty({B, NT, H, V, K}, opts_bf16);
            auto v_new  = torch::empty({B, T, H, V},     opts_bf16);
            k2args.ptr_h_snap = h_snap.data_ptr();
            k2args.ptr_v_new  = v_new.data_ptr();
            using OT = gdn_k2_traits<64, 128, 128, 32, 8>;  // out kernel: BV=32, nw=8
            dim3 out_grid(ceil_div(V, OT::BV), NT, B * H);
            dim3 scan_grid(ceil_div(V, OT::BV), B * H);
            // Scan num_warps: opus-template scan now supports nw=2 (H_E_M>1 fixed).
            // env OPUS_GDN_SPLIT_NW selects 2/4/8 (default 8). OPUS_GDN_HIP_SCAN=1
            // uses the pure-HIP single-warp scan instead (bit-exact, slow ref).
            int snw = 8;
            if (const char* e = std::getenv("OPUS_GDN_SPLIT_NW")) snw = atoi(e);
            bool hip_scan = false;
            if (const char* e = std::getenv("OPUS_GDN_HIP_SCAN")) hip_scan = atoi(e) != 0;
            bool scan32 = false;   // faithful triton port: 32x32x16 MFMA, nw=2
            if (const char* e = std::getenv("OPUS_GDN_SCAN32")) scan32 = atoi(e) != 0;
            // Reference FLA fwd_h as the scan (beats triton: ~353us; reads opus
            // token-major w/u/k directly, writes token-major v_new -> no transposes).
            // Default scan for the split path. BV: 16 when grid-starved (more
            // v-tiles -> higher occupancy), 32 at B*H>=32. OPUS_GDN_REF=0 disables.
            int ref_bv = 0;  // disabled: ref_fwd_h.hpp not available
#ifndef OPUS_GDN_NO_REF_SCAN
            ref_bv = (B * H >= 32) ? 32 : 16;
            if (const char* e = std::getenv("OPUS_GDN_REF")) ref_bv = atoi(e);  // 0=off, 16/32/64
#endif
#ifndef OPUS_GDN_NO_REF_SCAN
            if (ref_bv == 16 || ref_bv == 32 || ref_bv == 64) {
                // reference now reads opus token-major w/u/k directly (strided loads)
                // and writes v_new token-major [B,T,H,V] -> zero transposes.
                auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
                auto co = torch::arange(0, (int64_t)(B+1)*NT, (int64_t)NT, opts_i32);
                auto cu = torch::empty({0}, opts_i32);
                const int k_stride_t = H * K;
                const int64_t g_sb = (int64_t)T * H, g_sh = 1, g_st = H;
                dim3 rg(V / ref_bv, B * H), rb(256);
                const hip_bfloat16* kp = reinterpret_cast<const hip_bfloat16*>(k.data_ptr());
                const hip_bfloat16* wp = reinterpret_cast<const hip_bfloat16*>(w_bar.data_ptr());
                const hip_bfloat16* up = reinterpret_cast<const hip_bfloat16*>(u_bar.data_ptr());
                const float* gp = reinterpret_cast<const float*>(g_cumsum.data_ptr());
                const void* h0p = has_initial_state ? initial_state.data_ptr() : nullptr;
                hip_bfloat16* hsp = reinterpret_cast<hip_bfloat16*>(h_snap.data_ptr());
                hip_bfloat16* vnp = reinterpret_cast<hip_bfloat16*>(v_new.data_ptr());
                void* htp = output_final_state ? final_state.data_ptr() : nullptr;
                #define REF_LAUNCH(BVP, UINIT, SFIN) \
                    hipLaunchKernelGGL((chunk_gated_delta_rule_fwd_h_hip_kernel<BVP, UINIT, SFIN, true, false, false, false, false, false>), \
                        rg, rb, 0, stream, kp, wp, up, gp, (const float*)nullptr, h0p, hsp, vnp, htp, \
                        cu.data_ptr<int>(), co.data_ptr<int>(), B*NT, T, H, H, k_stride_t, g_sb, g_sh, g_st)
                #define REF_BV(UINIT, SFIN) do { \
                    if (ref_bv==16) REF_LAUNCH(16, UINIT, SFIN); \
                    else if (ref_bv==32) REF_LAUNCH(32, UINIT, SFIN); \
                    else REF_LAUNCH(64, UINIT, SFIN); } while(0)
                if (has_initial_state && output_final_state) REF_BV(true, true);
                else if (has_initial_state)                  REF_BV(true, false);
                else if (output_final_state)                 REF_BV(false, true);
                else                                         REF_BV(false, false);
                #undef REF_BV
                #undef REF_LAUNCH
                // v_new already written token-major into k2args.ptr_v_new.
                // out kernel is HBM-BW bound; larger V-tile (BV) cuts redundant
                // q/k re-reads + intra q@k^T recompute. Sweep via OPUS_GDN_OUT_BV.
                // BV=128 (single v-tile) is fastest: no q/k re-read or intra
                // q@k^T recompute. Fall back to 64 only when the out grid
                // (NT*B*H, since grid.x=V/BV=1 at BV=128) would starve the device.
                int out_bv = (NT * B * H >= 128) ? 128 : 64, out_nw = 8;
                if (const char* e = std::getenv("OPUS_GDN_OUT_BV")) out_bv = atoi(e);
                if (const char* e = std::getenv("OPUS_GDN_OUT_NW")) out_nw = atoi(e);
                #define LAUNCH_OUT(OBV, ONW) do { \
                    using OUTT = gdn_k2_traits<64,128,128,OBV,ONW>; \
                    dim3 og(ceil_div(V, OUTT::BV), NT, B * H); \
                    gdn_k2_out_kernel<OUTT><<<og, dim3(OUTT::BLOCK_SIZE), \
                        OUTT::smem_out_bytes(), stream>>>(k2args); } while(0)
                if      (out_bv == 128) { if (out_nw==4) LAUNCH_OUT(128,4); else LAUNCH_OUT(128,8); }
                else if (out_bv == 64)  { if (out_nw==4) LAUNCH_OUT(64,4);  else LAUNCH_OUT(64,8); }
                else                    { if (out_nw==4) LAUNCH_OUT(32,4);  else LAUNCH_OUT(32,8); }
                #undef LAUNCH_OUT
                return;
            }
#endif  // OPUS_GDN_NO_REF_SCAN
            #define DO_SCAN(NW) gdn_k2_scan_kernel<gdn_k2_traits<64,128,128,32,NW>> \
                <<<scan_grid, dim3((NW)*64), gdn_k2_traits<64,128,128,32,NW>::smem_scan_bytes(), stream>>>(k2args)
            if (scan32)      gdn_k2_scan32_kernel<OT><<<scan_grid, dim3(128), 0, stream>>>(k2args);
            else if (hip_scan) gdn_k2_scan_hip_kernel<OT><<<scan_grid, dim3(64), 0, stream>>>(k2args);
            else if (snw==2) DO_SCAN(2);
            else if (snw==4) DO_SCAN(4);
            else             DO_SCAN(8);
            #undef DO_SCAN
            gdn_k2_out_kernel<OT><<<out_grid, dim3(OT::BLOCK_SIZE),
                                    OT::smem_out_bytes(), stream>>>(k2args);
            return;
        }
    }
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
    int k1_algo,
    int k2_mode)
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
        initial_state, final_state, has_initial_state, output_final_state, k2_mode)

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
