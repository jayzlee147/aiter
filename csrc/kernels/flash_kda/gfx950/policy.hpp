#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

#include "../hip_common.hpp"
#include "../k1_kda_split_kernel.hpp"
#include "../k2_kda_csplit_bt64_kernel.hpp"
#include "k1_kda_bt16_fused_kernel.hpp"
#include "k1_kda_bt32_merge_kernel.hpp"
#include "k1_kda_bt64_merge_kernel.hpp"
#include "k1_kda_postprep_fused_kernel.hpp"
#include "k1_kda_split_solve_kernel.hpp"
#include "k2_kda_csplit_bt64_bv16_nw8_kernel.hpp"
#include "k2_kda_csplit_bt64_plain_kernel.hpp"
#include "k2_kda_csplit_segment_out_kernel.hpp"
#include "k2_kda_vsplit_rs_x32_kernel.hpp"

namespace flashkda_hip::gfx950 {

struct LaunchPolicy {
    static HipLaunchPolicy make(
            const FwdParams& p, const HipDeviceInfo& device) {
        const bool use_db = use_double_buffer(p, device);
        const bool use_pad = partial_padding_enabled();
        const bool use_tb3 = use_db && triple_buffer_enabled();
        const bool use_tiled_kr = tiled_kr_enabled();
        Csplit64ScanLauncher plain_scan = use_tb3
            ? (use_pad
                   ? (use_tiled_kr ? &launch_plain<3, true, true>
                                   : &launch_plain<3, true, false>)
                   : (use_tiled_kr ? &launch_plain<3, false, true>
                                   : &launch_plain<3, false, false>))
            : use_db
              ? (use_pad
                     ? (use_tiled_kr ? &launch_plain<2, true, true>
                                     : &launch_plain<2, true, false>)
                     : (use_tiled_kr ? &launch_plain<2, false, true>
                                     : &launch_plain<2, false, false>))
              : (use_pad ? &launch_plain<1, true, false>
                         : &launch_plain<1, false, false>);
        const bool use_replay_tb3 = output_gll_tb3_enabled();
        SegmentOutputLauncher segment_output = output_x32_enabled()
            ? &launch_segment_output_x32
            : output_gll_disabled()
                ? nullptr
                : output_gll_sin_disabled()
                    ? (use_replay_tb3
                        ? &launch_segment_output_gll<false, 3>
                        : &launch_segment_output_gll<false, 2>)
                    : (use_replay_tb3
                        ? &launch_segment_output_gll<true, 3>
                        : &launch_segment_output_gll<true, 2>);
        // CDNA4 global-to-LDS replay is the production default.  Setting
        // OUT_GLL=0 rolls back to the common operator; OUT_X32=1 has priority
        // as an explicit architecture-private diagnostic baseline.
        return {16, default_route(p), false,
                &launch_k6_nw8_x32,
                nullptr,
                plain_scan,
                &launch_plain_nw8,
                segment_output,
                &launch_plain_k1, &launch_vsplit_rs_x32,
                bt16_k1_disabled() ? nullptr : &launch_bt16_k1,
                false, false, nullptr, nullptr, nullptr, nullptr};
    }

private:
    enum class Bt16FusedMode {
        disabled,
        vector_x32,
        exact_x32,
        exact_x16,
    };

    template <bool VL, bool EXACT_PREP, bool USE_X32>
    static void launch_bt16_fused(
            const Bt16K1Launch& a, const dim3& grid) {
        k1_kda_bt16_fused_kernel<VL, EXACT_PREP, USE_X32>
            <<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.beta, a.A_log, a.dt_bias,
                a.scale, a.gate_scale, a.T_seq, a.H,
                a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv, a.mqk,
                VL ? a.cu_seqlens : nullptr,
                VL ? a.tile_prefix : nullptr, a.N, a.total_tiles);
    }

    static void launch_bt16_k1(const Bt16K1Launch& a) {
        const dim3 grid = a.is_varlen ? dim3(a.total_tiles, a.H)
                                      : dim3(a.NT, a.N * a.H);
        const Bt16FusedMode fused = bt16_fused_mode();
        if (fused != Bt16FusedMode::disabled) {
            auto launch = [&]<bool VL>() {
                if (fused == Bt16FusedMode::exact_x16)
                    launch_bt16_fused<VL, true, false>(a, grid);
                else if (fused == Bt16FusedMode::exact_x32)
                    launch_bt16_fused<VL, true, true>(a, grid);
                else
                    launch_bt16_fused<VL, false, true>(a, grid);
            };
            if (a.is_varlen) launch.template operator()<true>();
            else launch.template operator()<false>();
            return;
        }
        if (a.is_varlen) {
            k1_kda_split_prep_kernel<true><<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.A_log, a.dt_bias, a.scale, a.gate_scale,
                a.T_seq, a.H, a.kd, a.qd, a.kr, a.gt, a.kinv,
                reinterpret_cast<float*>(a.mqk), a.cu_seqlens,
                a.tile_prefix, a.N, a.total_tiles,
                nullptr, nullptr, nullptr, 0);
            k1_kda_split_solve_kernel<true><<<grid, 64, 0, a.stream>>>(
                a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                a.cu_seqlens, a.tile_prefix, a.N, a.total_tiles,
                a.T_seq, a.H);
        } else {
            k1_kda_split_prep_kernel<false><<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.A_log, a.dt_bias, a.scale, a.gate_scale,
                a.T_seq, a.H, a.kd, a.qd, a.kr, a.gt, a.kinv,
                reinterpret_cast<float*>(a.mqk), nullptr, nullptr,
                a.N, a.total_tiles, nullptr, nullptr, nullptr, 0);
            k1_kda_split_solve_kernel<false><<<grid, 64, 0, a.stream>>>(
                a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                nullptr, nullptr, a.N, a.total_tiles, a.T_seq, a.H);
        }
    }

    static bool bt16_k1_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_BT16_K1");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    static Bt16FusedMode bt16_fused_mode() {
        const char* value = std::getenv("FLASH_KDA_GFX950_BT16_FUSED");
        // Exact preparation restores the monolithic kernel's normalization
        // and decay rounding order, while CDNA4 K32 contraction is bit-exact
        // to the X16 solve for this operator.  Keep both the preservation
        // split and diagnostic template axes available without changing the
        // architecture-neutral route.
        if (value == nullptr || std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "exact_x32") == 0)
            return Bt16FusedMode::exact_x32;
        if (std::strcmp(value, "0") == 0)
            return Bt16FusedMode::disabled;
        if (std::strcmp(value, "vector_x32") == 0)
            return Bt16FusedMode::vector_x32;
        if (std::strcmp(value, "exact_x16") == 0)
            return Bt16FusedMode::exact_x16;
        return Bt16FusedMode::exact_x32;
    }

    static void launch_plain_k1(const PlainCsplit64K1Launch& a) {
        if (fused_k1_enabled()) {
            const bool padded = fused_k1_padded_enabled();
            if (a.is_varlen) {
                if (padded)
                    k1_kda_postprep_fused_kernel<true, true, true, true>
                        <<<dim3(a.total_segments, a.H), 256, 0, a.stream>>>(
                            a.beta, a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv,
                            a.mqk, a.cross32, a.cross64, a.cu_seqlens,
                            a.tile_prefix, a.pair_prefix, a.segment_prefix,
                            a.N, a.total_tiles, a.total_pairs,
                            a.total_segments, a.T_seq, a.H, a.NT);
                else
                    k1_kda_postprep_fused_kernel<true, true, true, false>
                        <<<dim3(a.total_segments, a.H), 256, 0, a.stream>>>(
                            a.beta, a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv,
                            a.mqk, a.cross32, a.cross64, a.cu_seqlens,
                            a.tile_prefix, a.pair_prefix, a.segment_prefix,
                            a.N, a.total_tiles, a.total_pairs,
                            a.total_segments, a.T_seq, a.H, a.NT);
            } else {
                if (padded)
                    k1_kda_postprep_fused_kernel<false, true, true, true>
                        <<<dim3((a.NT + 3) / 4, a.N * a.H), 256, 0,
                           a.stream>>>(
                            a.beta, a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv,
                            a.mqk, a.cross32, a.cross64, nullptr, nullptr,
                            nullptr, nullptr, a.N, a.total_tiles,
                            a.total_pairs, a.total_segments, a.T_seq, a.H,
                            a.NT);
                else
                    k1_kda_postprep_fused_kernel<false, true, true, false>
                        <<<dim3((a.NT + 3) / 4, a.N * a.H), 256, 0,
                           a.stream>>>(
                            a.beta, a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv,
                            a.mqk, a.cross32, a.cross64, nullptr, nullptr,
                            nullptr, nullptr, a.N, a.total_tiles,
                            a.total_pairs, a.total_segments, a.T_seq, a.H,
                            a.NT);
            }
            return;
        }

        if (a.is_varlen) {
            k1_kda_split_solve_kernel<true>
                <<<dim3(a.total_tiles, a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                    a.cu_seqlens, a.tile_prefix, a.N, a.total_tiles,
                    a.T_seq, a.H);
            k1_kda_bt32_merge_kernel<true>
                <<<dim3(a.total_pairs, a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cu_seqlens, a.tile_prefix, a.pair_prefix, a.N,
                    a.total_tiles, a.total_pairs, a.T_seq, a.H, a.NT);
            k1_kda_bt64_merge_kernel<true>
                <<<dim3(a.total_segments, a.H), 256, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                    a.segment_prefix, a.N, a.total_tiles, a.total_pairs,
                    a.total_segments, a.T_seq, a.H, a.NT);
        } else {
            k1_kda_split_solve_kernel<false>
                <<<dim3(a.NT, a.N * a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                    nullptr, nullptr, a.N, a.total_tiles, a.T_seq, a.H);
            k1_kda_bt32_merge_kernel<false>
                <<<dim3((a.NT + 1) / 2, a.N * a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_pairs, a.T_seq, a.H, a.NT);
            k1_kda_bt64_merge_kernel<false>
                <<<dim3((a.NT + 3) / 4, a.N * a.H), 256, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, nullptr, nullptr, nullptr, nullptr, a.N,
                    a.total_tiles, a.total_pairs, a.total_segments,
                    a.T_seq, a.H, a.NT);
        }
    }

    // Retain the former gfx950-private three-launch path as an exact A/B and
    // rollback hook.  Explicit common routes remain controlled by FLASH_KDA_K2.
    static bool fused_k1_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_FUSED_K1");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool fused_k1_padded_enabled() {
        // The padded gfx950 operator is the production default.  Keep the
        // former 38-KiB layout as an exact architecture-private A/B rollback.
        const char* value = std::getenv("FLASH_KDA_GFX950_FUSED_K1_PADDED");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool output_x32_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_X32");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool output_gll_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    static bool output_gll_sin_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL_SIN");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    // Three arenas fit two CTAs/CU even with the padded incoming-state cache
    // and remove one BT64 rendezvous.  Keep the former two-arena schedule as
    // an exact architecture-private A/B and rollback hook.
    static bool output_gll_tb3_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL_TB3");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    // A third LDS arena is the gfx950 low-grid default.  Delaying slot reuse by
    // two complete segments removes the double-buffer path's segment-end reuse
    // barrier without changing any numerical operation.  `0` is the exact
    // two-arena rollback used by paired performance and PMC checks.
    static bool triple_buffer_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_TB3");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    // gfx950 can form both state-carry MFMA operands with native LDS
    // transpose reads when kr is published in K16-major tiles.  Keep the
    // established row-major publication as an exact A/B and rollback path.
    static bool tiled_kr_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_KR_TR");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool use_double_buffer(
            const FwdParams& p, const HipDeviceInfo& device) {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_DB");
        if (value != nullptr) {
            if (value[0] == '0' && value[1] == '\0') return false;
            if (value[0] == '1' && value[1] == '\0') return true;
        }
        // Every multi-arena specialization admits one CTA/CU.  Select one only
        // when the complete V16 scan grid already fits in one device wave, so
        // the extra LDS cannot reduce residency, and only after enough BT64
        // segments exist to amortize the larger prologue.
        const int64_t scan_ctas = int64_t(p.N) * p.H * (128 / 16);
        const int T_seq = p.T_total / p.N;
        return T_seq >= 1024 && scan_ctas <= device.cu_count;
    }

    // Pitch-20 FP32 partial rows remove the half-wave pitch-16 bank alias.
    // Keep an exact rollback for PMC and regression A/B.
    static bool partial_padding_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_PARTIAL_PAD");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static K2DefaultRoute default_route(const FwdParams& p) {
        // Paired gfx950 measurements across the Kimi-K3 N x average-T plane
        // put the plain C-split crossover at 256 tokens.  Only the 128-token
        // bucket stays on register-state V-split; C-split wins single-256 and
        // the previously misrouted 16x1K/64x256 cases as well as long inputs.
        const long T_seq = p.T_total / p.N;
        const char* value = std::getenv("FLASH_KDA_GFX950_CSPLIT64_MIN_T");
        if (value != nullptr && *value != '\0') {
            errno = 0;
            char* end = nullptr;
            const long threshold = std::strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' ||
                threshold < 0 || threshold > INT_MAX)
                return K2DefaultRoute::vsplit_rs;
            if (threshold == 0)
                return K2DefaultRoute::vsplit_rs;
            return T_seq >= threshold ? K2DefaultRoute::csplit64
                                      : K2DefaultRoute::vsplit_rs;
        }
        return T_seq >= 256 ? K2DefaultRoute::csplit64
                            : K2DefaultRoute::vsplit_rs;
    }

    static void launch_k6_nw8_x32(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_bv16_nw8_x32_kernel<HI, HO, FP, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.decay, a.inv,
                    a.cross32, a.cross64, a.u, a.sin, a.init_state,
                    a.final_state, a.cu_seqlens, a.tile_prefix,
                    a.pair_prefix, a.segment_prefix, a.total_tiles,
                    a.total_pairs, a.total_segments, a.T_seq, a.H, a.NT,
                    a.scan_flags);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    template <typename RegBGemm, typename KrCarry>
    static void launch_vsplit_rs_variant(const VsplitRsLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_vsplit_rs_kernel<
                16, HI, HO, FP, VL, RegBGemm, KrCarry>
                <<<a.grid, 64, 0, a.stream>>>(
                    a.v, a.beta, a.out, a.kd, a.qd, a.kr, a.gt, a.inv,
                    a.mqk, a.init_state, a.final_state, a.cu_seqlens,
                    a.tile_prefix, a.total_tiles, a.T_seq, a.H, a.NT);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    // Short-sequence production path: compose the native K32 register-state
    // contraction with tiled kr transpose reads.  The two operators stay
    // independent template axes so the common recurrence contains no ISA or
    // architecture selection.
    static void launch_vsplit_rs_x32(const VsplitRsLaunch& a) {
        launch_vsplit_rs_variant<RegBX32, TiledKrCarryX16>(a);
    }

    // Production gfx950 plain BT64 scan.  PAD removes the FP32 reduction
    // arena's half-wave bank alias.  Multi-arena variants are selected only
    // when the complete scan grid fits in one device wave.  Keeping the arena
    // count compile-time leaves every hot loop free of policy predicates.
    template <int ARENAS, bool PAD, bool TILED_KR>
    static void launch_plain(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_plain_kernel<
                4, HI, HO, FP, VL, ARENAS, PAD, TILED_KR>
                <<<a.grid, 256, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, a.u, a.sin, a.init_state, a.final_state,
                    a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                    a.segment_prefix, a.total_tiles, a.total_pairs,
                    a.total_segments, a.T_seq, a.H, a.NT);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    // Diagnostic hook (`FLASH_KDA_K2=csplit64nw8`) for the plain pipeline.
    // It is intentionally architecture-private so a gfx950 x32/BV8 operator
    // can replace this implementation without touching common dispatch.
    static void launch_plain_nw8(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_scan_kernel<8, HI, HO, FP, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, a.u, a.sin, a.init_state, a.final_state,
                    a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                    a.segment_prefix, a.total_tiles, a.total_pairs,
                    a.total_segments, a.T_seq, a.H, a.NT);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    static void launch_segment_output_x32(
            const CsplitSegmentOutputLaunch& a) {
        if (a.is_varlen) {
            k2_kda_csplit_segment_out_x32_kernel<true>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                    a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT);
        } else {
            k2_kda_csplit_segment_out_x32_kernel<false>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_segments, a.T_seq, a.H, a.NT);
        }
    }

    template <bool StageSin, int Arenas>
    static void launch_segment_output_gll(
            const CsplitSegmentOutputLaunch& a) {
        if (a.is_varlen) {
            k2_kda_csplit_segment_out_gll_kernel<true, StageSin, Arenas>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                    a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT);
        } else {
            k2_kda_csplit_segment_out_gll_kernel<false, StageSin, Arenas>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_segments, a.T_seq, a.H, a.NT);
        }
    }
};

}  // namespace flashkda_hip::gfx950
