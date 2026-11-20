// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Pure launch-policy helpers for the gfx1250 context-parallel FlashKDA K2.
//
// Keep this header independent of the kernel implementation: its constexpr
// functions are shared by host launch code and CPU-only routing tests.
#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#define FLASH_KDA_CONTEXT_POLICY_HD __host__ __device__ __forceinline__
#else
#define FLASH_KDA_CONTEXT_POLICY_HD inline
#endif

namespace flashkda_hip::gfx1250::context_policy {

FLASH_KDA_CONTEXT_POLICY_HD constexpr uint64_t ceil_div_u64(
    uint64_t numerator,
    uint64_t denominator)
{
    return denominator == 0
        ? 0
        : numerator / denominator + (numerator % denominator != 0);
}

constexpr uint32_t kAutoRouteTokenChunk = 16;
constexpr uint32_t kAutoRouteG8WorkUnitsPerCu = 6;
constexpr uint32_t kMi455NumCus = 256;
constexpr uint32_t kAutoRouteG16GroupChunks = 16;
constexpr uint32_t kAutoRouteG16GroupTokens =
    kAutoRouteTokenChunk * kAutoRouteG16GroupChunks;
constexpr uint32_t kAutoRouteG16GvaQueryHeads = 3;
constexpr uint32_t kAutoRouteG16EqualQueryHeads = 12;

// The accepted automatic scan promotion is intentionally narrower than the
// set of shapes supported by the eight-wave kernel.  MI455 ABBA measurements
// cover HV=12 with final-state output; HV=1 is a repeatable regression and
// intermediate head counts have not yet been qualified.  Keep the gate pure
// so host-only tests can exhaust it without opening the GPU.
constexpr uint32_t kLeanScan8wValueHeads = 12;

FLASH_KDA_CONTEXT_POLICY_HD constexpr bool should_use_lean_scan_8w(
    uint32_t value_heads,
    bool has_state_out)
{
    return value_heads == kLeanScan8wValueHeads && has_state_out;
}

// G8/G16 has a repeatable alignment sawtooth rather than a monotonic
// crossover. Promote only fully measured single-sequence packed groups; in
// particular G11 and G13 remain on G8.  The wider historical points are exact
// measured groups as well, not an interpolation across the unqualified gaps.
FLASH_KDA_CONTEXT_POLICY_HD constexpr bool
is_measured_auto_g16_query_head_count(uint32_t query_heads)
{
    return query_heads == kAutoRouteG16GvaQueryHeads ||
        query_heads == kAutoRouteG16EqualQueryHeads;
}

FLASH_KDA_CONTEXT_POLICY_HD constexpr bool is_measured_auto_g16_group_count(
    uint64_t g16_groups)
{
    return g16_groups == 12 ||
        (g16_groups >= 14 && g16_groups <= 16) ||
        g16_groups == 24 || g16_groups == 32 ||
        g16_groups == 48 || g16_groups == 64;
}

FLASH_KDA_CONTEXT_POLICY_HD constexpr bool should_use_auto_g16(
    uint64_t total_tokens,
    uint32_t query_heads,
    uint32_t value_heads,
    uint32_t num_sequences,
    uint32_t num_cus,
    bool varlen,
    bool has_state_out,
    bool state_fp32)
{
    if(!varlen || num_sequences != 1 ||
       !is_measured_auto_g16_query_head_count(query_heads) ||
       value_heads != 12 ||
       num_cus != kMi455NumCus || !has_state_out || !state_fp32 ||
       total_tokens == 0 || total_tokens % kAutoRouteG16GroupTokens != 0)
        return false;
    return is_measured_auto_g16_group_count(
        total_tokens / kAutoRouteG16GroupTokens);
}

// Base MI455 routing selects between G4 and G8 after the narrow measured G16
// promotion above has failed.  Count one route work unit per (C16 token chunk,
// value head): G4 wins below six work units per CU, while equality and every
// larger grid select G8.  The policy is intentionally based on host-visible
// geometry, not device-resident cu_seqlens.
FLASH_KDA_CONTEXT_POLICY_HD constexpr uint32_t choose_auto_group_chunks(
    uint64_t total_tokens,
    uint32_t value_heads,
    uint32_t num_cus)
{
    // Production launch geometry is strictly positive.  Keep malformed direct
    // helper calls on the smaller of the two auto candidates rather than
    // introducing a third (C16) outcome.
    if(total_tokens == 0 || value_heads == 0 || num_cus == 0)
        return 4;

    const uint64_t total_token_chunks =
        ceil_div_u64(total_tokens, kAutoRouteTokenChunk);
    const uint64_t work_units = total_token_chunks * value_heads;
    const uint64_t g8_threshold =
        static_cast<uint64_t>(kAutoRouteG8WorkUnitsPerCu) * num_cus;
    return work_units < g8_threshold ? 4 : 8;
}

// Representative compile-time checks.  Runtime/unit tests should exhaust the
// full launch range; these keep accidental formula changes from compiling.
static_assert(choose_auto_group_chunks(2032, 12, 256) == 4);
static_assert(choose_auto_group_chunks(2033, 12, 256) == 8);
static_assert(choose_auto_group_chunks(2048, 12, 256) == 8);
static_assert(should_use_lean_scan_8w(12, true));
static_assert(!should_use_lean_scan_8w(1, true));
static_assert(!should_use_lean_scan_8w(12, false));
static_assert(should_use_auto_g16(3072, 12, 12, 1, 256, true, true, true));
static_assert(should_use_auto_g16(4096, 3, 12, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(2816, 12, 12, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(3328, 12, 12, 1, 256, true, true, true));
static_assert(should_use_auto_g16(4096, 12, 12, 1, 256, true, true, true));
static_assert(should_use_auto_g16(6144, 12, 12, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(4352, 12, 12, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(4096, 1, 12, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(4096, 12, 12, 1, 256, false, true, true));
static_assert(!should_use_auto_g16(4096, 12, 12, 32, 256, true, true, true));
static_assert(!should_use_auto_g16(4096, 12, 1, 1, 256, true, true, true));
static_assert(!should_use_auto_g16(4096, 12, 12, 1, 256, true, false, true));
static_assert(!should_use_auto_g16(4096, 12, 12, 1, 256, true, true, false));

} // namespace flashkda_hip::gfx1250::context_policy

#undef FLASH_KDA_CONTEXT_POLICY_HD
