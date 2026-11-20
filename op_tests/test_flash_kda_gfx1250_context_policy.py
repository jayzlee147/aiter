# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU/source contracts for the production gfx1250 context routes."""

import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY_SOURCE = (
    _REPO_ROOT
    / "csrc/kernels/flash_kda/gfx1250/k2_kda_context_gfx1250_policy.hpp"
)
_BACKEND_SOURCE = (
    _REPO_ROOT
    / "csrc/kernels/flash_kda/gfx1250/flash_kda_gfx1250_backend.cu"
)
_ADAPTER_SOURCE = _REPO_ROOT / "csrc/kernels/flash_kda/flash_kda_aiter.cu"
_COMPILER_CONFIG = _REPO_ROOT / "aiter/jit/optCompilerConfig.json"


def _ceil_div(numerator: int, denominator: int) -> int:
    return numerator // denominator + (numerator % denominator != 0)


def _choose_auto_group_chunks(
    total_tokens: int,
    value_heads: int,
    num_cus: int,
) -> int:
    if total_tokens <= 0 or value_heads <= 0 or num_cus <= 0:
        return 4
    total_token_chunks = _ceil_div(total_tokens, 16)
    return 4 if total_token_chunks * value_heads < 6 * num_cus else 8


def _should_use_lean_scan_8w(
    value_heads: int,
    has_state_out: bool,
) -> bool:
    return value_heads == 12 and has_state_out


def _should_use_auto_g16(
    total_tokens: int,
    query_heads: int,
    value_heads: int,
    num_sequences: int,
    num_cus: int,
    *,
    varlen: bool,
    has_state_out: bool,
    state_fp32: bool,
) -> bool:
    measured_groups = {12, 14, 15, 16, 24, 32, 48, 64}
    return (
        varlen
        and num_sequences == 1
        and query_heads in {3, 12}
        and value_heads == 12
        and num_cus == 256
        and has_state_out
        and state_fp32
        and total_tokens > 0
        and total_tokens % 256 == 0
        and total_tokens // 256 in measured_groups
    )


def test_gfx1250_module_preserves_single_sequence_packed_metadata():
    adapter = _ADAPTER_SOURCE.read_text()
    config = json.loads(_COMPILER_CONFIG.read_text())["module_flash_kda_gfx1250"]

    define = "'-DFLASH_KDA_GFX1250_BACKEND=1'"
    assert define in config["flags_extra_cc"]
    assert define in config["flags_extra_hip"]
    assert "#if defined(FLASH_KDA_GFX1250_BACKEND)" in adapter
    assert "kPreserveSingleSequencePackedMetadata = true" in adapter
    assert adapter.count("!kPreserveSingleSequencePackedMetadata") == 2
    assert adapter.count("single_sequence_dense ? nullptr") == 2


def test_gfx1250_context_policy_source_contract():
    source = _POLICY_SOURCE.read_text()
    backend = _BACKEND_SOURCE.read_text()

    assert "uint32_t choose_auto_group_chunks" in source
    assert "bool should_use_lean_scan_8w" in source
    assert "bool should_use_auto_g16" in source
    assert "bool\nis_measured_auto_g16_query_head_count" in source
    assert "bool is_measured_auto_g16_group_count" in source
    assert "kAutoRouteTokenChunk = 16" in source
    assert "kAutoRouteG8WorkUnitsPerCu = 6" in source
    assert "kMi455NumCus = 256" in source
    assert "kAutoRouteG16GroupChunks = 16" in source
    assert "kAutoRouteG16GvaQueryHeads = 3" in source
    assert "kAutoRouteG16EqualQueryHeads = 12" in source
    assert "kLeanScan8wValueHeads = 12" in source
    assert "total_token_chunks * value_heads" in source
    assert "work_units < g8_threshold ? 4 : 8" in source
    assert "value_heads == kLeanScan8wValueHeads && has_state_out" in source
    assert "g16_groups == 12" in source
    assert "g16_groups >= 14 && g16_groups <= 16" in source
    assert "g16_groups == 24" in source
    assert "g16_groups == 32" in source
    assert "g16_groups == 48" in source
    assert "g16_groups == 64" in source
    assert "query_heads == kAutoRouteG16GvaQueryHeads" in source
    assert "query_heads == kAutoRouteG16EqualQueryHeads" in source

    auto_start = backend.index("const bool auto_use_g16")
    auto_end = backend.index("const bool use_context", auto_start)
    auto_call = backend[auto_start:auto_end]
    assert "static_cast<uint64_t>(T_total)" in auto_call
    assert "static_cast<uint32_t>(H_q)" in auto_call
    assert "static_cast<uint32_t>(H)" in auto_call
    assert "device_props.multiProcessorCount" in auto_call
    assert "cu_seqlens" not in auto_call
    assert "max_seqlen_upper_bound" not in auto_call
    assert "context_policy::should_use_auto_g16(" in auto_call
    assert "auto_use_g16" in auto_call
    assert "K2_CONTEXT_GFX1250_GROUP16" in auto_call
    assert "FLASH_KDA_GFX1250_AUTO_G16" not in backend
    assert 'route == nullptr || std::strcmp(route, "auto") == 0' in backend
    assert (
        'route != nullptr && std::strcmp(route, "c16") == 0' in backend
    )
    assert "FLASH_KDA_GFX1250_ROUTE must be auto or c16" in backend
    assert "const bool use_context = request_auto" in backend
    for unsupported_route in (
        '"bt64"',
        '"context4"',
        '"context8"',
        '"context16"',
        '"context32"',
        '"context64"',
    ):
        assert unsupported_route not in backend

def test_gfx1250_auto_group_policy_boundaries():
    # For 256 CUs and HV=12, 6*CU == 1536 work units.  The last 127-chunk
    # shape selects G4; the next token starts chunk 128 and equality selects G8.
    assert _choose_auto_group_chunks(2032, 12, 256) == 4
    assert _choose_auto_group_chunks(2033, 12, 256) == 8
    assert _choose_auto_group_chunks(2048, 12, 256) == 8

    # The base G4/G8 helper scales with value heads.  The separate measured
    # G16 promotion gate is checked below.
    assert _choose_auto_group_chunks(24560, 1, 256) == 4
    assert _choose_auto_group_chunks(24561, 1, 256) == 8
    supported = {4, 8}
    decisions = {
        _choose_auto_group_chunks(tokens, value_heads, 256)
        for tokens in (1, 256, 1024, 2032, 2033, 4096, 16384)
        for value_heads in (1, 2, 4, 8, 12, 16, 64)
    }
    assert decisions <= supported


def test_gfx1250_auto_g16_policy_is_an_exact_measured_envelope():
    selected_groups = {12, 14, 15, 16, 24, 32, 48, 64}
    for query_heads in (3, 12):
        for group_count in range(1, 65):
            assert _should_use_auto_g16(
                group_count * 256,
                query_heads,
                12,
                1,
                256,
                varlen=True,
                has_state_out=True,
                state_fp32=True,
            ) == (group_count in selected_groups)

    base = dict(
        total_tokens=4096,
        query_heads=12,
        value_heads=12,
        num_sequences=1,
        num_cus=256,
        varlen=True,
        has_state_out=True,
        state_fp32=True,
    )
    for changes in (
        {"total_tokens": 4097},
        {"query_heads": 1},
        {"query_heads": 2},
        {"query_heads": 4},
        {"query_heads": 6},
        {"value_heads": 1},
        {"num_sequences": 2},
        {"num_cus": 304},
        {"varlen": False},
        {"has_state_out": False},
        {"state_fp32": False},
    ):
        assert not _should_use_auto_g16(**(base | changes))


def test_gfx1250_lean_scan_auto_policy_is_narrow_and_state_out_only():
    assert _should_use_lean_scan_8w(12, True)
    assert not _should_use_lean_scan_8w(12, False)
    for value_heads in (0, 1, 2, 3, 4, 8, 16, 64):
        assert not _should_use_lean_scan_8w(value_heads, True)
