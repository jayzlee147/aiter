# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx


@compile_ops("module_opus_gdn_prefill")
def _opus_gdn_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    has_initial_state: bool,
    output_final_state: bool,
    BT: int,
    BV: int,
    num_warps: int,
    k1_algo: int,
    pipeline: bool,
    occ_hint: int,
) -> None: ...


@compile_ops("module_opus_gdn_prefill")
def _opus_gdn_prefill_split_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    has_initial_state: bool,
    output_final_state: bool,
    BT: int,
    BV: int,
    num_warps: int,
) -> None: ...


@compile_ops("module_opus_gdn_prefill")
def _opus_gdn_wavefront_h_fwd(
    k: torch.Tensor,
    w_bar: torch.Tensor,
    u_bar: torch.Tensor,
    g_cumsum: torch.Tensor,
    h_out: torch.Tensor,
    v_new_out: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    has_initial_state: bool,
    output_final_state: bool,
    S: int,
    BT: int,
    q: torch.Tensor,
    o: torch.Tensor,
    scale: float,
) -> None: ...


def opus_gdn_wavefront_h_fwd(
    k: torch.Tensor,
    w_bar: torch.Tensor,
    u_bar: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    save_v_new: bool = True,
    S: int = 8,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Wavefront-parallel h-state scan for GDN prefill (forward only, scan-only).

    Args:
        k: [B, T, H, K] bf16 — keys
        w_bar: [B, T, H, K] bf16 — WY factor w
        u_bar: [B, T, H, V] bf16 — WY factor u
        g_cumsum: [B, T, H] fp32 — cumulative gate sums
        initial_state: [B, H, V, K] fp32 or None
        output_final_state: whether to return final h-state
        save_v_new: whether to output corrected values
        S: chunks per super-chunk (wavefront segment size)
        BT: chunk size (64 or 128)

    Returns:
        (h, v_new, final_state):
            h: [B, NT, H, K, V] bf16 — h snapshots at each chunk boundary
            v_new: [B, T, H, V] bf16 or None
            final_state: [B, H, V, K] fp32 or None
    """
    B, T, H, K = k.shape
    V = u_bar.shape[-1]
    NT = T // BT
    assert T % BT == 0, f"T={T} must be a multiple of BT={BT}"
    assert NT % S == 0, f"NT={NT} must be a multiple of S={S}"

    h_out = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device=k.device)
    v_new_out = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=k.device) if save_v_new else torch.empty(0, device=k.device, dtype=torch.bfloat16)

    has_init = initial_state is not None
    init_st = initial_state.contiguous().float() if has_init else torch.empty(0, device=k.device, dtype=torch.float32)
    final_st = torch.empty(B, H, V, K, dtype=torch.float32, device=k.device) if output_final_state else torch.empty(0, device=k.device, dtype=torch.float32)

    empty_bf16 = torch.empty(0, device=k.device, dtype=torch.bfloat16)
    _opus_gdn_wavefront_h_fwd(
        k, w_bar, u_bar, g_cumsum,
        h_out, v_new_out,
        init_st, final_st,
        has_init, output_final_state, S, BT,
        empty_bf16, empty_bf16, 0.0,
    )

    return h_out, v_new_out if save_v_new else None, final_st if output_final_state else None


def opus_gdn_wavefront_fused_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w_bar: torch.Tensor,
    u_bar: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    S: int = 8,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fused wavefront scan + output for GDN prefill.
    Computes h-state scan and output in a single kernel, eliminating
    intermediate h snapshot and v_new stores.

    Returns:
        (o, final_state):
            o: [B, T, H, V] bf16
            final_state: [B, H, V, K] fp32 or None
    """
    B, T, H, K = k.shape
    V = u_bar.shape[-1]
    NT = T // BT
    assert T % BT == 0
    assert NT % S == 0

    o = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=k.device)
    h_out = torch.empty(0, device=k.device, dtype=torch.bfloat16)
    v_new_out = torch.empty(0, device=k.device, dtype=torch.bfloat16)

    has_init = initial_state is not None
    init_st = initial_state.contiguous().float() if has_init else torch.empty(0, device=k.device, dtype=torch.float32)
    final_st = torch.empty(B, H, V, K, dtype=torch.float32, device=k.device) if output_final_state else torch.empty(0, device=k.device, dtype=torch.float32)

    _opus_gdn_wavefront_h_fwd(
        k, w_bar, u_bar, g_cumsum,
        h_out, v_new_out,
        init_st, final_st,
        has_init, output_final_state, S, BT,
        q, o, scale,
    )

    return o, final_st if output_final_state else None


def opus_gdn_prefill_split_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    BT: int = 64,
    BV: int = 64,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    o = torch.empty_like(v)

    has_init = initial_state is not None
    init_st = (
        initial_state.contiguous().float()
        if has_init
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )
    final_st = (
        torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
        if output_final_state
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )

    _opus_gdn_prefill_split_fwd(
        q, k, v, g, beta, o, scale,
        init_st, final_st, has_init, output_final_state, BT, BV, num_warps,
    )

    if pad_len > 0:
        o = o[:, :T]

    return o, final_st if output_final_state else None


def opus_gdn_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    BT: int = 32,
    BV: int = 64,
    num_warps: int = 4,
    k1_algo: int = 1,
    pipeline: bool = False,
    occ_hint: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Opus HIP kernel for Gated DeltaNet prefill (forward only).

    Fuses all 4 chunkwise steps into 2 HIP kernels (K1 + K2).
    K=V=128 specialized, gfx942/gfx950.

    Args:
        q: [B, T, H, K] bf16
        k: [B, T, H, K] bf16
        v: [B, T, H, V] bf16
        g: [B, T, H] — gate (log-space decay), any dtype (cast to fp32)
        beta: [B, T, H] — sigmoid gating, any dtype (cast to fp32)
        scale: 1/sqrt(K) if None
        initial_state: [B, H, V, K] fp32 or None
        output_final_state: whether to return final hidden state
        BT: chunk size, 32 (default), 64, 16, or 128 (gfx950 only)
        occ_hint: 0=auto, 2=force OCC=2 (serialized b/c GEMMs, reduced LDS)

    Returns:
        (o, final_state): o is [B, T, H, V] bf16, final_state is [B, H, V, K] fp32 or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    if BT == 128 and get_gfx() != "gfx950":
        raise ValueError("BT=128 requires gfx950 (MI350) — LDS exceeds gfx942 64KB limit")

    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    o = torch.empty_like(v)

    has_init = initial_state is not None
    init_st = (
        initial_state.contiguous().float()
        if has_init
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )
    final_st = (
        torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)
        if output_final_state
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )

    _opus_gdn_prefill_fwd(
        q, k, v, g, beta, o, scale,
        init_st, final_st, has_init, output_final_state, BT, BV, num_warps,
        k1_algo, pipeline, occ_hint,
    )

    if pad_len > 0:
        o = o[:, :T]

    return o, final_st if output_final_state else None
