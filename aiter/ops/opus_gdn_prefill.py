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
    k2_mode: int,
) -> None: ...


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
    k2_mode: int = 0,
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
        k1_algo, k2_mode,
    )

    if pad_len > 0:
        o = o[:, :T]

    return o, final_st if output_final_state else None
