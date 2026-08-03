# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx


# Explicit W/U backend selection.  Keep the integer ABI stable so the adapter
# layer can select a path without relying on process-global environment state.
OPUS_GDN_K2_AUTO = 0
OPUS_GDN_K2_WU_FUSED = 1
OPUS_GDN_K2_SPLIT = 2
OPUS_GDN_SUPPORTED_K2_MODES = (
    OPUS_GDN_K2_AUTO,
    OPUS_GDN_K2_WU_FUSED,
    OPUS_GDN_K2_SPLIT,
)


@compile_ops("module_opus_gdn_wu_prefill")
def _opus_gdn_wu_prefill_fwd(
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
    use_env_overrides: bool = True,
) -> None: ...


def opus_gdn_wu_prefill_fwd(
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
    num_warps: int = None,
    k1_algo: int = 1,
    k2_mode: int = 0,
    out: torch.Tensor = None,
    use_env_overrides: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Opus HIP kernel for Gated DeltaNet prefill (forward only).

    Fuses all 4 chunkwise steps into specialized HIP kernels.
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
        BT: chunk size, 64 (tuned default), 32, 16, or 128 (gfx950 only)
        num_warps: K2 wave count. If omitted, uses 8 for the tuned BT64
            WS/WF paths and 4 for the legacy BT16/32/128 configurations.
        k2_mode: 0 selects between the W/U fused and split families using the
            measured gfx942 T/B*H/state envelope; 1 forces fused (WF); 2
            forces split scan/output (WS). Forced split requires BT=64.
        out: optional preallocated contiguous bf16 output tensor. Supplying an
            output buffer requires an already BT-aligned sequence length.
        use_env_overrides: whether the native W/U launcher may read its
            OPUS_GDN_* benchmark overrides. Defaults to True for direct
            backend A/B compatibility; production adapters should pass False.

    Returns:
        (o, final_state): o is [B, T, H, V] bf16, final_state is [B, H, V, K] fp32 or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if num_warps is None:
        num_warps = 8 if BT == 64 else 4

    if k2_mode not in OPUS_GDN_SUPPORTED_K2_MODES:
        raise ValueError(
            f"Unsupported k2_mode={k2_mode}; expected one of "
            f"{OPUS_GDN_SUPPORTED_K2_MODES}"
        )
    if k2_mode == OPUS_GDN_K2_SPLIT and BT != 64:
        raise ValueError("k2_mode=2 (WS) requires BT=64")

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

    if out is not None:
        if pad_len:
            raise ValueError("a preallocated output requires T to be BT-aligned")
        if (
            out.shape != v.shape
            or out.dtype != torch.bfloat16
            or out.device != v.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be a contiguous bf16 tensor matching v shape/device"
            )
        o = out
    else:
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

    _opus_gdn_wu_prefill_fwd(
        q, k, v, g, beta, o, scale,
        init_st, final_st, has_init, output_final_state, BT, BV, num_warps,
        k1_algo, k2_mode, use_env_overrides,
    )

    if pad_len > 0:
        o = o[:, :T]

    return o, final_st if output_final_state else None
