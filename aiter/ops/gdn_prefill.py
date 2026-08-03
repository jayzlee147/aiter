# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense Gated DeltaNet prefill dispatcher with a canonical Triton fallback.

The Opus routes in this module are intentionally limited to the measured dense
gfx942 domain.  Unsupported inputs are delegated, unchanged, to
``chunk_gated_delta_rule_opt_vk``.  Models that already own a workload policy
may request one of the four implementation families explicitly through
``path``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from ._gdn_dense_gfx942_routes import lookup_dense_gfx942_path
from .opus_gdn_c_prefill import (
    OPUS_GDN_C_FUSED,
    OPUS_GDN_C_SPLIT,
    opus_gdn_c_prefill_fwd,
)
from .opus_gdn_wu_prefill import (
    OPUS_GDN_K2_SPLIT,
    OPUS_GDN_K2_WU_FUSED,
    opus_gdn_wu_prefill_fwd,
)
from .triton.gated_delta_net import chunk_gated_delta_rule_opt_vk


GdnPrefillPath = Literal[
    "auto", "c", "cf", "cs", "wu", "wf", "ws", "triton"
]
_EXPLICIT_OPUS_PATHS = frozenset(("c", "cf", "cs", "wu", "wf", "ws"))
_ALL_PATHS = _EXPLICIT_OPUS_PATHS | {"auto", "triton"}
_DENSE_BT = 64
_DENSE_DIM = 128
_MEASURED_GFX = "gfx942"
_MEASURED_CU_COUNT = 80


def _runtime_target(q: torch.Tensor) -> tuple[str, int]:
    properties = torch.cuda.get_device_properties(q.device)
    gfx = properties.gcnArchName.split(":", 1)[0]
    return gfx, int(properties.multi_processor_count)


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _opus_input_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    allow_padding: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.LongTensor | None,
    use_chunk_hip: bool,
    use_chunk_flydsl: bool,
    state_dtype: torch.dtype | None,
    use_exp2: bool,
    num_decodes: int,
    num_decode_tokens: int,
) -> str | None:
    tensors = (("q", q), ("k", k), ("v", v))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if not tensor.is_cuda:
            return f"{name} must be a HIP tensor"
        if tensor.dtype != torch.bfloat16:
            return f"{name} must have dtype bfloat16"
        if not tensor.is_contiguous():
            return f"{name} must be contiguous"

    if q.ndim != 4:
        return "q must have shape [B, T, H, 128]"
    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        return "B, T, and H must be positive"
    if K != _DENSE_DIM:
        return "q feature size must be 128"
    expected_vector_shape = (B, T, H, _DENSE_DIM)
    if tuple(k.shape) != expected_vector_shape:
        return "k must match q shape [B, T, H, 128]"
    if tuple(v.shape) != expected_vector_shape:
        return "v must match q shape [B, T, H, 128]"
    if k.device != q.device or v.device != q.device:
        return "q, k, and v must be on the same device"
    if T % _DENSE_BT and (not allow_padding or o is not None):
        return "T must be divisible by 64"

    expected_scalar_shape = (B, T, H)
    for name, tensor in (("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if tuple(tensor.shape) != expected_scalar_shape:
            return f"{name} must have shape [B, T, H]"
        if tensor.device != q.device:
            return f"{name} must be on the same device as q"
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            return f"{name} must have dtype bfloat16 or float32"
        if not tensor.is_contiguous():
            return f"{name} must be contiguous"

    if o is not None:
        if not isinstance(o, torch.Tensor):
            return "o must be a torch.Tensor"
        if (
            tuple(o.shape) != expected_vector_shape
            or o.device != q.device
            or o.dtype != torch.bfloat16
            or not o.is_contiguous()
        ):
            return "o must be contiguous bf16 and match v shape/device"

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            return "initial_state must be a torch.Tensor"
        if tuple(initial_state.shape) != (B, H, _DENSE_DIM, _DENSE_DIM):
            return "initial_state must have shape [B, H, 128, 128]"
        if initial_state.device != q.device:
            return "initial_state must be on the same device as q"
        if initial_state.dtype != torch.float32:
            return "initial_state must have dtype float32"
        if not initial_state.is_contiguous():
            return "initial_state must be contiguous"

    if state_dtype not in (None, torch.float32):
        return "state_dtype must be None or float32"
    if use_qk_l2norm_in_kernel:
        return "use_qk_l2norm_in_kernel is not supported by Opus dense paths"
    if cu_seqlens is not None:
        return "cu_seqlens/varlen is not supported by Opus dense paths"
    if use_chunk_hip or use_chunk_flydsl:
        return "explicit chunk HIP/FlyDSL selection requires the fallback"
    if not use_exp2:
        return "use_exp2=False is not supported by Opus dense paths"
    if num_decodes != 0 or num_decode_tokens != 0:
        return "decode-prefix inputs are not supported by Opus dense paths"
    return None


def _select_wu_path(T: int, batch_heads: int, with_state_io: bool) -> str:
    """Deterministic WS/WF envelope from the dense closeout."""
    if T <= 64:
        return "wf"
    if T <= 256:
        threshold = 24 if with_state_io else 21
    elif T <= 512:
        threshold = 40
    elif T <= 8192:
        threshold = 48
    else:
        threshold = 64
    return "wf" if batch_heads >= threshold else "ws"


def _select_c_path(T: int, batch_heads: int) -> str:
    """Mirror the standalone C backend's conservative deterministic policy."""
    use_split = (T >= 256 and batch_heads <= 20) or (
        T >= 128 and batch_heads <= 8
    )
    return "cs" if use_split else "cf"


def select_gdn_prefill_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_chunk_hip: bool = False,
    use_chunk_flydsl: bool = False,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    *,
    path: GdnPrefillPath = "auto",
) -> str:
    """Return the concrete implementation family without launching a kernel."""
    normalized_path = str(path).lower()
    if normalized_path not in _ALL_PATHS:
        raise ValueError(
            f"unsupported path={path!r}; expected one of {sorted(_ALL_PATHS)}"
        )
    if normalized_path == "triton":
        return "triton"

    explicit = normalized_path in _EXPLICIT_OPUS_PATHS
    error = _opus_input_error(
        q,
        k,
        v,
        o,
        g,
        beta,
        initial_state,
        allow_padding=explicit,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_chunk_hip=use_chunk_hip,
        use_chunk_flydsl=use_chunk_flydsl,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )
    if error is not None:
        if explicit:
            raise ValueError(f"path={normalized_path!r} is unavailable: {error}")
        return "triton"

    gfx, cu_count = _runtime_target(q)
    if explicit:
        if normalized_path in ("c", "cf", "cs") and gfx != "gfx942":
            raise ValueError(
                f"path={normalized_path!r} requires gfx942, got {gfx}"
            )
        if normalized_path in ("wu", "wf", "ws") and gfx not in (
            "gfx942",
            "gfx950",
        ):
            raise ValueError(
                f"path={normalized_path!r} requires gfx942/gfx950, got {gfx}"
            )
    elif gfx != _MEASURED_GFX or cu_count != _MEASURED_CU_COUNT:
        return "triton"

    B, T, H, _ = q.shape
    padded_T = ((T + _DENSE_BT - 1) // _DENSE_BT) * _DENSE_BT
    with_state_io = initial_state is not None or output_final_state

    if normalized_path == "c":
        selected = _select_c_path(padded_T, B * H)
    elif normalized_path == "wu":
        selected = _select_wu_path(padded_T, B * H, with_state_io)
    elif explicit:
        selected = normalized_path
    else:
        measured_state: bool | None
        if initial_state is None and not output_final_state:
            measured_state = False
        elif initial_state is not None and output_final_state:
            measured_state = True
        else:
            measured_state = None
        selected = None
        if measured_state is not None:
            selected = lookup_dense_gfx942_path(B, T, H, measured_state)
        if selected is None:
            selected = _select_wu_path(T, B * H, with_state_io)

    if selected in ("cf", "cs") and o is not None and _shares_storage(o, v):
        if explicit:
            raise ValueError(
                f"path={selected!r} requires o not to alias v storage"
            )
        return "triton"
    return selected


def gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_chunk_hip: bool = False,
    use_chunk_flydsl: bool = False,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    *,
    path: GdnPrefillPath = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run dense GDN prefill through Opus when eligible, otherwise opt_vk.

    ``path="auto"`` uses the exact 477-shape gfx942/80-CU winner table.  A
    table miss uses the fixed W/U WS/WF envelope; it never consults process
    environment variables for family selection.  ``path="triton"`` forces the
    canonical fallback.  Explicit Opus paths raise on unsupported inputs rather
    than silently changing the caller's requested implementation.
    """
    selected = select_gdn_prefill_path(
        q,
        k,
        v,
        o=o,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_chunk_hip=use_chunk_hip,
        use_chunk_flydsl=use_chunk_flydsl,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        path=path,
    )

    if selected == "triton":
        return chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            o=o,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_chunk_hip=use_chunk_hip,
            use_chunk_flydsl=use_chunk_flydsl,
            state_dtype=state_dtype,
            use_exp2=use_exp2,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )

    if scale is not None and not math.isfinite(float(scale)):
        raise ValueError("scale must be finite for an Opus path")
    if selected in ("cf", "cs"):
        return opus_gdn_c_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            c_mode=(
                OPUS_GDN_C_FUSED if selected == "cf" else OPUS_GDN_C_SPLIT
            ),
            out=o,
            use_env_overrides=False,
        )

    return opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        BT=_DENSE_BT,
        BV=64,
        num_warps=8 if selected == "wf" else 4,
        k1_algo=1,
        k2_mode=(
            OPUS_GDN_K2_WU_FUSED if selected == "wf" else OPUS_GDN_K2_SPLIT
        ),
        out=o,
        use_env_overrides=False,
    )


__all__ = ["GdnPrefillPath", "gdn_prefill", "select_gdn_prefill_path"]
