# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from flydsl.runtime.device import get_rocm_arch

from .kernels.gdr_decode import create_vk_gdr_decode_kernel
from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "flydsl_gdr_decode",
]


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()


def get_default_kwargs(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    is_kda=False,
):
    d = {}
    d["NUM_BLOCKS_PER_V_DIM"] = 1
    d["NUM_WARPS"] = 4
    d["WARP_THREADS_K"] = 8
    d["SHARE_KDA_GATE_LDS"] = False
    # Kimi-K3 TP8 small-batch decode needs extra V parallelism. Scale the split
    # down quickly as batch itself supplies enough workgroups; retaining the
    # batch-1 split at larger batches increases VMEM latency and loses badly.
    # These shapes were tuned on gfx942/MI308X.
    if (
        is_kda
        and GDR_GPU_ARCH == "gfx942"
        and state_dtype_str == "torch.float32"
        and batch_size in (1, 2, 4)
        and seq_length == 1
        and num_k_heads == num_v_heads == 8
        and head_k_dim == head_v_dim == 128
    ):
        if batch_size == 1:
            d.update(NUM_BLOCKS_PER_V_DIM=16, NUM_WARPS=4, WARP_THREADS_K=32)
        elif batch_size == 2:
            d.update(NUM_BLOCKS_PER_V_DIM=8, NUM_WARPS=4, WARP_THREADS_K=16)
        else:  # batch_size == 4
            d.update(NUM_BLOCKS_PER_V_DIM=2, NUM_WARPS=4, WARP_THREADS_K=16)
    elif (
        is_kda
        and GDR_GPU_ARCH == "gfx942"
        and state_dtype_str == "torch.float32"
        and batch_size >= 8
        and seq_length == 1
        and num_k_heads == num_v_heads == 8
        and head_k_dim == head_v_dim == 128
    ):
        # Keep roughly 128 blocks at B8, then switch to full-V blocks once the
        # batch supplies that grid parallelism itself. Sixteen K lanes halves
        # the K-loop depth versus the generic 8-lane layout.
        d.update(
            NUM_BLOCKS_PER_V_DIM=2 if batch_size == 8 else 1,
            NUM_WARPS=4,
            WARP_THREADS_K=16,
        )
    global GDR_GLOBAL_CONFIG_MAP
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        fname = os.path.join(Path(__file__).resolve().parent, "gdr_decode_tuned.csv")
        with open(fname, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = dict(row)
                arch, b, sq, nkh, nvh, khd, vhd = (
                    obj["arch"],
                    int(obj["b"]),
                    int(obj["sq"]),
                    int(obj["num_k_heads"]),
                    int(obj["num_v_heads"]),
                    int(obj["head_k_dim"]),
                    int(obj["head_v_dim"]),
                )
                d_str, sd_str = obj["dtype"], obj["state_dtype"]
                if float(obj["duration"]) < 10000.0:
                    _dict[(d_str, sd_str, arch, b, sq, nkh, nvh, khd, vhd)] = {
                        "NUM_BLOCKS_PER_V_DIM": int(obj["NUM_BLOCKS_PER_V_DIM"]),
                        "NUM_WARPS": int(obj["NUM_WARPS"]),
                        "WARP_THREADS_K": int(obj["WARP_THREADS_K"]),
                    }
        GDR_GLOBAL_CONFIG_MAP = _dict
    config = GDR_GLOBAL_CONFIG_MAP.get(
        (
            dtype_str,
            state_dtype_str,
            GDR_GPU_ARCH,
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        ),
        None,
    )
    if config:
        d.update(config)
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    is_kda: bool = False,
    lower_bound: float = -5.0,
    stream: torch.cuda.Stream = None,
    read_indices: torch.Tensor | None = None,
    write_indices: torch.Tensor | None = None,
):
    if is_kda and query.shape[1] != 1:
        raise ValueError(
            "FlyDSL KDA currently supports single-token decode only; "
            f"got seq_length={query.shape[1]}"
        )
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    dtype = query.dtype
    read_indices = indices if read_indices is None else read_indices
    write_indices = indices if write_indices is None else write_indices
    for input in [
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        read_indices,
        write_indices,
        out,
    ]:
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, dt_bias, out]:
        assert input.dtype == dtype
    if not is_kda:
        assert b.dtype == dtype
    else:
        assert b.dtype in [dtype, torch.float32]
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert read_indices.dtype == torch.int32
    assert write_indices.dtype == torch.int32
    if query.stride(-1) != 1:
        raise ValueError(
            "`query` must have a contiguous last dimension for vectorized loads; "
            f"got stride {query.stride()}."
        )
    if key.stride(-1) != 1:
        raise ValueError(
            "`key` must have a contiguous last dimension for vectorized loads; "
            f"got stride {key.stride()}."
        )

    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs_ = get_default_kwargs(
        str(dtype),
        str(state_.dtype),
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        is_kda,
    )
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(b.dtype),
        get_dtype_str(state_.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        query.stride(),
        key.stride(),
        value.stride(),
        state_.stride(),
        a.stride(),
        b.stride(),
        use_qk_l2norm,
        is_kda=is_kda,
        lower_bound=lower_bound,
        **kwargs_,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            query,
            key,
            value,
            a,
            b,
            dt_bias.contiguous(),
            A_log.contiguous(),
            read_indices.contiguous(),
            write_indices.contiguous(),
            state_,
            out,
            batch_size,
            stream,
        )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)
