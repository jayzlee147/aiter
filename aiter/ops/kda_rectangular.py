# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 KDA inference on gfx950 with 64 heads, 64 key and 128 value channels.

States are FP32 and V-first. Warm up the target operation before HIP graph
capture. Prefill metadata is specific to the 16-token chunks used here.
"""

import functools
import threading

import torch


@functools.cache
def _module(device):
    from aiter import AITER_TRITON_ONLY

    if AITER_TRITON_ONLY or not torch.version.hip:
        raise RuntimeError("Rectangular HIP KDA requires an AITER HIP build")
    from aiter.jit import core

    with torch.cuda.device(device):
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise RuntimeError(f"Rectangular HIP KDA requires gfx950, got {arch}")
        name = "module_kda_rectangular"
        if not core.AITER_REBUILD:
            try:
                return core.get_module(name)
            except ModuleNotFoundError:
                pass
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up rectangular HIP KDA before graph capture")
        cfg = core.get_args_of_build(name)
        core.build_module(
            name,
            cfg["srcs"],
            cfg["flags_extra_cc"],
            cfg["flags_extra_hip"],
            cfg["blob_gen_cmd"],
            cfg["extra_include"],
            cfg["extra_ldflags"],
            cfg["verbose"],
            cfg["is_python_module"],
            cfg["is_standalone"],
            cfg["torch_exclude"],
            cfg["third_party"],
        )
        return core.get_module(name)


def _check(t, shape=None, dtype=None, name="tensor", dense=True):
    if not t.is_cuda or (shape is not None and tuple(t.shape) != tuple(shape)):
        raise ValueError(f"Invalid {name} device/shape: {t.device}, {tuple(t.shape)}")
    if dtype is not None and t.dtype not in (
        (dtype,) if isinstance(dtype, torch.dtype) else dtype
    ):
        raise ValueError(f"Invalid {name} dtype: {t.dtype}")
    if dense and not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _same_device(ref, *tensors):
    if any(t is not None and t.device != ref.device for t in tensors):
        raise ValueError("All HIP KDA tensors must be on the same device")


def prepare_metadata(seq_lens, device):
    """Create packed BT16 chunk indices/offsets outside graph capture."""
    ci = []
    co = [0]
    for seq, n in enumerate(seq_lens):
        if n < 0:
            raise ValueError("Sequence lengths must be nonnegative")
        ci.extend((seq, c) for c in range((n + 15) // 16))
        co.append(len(ci))
    return (
        torch.tensor(ci, device=device, dtype=torch.int32).reshape(-1, 2),
        torch.tensor(co, device=device, dtype=torch.int64),
    )


# Host resources for the side stream are created on first use. Each thread,
# device and submitting stream gets its own stream and dependency events.
_pipeline_tls = threading.local()


def _pipeline_capture(device):
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _pipeline_resources(stream):
    cache = getattr(_pipeline_tls, "resources", None)
    if cache is None:
        cache = _pipeline_tls.resources = {}
    key = (stream.device.index, stream.cuda_stream)
    if key not in cache:
        cache[key] = (
            torch.cuda.Stream(device=stream.device),
            torch.cuda.Event(),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
    return cache[key]


def _prepare_walk_pair(
    mod,
    inputs,
    scratch,
    cu,
    indices,
    offsets,
    initial,
    final,
    out,
    lb,
    scale,
    second_conv=None,
):
    main = torch.cuda.current_stream(out.device)
    side, start, first_ready, last_ready = _pipeline_resources(main)
    carry = torch.empty((1, 64, 128, 64), device=out.device, dtype=torch.float32)
    start.record(main)
    side.wait_event(start)
    middle = indices.size(0) // 2
    try:
        for begin, end, prev, dest, ready in (
            (0, middle, initial, carry, first_ready),
            (middle, indices.size(0), carry, final, last_ready),
        ):
            tiles = tuple(t[begin:end] for t in scratch)
            with torch.cuda.stream(side):
                if begin and second_conv is not None:
                    mod.conv_half(*second_conv, 1)
                mod.prepare_for_blocked(
                    *inputs, cu, indices[begin:end], *tiles, lb, scale
                )
                ready.record(side)
            main.wait_event(ready)
            mod.walk_part(*tiles, cu, offsets, prev, dest, out, begin)
    except BaseException:
        # Workspace allocations belong to main. Another call may recycle them
        # after this exception, so preserve the side-stream lifetime dependency.
        main.wait_stream(side)
        raise


@torch.no_grad()
def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    lower_bound,
    cu_seqlens,
    chunk_indices=None,
    chunk_offsets=None,
    *,
    seq_lens=None,
    initial_state=None,
    output_final_state=False,
    out_gate=None,
    norm_weight=None,
    norm_eps=1e-5,
    scale=None,
    out=None,
    _validate_only=False,
    _from_mixer=False,
    _blocked_qkv=False,
    _second_conv=None,
):
    """BT16 HIP/MFMA core. Packed [1,T,H,K/V], FP32 V-first state.

    Use prepare_metadata for the BT16 indices and offsets needed for capture.
    Q/K/g may have padded token strides. Initial/final state is [B,H,128,64].
    Optional output norm uses BF16 core output and sigmoid(out_gate).
    """
    if q.ndim != 4:
        raise ValueError("q must be [1,T,H,64]")
    B, T, H, K = q.shape
    if B != 1 or K != 64 or H != 64:
        raise ValueError("Expected [1,T,64,64] q")
    n = cu_seqlens.numel() - 1
    if n < 1:
        raise ValueError("cu_seqlens must describe at least one sequence")
    for name, t, shape in (
        ("q", q, (1, T, H, 64)),
        ("k", k, (1, T, H, 64)),
        ("v", v, (1, T, H, 128)),
        ("g", g, (1, T, H, 64)),
    ):
        _check(t, shape, torch.bfloat16, name, dense=False)
        if t.stride()[-2:] != (shape[-1], 1):
            raise ValueError(f"{name} must be dense per head")
    _check(beta, (1, T, H), (torch.bfloat16, torch.float32), "beta", False)
    if beta.stride(-1) != 1:
        raise ValueError("beta must be dense over heads")
    _check(A_log, (H,), torch.float32, "A_log")
    _check(dt_bias, (H * 64,), torch.float32, "dt_bias")
    _check(cu_seqlens, (n + 1,), (torch.int32, torch.int64), "cu_seqlens")
    if not -5 <= lower_bound < 0:
        raise ValueError("lower_bound must be in [-5,0)")
    if (chunk_indices is None) != (chunk_offsets is None):
        raise ValueError("Pass both metadata tensors")
    if chunk_indices is None:
        if seq_lens is None or _pipeline_capture(q.device):
            raise ValueError(
                "Pass precomputed HIP BT16 metadata, or host seq_lens outside capture"
            )
        chunk_indices, chunk_offsets = prepare_metadata(seq_lens, q.device)
    _check(chunk_indices, None, torch.int32, "chunk_indices")
    _check(chunk_offsets, (n + 1,), torch.int64, "chunk_offsets")
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must be [NT,2]")
    if seq_lens is not None and (
        len(seq_lens) != n
        or sum(seq_lens) != T
        or chunk_indices.size(0) != sum((x + 15) // 16 for x in seq_lens)
    ):
        raise ValueError("HIP BT16 metadata and sequence lengths do not agree")
    if initial_state is not None:
        _check(initial_state, (n, H, 128, 64), torch.float32, "initial_state")
    if (out_gate is None) != (norm_weight is None):
        raise ValueError("Pass output gate and norm weight together")
    if out_gate is not None:
        _check(out_gate, (1, T, H, 128), torch.bfloat16, "out_gate", False)
        if out_gate.stride()[-2:] != (128, 1):
            raise ValueError("out_gate must be dense per head")
        _check(norm_weight, (128,), (torch.bfloat16, torch.float32), "norm_weight")
    _same_device(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        initial_state,
        out_gate,
        norm_weight,
        out,
    )
    if out is not None:
        _check(out, (1, T, H, 128), torch.bfloat16, "out")
    if _validate_only:
        return None, None
    caller_out = out is not None
    if caller_out and out_gate is not None:
        # The state walk writes out before the norm reads its gate and weight.
        if torch._C._overlaps(out, out_gate):
            out_gate = out_gate.clone()
        if torch._C._overlaps(out, norm_weight):
            norm_weight = norm_weight.clone()
    if out is None:
        out = torch.empty_like(v, memory_format=torch.contiguous_format)
    _check(out, (1, T, H, 128), torch.bfloat16, "out")
    fs = (
        torch.empty((n, H, 128, 64), device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    nt = chunk_indices.size(0)
    qg = torch.empty((nt, H, 16, 64), device=q.device, dtype=torch.bfloat16)
    w = torch.empty_like(qg)
    kg = torch.empty_like(qg)
    u = torch.empty((nt, H, 16, 128), device=q.device, dtype=torch.bfloat16)
    aq = torch.empty((nt, H, 16, 16), device=q.device, dtype=torch.bfloat16)
    decay = torch.empty((nt, H, 64), device=q.device, dtype=torch.float32)
    mod = _module(q.device)
    # Paired measurements favor two streams only inside a captured graph.
    if (
        _blocked_qkv
        and n == 1
        and T == 32768
        and nt == 2048
        and _pipeline_capture(out.device)
    ):
        _prepare_walk_pair(
            mod,
            (q, k, v, g, beta, A_log, dt_bias),
            (qg, w, u, kg, aq, decay),
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            initial_state,
            fs,
            out,
            lower_bound,
            64**-0.5 if scale is None else scale,
            _second_conv,
        )
    else:
        prepare = mod.prepare_for_walk if _from_mixer else mod.prepare_for_core
        if _blocked_qkv:
            prepare = mod.prepare_for_blocked
        prepare(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            chunk_indices,
            qg,
            w,
            u,
            kg,
            aq,
            decay,
            lower_bound,
            64**-0.5 if scale is None else scale,
        )
        mod.walk(
            qg, w, u, kg, aq, decay, cu_seqlens, chunk_offsets, initial_state, fs, out
        )
    if out_gate is not None:
        if caller_out:
            mod.norm_live(out, out_gate, norm_weight, out, cu_seqlens, norm_eps)
        else:
            mod.norm(out, out_gate, norm_weight, out, norm_eps)
    return out, fs


@torch.no_grad()
def prefill(
    mixed_qkv,
    conv_weight,
    gate,
    beta,
    out_gate,
    A_log,
    dt_bias,
    norm_weight,
    cu_seqlens,
    seq_lens,
    *,
    conv_state=None,
    has_initial_state=None,
    initial_state=None,
    chunk_indices=None,
    chunk_offsets=None,
    output_final_state=True,
    norm_eps=1e-5,
    lower_bound=-5.0,
):
    """Run the complete packed mixer and return (output, optional FP32 final state).

    mixed_qkv is [T, 2*H*K+H*V], gate is [1,T,H,K], beta is [1,T,H],
    and out_gate is [1,T,H,V]. conv_weight is [2*H*K+H*V,4]. When supplied,
    conv_state [B,2*H*K+H*V,3] receives the updated raw convolution history.
    has_initial_state selects which input histories are used. initial_state
    is independently optional and has layout [B,H,V,K]. The output is BF16
    [1,T,H,V]. Pass matching host seq_lens, device cumulative lengths and
    BT16 metadata from prepare_metadata for graph capture.
    """
    T = mixed_qkv.size(0)
    n = len(seq_lens)
    if n < 1:
        raise ValueError("seq_lens must describe at least one sequence")
    _check(mixed_qkv, (T, 16384), torch.bfloat16, "mixed_qkv", False)
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv channels must be dense")
    _check(conv_weight, (16384, 4), torch.bfloat16, "conv_weight", False)
    _check(cu_seqlens, (n + 1,), (torch.int32, torch.int64), "cu_seqlens")
    if sum(seq_lens) != T or any(x < 0 for x in seq_lens):
        raise ValueError("Invalid sequence lengths")
    if conv_state is not None:
        _check(conv_state, (n, 16384, 3), torch.bfloat16, "conv_state", False)
    if has_initial_state is not None:
        _check(has_initial_state, (n,), torch.bool, "has_initial_state")
    _same_device(mixed_qkv, conv_weight, conv_state, cu_seqlens, has_initial_state)
    x = torch.empty((T, 16384), device=mixed_qkv.device, dtype=torch.bfloat16)
    q, k, v = x.split((4096, 4096, 8192), -1)
    if (chunk_indices is None) != (chunk_offsets is None):
        raise ValueError("Pass both metadata tensors")
    if chunk_indices is None:
        if _pipeline_capture(mixed_qkv.device):
            raise ValueError("Prepare HIP metadata before capture")
        chunk_indices, chunk_offsets = prepare_metadata(seq_lens, mixed_qkv.device)
    args = dict(
        q=q.view(1, T, 64, 64),
        k=k.view(1, T, 64, 64),
        v=v.view(1, T, 64, 128),
        g=gate,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_lens=seq_lens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        out_gate=out_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    chunk_kda(**args, _validate_only=True)
    mod = _module(mixed_qkv.device)
    if (
        initial_state is not None
        and n == 1
        and T == 32768
        and chunk_indices.size(0) == 2048
        and _pipeline_capture(x.device)
    ):
        conv_args = (
            mixed_qkv,
            conv_weight,
            conv_state,
            cu_seqlens,
            has_initial_state,
            x,
        )
        if mod.conv_half(*conv_args, 0):
            return chunk_kda(
                **args, _from_mixer=True, _blocked_qkv=True, _second_conv=conv_args
            )
    blocked = mod.conv_for_mixer(
        mixed_qkv, conv_weight, conv_state, cu_seqlens, has_initial_state, x
    )
    return chunk_kda(**args, _from_mixer=True, _blocked_qkv=blocked)


@torch.no_grad()
def fused_kda_decode(
    mixed_qkv,
    conv_state,
    conv_weight,
    gate,
    beta,
    out_gate,
    A_log,
    dt_bias,
    ssm_state,
    ssm_state_indices,
    cu_seqlens,
    norm_weight,
    norm_eps,
    head_dim=64,
    num_local_heads=64,
    lower_bound=-5.0,
    num_accepted_tokens=None,
    conv_state_indices=None,
    out=None,
    value_head_dim=128,
):
    """Run packed decode and update the convolution and recurrent caches in place.

    mixed_qkv is [T,H*(2*K+V)], out_gate and the BF16 output are [T,H*V].
    The FP32 state cache has layout [slots,H,V,K]. Ordinary decode takes one
    state index per sequence and three convolution history entries. For MTP,
    pass a [B,2..8] checkpoint table, accepted counts and convolution row
    indices. A width of MTP+1 permits up to that many input tokens per row.
    MTP uses positive cache indices and selects the starting checkpoint with
    max(0, accepted-1). cu_seqlens selects live tokens. Supply valid cumulative
    lengths and in-range cache indices on the same device for each replay.
    """
    if (head_dim, value_head_dim, num_local_heads) != (64, 128, 64):
        raise ValueError("Expected H64/K64/V128")
    T = mixed_qkv.size(0)
    n = cu_seqlens.numel() - 1
    spec = num_accepted_tokens is not None
    if n < 1:
        raise ValueError("cu_seqlens must describe at least one sequence")
    for name, t, shape in (
        ("mixed_qkv", mixed_qkv, (T, 16384)),
        ("gate", gate, (1, T, 64, 64)),
        ("out_gate", out_gate, (T, 8192)),
    ):
        _check(t, shape, torch.bfloat16, name, False)
        if t.stride(-1) != 1:
            raise ValueError(f"{name} innermost dimension must be dense")
    if not gate.is_contiguous():
        raise ValueError("gate must be contiguous")
    _check(conv_weight, (16384, 4), torch.bfloat16, "conv_weight", False)
    _check(conv_state, None, torch.bfloat16, "conv_state", False)
    if conv_state.ndim != 3 or conv_state.size(1) != 16384:
        raise ValueError("Invalid conv state shape")
    if any(s <= 0 for s in conv_state.stride()):
        raise ValueError("Convolution cache strides must be positive")
    _check(beta, (1, T, 64), (torch.bfloat16, torch.float32), "beta", False)
    if beta.stride(-1) != 1:
        raise ValueError("beta must be dense over heads")
    _check(A_log, (64,), torch.float32, "A_log")
    _check(dt_bias, (4096,), torch.float32, "dt_bias")
    _check(norm_weight, (128,), (torch.bfloat16, torch.float32), "norm_weight")
    _check(ssm_state, None, torch.float32, "ssm_state", False)
    if ssm_state.shape[1:] != (64, 128, 64) or ssm_state.stride()[1:] != (8192, 64, 1):
        raise ValueError("State must be dense [N,64,128,64]")
    if ssm_state.stride(0) < 64 * 128 * 64:
        raise ValueError("State cache rows must not overlap")
    _check(cu_seqlens, (n + 1,), (torch.int32, torch.int64), "cu_seqlens")
    _check(ssm_state_indices, None, (torch.int32, torch.int64), "state_indices", False)
    if any(s <= 0 for s in ssm_state_indices.stride()):
        raise ValueError("State index strides must be positive")
    if spec:
        if (
            ssm_state_indices.ndim != 2
            or ssm_state_indices.size(0) != n
            or not 2 <= ssm_state_indices.size(1) <= 8
        ):
            raise ValueError("MTP state indices must be [B,2..8]")
        _check(num_accepted_tokens, (n,), (torch.int32, torch.int64), "accepted")
        if conv_state_indices is None:
            raise ValueError("Spec decode requires conv_state_indices")
        _check(
            conv_state_indices,
            (n,),
            (torch.int32, torch.int64),
            "conv_state_indices",
            False,
        )
        if conv_state_indices.stride(0) <= 0:
            raise ValueError("Convolution index stride must be positive")
        if conv_state.size(2) < ssm_state_indices.size(1) + 2:
            raise ValueError("Insufficient conv rollback history")
    else:
        if ssm_state_indices.shape != (n,) or conv_state.size(2) != 3:
            raise ValueError("Normal decode requires [B] indices and history length 3")
    if not -5 <= lower_bound < 0:
        raise ValueError("Invalid gate lower bound")
    if out is None:
        out = (
            torch.zeros((T, 8192), device=mixed_qkv.device, dtype=torch.bfloat16)
            if spec
            else torch.empty((T, 8192), device=mixed_qkv.device, dtype=torch.bfloat16)
        )
    _check(out, (T, 8192), torch.bfloat16, "out")
    _same_device(
        mixed_qkv,
        conv_state,
        conv_weight,
        gate,
        beta,
        out_gate,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        num_accepted_tokens,
        conv_state_indices,
        out,
    )
    _module(mixed_qkv.device).decode(
        mixed_qkv,
        conv_state,
        conv_weight,
        gate,
        beta,
        out_gate,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        num_accepted_tokens,
        conv_state_indices,
        out,
        lower_bound,
        norm_eps,
    )
    return out
