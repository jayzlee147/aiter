# SPDX-License-Identifier: MIT
"""Safety/correctness checks for the experimental FlashKDA raw-pointer ABI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiter.ops.flash_kda as flash_kda
from aiter.jit.core import get_module


def make_inputs(
    tokens: int, heads: int, device: torch.device | str = torch.device("cuda")
) -> dict[str, Any]:
    """Build one deterministic input without depending on diagnostic scripts."""

    device = torch.device(device)
    shape = (1, tokens, heads, 128)
    torch.manual_seed(20260817)
    return {
        "q": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "k": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "v": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "g": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "beta": torch.randn(
            (1, tokens, heads), device=device, dtype=torch.float32
        ),
        "A_log": torch.empty(heads, device=device, dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        "dt_bias": torch.randn(heads * 128, device=device, dtype=torch.float32),
        "initial_state": torch.zeros(
            1, heads, 128, 128, device=device, dtype=torch.float32
        ),
        "cu_seqlens": torch.tensor([0, tokens], device=device, dtype=torch.int32),
        "scale": 128**-0.5,
        "lower_bound": -5.0,
    }


def public_call(x: dict[str, Any]):
    return flash_kda.flash_kda_fwd(
        q=x["q"],
        k=x["k"],
        v=x["v"],
        g=x["g"],
        beta=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        scale=x["scale"],
        initial_state=x["initial_state"],
        output_final_state=True,
        lower_bound=x["lower_bound"],
        cu_seqlens=x["cu_seqlens"],
    )


def rejection_reason(x: dict[str, Any]):
    return flash_kda._native_rejection_reason(
        q=x["q"],
        k=x["k"],
        v=x["v"],
        g=x["g"],
        beta=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        initial_state=x["initial_state"],
        output_final_state=True,
        lower_bound=x["lower_bound"],
        state_v_first=True,
        cu_seqlens=x["cu_seqlens"],
    )


def allocate(x: dict[str, Any]):
    _, tokens, heads, _ = x["q"].shape
    out = torch.empty_like(x["v"])
    final = torch.empty_like(x["initial_state"])
    workspace = torch.empty(
        flash_kda.flash_kda_workspace_size(tokens, heads, 1),
        device=x["q"].device,
        dtype=torch.uint8,
    )
    return out, final, workspace


def raw_args(x: dict[str, Any], out, final, workspace) -> tuple[Any, ...]:
    B, T, H, _ = x["q"].shape
    device = x["q"].device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        x["q"].data_ptr(),
        x["k"].data_ptr(),
        x["v"].data_ptr(),
        x["g"].data_ptr(),
        x["beta"].data_ptr(),
        out.data_ptr(),
        workspace.data_ptr(),
        x["A_log"].data_ptr(),
        x["dt_bias"].data_ptr(),
        x["initial_state"].data_ptr(),
        final.data_ptr(),
        x["cu_seqlens"].data_ptr(),
        B,
        T,
        H,
        x["cu_seqlens"].numel() - 1,
        workspace.nbytes,
        x["scale"],
        x["lower_bound"],
        True,
        True,
        True,
        x["initial_state"].dtype == torch.float32,
        device_index,
        torch.cuda.current_stream(device).cuda_stream,
    )


def raw_call(module, x, out, final, workspace):
    reason = rejection_reason(x)
    if reason is not None:
        raise RuntimeError(f"Python admission unexpectedly rejected valid input: {reason}")
    with torch.cuda.device(x["q"].device):
        module.flash_kda_fwd_hip_raw(*raw_args(x, out, final, workspace))


def assert_same(actual, reference, label: str):
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=label)


def check_one_device(module, device: torch.device, tokens: int, heads: int):
    with torch.cuda.device(device):
        x = make_inputs(tokens, heads, device)
        reference_out, reference_final = public_call(x)
        out, final, workspace = allocate(x)
        raw_call(module, x, out, final, workspace)
        torch.cuda.synchronize(device)
        assert_same(out, reference_out, f"raw output mismatch on {device}")
        assert reference_final is not None
        assert_same(final, reference_final, f"raw final state mismatch on {device}")
    return x, reference_out, reference_final


def expect_rejection(label: str, operation):
    try:
        operation()
    except (ValueError, RuntimeError) as error:
        print(f"PASS reject {label}: {type(error).__name__}: {error}")
        return
    raise AssertionError(f"raw ABI accepted invalid {label}")


def check_graph(module, x, reference_out, reference_final):
    device = x["q"].device
    out, final, workspace = allocate(x)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        raw_call(module, x, out, final, workspace)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw_call(module, x, out, final, workspace)
    graph.replay()
    torch.cuda.synchronize(device)
    assert_same(out, reference_out, "graph replay output mismatch")
    assert_same(final, reference_final, "graph replay final-state mismatch")
    print("PASS graph capture/replay")


def check_multistream(module, x, reference_out, reference_final):
    device = x["q"].device
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    buffers_a = allocate(x)
    buffers_b = allocate(x)
    stream_a.wait_stream(torch.cuda.current_stream(device))
    stream_b.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream_a):
        raw_call(module, x, *buffers_a)
    with torch.cuda.stream(stream_b):
        raw_call(module, x, *buffers_b)
    torch.cuda.current_stream(device).wait_stream(stream_a)
    torch.cuda.current_stream(device).wait_stream(stream_b)
    torch.cuda.synchronize(device)
    for name, (out, final, _) in (("stream_a", buffers_a), ("stream_b", buffers_b)):
        assert_same(out, reference_out, f"{name} output mismatch")
        assert_same(final, reference_final, f"{name} final-state mismatch")
    print("PASS concurrent two-stream calls with disjoint workspaces")


def check_invalid_metadata(module, x):
    out, final, workspace = allocate(x)
    args = list(raw_args(x, out, final, workspace))
    expect_rejection(
        "null q pointer", lambda: module.flash_kda_fwd_hip_raw(0, *args[1:])
    )
    too_small = list(args)
    too_small[16] = workspace.nbytes - 1
    expect_rejection(
        "short workspace", lambda: module.flash_kda_fwd_hip_raw(*too_small)
    )
    invalid_scale = list(args)
    invalid_scale[17] = 0.0
    expect_rejection(
        "nonpositive scale", lambda: module.flash_kda_fwd_hip_raw(*invalid_scale)
    )
    bad_dense_relation = list(args)
    bad_dense_relation[21] = False
    bad_dense_relation[12] = 2
    expect_rejection(
        "dense N != B", lambda: module.flash_kda_fwd_hip_raw(*bad_dense_relation)
    )


def check_two_devices(module, tokens: int, heads: int):
    if torch.cuda.device_count() < 2:
        print("SKIP multi-GPU: expose at least two GPUs")
        return
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    x0, _, _ = check_one_device(module, device0, tokens, heads)
    x1, _, _ = check_one_device(module, device1, tokens, heads)
    print("PASS correct launches after cuda:0 -> cuda:1 switching")

    with torch.cuda.device(device1):
        out1, final1, workspace1 = allocate(x1)
        args1 = raw_args(x1, out1, final1, workspace1)
    with torch.cuda.device(device0):
        expect_rejection(
            "active-device mismatch",
            lambda: module.flash_kda_fwd_hip_raw(*args1),
        )

    with torch.cuda.device(device0):
        out0, final0, workspace0 = allocate(x0)
        args0 = list(raw_args(x0, out0, final0, workspace0))
    with torch.cuda.device(device1):
        # Legacy/default stream handles may be the universal null handle, so use
        # a real non-default stream whose owning device is unambiguous.
        foreign_stream = torch.cuda.Stream(device=device1)
        args0[-1] = foreign_stream.cuda_stream
    with torch.cuda.device(device0):
        expect_rejection(
            "foreign-device stream",
            lambda: module.flash_kda_fwd_hip_raw(*args0),
        )
    print("PASS multi-GPU device/stream rejection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    module = get_module(flash_kda.MD_NAME)
    if not hasattr(module, "flash_kda_fwd_hip_raw"):
        raise RuntimeError("module_flash_kda_hip was not rebuilt with the raw ABI")

    x, reference_out, reference_final = check_one_device(
        module, torch.device("cuda:0"), args.tokens, args.heads
    )
    print("PASS raw vs regular ABI bitwise correctness")
    check_graph(module, x, reference_out, reference_final)
    check_multistream(module, x, reference_out, reference_final)
    check_invalid_metadata(module, x)
    check_two_devices(module, min(args.tokens, 512), args.heads)
    print("ALL RAW ABI CHECKS PASSED")


if __name__ == "__main__":
    main()
