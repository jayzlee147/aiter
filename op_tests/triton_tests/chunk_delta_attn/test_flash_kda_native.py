# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 contract tests for the native gfx942/gfx950 FlashKDA path.

The native implementation has a deliberately narrower surface than the public
KDA wrapper.  These tests keep the production contract explicit: packed B=1
BF16 activations, FP32 beta/gate parameters, one-dimensional ``dt_bias``, GPU
int32 sequence metadata, and an FP32 V-first recurrent state.  The numerical
reference is the PR #4683 Triton FlashKDA implementation, called directly so
the public ``auto`` selector cannot accidentally compare one backend with
itself.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.flash_kda import (
    flash_kda_fwd as flash_kda_native_fwd,
    flash_kda_native_supported,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
    flash_kda_fwd as flash_kda_triton_fwd,
)
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

_NATIVE_MODULE = importlib.import_module("aiter.ops.flash_kda")
_PUBLIC_MODULE = importlib.import_module(
    "aiter.ops.triton.kimi_delta_attn.chunk_delta_attn"
)

HEAD_DIM = 128
LOWER_BOUND = -5.0
_ROCM_GPU_AVAILABLE = torch.cuda.is_available() and torch.version.hip is not None
requires_rocm_gpu = pytest.mark.skipif(
    not _ROCM_GPU_AVAILABLE, reason="native FlashKDA tests require a ROCm GPU"
)


def _make_v_first_state(
    n: int, h: int, *, device: str | torch.device, nonzero: bool
) -> torch.Tensor:
    """Make an asymmetric ``[N,H,V,K]`` state.

    K and V are both 128, so shape checks alone cannot catch a transposed state.
    Different V- and K-axis ramps make that layout error numerically obvious.
    """

    state = torch.zeros(
        n, h, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
    )
    if nonzero:
        state.normal_(mean=0.0, std=0.02)
        v_axis = torch.linspace(-0.04, 0.03, HEAD_DIM, device=device).view(
            1, 1, HEAD_DIM, 1
        )
        k_axis = torch.linspace(0.02, -0.01, HEAD_DIM, device=device).view(
            1, 1, 1, HEAD_DIM
        )
        state.add_(v_axis).add_(0.37 * k_axis)
        assert not torch.equal(state, state.transpose(-1, -2))
    return state


def _make_k3_inputs(
    seq_lens: tuple[int, ...],
    *,
    heads: int = 2,
    resume: bool = False,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> dict[str, object]:
    torch.manual_seed(seed)
    total = sum(seq_lens)
    shape = (1, total, heads, HEAD_DIM)

    # K3 obtains these tensors from BF16 projections.  Producing q/k/v through
    # SiLU also gives a less synthetic distribution than four plain Gaussians.
    def projection() -> torch.Tensor:
        return F.silu(torch.randn(shape, device=device, dtype=torch.float32)).to(
            torch.bfloat16
        )

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)

    return {
        "q": projection(),
        "k": projection(),
        "v": projection(),
        "g": torch.randn(shape, device=device, dtype=torch.float32).to(
            torch.bfloat16
        ),
        # ATOM widens the BF16 beta projection before invoking KDA.
        "beta": torch.randn(
            (1, total, heads), device=device, dtype=torch.float32
        )
        .to(torch.bfloat16)
        .float(),
        "A_log": torch.empty(heads, device=device, dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        # Keep this one-dimensional: that is the real K3 parameter layout.
        "dt_bias": torch.randn(
            heads * HEAD_DIM, device=device, dtype=torch.float32
        ),
        "scale": 1.0 / math.sqrt(HEAD_DIM),
        "initial_state": _make_v_first_state(
            len(seq_lens), heads, device=device, nonzero=resume
        ),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "state_v_first": True,
        # ATOM constructs and retains this metadata on the GPU as int32.
        "cu_seqlens": torch.tensor(offsets, device=device, dtype=torch.int32),
    }


def _native_kwargs(inputs: dict[str, object]) -> dict[str, object]:
    keys = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "A_log",
        "dt_bias",
        "scale",
        "initial_state",
        "output_final_state",
        "lower_bound",
        "cu_seqlens",
    )
    return {key: inputs[key] for key in keys}


def _native_support_kwargs(inputs: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in inputs.items()
        if key
        in {
            "q",
            "k",
            "v",
            "g",
            "beta",
            "A_log",
            "dt_bias",
            "initial_state",
            "output_final_state",
            "use_qk_l2norm_in_kernel",
            "use_gate_in_kernel",
            "use_beta_sigmoid_in_kernel",
            "safe_gate",
            "lower_bound",
            "state_v_first",
            "cu_seqlens",
        }
    }


def _triton_call(inputs: dict[str, object]):
    return flash_kda_triton_fwd(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        lower_bound=inputs["lower_bound"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=inputs["cu_seqlens"],
    )


def _relative_rms(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual = actual.detach().float()
    reference = reference.detach().float()
    rms = (actual - reference).square().mean().sqrt()
    denominator = reference.square().mean().sqrt().clamp_min(1e-8)
    return float((rms / denominator).item())


def _require_native(inputs: dict[str, object]) -> None:
    if not flash_kda_native_supported(**_native_support_kwargs(inputs)):
        reason = _NATIVE_MODULE._native_rejection_reason(  # noqa: SLF001
            **_native_support_kwargs(inputs)
        )
        pytest.skip(f"native FlashKDA is unavailable: {reason}")


@pytest.mark.parametrize(
    ("seq_lens", "resume"),
    [
        pytest.param((96,), False, id="fresh-tail"),
        pytest.param((129,), True, id="resume-tail"),
        pytest.param((17, 64, 95), True, id="resume-ragged-tail"),
    ],
)
@requires_rocm_gpu
def test_native_matches_direct_triton_k3_contract(seq_lens, resume):
    """Cover the fresh/resume, packed-ragged, and partial-tile K3 paths."""

    inputs = _make_k3_inputs(seq_lens, resume=resume)
    _require_native(inputs)
    initial_copy = inputs["initial_state"].clone()

    native_o, native_ht = flash_kda_native_fwd(**_native_kwargs(inputs))
    triton_o, triton_ht = _triton_call(inputs)
    torch.cuda.synchronize()

    assert all(
        inputs[name].dtype == torch.bfloat16 for name in ("q", "k", "v", "g")
    )
    assert inputs["beta"].dtype == torch.float32
    assert inputs["A_log"].dtype == torch.float32
    assert inputs["dt_bias"].dtype == torch.float32
    assert inputs["dt_bias"].ndim == 1
    assert inputs["cu_seqlens"].is_cuda
    assert inputs["cu_seqlens"].dtype == torch.int32
    assert torch.equal(inputs["initial_state"], initial_copy), "state input was mutated"

    assert native_o.shape == inputs["v"].shape
    assert native_o.dtype == torch.bfloat16
    assert native_ht is not None and triton_ht is not None
    assert native_ht.shape == inputs["initial_state"].shape
    assert native_ht.dtype == torch.float32
    assert torch.isfinite(native_o).all()
    assert torch.isfinite(native_ht).all()

    assert _relative_rms(native_o, triton_o) < 2.5e-2
    assert _relative_rms(native_ht, triton_ht) < 2.5e-2


@requires_rocm_gpu
def test_native_fresh_none_matches_explicit_zero_state():
    """Cover the standalone API's no-state fresh path as well as ATOM's zero cache."""

    zero_inputs = _make_k3_inputs((33,), heads=1, resume=False, seed=11)
    none_inputs = {**zero_inputs, "initial_state": None}
    _require_native(zero_inputs)
    _require_native(none_inputs)

    zero_o, zero_ht = flash_kda_native_fwd(**_native_kwargs(zero_inputs))
    none_o, none_ht = flash_kda_native_fwd(**_native_kwargs(none_inputs))
    triton_o, triton_ht = _triton_call(none_inputs)
    torch.cuda.synchronize()

    assert zero_ht is not None and none_ht is not None and triton_ht is not None
    assert none_ht.dtype == torch.float32
    assert _relative_rms(none_o, zero_o) < 5e-3
    assert _relative_rms(none_ht, zero_ht) < 5e-3
    assert _relative_rms(none_o, triton_o) < 2.5e-2
    assert _relative_rms(none_ht, triton_ht) < 2.5e-2


@requires_rocm_gpu
def test_native_nonzero_v_first_state_matches_fp32_recurrence():
    """A small absolute reference catches K/V transposition despite K == V."""

    inputs = _make_k3_inputs((9, 17), heads=1, resume=True, seed=7)
    _require_native(inputs)
    native_o, native_ht = flash_kda_native_fwd(**_native_kwargs(inputs))
    gold_o, gold_ht = chunk_kda_ref(**inputs)
    torch.cuda.synchronize()

    assert native_ht is not None and gold_ht is not None
    assert _relative_rms(native_o, gold_o) < 4e-2
    assert _relative_rms(native_ht, gold_ht) < 4e-2


@requires_rocm_gpu
def test_public_native_backend_reaches_real_kernel():
    """Verify the production wrapper, allocation adapter, and native JIT together."""

    inputs = _make_k3_inputs((33, 71), heads=2, resume=True, seed=11)
    _require_native(inputs)
    direct_o, direct_ht = flash_kda_native_fwd(**_native_kwargs(inputs))
    public_o, public_ht = _PUBLIC_MODULE.chunk_kimi_delta_attn(
        **inputs, backend="native"
    )
    torch.cuda.synchronize()

    assert direct_ht is not None and public_ht is not None
    assert torch.equal(public_o, direct_o)
    assert torch.equal(public_ht, direct_ht)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_native_support_validator_accepts_exact_k3_metadata(monkeypatch, arch):
    """Exercise eligibility without loading or compiling the HIP extension."""

    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(_NATIVE_MODULE, "_device_arch", lambda _device: arch)
    inputs = _make_k3_inputs((3, 5), heads=1, resume=True, device="cpu")
    kwargs = _native_support_kwargs(inputs)

    assert inputs["dt_bias"].ndim == 1
    assert inputs["cu_seqlens"].dtype == torch.int32
    assert flash_kda_native_supported(**kwargs)

    assert not flash_kda_native_supported(**{**kwargs, "q": inputs["q"].float()})
    assert not flash_kda_native_supported(**{**kwargs, "chunk_size": 64})
    assert not flash_kda_native_supported(**{**kwargs, "state_v_first": False})
    assert not flash_kda_native_supported(
        **{**kwargs, "dt_bias": inputs["dt_bias"][:-1]}
    )

    monkeypatch.setattr(_NATIVE_MODULE, "_device_arch", lambda _device: "gfx90a")
    assert not flash_kda_native_supported(**kwargs)


def test_public_backend_selector_and_unsupported_fallback(monkeypatch):
    """The selector is testable on CPU and never reaches a compiled kernel."""

    monkeypatch.delenv("AITER_KDA_BACKEND", raising=False)
    inputs = _make_k3_inputs((3, 5), heads=1, resume=True, device="cpu")
    calls: list[tuple[str, dict[str, object]]] = []
    support = {"value": True}

    def fake_supported(**_kwargs):
        return support["value"]

    def fake_native(**kwargs):
        calls.append(("native", kwargs))
        return torch.full_like(kwargs["v"], 1), kwargs["initial_state"].clone()

    def fake_triton(**kwargs):
        calls.append(("triton", kwargs))
        return torch.full_like(kwargs["v"], 2), kwargs["initial_state"].clone()

    monkeypatch.setattr(_PUBLIC_MODULE, "flash_kda_native_supported", fake_supported)
    monkeypatch.setattr(_PUBLIC_MODULE, "flash_kda_native_fwd", fake_native)
    monkeypatch.setattr(_PUBLIC_MODULE, "chunk_delta_attn_fwd", fake_triton)
    monkeypatch.setattr(
        _PUBLIC_MODULE,
        "_native_rejection_reason",
        lambda **_kwargs: "synthetic unsupported input",
    )

    # An unvalidated architecture preserves the Triton default without even
    # querying native eligibility.
    monkeypatch.setattr(
        _PUBLIC_MODULE,
        "flash_kda_native_supported",
        lambda **_kwargs: pytest.fail("default Triton path queried native support"),
    )
    output, _ = _PUBLIC_MODULE.chunk_kimi_delta_attn(**inputs)
    assert torch.equal(output, torch.full_like(inputs["v"], 2))
    assert calls[0][0] == "triton"
    assert calls[0][1]["allow_flash_kda"] is True
    calls.clear()

    # gfx950 is compile-validated but has no real-hardware promotion data yet.
    monkeypatch.setattr(_PUBLIC_MODULE, "_device_arch", lambda _device: "gfx950")
    output, _ = _PUBLIC_MODULE.chunk_kimi_delta_attn(**inputs)
    assert torch.equal(output, torch.full_like(inputs["v"], 2))
    assert calls[0][0] == "triton"
    calls.clear()

    monkeypatch.setattr(_PUBLIC_MODULE, "flash_kda_native_supported", fake_supported)

    # gfx942 completed the production promotion matrix, so the same zero-env
    # call selects the already-validated private native fast path.
    monkeypatch.setattr(_PUBLIC_MODULE, "_device_arch", lambda _device: "gfx942")
    output, final_state = _PUBLIC_MODULE.chunk_kimi_delta_attn(**inputs)
    assert [name for name, _ in calls] == ["native"]
    assert torch.equal(output, torch.full_like(inputs["v"], 1))
    assert final_state is not None
    calls.clear()

    for backend, expected, marker, allow_flash in (
        ("auto", "native", 1, None),
        ("native", "native", 1, None),
        ("triton", "triton", 2, True),
        ("baseline", "triton", 2, False),
    ):
        calls.clear()
        output, final_state = _PUBLIC_MODULE.chunk_kimi_delta_attn(
            **inputs, backend=backend
        )
        assert [name for name, _ in calls] == [expected]
        assert torch.equal(output, torch.full_like(inputs["v"], marker))
        assert final_state is not None
        if expected == "native":
            native_call = calls[0][1]
            assert native_call["dt_bias"].ndim == 1
            assert native_call["cu_seqlens"].dtype == torch.int32
            assert native_call["initial_state"].dtype == torch.float32
        else:
            assert calls[0][1]["allow_flash_kda"] is allow_flash

    # ``auto`` must fall back, while an explicit native request must fail loud.
    support["value"] = False
    calls.clear()
    _PUBLIC_MODULE.chunk_kimi_delta_attn(**inputs, backend="auto")
    assert [name for name, _ in calls] == ["triton"]
    assert calls[0][1]["allow_flash_kda"] is True

    calls.clear()
    with pytest.raises(ValueError, match="synthetic unsupported input"):
        _PUBLIC_MODULE.chunk_kimi_delta_attn(**inputs, backend="native")
    assert not calls
