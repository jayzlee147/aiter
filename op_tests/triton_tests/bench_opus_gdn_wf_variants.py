# SPDX-License-Identifier: MIT
"""Correctness and interleaved timing for dense gfx942 W/U-fused variants."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.profiler as tprof

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.ops.opus_gdn_wu_prefill import opus_gdn_wu_prefill_fwd


def require_opus_runtime() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("Opus GDN W/U benchmarks require a ROCm GPU")
    gfx = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":", 1)[0]
    if gfx not in ("gfx942", "gfx950"):
        raise RuntimeError(f"unsupported GPU architecture: {gfx}")
    return gfx


def make_dense_inputs(
    B: int,
    T: int,
    H: int,
    *,
    with_initial_state: bool,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    D = 128
    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    g = F.logsigmoid(
        torch.randn(B, T, H, dtype=torch.float32, device="cuda")
    )
    beta = torch.randn(B, T, H, dtype=torch.float32, device="cuda").sigmoid()
    h0_vk = (
        torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.1
        if with_initial_state
        else None
    )
    return q, k, v, g, beta, None, h0_vk


def call_full_path_variant(
    inputs: tuple[torch.Tensor, ...],
    output_final_state: bool,
):
    q, k, v, g, beta, _h0_kv, h0_vk = inputs
    return opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0_vk,
        output_final_state=output_final_state,
        BT=64,
        BV=64,
        num_warps=8,
        k1_algo=1,
        k2_mode=1,
    )


def call_variant(inputs: tuple[torch.Tensor, ...], state: bool, variant: int):
    os.environ["OPUS_GDN_WF_VARIANT"] = str(variant)
    return call_full_path_variant(inputs, state)


def balanced_orders(variants: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    rotations = tuple(
        variants[shift:] + variants[:shift] for shift in range(len(variants))
    )
    reverse = variants[::-1]
    reverse_rotations = tuple(
        reverse[shift:] + reverse[:shift] for shift in range(len(variants))
    )
    return tuple(dict.fromkeys(rotations + reverse_rotations))


def interleaved_timing(
    inputs: tuple[torch.Tensor, ...],
    state: bool,
    variants: tuple[int, ...],
    *,
    warmup: int,
    repeat: int,
) -> tuple[
    dict[int, tuple[int, float, float, float]],
    dict[int, tuple[int, float, float, float]],
]:
    """Return balanced complete-call and K2-only GPU timings in microseconds."""
    orders = balanced_orders(variants)
    full_pending: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
    launch_order: list[int] = []
    with torch.inference_mode():
        for variant in variants:
            for _ in range(warmup):
                call_variant(inputs, state, variant)
        torch.cuda.synchronize()

        for iteration in range(repeat):
            for variant in orders[iteration % len(orders)]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = call_variant(inputs, state, variant)
                end.record()
                full_pending.append((variant, start, end))
                del output
        torch.cuda.synchronize()

        with tprof.profile(
            activities=[tprof.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for iteration in range(repeat):
                for variant in orders[iteration % len(orders)]:
                    call_variant(inputs, state, variant)
                    launch_order.append(variant)
            torch.cuda.synchronize()

    full_samples: dict[int, list[float]] = defaultdict(list)
    for variant, start, end in full_pending:
        full_samples[variant].append(start.elapsed_time(end) * 1000.0)

    k2_events = [
        event
        for event in prof.events()
        if event.device_type.name == "CUDA"
        and "gdn_k2_kernel" in event.name
    ]
    if len(k2_events) != len(launch_order):
        names = sorted({event.name for event in k2_events})
        raise RuntimeError(
            f"expected {len(launch_order)} fused K2 events, got "
            f"{len(k2_events)}: {names}"
        )
    k2_samples: dict[int, list[float]] = defaultdict(list)
    for variant, event in zip(launch_order, k2_events, strict=True):
        k2_samples[variant].append(event.device_time_total)

    def summarize(samples: dict[int, list[float]]):
        return {
            variant: (
                len(values),
                statistics.median(values),
                min(values),
                max(values),
            )
            for variant, values in samples.items()
        }

    return summarize(full_samples), summarize(k2_samples)


def interleaved_full_timing(
    inputs: tuple[torch.Tensor, ...],
    state: bool,
    variants: tuple[int, ...],
    *,
    warmup: int,
    repeat: int,
) -> dict[int, tuple[int, float, float, float]]:
    """Low-overhead balanced complete-call timing for shape sweeps."""
    orders = balanced_orders(variants)
    pending: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        for variant in variants:
            for _ in range(warmup):
                call_variant(inputs, state, variant)
        torch.cuda.synchronize()
        for iteration in range(repeat):
            for variant in orders[iteration % len(orders)]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = call_variant(inputs, state, variant)
                end.record()
                pending.append((variant, start, end))
                del output
        torch.cuda.synchronize()
    samples: dict[int, list[float]] = defaultdict(list)
    for variant, start, end in pending:
        samples[variant].append(start.elapsed_time(end) * 1000.0)
    return {
        variant: (
            len(values),
            statistics.median(values),
            min(values),
            max(values),
        )
        for variant, values in samples.items()
    }


def tensor_diff(reference: torch.Tensor | None, candidate: torch.Tensor | None):
    if reference is None or candidate is None:
        return reference is candidate, 0, 0.0
    exact = torch.equal(reference, candidate)
    mismatch = int(torch.count_nonzero(reference != candidate).item())
    max_abs = float(
        (reference.float() - candidate.float()).abs().max().item()
    )
    return exact, mismatch, max_abs


def correctness_case(B: int, T: int, H: int, state: bool) -> None:
    inputs = make_dense_inputs(
        B,
        T,
        H,
        with_initial_state=state,
        seed=20260731 + B * 1_000_003 + T * 1009 + H * 97 + int(state),
    )
    outputs = {}
    with torch.inference_mode():
        for variant in range(7):
            outputs[variant] = call_variant(inputs, state, variant)
        torch.cuda.synchronize()
    ref_o, ref_ht = outputs[0]
    for variant in range(1, 7):
        out_exact, out_mismatch, out_max = tensor_diff(ref_o, outputs[variant][0])
        ht_exact, ht_mismatch, ht_max = tensor_diff(ref_ht, outputs[variant][1])
        print(
            f"correct B={B} T={T} H={H} state={state} v={variant} "
            f"o_exact={out_exact} o_mismatch={out_mismatch} o_max={out_max:.9g} "
            f"ht_exact={ht_exact} ht_mismatch={ht_mismatch} ht_max={ht_max:.9g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--variants", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=24)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--sweep-tokens", type=int, nargs="+")
    parser.add_argument("--sweep-heads", type=int, nargs="+")
    args = parser.parse_args()

    require_opus_runtime()
    old_variant = os.environ.get("OPUS_GDN_WF_VARIANT")
    try:
        if args.correctness:
            for state in (False, True):
                for B, T, H in (
                    (1, 64, 1),
                    (1, 65, 2),
                    (1, 128, 8),
                    (1, 4096, 64),
                ):
                    correctness_case(B, T, H, state)
            return

        if args.sweep_tokens or args.sweep_heads:
            tokens = args.sweep_tokens or [args.tokens]
            heads = args.sweep_heads or [args.heads]
            state = not args.no_state
            variants = tuple(dict.fromkeys(args.variants))
            for T in tokens:
                for H in heads:
                    inputs = make_dense_inputs(
                        args.batch,
                        T,
                        H,
                        with_initial_state=state,
                        seed=20260731 + T * 1009 + H * 97 + int(state),
                    )
                    result = interleaved_full_timing(
                        inputs,
                        state,
                        variants,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    medians = " ".join(
                        f"v{variant}={result[variant][1]:.3f}us"
                        for variant in variants
                    )
                    best = min(variants, key=lambda variant: result[variant][1])
                    print(
                        f"sweep B={args.batch} T={T} H={H} state={state} "
                        f"best=v{best} {medians}",
                        flush=True,
                    )
            return

        state = not args.no_state
        inputs = make_dense_inputs(
            args.batch,
            args.tokens,
            args.heads,
            with_initial_state=state,
            seed=20260731,
        )
        variants = tuple(dict.fromkeys(args.variants))
        full, k2 = interleaved_timing(
            inputs,
            state,
            variants,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(
            f"B={args.batch} T={args.tokens} H={args.heads} state={state}",
            flush=True,
        )
        for variant in variants:
            nf, mf, lof, hif = full[variant]
            nk, mk, lok, hik = k2[variant]
            print(
                f"variant={variant} full={mf:.3f}us [{lof:.3f},{hif:.3f}] n={nf} "
                f"k2={mk:.3f}us [{lok:.3f},{hik:.3f}] n={nk}",
                flush=True,
            )
    finally:
        if old_variant is None:
            os.environ.pop("OPUS_GDN_WF_VARIANT", None)
        else:
            os.environ["OPUS_GDN_WF_VARIANT"] = old_variant


if __name__ == "__main__":
    main()
