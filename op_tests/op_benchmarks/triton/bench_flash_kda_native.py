# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare native HIP FlashKDA with both aiter Triton KDA implementations.

All backends receive the same Kimi-K3 TP8-style packed tensors in one process:

* ``native`` calls :func:`aiter.ops.flash_kda.flash_kda_fwd` directly;
* ``triton`` calls the PR #4683 Triton FlashKDA function directly; and
* ``baseline`` forces the original Triton chunk pipeline at BT=64.

The default mode calls each implementation directly so an eligibility change
cannot silently turn a comparison into the same kernel on both sides.  Pass
``--public-k3`` for the final production-routing gate: its ``native`` row uses
the zero-environment public K3 default, proves it is bitwise equal to explicit
native, and compares it with the same public wrapper forced to Triton.
Latencies are collected with alternating backend order and are reported
together with token throughput, speedup versus Triton FlashKDA, incremental
peak allocated memory, and output/final-state error.

Examples::

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --case single-2k --case ragged-16k --warmup 5 --repeat 30

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --seq-lens 127 511 1361 --resume --csv result.csv --json result.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make direct execution from any working directory resolve the source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from aiter.ops.flash_kda import (
    flash_kda_fwd as flash_kda_native_fwd,
    flash_kda_native_supported,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_delta_attn_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
    flash_kda_fwd as flash_kda_triton_fwd,
)
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

HEAD_DIM = 128
K3_GLOBAL_HEADS = 96
K3_TP_SIZE = 8
K3_LOCAL_HEADS = K3_GLOBAL_HEADS // K3_TP_SIZE
LOWER_BOUND = -5.0
BACKEND_ORDER = ("native", "triton", "baseline")


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple[int, ...]
    resume: bool = False


# Shapes used by the K3 integration study.  Every case is packed B=1, including
# a single sequence, because that is the metadata/layout ATOM passes to prefill.
CASES = (
    Case("single-128", (128,)),
    Case("single-256", (256,)),
    Case("single-512", (512,)),
    Case("single-1k", (1024,)),
    Case("single-2k", (2048,)),
    Case("single-8k", (8192,)),
    Case("single-16k", (16384,)),
    Case("batch-16x1k", (1024,) * 16),
    Case("batch-64x256", (256,) * 64),
    Case("ragged-16k", (127, 255, 511, 1023, 2047, 3073, 4095, 5253)),
    Case("resume-4x4k", (4096,) * 4, resume=True),
)


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _errors(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    relative_rms = (
        difference.square().mean().sqrt()
        / reference.square().mean().sqrt().clamp_min(1e-8)
    )
    return float(relative_rms.item()), float(difference.abs().max().item())


def _make_state(
    sequences: int, heads: int, *, resume: bool, device: torch.device
) -> torch.Tensor:
    state = torch.zeros(
        sequences,
        heads,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    if resume:
        state.normal_(mean=0.0, std=0.02)
        # Make V and K physically distinguishable even though both dimensions
        # happen to be 128 in K3.
        v_axis = torch.linspace(-0.04, 0.03, HEAD_DIM, device=device).view(
            1, 1, HEAD_DIM, 1
        )
        k_axis = torch.linspace(0.02, -0.01, HEAD_DIM, device=device).view(
            1, 1, 1, HEAD_DIM
        )
        state.add_(v_axis).add_(0.37 * k_axis)
    return state


def _make_inputs(
    case: Case, *, heads: int, seed: int, device: torch.device
) -> dict[str, object]:
    torch.manual_seed(seed)
    total = sum(case.seq_lens)
    shape = (1, total, heads, HEAD_DIM)

    def projection() -> torch.Tensor:
        return F.silu(torch.randn(shape, device=device, dtype=torch.float32)).to(
            torch.bfloat16
        )

    offsets = [0]
    for length in case.seq_lens:
        offsets.append(offsets[-1] + length)

    inputs: dict[str, object] = {
        "q": projection(),
        "k": projection(),
        "v": projection(),
        "g": torch.randn(shape, device=device, dtype=torch.bfloat16),
        # This mirrors ATOM: the beta projection is BF16 and is widened before
        # the fused in-kernel sigmoid.
        "beta": torch.randn(
            (1, total, heads), device=device, dtype=torch.bfloat16
        ).float(),
        "A_log": torch.empty(heads, device=device, dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        "dt_bias": torch.randn(
            heads * HEAD_DIM, device=device, dtype=torch.float32
        ),
        "scale": HEAD_DIM**-0.5,
        "initial_state": _make_state(
            len(case.seq_lens), heads, resume=case.resume, device=device
        ),
        "cu_seqlens": torch.tensor(offsets, device=device, dtype=torch.int32),
    }
    assert inputs["dt_bias"].ndim == 1
    assert inputs["cu_seqlens"].dtype == torch.int32
    return inputs


def _native_supported(inputs: dict[str, object]) -> bool:
    return flash_kda_native_supported(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=inputs["cu_seqlens"],
    )


def _build_backends(
    inputs: dict[str, object], selected: tuple[str, ...], *, public_k3: bool = False
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]]:
    def public(backend: str | None):
        return chunk_kimi_delta_attn(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            scale=inputs["scale"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            state_v_first=True,
            cu_seqlens=inputs["cu_seqlens"],
            backend=backend,
        )

    def native():
        if public_k3:
            # ``None`` deliberately exercises the production zero-environment
            # resolver instead of forcing the backend under test.
            return public(None)
        return flash_kda_native_fwd(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            scale=inputs["scale"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            lower_bound=LOWER_BOUND,
            cu_seqlens=inputs["cu_seqlens"],
        )

    def triton_flash():
        if public_k3:
            return public("triton")
        return flash_kda_triton_fwd(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            scale=inputs["scale"],
            lower_bound=LOWER_BOUND,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=inputs["cu_seqlens"],
        )

    def baseline():
        if public_k3:
            return public("baseline")
        output, final_state, *_ = chunk_delta_attn_fwd(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            scale=inputs["scale"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            cu_seqlens=inputs["cu_seqlens"],
            chunk_size=64,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            # This is the important isolation switch: BT64 must never enter
            # either FlashKDA path even if process-wide defaults change.
            allow_flash_kda=False,
        )
        return output, final_state

    implementations = {
        "native": native,
        "triton": triton_flash,
        "baseline": baseline,
    }
    return {name: implementations[name] for name in selected}


def _time_once(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    milliseconds = float(start.elapsed_time(end))
    del result
    return milliseconds


def _peak_memory_mib(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    del result
    torch.cuda.synchronize()
    return peak / (1024**2)


@torch.inference_mode()
def _benchmark_case(
    case: Case,
    *,
    heads: int,
    selected: tuple[str, ...],
    warmup: int,
    repeat: int,
    seed: int,
    check: bool,
    tolerance: float,
    public_k3: bool = False,
    raw_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    inputs = _make_inputs(case, heads=heads, seed=seed, device=torch.device("cuda"))
    if "native" in selected and not _native_supported(inputs):
        raise RuntimeError(
            "native FlashKDA rejected the exact K3 inputs; check gfx942/gfx950, "
            "AITER_TRITON_ONLY, dtypes, and the extension build"
        )
    backends = _build_backends(inputs, selected, public_k3=public_k3)

    correctness: dict[str, dict[str, float | str]] = {}
    if check:
        outputs = {name: fn() for name, fn in backends.items()}
        if public_k3 and "native" in outputs:
            # Prove that the zero-env default actually selected native rather
            # than merely producing a numerically close Triton result.
            explicit_native = chunk_kimi_delta_attn(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                g=inputs["g"],
                beta=inputs["beta"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                scale=inputs["scale"],
                initial_state=inputs["initial_state"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=LOWER_BOUND,
                state_v_first=True,
                cu_seqlens=inputs["cu_seqlens"],
                backend="native",
            )
            default_o, default_state = outputs["native"]
            explicit_o, explicit_state = explicit_native
            if not (
                torch.equal(default_o, explicit_o)
                and default_state is not None
                and explicit_state is not None
                and torch.equal(default_state, explicit_state)
            ):
                raise RuntimeError(
                    f"{case.name}: zero-env K3 default is not bitwise native"
                )
            del explicit_native, explicit_o, explicit_state
        torch.cuda.synchronize()
        reference_name = "triton" if "triton" in outputs else selected[0]
        reference_o, reference_state = outputs[reference_name]
        assert reference_state is not None
        for name, (output, final_state) in outputs.items():
            assert final_state is not None
            output_rms, output_max = _errors(output, reference_o)
            state_rms, state_max = _errors(final_state, reference_state)
            correctness[name] = {
                "error_reference": reference_name,
                "output_relative_rms": output_rms,
                "output_max_abs": output_max,
                "state_relative_rms": state_rms,
                "state_max_abs": state_max,
            }
            if name != reference_name and max(output_rms, state_rms) > tolerance:
                raise RuntimeError(
                    f"{case.name}/{name} exceeds relative-RMS tolerance "
                    f"{tolerance}: output={output_rms:.6g}, state={state_rms:.6g}"
                )
        del outputs, reference_o, reference_state, output, final_state

    peak_memory = {name: _peak_memory_mib(fn) for name, fn in backends.items()}

    # Rotate the first backend in every round.  This prevents one implementation
    # from always measuring at the same clock/temperature point.
    names = list(backends)
    for index in range(warmup):
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            result = backends[name]()
            del result
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(repeat):
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        for order_index, name in enumerate(order):
            elapsed = _time_once(backends[name])
            samples[name].append(elapsed)
            if raw_rows is not None:
                raw_rows.append(
                    {
                        "case": case.name,
                        "backend": name,
                        "round": index,
                        "order": order_index,
                        "sequences": len(case.seq_lens),
                        "tokens": sum(case.seq_lens),
                        "state": "resume" if case.resume else "fresh",
                        "latency_ms": elapsed,
                    }
                )

    medians = {name: statistics.median(values) for name, values in samples.items()}
    triton_median = medians.get("triton")
    total_tokens = sum(case.seq_lens)
    rows: list[dict[str, object]] = []
    for name in names:
        values = samples[name]
        median = medians[name]
        row: dict[str, object] = {
            "case": case.name,
            "backend": name,
            "sequences": len(case.seq_lens),
            "tokens": total_tokens,
            "heads": heads,
            "state": "resume" if case.resume else "fresh",
            "latency_min_ms": min(values),
            "latency_p10_ms": _percentile(values, 0.10),
            "latency_median_ms": median,
            "latency_p90_ms": _percentile(values, 0.90),
            "latency_mean_ms": statistics.mean(values),
            "latency_std_ms": statistics.pstdev(values),
            "tokens_per_second": total_tokens * 1000.0 / median,
            "speedup_vs_triton": (
                triton_median / median if triton_median is not None else None
            ),
            "peak_memory_mib": peak_memory[name],
            "samples": len(values),
        }
        row.update(correctness.get(name, {}))
        rows.append(row)

    del backends, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _print_environment(heads: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(properties, "gcnArchName", "unknown")
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "arch": arch,
        "compute_units": properties.multi_processor_count,
        "pytorch": torch.__version__,
        "rocm": torch.version.hip,
        "heads": heads,
        "head_dim": HEAD_DIM,
        "tp_size": K3_TP_SIZE,
    }
    print(
        f"GPU: {metadata['gpu']} ({arch}, {metadata['compute_units']} CUs); "
        f"PyTorch {torch.__version__}; ROCm {torch.version.hip}"
    )
    print(
        f"K3 per rank: TP={K3_TP_SIZE}, H={heads}, K=V={HEAD_DIM}; "
        "packed cu_seqlens=int32, beta/state=fp32"
    )
    return metadata


def _print_rows(rows: list[dict[str, object]]) -> None:
    print(
        "\n| case | backend | N | tokens | state | p10 / p50 / p90 ms | "
        "Mtoken/s | vs Triton | peak MiB | out RMS | state RMS |"
    )
    print(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        speedup = row["speedup_vs_triton"]
        speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
        output_rms = row.get("output_relative_rms")
        state_rms = row.get("state_relative_rms")
        output_text = "-" if output_rms is None else f"{output_rms:.3e}"
        state_text = "-" if state_rms is None else f"{state_rms:.3e}"
        print(
            f"| {row['case']} | {row['backend']} | {row['sequences']} | "
            f"{row['tokens']} | {row['state']} | {row['latency_p10_ms']:.4f} / "
            f"{row['latency_median_ms']:.4f} / {row['latency_p90_ms']:.4f} | "
            f"{row['tokens_per_second'] / 1e6:.3f} | {speedup_text} | "
            f"{row['peak_memory_mib']:.1f} | {output_text} | {state_text} |"
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {path}")


def _write_json(
    path: Path,
    *,
    metadata: dict[str, object],
    args: argparse.Namespace,
    cases: list[Case],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": metadata,
        "configuration": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "backends": list(args.backend),
            "public_k3": args.public_k3,
        },
        "cases": [
            {"name": case.name, "seq_lens": list(case.seq_lens), "resume": case.resume}
            for case in cases
        ],
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote JSON: {path}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Benchmark only this named case; repeat the option for multiple cases.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        metavar="T",
        help="Use one custom packed case instead of the named case sweep.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Use a nonzero asymmetric V-first state for --seq-lens.",
    )
    parser.add_argument("--heads", type=int, default=K3_LOCAL_HEADS)
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKEND_ORDER,
        help="Backend to run; repeat as needed. Defaults to all three.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Do not compare output and final state before timing.",
    )
    parser.add_argument(
        "--public-k3",
        action="store_true",
        help=(
            "Time the public chunk_kimi_delta_attn resolver: the 'native' row "
            "uses its zero-env default and the 'triton' row forces Triton."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.04,
        help="Maximum output/state relative RMS versus Triton FlashKDA.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="?",
        const=Path("bench_flash_kda_native.csv"),
        help="Write flat results to this path.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        help="Write one latency row per alternating backend/round.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        nargs="?",
        const=Path("bench_flash_kda_native.json"),
        help="Write metadata and results to this path.",
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="Print named cases and exit."
    )
    args = parser.parse_args(argv)
    if args.backend is None:
        args.backend = list(BACKEND_ORDER)
    else:
        args.backend = list(dict.fromkeys(args.backend))
    if args.heads <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error(
            "--heads and --repeat must be positive; --warmup must be nonnegative"
        )
    if args.seq_lens is not None and any(length <= 0 for length in args.seq_lens):
        parser.error("all --seq-lens values must be positive")
    if args.case and args.seq_lens:
        parser.error("--case and --seq-lens are mutually exclusive")
    if args.resume and args.seq_lens is None:
        parser.error("--resume only applies to a custom --seq-lens case")
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.list_cases:
        for case in CASES:
            print(
                f"{case.name:16s} N={len(case.seq_lens):2d} "
                f"tokens={sum(case.seq_lens):6d} "
                f"state={'resume' if case.resume else 'fresh'}"
            )
        return
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This benchmark requires a ROCm GPU.")
    if os.getenv("AITER_TRITON_ONLY", "0") == "1" and "native" in args.backend:
        raise SystemExit("Unset AITER_TRITON_ONLY to benchmark native FlashKDA.")
    if args.public_k3 and os.getenv("AITER_KDA_BACKEND"):
        raise SystemExit("Unset AITER_KDA_BACKEND for a zero-env public K3 benchmark.")

    if args.seq_lens is not None:
        cases = [Case("custom", tuple(args.seq_lens), resume=args.resume)]
    else:
        selected_names = set(args.case or [case.name for case in CASES])
        cases = [case for case in CASES if case.name in selected_names]

    metadata = _print_environment(args.heads)
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    selected_backends = tuple(args.backend)
    for index, case in enumerate(cases):
        print(
            f"Running {case.name}: N={len(case.seq_lens)}, "
            f"tokens={sum(case.seq_lens)}, state="
            f"{'resume' if case.resume else 'fresh'}",
            flush=True,
        )
        rows.extend(
            _benchmark_case(
                case,
                heads=args.heads,
                selected=selected_backends,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed + index,
                check=not args.skip_correctness,
                tolerance=args.tolerance,
                public_k3=args.public_k3,
                raw_rows=raw_rows,
            )
        )

    _print_rows(rows)
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.raw_csv is not None:
        _write_csv(args.raw_csv, raw_rows)
    if args.json is not None:
        _write_json(
            args.json, metadata=metadata, args=args, cases=cases, rows=rows
        )


if __name__ == "__main__":
    main()
