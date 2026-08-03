# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Balanced full-call benchmark for the standalone C-input GDN backend."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.ops.opus_gdn_c_prefill import (  # noqa: E402
    OPUS_GDN_C_AUTO,
    OPUS_GDN_C_FUSED,
    OPUS_GDN_C_SPLIT,
    opus_gdn_c_prefill_fwd,
)
from op_tests.triton_tests.test_opus_gdn_c_prefill import (  # noqa: E402
    make_dense_inputs,
    require_c_runtime,
)


MODE_BY_NAME = {
    "auto": OPUS_GDN_C_AUTO,
    "cf": OPUS_GDN_C_FUSED,
    "cs": OPUS_GDN_C_SPLIT,
}


def call_mode(
    inputs: tuple[torch.Tensor, ...],
    output_final_state: bool,
    mode: int,
):
    q, k, v, g, beta, _state_kv, state_vk = inputs
    return opus_gdn_c_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state_vk,
        output_final_state=output_final_state,
        c_mode=mode,
    )


def interleaved_timing(
    inputs: tuple[torch.Tensor, ...],
    output_final_state: bool,
    names: tuple[str, ...],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, tuple[int, float, float, float]]:
    with torch.inference_mode():
        for name in names:
            for _ in range(warmup):
                call_mode(inputs, output_final_state, MODE_BY_NAME[name])
        torch.cuda.synchronize()

        rotations = tuple(
            names[shift:] + names[:shift] for shift in range(len(names))
        )
        reverse = names[::-1]
        reverse_rotations = tuple(
            reverse[shift:] + reverse[:shift] for shift in range(len(names))
        )
        orders = tuple(dict.fromkeys(rotations + reverse_rotations))
        pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for iteration in range(repeat):
            for name in orders[iteration % len(orders)]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = call_mode(
                    inputs, output_final_state, MODE_BY_NAME[name]
                )
                end.record()
                pending.append((name, start, end))
                del result
        torch.cuda.synchronize()

    samples: dict[str, list[float]] = defaultdict(list)
    for name, start, end in pending:
        samples[name].append(start.elapsed_time(end) * 1000.0)
    return {
        name: (
            len(values),
            statistics.median(values),
            min(values),
            max(values),
        )
        for name, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_BY_NAME),
        default=("auto", "cf", "cs"),
    )
    args = parser.parse_args()

    require_c_runtime()
    with_state = not args.no_state
    inputs = make_dense_inputs(
        args.batch,
        args.tokens,
        args.heads,
        with_initial_state=with_state,
        seed=20260803,
    )
    names = tuple(dict.fromkeys(args.modes))
    results = interleaved_timing(
        inputs,
        with_state,
        names,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(
        f"B={args.batch} T={args.tokens} H={args.heads} "
        f"state={with_state} complete-call GPU time"
    )
    for name in names:
        count, median_us, minimum_us, maximum_us = results[name]
        print(
            f"{name}: {median_us:.3f} us "
            f"[{minimum_us:.3f}, {maximum_us:.3f}] ({count} samples)"
        )


if __name__ == "__main__":
    main()
