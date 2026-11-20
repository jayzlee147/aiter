# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measure whole-mixer HIP graph latency for H64/K64/V128 on gfx950.

Each graph contains one public call per independent input/state buffer. Each
sample times several graph replays using GPU events and divides by the number
of mixer calls. The reported latency is the median of these samples. Metadata,
JIT compilation, capture and model projections are outside the timed region.
Decode updates its FP32 checkpoint and convolution buffers in place.
"""

import argparse
from itertools import accumulate
import json
from pathlib import Path
from statistics import median

import torch

H, K, V = 64, 64, 128
D = H * (2 * K + V)


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype, device="cuda").mul_(0.3)


def common(total, weights):
    return dict(
        **weights,
        mixed_qkv=rand(total, D),
        gate=rand(1, total, H, K),
        beta=rand(1, total, H),
        out_gate=rand(1, total, H, V),
    )


def prefill_inputs(api, weights, length, resumed):
    args = common(length, weights)
    ci, co = api.prepare_metadata([length], device="cuda")
    args.update(
        cu_seqlens=torch.tensor([0, length], device="cuda"),
        seq_lens=[length],
        chunk_indices=ci,
        chunk_offsets=co,
        conv_state=rand(1, D, 3),
        has_initial_state=torch.tensor([resumed], device="cuda"),
        initial_state=rand(1, H, V, K, dtype=torch.float32) if resumed else None,
        output_final_state=True,
    )
    return args


def decode_inputs(weights, batch, spec, lengths):
    steps = spec + 1
    total = sum(lengths)
    args = common(total, weights)
    seq = torch.arange(batch, device="cuda", dtype=torch.int32)
    slots = torch.arange(1, batch * steps + 1, device="cuda", dtype=torch.int32)
    args.update(
        out_gate=args["out_gate"].view(total, H * V),
        cu_seqlens=torch.tensor(list(accumulate(lengths, initial=0)), device="cuda"),
        conv_state=rand(batch + 1, D, steps + 2),
        ssm_state=rand(batch * steps + 1, H, V, K, dtype=torch.float32),
        ssm_state_indices=slots.view(batch, steps) if spec else slots,
        out=torch.empty((total, H * V), device="cuda", dtype=torch.bfloat16),
    )
    if spec:
        args.update(num_accepted_tokens=1 + seq % steps, conv_state_indices=1 + seq)
    return args


def bench(fn, pool, args):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(**pool[0])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # Keep every graph output alive for the complete timing window.
        outputs = [fn(**data) for data in pool]
    for _ in range(args.warmup):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(args.rounds):
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (args.replays * len(outputs)))
    return {"median_us": median(samples), "samples_us": samples}


def run(args):
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a ROCm gfx950 GPU.")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.gcnArchName.split(":")[0] != "gfx950":
        raise SystemExit(
            f"This benchmark requires gfx950. Device is {props.gcnArchName}."
        )
    from aiter.ops import kda_rectangular as api

    torch.manual_seed(args.seed)
    weights = dict(
        conv_weight=rand(D, 4),
        A_log=rand(H, dtype=torch.float32),
        dt_bias=rand(H * K, dtype=torch.float32),
        norm_weight=rand(V),
        norm_eps=1e-5,
        lower_bound=-5.0,
    )
    rows = []
    if args.mode in ("all", "prefill"):
        for resumed in (False, True):
            pool = [
                prefill_inputs(api, weights, args.length, resumed)
                for _ in range(args.pool_size)
            ]
            row = dict(op="prefill", batch=1, tokens=args.length, resumed=resumed)
            rows.append({**row, **bench(api.prefill, pool, args)})
            del pool
    if args.mode in ("all", "decode"):
        for batch in args.batch:
            for spec in args.spec:
                dynamic = bool(spec and args.dynamic)
                steps = spec + 1
                lengths = [1 + i % steps if dynamic else steps for i in range(batch)]
                pool = [
                    decode_inputs(weights, batch, spec, lengths)
                    for _ in range(args.pool_size)
                ]
                row = dict(
                    op="decode",
                    batch=batch,
                    spec=spec,
                    dynamic=dynamic,
                    tokens=sum(lengths),
                )
                rows.append({**row, **bench(api.fused_kda_decode, pool, args)})
                del pool
    report = dict(
        device=props.name,
        arch=props.gcnArchName,
        compute_units=props.multi_processor_count,
        torch=str(torch.__version__),
        hip=torch.version.hip,
        seed=args.seed,
        shape=dict(heads=H, key_dim=K, value_dim=V),
        pool_size=args.pool_size,
        warmup_replays=args.warmup,
        rounds=args.rounds,
        replays_per_round=args.replays,
        rows=rows,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--length", type=int, default=32768, help="tokens per prefill")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument(
        "--spec", type=int, nargs="+", choices=range(8), default=[0, 3, 7]
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="cycle decode lengths from 1 to spec+1"
    )
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    args = parser.parse_args()
    if (
        min(args.length, args.pool_size, args.rounds, args.replays) < 1
        or args.warmup < 0
    ):
        parser.error(
            "length, pool size, rounds and replays must be positive; warmup must be nonnegative"
        )
    if any(not 1 <= b <= 256 for b in args.batch):
        parser.error("batch sizes must be in [1, 256]")
    return args


if __name__ == "__main__":
    run(parse_args())
