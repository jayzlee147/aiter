# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# E2E performance benchmark: opus GDN (2 kernels: K1+K2) vs Triton (5 kernels)
# with per-kernel breakdown for Triton.
#
# Usage:
#   python -m pytest op_tests/triton_tests/bench_opus_gdn.py -v -s
#   python -m pytest op_tests/triton_tests/bench_opus_gdn.py -v -s -k "32768"
#   python op_tests/triton_tests/bench_opus_gdn.py          # direct run

import pytest
import torch
import torch.nn.functional as F

from aiter.jit.core import get_gfx
from aiter.ops.opus_gdn_wu_prefill import opus_gdn_wu_prefill_fwd

# Detected GPU arch → friendly label for the benchmark header
_GFX_NAMES = {"gfx942": "MI300X", "gfx950": "MI350"}
_GFX = get_gfx()
_GFX_LABEL = f"{_GFX} ({_GFX_NAMES[_GFX]})" if _GFX in _GFX_NAMES else _GFX
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill import (
    chunk_gated_delta_rule_fwd_h,
    chunk_fwd_o,
    fused_cumsum_kkt,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    solve_tril,
    recompute_w_u_fwd,
)

device = "cuda"
D = 128  # K = V = 128, hardcoded in opus kernel


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
# Format: (B, T, H, tag)
#   tag: human-readable label for grouping/filtering
#
# Design rationale:
#   - "colleague-32k" configs match the colleague's K5 benchmark:
#     total_tokens ≈ 32768, seqlen = 1k / 5k / 10k
#   - "model-*" configs reflect real GDN model head counts (H=16/32/40)
#     at practical sequence lengths
#   - "stress-long" configs test long-context scenarios (T=8k/16k/32k)

BENCH_CONFIGS = [
    # ---- Colleague's highlighted parameter set (total ≈ 32k tokens) ----
    (32, 1024, 40, "colleague-32k-seq1k"),
    (6, 5120, 40, "colleague-32k-seq5k"),
    (3, 10240, 40, "colleague-32k-seq10k"),

    # ---- Standard model configs — short seq ----
    (8, 512, 16, "small-H16-seq512"),
    (8, 512, 32, "medium-H32-seq512"),

    # ---- Standard model configs — medium seq ----
    (4, 2048, 16, "small-H16-seq2k"),
    (4, 2048, 32, "medium-H32-seq2k"),
    (4, 2048, 40, "large-H40-seq2k"),

    # ---- Standard model configs — long seq ----
    (2, 4096, 32, "medium-H32-seq4k"),
    (2, 4096, 40, "large-H40-seq4k"),
    (1, 8192, 32, "medium-H32-seq8k"),
    (1, 8192, 40, "large-H40-seq8k"),

    # ---- Stress: long context ----
    (1, 16384, 32, "stress-H32-seq16k"),
    (1, 32768, 32, "stress-H32-seq32k"),
]


def _make_inputs(B, T, H):
    """Create random GDN inputs on GPU."""
    torch.manual_seed(42)
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    return q, k, v, g, beta


def _bench_fn(fn, warmup=5, repeat=20):
    """Time a GPU function in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    # Use median of middle 60% to reduce outlier noise
    lo = len(times) // 5
    hi = len(times) - lo
    return sum(times[lo:hi]) / (hi - lo)


# ---------------------------------------------------------------------------
# Triton per-kernel breakdown
# ---------------------------------------------------------------------------
def _triton_breakdown(q, k, v, g, beta, scale):
    """Run triton's 5 sub-kernels individually, return per-kernel times (ms)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64  # triton hardcodes chunk_size=64

    # Step 1: cumsum
    def fn_cumsum():
        return chunk_local_cumsum(g.clone(), chunk_size=BT)
    t_cumsum = _bench_fn(fn_cumsum)
    g_cs = chunk_local_cumsum(g, chunk_size=BT)

    # Step 2a: KKT
    def fn_kkt():
        return chunk_scaled_dot_kkt_fwd(
            k=k, g=g_cs, beta=beta, cu_seqlens=None, output_dtype=torch.float32
        )
    t_kkt = _bench_fn(fn_kkt)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g_cs, beta=beta, cu_seqlens=None, output_dtype=torch.float32)

    # Step 2b: solve_tril
    def fn_trisol():
        return solve_tril(A=A.clone(), cu_seqlens=None, output_dtype=k.dtype)
    t_trisol = _bench_fn(fn_trisol)
    A_solved = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)

    # Step 2c: recompute_w_u (WY factors)
    def fn_wy():
        return recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_solved, g=g_cs)
    t_wy = _bench_fn(fn_wy)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_solved, g=g_cs)

    # Step 3: chunk_fwd_h (K5 — the colleague's target)
    def fn_fwd_h():
        return chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g_cs,
            initial_state=None, output_final_state=False,
        )
    t_fwd_h = _bench_fn(fn_fwd_h)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cs,
        initial_state=None, output_final_state=False,
    )

    # Step 4: chunk_fwd_o (output)
    def fn_fwd_o():
        return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cs, scale=scale)
    t_fwd_o = _bench_fn(fn_fwd_o)

    return {
        "cumsum": t_cumsum,
        "kkt": t_kkt,
        "trisol": t_trisol,
        "wy": t_wy,
        "fwd_h": t_fwd_h,
        "fwd_o": t_fwd_o,
    }


# ---------------------------------------------------------------------------
# Opus e2e timing
# ---------------------------------------------------------------------------
def _opus_time(q, k, v, g, beta, BT=64, BV=64):
    """Time opus K1+K2 e2e in ms."""
    beta_bf16 = beta.to(torch.bfloat16)
    def fn():
        return opus_gdn_wu_prefill_fwd(q, k, v, g, beta_bf16, BT=BT, BV=BV)
    return _bench_fn(fn)


# ---------------------------------------------------------------------------
# Triton e2e timing (all 5 kernels through the public API)
# ---------------------------------------------------------------------------
def _triton_e2e_time(q, k, v, g, beta):
    """Time triton chunk_gated_delta_rule e2e in ms."""
    beta_bf16 = beta.to(torch.bfloat16)
    def fn():
        return chunk_gated_delta_rule(
            q=q.clone(), k=k.clone(), v=v.clone(),
            g=g.clone(), beta=beta_bf16.clone(),
        )
    return _bench_fn(fn)


# ---------------------------------------------------------------------------
# Pytest parametrized benchmark
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("B", "T", "H", "tag"),
    [pytest.param(*c, id=c[3]) for c in BENCH_CONFIGS],
)
class TestGDNPerformance:
    """E2E performance: opus (K1+K2) vs triton (5 kernels) with breakdown."""

    def test_e2e_comparison(self, B, T, H, tag):
        q, k, v, g, beta = _make_inputs(B, T, H)
        scale = D ** -0.5

        # Triton breakdown
        tri_times = _triton_breakdown(q, k, v, g, beta, scale)
        tri_total = sum(tri_times.values())

        # Triton e2e (through public API — includes overhead)
        tri_e2e = _triton_e2e_time(q, k, v, g, beta)

        # Opus e2e (BT=64, BV=64)
        opus_ms = _opus_time(q, k, v, g, beta, BT=64, BV=64)
        speedup = tri_e2e / opus_ms

        # Print results
        total_tokens = B * T
        highlight = " ★★★" if "colleague" in tag else ""
        print(f"\n{'='*80}")
        print(f"  {tag}  B={B} T={T} H={H} D={D}  tokens={total_tokens}{highlight}")
        print(f"{'='*80}")
        print(f"  Triton per-kernel breakdown (ms):")
        print(f"    cumsum   : {tri_times['cumsum']:8.3f}")
        print(f"    kkt      : {tri_times['kkt']:8.3f}")
        print(f"    trisol   : {tri_times['trisol']:8.3f}")
        print(f"    wy       : {tri_times['wy']:8.3f}")
        print(f"    fwd_h(K5): {tri_times['fwd_h']:8.3f}")
        print(f"    fwd_o    : {tri_times['fwd_o']:8.3f}")
        print(f"    ─────────────────────────")
        print(f"    sum      : {tri_total:8.3f}")
        print(f"    e2e API  : {tri_e2e:8.3f}")
        print()
        print(f"  Opus e2e   : {opus_ms:8.3f}   speedup vs triton: {speedup:.2f}x")
        print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Direct-run mode: run all configs and print a summary table
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*120}")
    print(f"  GDN Prefill Benchmark — opus (2 kernels) vs triton (5 kernels)")
    print(f"  D=K=V={D}, bf16, {_GFX_LABEL}")
    print(f"{'='*120}")

    header = (
        f"{'tag':>28s} | {'B':>3s} {'T':>6s} {'H':>3s} {'tokens':>7s} | "
        f"{'cumsum':>7s} {'kkt':>7s} {'trisol':>7s} {'wy':>7s} {'fwd_h':>7s} {'fwd_o':>7s} | "
        f"{'tri_e2e':>8s} {'opus':>8s} | {'spd':>5s}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for B, T, H, tag in BENCH_CONFIGS:
        q, k, v, g, beta = _make_inputs(B, T, H)
        scale = D ** -0.5

        tri_times = _triton_breakdown(q, k, v, g, beta, scale)
        tri_e2e = _triton_e2e_time(q, k, v, g, beta)
        opus_ms = _opus_time(q, k, v, g, beta, BT=64, BV=64)
        spd = tri_e2e / opus_ms

        mark = "★" if "colleague" in tag else " "
        print(
            f"{mark}{tag:>27s} | {B:3d} {T:6d} {H:3d} {B*T:7d} | "
            f"{tri_times['cumsum']:7.3f} {tri_times['kkt']:7.3f} {tri_times['trisol']:7.3f} "
            f"{tri_times['wy']:7.3f} {tri_times['fwd_h']:7.3f} {tri_times['fwd_o']:7.3f} | "
            f"{tri_e2e:8.3f} {opus_ms:8.3f} | {spd:5.2f}"
        )

    print()
    print(f"NOTE: Both K1 and K2 use MFMA bf16 16x16x16 on {_GFX}.")
    print("      Speedup column shows triton_e2e / opus_e2e (>1 = opus faster).")
    print()


if __name__ == "__main__":
    main()
