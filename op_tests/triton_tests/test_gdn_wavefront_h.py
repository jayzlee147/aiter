#!/usr/bin/env python3
"""Test & benchmark for GDN wavefront H-scan kernel."""

import math
import time
import torch
import triton
import pytest


def ref_h_scan(k, w_bar, u_bar, g_cumsum, initial_state=None, BT=64):
    """
    Reference PyTorch implementation of h-state scan.

    Args:
        k:        [B, T, H, K] bf16 — keys (gated by g within chunk)
        w_bar:    [B, T, H, K] bf16 — WY factor w
        u_bar:    [B, T, H, V] bf16 — WY factor u (corrected values before scan)
        g_cumsum: [B, T, H]    fp32 — intra-chunk cumulative gate sums
        initial_state: [B, H, V, K] fp32 or None

    Returns:
        h_out:  [B, NT, H, V, K] fp32 — h snapshot before each chunk's update
        v_new:  [B, T, H, V]     bf16 — corrected values
    """
    B, T, H, K = k.shape
    V = u_bar.shape[-1]
    NT = T // BT

    h = torch.zeros(B, H, V, K, dtype=torch.float32, device=k.device)
    if initial_state is not None:
        h = initial_state.clone().float()

    h_out = torch.empty(B, NT, H, V, K, dtype=torch.float32, device=k.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=k.device)

    for i_t in range(NT):
        t_start = i_t * BT
        t_end = t_start + BT

        # Store h snapshot (pre-update)
        h_out[:, i_t] = h.clone()

        # Chunk slices
        w_chunk = w_bar[:, t_start:t_end].float()   # [B, BT, H, K]
        u_chunk = u_bar[:, t_start:t_end].float()   # [B, BT, H, V]
        k_chunk = k[:, t_start:t_end].float()       # [B, BT, H, K]
        g_chunk = g_cumsum[:, t_start:t_end]         # [B, BT, H]

        # Phase b: retrieve = w_bar @ h → [B, BT, H, V]
        # h is [B, H, V, K], w is [B, BT, H, K]
        # retrieve[b, t, h, v] = sum_k w[b,t,h,k] * h[b,h,v,k]
        retrieve = torch.einsum('bthk,bhvk->bthv', w_chunk, h)

        # Phase b': v_new = u_bar - retrieve
        v_new_chunk = (u_chunk - retrieve).to(torch.bfloat16)
        v_new[:, t_start:t_end] = v_new_chunk

        # Phase d: gate decay + accumulate
        # g_last = g_cumsum at last position of chunk
        g_last = g_chunk[:, -1, :]  # [B, H]
        # h *= exp(g_last)  — h is [B, H, V, K]
        decay = torch.exp(g_last)[:, :, None, None]  # [B, H, 1, 1]
        h = h * decay

        # Gate k by g_cumsum: k_gated[t] = k[t] * exp(g_last - g[t])
        g_diff = g_last[:, None, :] - g_chunk  # [B, BT, H]
        k_gated = k_chunk * torch.exp(g_diff).unsqueeze(-1)  # [B, BT, H, K]

        # h += k_gated^T @ v_new → [B, H, V, K]
        # k_gated: [B, BT, H, K], v_new_chunk: [B, BT, H, V]
        # h[b,h,v,k] += sum_t k_gated[b,t,h,k] * v_new_chunk[b,t,h,v]
        h = h + torch.einsum('bthk,bthv->bhvk', k_gated, v_new_chunk.float())

    return h_out, v_new, h  # h_out, v_new, final_state


@pytest.mark.parametrize("B,T,H", [
    (1, 512, 4),
    (1, 1024, 8),
    (2, 2048, 4),
    (1, 4096, 16),
    (1, 8192, 32),
])
def test_wavefront_correctness(B, T, H):
    """Compare wavefront kernel output against PyTorch reference."""
    from aiter.ops.opus_gdn_prefill import opus_gdn_wavefront_h_fwd

    K = V = 128
    BT = 64
    NT = T // BT
    S = 8

    if NT < S:
        S = NT
    assert NT % S == 0

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
    w_bar = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
    u_bar = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda') * 0.1
    # g_cumsum is cumsum of log-space gate (negative), so monotonically decreasing within chunk
    g_raw = -torch.rand(B, NT, BT, H, dtype=torch.float32, device='cuda') * 0.1
    g_cumsum = g_raw.cumsum(dim=2).view(B, T, H)

    # Reference
    h_ref, v_new_ref, fs_ref = ref_h_scan(k, w_bar, u_bar, g_cumsum)

    # Wavefront kernel
    h_wf, v_new_wf, _ = opus_gdn_wavefront_h_fwd(
        k, w_bar, u_bar, g_cumsum, S=S
    )

    # Compare h snapshots — kernel outputs [B,NT,H,K,V], ref outputs [B,NT,H,V,K]
    h_wf_cmp = h_wf.permute(0, 1, 2, 4, 3)  # [B,NT,H,K,V] → [B,NT,H,V,K]
    h_abs_err = (h_wf_cmp.float() - h_ref.float()).abs().max().item()

    # Compare v_new
    v_abs_err = (v_new_wf.float() - v_new_ref.float()).abs().max().item()

    print(f"B={B} T={T} H={H}: h max_abs_err={h_abs_err:.6f}, v_new max_abs_err={v_abs_err:.6f}")

    assert h_abs_err < 0.05, f"h absolute error too large: {h_abs_err}"
    assert v_abs_err < 0.01, f"v_new absolute error too large: {v_abs_err}"


@pytest.mark.parametrize("B,T,H", [
    (1, 512, 4),
    (1, 4096, 16),
])
def test_wavefront_with_initial_state(B, T, H):
    """Test with non-zero initial state."""
    from aiter.ops.opus_gdn_prefill import opus_gdn_wavefront_h_fwd

    K = V = 128
    BT = 64
    NT = T // BT
    S = 8
    if NT < S:
        S = NT

    torch.manual_seed(123)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
    w_bar = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
    u_bar = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda') * 0.1
    g_raw = -torch.rand(B, NT, BT, H, dtype=torch.float32, device='cuda') * 0.1
    g_cumsum = g_raw.cumsum(dim=2).view(B, T, H)

    initial_state = torch.randn(B, H, V, K, dtype=torch.float32, device='cuda') * 0.1

    h_ref, v_new_ref, fs_ref = ref_h_scan(k, w_bar, u_bar, g_cumsum, initial_state=initial_state)
    h_wf, v_new_wf, _ = opus_gdn_wavefront_h_fwd(
        k, w_bar, u_bar, g_cumsum, initial_state=initial_state, S=S
    )

    h_wf_cmp = h_wf.permute(0, 1, 2, 4, 3)  # [B,NT,H,K,V] → [B,NT,H,V,K]
    h_abs_err = (h_wf_cmp.float() - h_ref.float()).abs().max().item()
    v_abs_err = (v_new_wf.float() - v_new_ref.float()).abs().max().item()

    print(f"[init_state] B={B} T={T} H={H}: h max_abs_err={h_abs_err:.6f}, v_new max_abs_err={v_abs_err:.6f}")

    assert h_abs_err < 0.05, f"h absolute error too large: {h_abs_err}"
    assert v_abs_err < 0.01, f"v_new absolute error too large: {v_abs_err}"


def benchmark_wavefront_vs_serial():
    """Benchmark wavefront kernel vs Triton serial scan."""
    from aiter.ops.opus_gdn_prefill import opus_gdn_wavefront_h_fwd
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )

    configs = [
        (1, 2048, 32, 128, 128),
        (1, 4096, 32, 128, 128),
        (1, 8192, 32, 128, 128),
        (4, 2048, 32, 128, 128),
        (4, 4096, 32, 128, 128),
    ]

    print(f"\n{'Config':>25s}  {'Serial(ms)':>12s}  {'Wavefront(ms)':>14s}  {'Speedup':>8s}")
    print("-" * 70)

    for B, T, H, K, V in configs:
        BT = 64
        NT = T // BT
        S = 8
        if NT < S:
            S = NT

        torch.manual_seed(0)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
        w_bar = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda') * 0.02
        u_bar = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda') * 0.1
        g_raw = -torch.rand(B, NT, BT, H, dtype=torch.float32, device='cuda') * 0.1
        g_cumsum = g_raw.cumsum(dim=2).view(B, T, H)

        # Warmup
        for _ in range(3):
            opus_gdn_wavefront_h_fwd(k, w_bar, u_bar, g_cumsum, S=S)
        torch.cuda.synchronize()

        # Benchmark wavefront
        N_ITER = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            opus_gdn_wavefront_h_fwd(k, w_bar, u_bar, g_cumsum, S=S)
        torch.cuda.synchronize()
        wf_ms = (time.perf_counter() - t0) / N_ITER * 1000

        # Benchmark serial (Triton)
        h_serial = k.new_empty(B, NT, H, K, V)
        v_new_serial = torch.empty_like(u_bar)

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), B * H)

        for _ in range(3):
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
                k=k, v=u_bar, w=w_bar, v_new=v_new_serial,
                g=g_cumsum, gk=None,
                h=h_serial, h0=None, ht=None,
                cu_seqlens=None, chunk_offsets=None,
                T=T, H=H, K=K, V=V, BT=BT,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_ITER):
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
                k=k, v=u_bar, w=w_bar, v_new=v_new_serial,
                g=g_cumsum, gk=None,
                h=h_serial, h0=None, ht=None,
                cu_seqlens=None, chunk_offsets=None,
                T=T, H=H, K=K, V=V, BT=BT,
            )
        torch.cuda.synchronize()
        serial_ms = (time.perf_counter() - t0) / N_ITER * 1000

        speedup = serial_ms / wf_ms if wf_ms > 0 else float('inf')
        cfg = f"B={B} T={T} H={H}"
        print(f"{cfg:>25s}  {serial_ms:>12.3f}  {wf_ms:>14.3f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    # Run correctness tests first
    print("=== Correctness Tests ===")
    for B, T, H in [(1, 512, 4), (1, 1024, 8), (2, 2048, 4), (1, 4096, 16)]:
        test_wavefront_correctness(B, T, H)
    print("\n=== Initial State Tests ===")
    for B, T, H in [(1, 512, 4), (1, 4096, 16)]:
        test_wavefront_with_initial_state(B, T, H)

    print("\n=== Performance Benchmark ===")
    benchmark_wavefront_vs_serial()
