"""Microbenchmark Kimi-KDA decode: FlyDSL versus ATOM Triton."""

import argparse

import torch

import aiter.ops.flydsl.linear_attention_kernels as lak
from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)


def bench_graph(fn, warmup=50, iters=1000):
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--vs", type=int)
    parser.add_argument("--warps", type=int)
    parser.add_argument("--wk", type=int)
    parser.add_argument("--no-share-gate", action="store_true")
    parser.add_argument("--triton-bv", type=int)
    parser.add_argument("--triton-warps", type=int)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    B, T, H, D = args.batch, args.seq, 8, 128
    if args.vs is not None:
        lak.get_default_kwargs("", "", 0, 0, 0, 0, 0, 0)
        lak.GDR_GLOBAL_CONFIG_MAP[
            ("torch.bfloat16", "torch.float32", lak.GDR_GPU_ARCH, B, T, H, H, D, D)
        ] = {
            "NUM_BLOCKS_PER_V_DIM": args.vs,
            "NUM_WARPS": args.warps,
            "WARP_THREADS_K": args.wk,
            "SHARE_KDA_GATE_LDS": not args.no_share_gate,
        }
    torch.manual_seed(1)
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    k, v, a = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    indices = torch.arange(B, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(B + 1, device="cuda", dtype=torch.int32) * T
    state0 = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32)
    fly_state, tri_state = state0.clone(), state0.clone()
    fly_out = torch.empty_like(v)
    tri_out = torch.empty(B * T, H, D, device="cuda", dtype=v.dtype)

    def fly():
        flydsl_gdr_decode(
            q,
            k,
            v,
            a,
            beta,
            dt_bias,
            A_log,
            indices,
            fly_state,
            fly_out,
            use_qk_l2norm=True,
            need_shuffle_state=False,
            is_kda=True,
            lower_bound=-5.0,
        )

    def triton():
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a.view(1, B * T, H, D),
            b=beta.view(1, B * T, H),
            dt_bias=dt_bias.view(-1),
            q=q.view(1, B * T, H, D),
            k=k.view(1, B * T, H, D),
            v=v.view(1, B * T, H, D),
            o=tri_out,
            initial_state=tri_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            lower_bound=-5.0,
            bv_override=args.triton_bv,
            num_warps_override=args.triton_warps,
        )

    fly()
    triton()
    torch.cuda.synchronize()
    print("out_max", (fly_out.view_as(tri_out).float() - tri_out.float()).abs().max().item())
    state_diff = (fly_state - tri_state).abs()
    print("state_max", state_diff.max().item())
    if state_diff.max().item() > 1e-4:
        print(
            "state_max_per_batch",
            state_diff.flatten(1).amax(1).cpu().tolist(),
        )
        print(
            "state_transpose_max",
            (fly_state - tri_state.transpose(-1, -2)).abs().max().item(),
        )
    fly_state.copy_(state0)
    tri_state.copy_(state0)
    if args.sweep:
        lak.get_default_kwargs("", "", 0, 0, 0, 0, 0, 0)
        key_cfg = (
            "torch.bfloat16", "torch.float32", lak.GDR_GPU_ARCH,
            B, T, H, H, D, D,
        )
        tri_us = bench_graph(triton, iters=args.iters)
        print(f"triton_us={tri_us:.3f}")
        for nw in (1, 2, 4, 8):
            for wk in (4, 8, 16, 32, 64):
                group_v = nw * (64 // wk)
                if group_v > D or D % group_v or wk * 4 > D:
                    continue
                lak.GDR_GLOBAL_CONFIG_MAP[key_cfg] = {
                    "NUM_BLOCKS_PER_V_DIM": 1,
                    "NUM_WARPS": nw,
                    "WARP_THREADS_K": wk,
                    "SHARE_KDA_GATE_LDS": False,
                }
                fly_state.copy_(state0)
                us = bench_graph(fly, iters=args.iters)
                print(
                    f"warps={nw} wk={wk} flydsl_us={us:.3f} "
                    f"speedup={tri_us/us:.3f}x"
                )
        return
    fly_us = bench_graph(fly, iters=args.iters)
    tri_us = bench_graph(triton, iters=args.iters)
    print(
        f"batch={B} seq={T} flydsl_us={fly_us:.3f} "
        f"triton_us={tri_us:.3f} speedup={tri_us/fly_us:.3f}x"
    )


if __name__ == "__main__":
    main()
