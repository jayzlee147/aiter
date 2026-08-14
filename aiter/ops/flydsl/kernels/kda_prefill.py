# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent recurrent KDA prefill kernel.

This is the correctness-first FlyDSL implementation used to establish the
Kimi-K3 prefill ABI.  One wave owns sixteen V rows; four groups of sixteen
lanes reduce the K=128 dot products independently.  The recurrent FP32 state
stays in registers while a CTA walks one variable-length sequence.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import GTensor, _run_compiled, _to_raw, get_dtype_in_kernel


@functools.lru_cache(maxsize=128)
def create_kda_prefill_recurrent_kernel(
    *,
    num_heads: int,
    head_dim: int = 128,
    dt_bias_dtype: str = "bf16",
    lower_bound: float = -5.0,
):
    if head_dim != 128:
        raise ValueError("FlyDSL KDA recurrent prefill currently requires K=V=128")

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = WARP_SIZE * NUM_WARPS
    K_LANES = 32
    V_ROWS_PER_WARP = WARP_SIZE // K_LANES
    V_ROWS_PER_BLOCK = NUM_WARPS * V_ROWS_PER_WARP
    V_TILES = head_dim // V_ROWS_PER_BLOCK
    ELEMS_PER_LANE = head_dim // K_LANES
    LOG2E = math.log2(math.e)
    SCALE = head_dim**-0.5

    @flyc.kernel(name="kda_prefill_recurrent_flydsl")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        gate: fx.Tensor,
        beta: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        cu_seqlens: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        num_sequences: fx.Int32,
    ):
        del num_sequences
        q_ = GTensor(q, dtype=T.bf16, shape=(-1,))
        k_ = GTensor(k, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v, dtype=T.bf16, shape=(-1,))
        gate_ = GTensor(gate, dtype=T.bf16, shape=(-1,))
        beta_ = GTensor(beta, dtype=T.f32, shape=(-1,))
        A_ = GTensor(A_log, dtype=T.f32, shape=(-1,))
        dt_t = get_dtype_in_kernel(dt_bias_dtype)
        dt_ = GTensor(dt_bias, dtype=dt_t, shape=(-1,))
        cu_ = GTensor(cu_seqlens, dtype=T.i64, shape=(-1,))
        state_ = GTensor(state, dtype=T.f32, shape=(-1,))
        out_ = GTensor(out, dtype=T.bf16, shape=(-1,))

        block = fx.block_idx.x
        v_tile = block % fx.Int32(V_TILES)
        nh = block // fx.Int32(V_TILES)
        head = nh % fx.Int32(num_heads)
        seq = nh // fx.Int32(num_heads)

        tid = fx.thread_idx.x
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        k_lane = lane % fx.Int32(K_LANES)
        v_in_warp = lane // fx.Int32(K_LANES)
        v_row = (
            v_tile * fx.Int32(V_ROWS_PER_BLOCK)
            + warp * fx.Int32(V_ROWS_PER_WARP)
            + v_in_warp
        )
        k_base = k_lane * fx.Int32(ELEMS_PER_LANE)

        start = cu_[seq]
        end = cu_[seq + fx.Int32(1)]
        state_base = (
            ((fx.Int64(seq) * fx.Int64(num_heads) + fx.Int64(head))
             * fx.Int64(head_dim) + fx.Int64(v_row))
            * fx.Int64(head_dim)
            + fx.Int64(k_base)
        )
        state_vec = state_.vec_load((state_base,), ELEMS_PER_LANE)

        def fast_exp(x):
            return fx.Float32(rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * LOG2E)))

        A = fast_exp(A_[head])
        init = [_to_raw(state_vec[i]) for i in range_constexpr(ELEMS_PER_LANE)]
        for token, carried in range(start, end, 1, init=init):
            h_vec = fx.Vector.from_elements(list(carried), fx.Float32)
            token_head = fx.Int64(token) * fx.Int64(num_heads) + fx.Int64(head)
            qk_base = token_head * fx.Int64(head_dim) + fx.Int64(k_base)

            q_vec = q_.vec_load((qk_base,), ELEMS_PER_LANE).extf(
                T.vec(ELEMS_PER_LANE, T.f32)
            )
            k_vec = k_.vec_load((qk_base,), ELEMS_PER_LANE).extf(
                T.vec(ELEMS_PER_LANE, T.f32)
            )
            g_vec = gate_.vec_load((qk_base,), ELEMS_PER_LANE).extf(
                T.vec(ELEMS_PER_LANE, T.f32)
            )
            dt_vec = dt_.vec_load(
                (fx.Int64(head) * fx.Int64(head_dim) + fx.Int64(k_base),),
                ELEMS_PER_LANE,
            )
            if const_expr(dt_bias_dtype != "f32"):
                dt_vec = dt_vec.extf(T.vec(ELEMS_PER_LANE, T.f32))

            q_ss = fx.Vector.from_elements(
                [fx.Float32(0.0) for _ in range_constexpr(ELEMS_PER_LANE)],
                fx.Float32,
            )
            k_ss = fx.Vector.from_elements(
                [fx.Float32(0.0) for _ in range_constexpr(ELEMS_PER_LANE)],
                fx.Float32,
            )
            q_ss = fx.math.fma(q_vec, q_vec, q_ss).reduce(fx.ReductionOp.ADD)
            k_ss = fx.math.fma(k_vec, k_vec, k_ss).reduce(fx.ReductionOp.ADD)
            for offset in (16, 8, 4, 2, 1):
                q_ss = q_ss + q_ss.shuffle_xor(offset, WARP_SIZE)
                k_ss = k_ss + k_ss.shuffle_xor(offset, WARP_SIZE)
            inv_q = fx.Float32(rocdl.rsq(T.f32, _to_raw(q_ss + 1e-6))) * SCALE
            inv_k = fx.Float32(rocdl.rsq(T.f32, _to_raw(k_ss + 1e-6)))
            q_vec = q_vec * fx.Vector.filled(ELEMS_PER_LANE, inv_q, fx.Float32)
            k_vec = k_vec * fx.Vector.filled(ELEMS_PER_LANE, inv_k, fx.Float32)

            gate_elems = []
            for elem in range_constexpr(ELEMS_PER_LANE):
                sigmoid = fx.Float32(
                    rocdl.rcp(
                        T.f32,
                        _to_raw(fx.Float32(1.0) + fast_exp(-A * (g_vec[elem] + dt_vec[elem]))),
                    )
                )
                gate_elems.append(fast_exp(lower_bound * sigmoid))
            h_vec = h_vec * fx.Vector.from_elements(gate_elems, fx.Float32)

            dot_hk = fx.math.fma(
                h_vec,
                k_vec,
                fx.Vector.from_elements(
                    [fx.Float32(0.0) for _ in range_constexpr(ELEMS_PER_LANE)],
                    fx.Float32,
                ),
            ).reduce(fx.ReductionOp.ADD)
            for offset in (16, 8, 4, 2, 1):
                dot_hk = dot_hk + dot_hk.shuffle_xor(offset, WARP_SIZE)

            value_off = token_head * fx.Int64(head_dim) + fx.Int64(v_row)
            value = v_[value_off].extf(T.f32)
            beta_logit = beta_[token_head]
            beta_value = fx.Float32(
                rocdl.rcp(T.f32, _to_raw(fx.Float32(1.0) + fast_exp(-beta_logit)))
            )
            update = (value - dot_hk) * beta_value
            h_vec = fx.math.fma(
                k_vec,
                fx.Vector.filled(ELEMS_PER_LANE, fx.Float32(update), fx.Float32),
                h_vec,
            )

            dot_hq = fx.math.fma(
                h_vec,
                q_vec,
                fx.Vector.from_elements(
                    [fx.Float32(0.0) for _ in range_constexpr(ELEMS_PER_LANE)],
                    fx.Float32,
                ),
            ).reduce(fx.ReductionOp.ADD)
            for offset in (16, 8, 4, 2, 1):
                dot_hq = dot_hq + dot_hq.shuffle_xor(offset, WARP_SIZE)
            def _write_output(_off=value_off, _value=dot_hq):
                out_[_off] = _value.to(fx.BFloat16)

            if k_lane == 0:
                _write_output()
            results = yield [
                _to_raw(h_vec[i]) for i in range_constexpr(ELEMS_PER_LANE)
            ]

        final_vec = fx.Vector.from_elements(list(results), fx.Float32)
        state_.vec_store((state_base,), final_vec, ELEMS_PER_LANE)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        gate: fx.Tensor,
        beta: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        cu_seqlens: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        num_sequences: fx.Int32,
        stream: fx.Stream,
    ):
        gx = num_sequences * num_heads * V_TILES
        kernel(
            q,
            k,
            v,
            gate,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            state,
            out,
            num_sequences,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def run_kda_prefill_recurrent(
    q,
    k,
    v,
    gate,
    beta,
    A_log,
    dt_bias,
    cu_seqlens,
    state,
    out,
    lower_bound=-5.0,
    stream=None,
):
    if stream is None:
        import torch

        stream = torch.cuda.current_stream()
    launch = create_kda_prefill_recurrent_kernel(
        num_heads=q.shape[2],
        head_dim=q.shape[-1],
        dt_bias_dtype="f32" if str(dt_bias.dtype).endswith("float32") else "bf16",
        lower_bound=float(lower_bound),
    )
    _run_compiled(
        launch,
        q,
        k,
        v,
        gate,
        beta,
        A_log,
        dt_bias,
        cu_seqlens,
        state,
        out,
        len(cu_seqlens) - 1,
        stream,
    )
