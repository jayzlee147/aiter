# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import (
    gpu as mlir_gpu,
)
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import (
    GTensor,
    _to_raw,
    get_dtype_bytes,
    get_dtype_in_kernel,
)


@functools.lru_cache(maxsize=1024)
def create_vk_gdr_decode_kernel(
    dtype: str,
    A_log_dtype: str,
    b_dtype: str,
    state_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_strides: tuple,
    k_strides: tuple,
    v_strides: tuple,
    state_strides: tuple,
    a_strides: tuple,
    b_strides: tuple,
    use_qk_l2norm: bool,
    is_kda: bool = False,
    lower_bound: float = -5.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
    SHARE_KDA_GATE_LDS: bool = False,
):
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]

    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2

    KERNEL_NAME = f"gdr_decode_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"
    if SHARE_KDA_GATE_LDS:
        KERNEL_NAME += "_kda_gate_lds"

    @fx.struct
    class SharedStorage:
        kda_gate: fx.Array[fx.Float32, head_k_dim, 16]

    @flyc.kernel
    def gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        dtype_ = get_dtype_in_kernel(dtype)
        fx_dtype_ = fx.BFloat16 if dtype == "bf16" else fx.Float16
        A_log_dtype_ = get_dtype_in_kernel(A_log_dtype)
        state_dtype_ = get_dtype_in_kernel(state_dtype)
        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        width_i32 = _to_raw(fx.Int32(WARP_SIZE))
        vec_t = T.vec(VALUES_PER_THREAD_K, dtype_)
        acc_vec_t = T.vec(VALUES_PER_THREAD_K, T.f32)
        if const_expr(is_kda and SHARE_KDA_GATE_LDS):
            kda_gate_lds = fx.SharedAllocator().allocate(SharedStorage).peek().kda_gate

        tidx = fx.thread_idx.x
        bidx = fx.block_idx.x
        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        read_indices_tensor = GTensor(read_indices, dtype=T.i32, shape=(-1,))
        write_indices_tensor = GTensor(write_indices, dtype=T.i32, shape=(-1,))
        read_pool_idx = fx.Int32(read_indices_tensor[b_i])
        write_pool_idx = fx.Int32(write_indices_tensor[b_i])

        q_tensor = GTensor(
            query,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=q_strides,
        )
        k_tensor = GTensor(
            key,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=k_strides,
        )
        v_tensor = GTensor(
            value,
            dtype=dtype_,
            shape=(-1, seq_length, num_v_heads, head_v_dim),
            stride=v_strides,
        )
        a_tensor = GTensor(
            a,
            dtype=dtype_,
            stride=(a_strides[0], a_strides[1], a_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        b_dtype_ = get_dtype_in_kernel(b_dtype)
        b_tensor = GTensor(
            b,
            dtype=b_dtype_,
            stride=(b_strides[0], b_strides[1], b_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        if const_expr(is_kda):
            a_tensor = GTensor(
                a,
                dtype=dtype_,
                stride=(a_strides[0], a_strides[1], a_strides[2], a_strides[3]),
                shape=(-1, seq_length, num_v_heads, head_k_dim),
            )
            dt_bias_tensor = GTensor(
                dt_bias, dtype=dtype_, shape=(num_v_heads, head_k_dim)
            )
        else:
            dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=A_log_dtype_, shape=(num_v_heads,))
        out_tensor = GTensor(
            out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim)
        )
        read_state_tensor = GTensor(
            state,
            dtype=state_dtype_,
            shape=(num_v_heads, head_v_dim, head_k_dim),
            stride=(state_strides[1], state_strides[2], state_strides[3]),
            static_bytes_offset_i64=fx.Int64(read_pool_idx)
            * fx.Int64(state_strides[0])
            * get_dtype_bytes(state_dtype),
        )
        write_state_tensor = GTensor(
            state,
            dtype=state_dtype_,
            shape=(num_v_heads, head_v_dim, head_k_dim),
            stride=(state_strides[1], state_strides[2], state_strides[3]),
            static_bytes_offset_i64=fx.Int64(write_pool_idx)
            * fx.Int64(state_strides[0])
            * get_dtype_bytes(state_dtype),
        )

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * log2e))
            return fx.math.exp(x, fastmath=fx.FastMathFlags.fast)

        def fast_log1p(x):
            return fx.math.log1p(x, fastmath=fx.FastMathFlags.fast)

        # Skip CG-pad slots (indices sentinel < 0). The guarded body is a
        # closure so the runtime `if` sees an opaque call (no GTensor "state"
        # to thread through an scf.if yield) -- lowers to scf.if, no raw region.
        def _do_decode():
            if const_expr("f32" in A_log_dtype):
                r_A_log = A_log_tensor[hv_i]
            else:
                r_A_log = A_log_tensor[hv_i].extf(T.f32)
            # A_log is invariant across sequence, K and V. Keep exp(A_log)
            # explicitly outside the element loop so the generated ISA cannot
            # duplicate this transcendental while unrolling gate vectors.
            r_A = fast_exp(r_A_log)
            if const_expr(not is_kda):
                r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = (
                        read_state_tensor.vec_load(
                            (hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K
                        )
                    )
                    if const_expr("f32" not in state_dtype):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].extf(acc_vec_t)

            for sq_i in range_constexpr(seq_length):
                r_b_raw = b_tensor[b_i, sq_i, hv_i]
                r_b = r_b_raw if const_expr("f32" in b_dtype) else r_b_raw.extf(T.f32)
                r_beta = fx.Float32(
                    rocdl.rcp(T.f32, _to_raw(f32_1 + fast_exp(-r_b)))
                )

                if const_expr(not is_kda):
                    r_a = a_tensor[b_i, sq_i, hv_i].extf(T.f32)
                    x = r_a + r_dt_bias
                    beta_x = softplus_beta_ * x

                    # softplus with the large-x identity: for beta_x > threshold,
                    # softplus(x) == x. select computes both arms (the overflow arm
                    # is discarded) -> bit-identical to the old branch.
                    softplus_big = (f32_1 / softplus_beta_) * fast_log1p(
                        fast_exp(beta_x)
                    )
                    softplus_x = (
                        fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)
                    ).select(softplus_big, x)

                    r_g_value = -r_A * softplus_x
                    r_g = fast_exp(r_g_value)
                    r_g_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                    )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(scale), fx.Float32
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    k_vec = k_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                    if const_expr(is_kda):
                        def _compute_gate_elems():
                            a_vec = a_tensor.vec_load(
                                (b_i, sq_i, hv_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            ).extf(acc_vec_t)
                            dt_vec = dt_bias_tensor.vec_load(
                                (hv_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            ).extf(acc_vec_t)
                            gate_elems = []
                            for elem_i in range_constexpr(VALUES_PER_THREAD_K):
                                a_elem = fx.Float32(
                                    a_vec[elem_i]
                                )
                                dt_elem = fx.Float32(
                                    dt_vec[elem_i]
                                )
                                gate_log = fx.Float32(
                                    lower_bound
                                    * fx.Float32(
                                        rocdl.rcp(
                                            T.f32,
                                            _to_raw(
                                                f32_1
                                                + fast_exp(
                                                    -r_A * (a_elem + dt_elem)
                                                )
                                            ),
                                        )
                                    )
                                )
                                gate_elems.append(fast_exp(gate_log))
                            return gate_elems

                        if const_expr(SHARE_KDA_GATE_LDS):
                            # Only the first K-lane group computes the 128 decay
                            # factors; all V rows and warps reuse them from LDS.
                            if tidx < WARP_THREADS_K:
                                gate_elems = _compute_gate_elems()
                                for elem_i in range_constexpr(VALUES_PER_THREAD_K):
                                    fx.ptr_store(
                                        gate_elems[elem_i],
                                        kda_gate_lds.ptr
                                        + fx.Int32(warp_k_vec_i + elem_i),
                                    )
                        else:
                            state_gate_vec = fx.Vector.from_elements(
                                _compute_gate_elems(), fx.Float32
                            )
                            for vi in range_constexpr(WARP_TILE_V_ITERS):
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] *= state_gate_vec

                if const_expr(is_kda and SHARE_KDA_GATE_LDS):
                    gpu.barrier()
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        gate_elems = []
                        for elem_i in range_constexpr(VALUES_PER_THREAD_K):
                            gate_elems.append(
                                fx.ptr_load(
                                    kda_gate_lds.ptr
                                    + fx.Int32(warp_k_vec_i + elem_i)
                                )
                            )
                        state_gate_vec = fx.Vector.from_elements(gate_elems, fx.Float32)
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            state_vecs[vi * WARP_TILE_K_ITERS + ki] *= state_gate_vec

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_k_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = (
                            sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        )
                        sum_k_partial_vec = (
                            sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        )
                    sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                        sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                    local_sum_q = mlir_gpu.ShuffleOp(
                        _to_raw(sum_q_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(
                        _to_raw(sum_k_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    inv_norm_q = rocdl.rsq(T.f32, _to_raw(local_sum_q + 1e-6))
                    inv_norm_k = rocdl.rsq(T.f32, _to_raw(local_sum_k + 1e-6))
                    inv_norm_q_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
                    )
                    inv_norm_k_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
                )
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
                dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = v_tensor[b_i, sq_i, hv_i, global_v_i].extf(T.f32)

                    sum_hk = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_hq_old = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        if const_expr(not is_kda):
                            state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                        sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                    sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                    sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                        sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(
                            offset, WARP_SIZE
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(
                        _to_raw(v_new),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = fx.math.fma(
                            sk_vecs[ki],
                            v_new_bcast,
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                        )
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                    sum_hq = sum_hq.to(fx_dtype_)

                    # Only k-vec lane 0 writes the q output; closure keeps the
                    # GTensor store opaque to the runtime-if state analysis.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        out_tensor[b_i, _sq, hv_i, _gv] = _sum_hq

                    if warp_k_vec_start == 0:
                        _write_q()

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    if const_expr("f32" in state_dtype):
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    else:
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki].truncf(vec_t)
                    write_state_tensor.vec_store(
                        (hv_i, global_v_i, warp_k_vec_i), out_vec, VALUES_PER_THREAD_K
                    )

        if (read_pool_idx >= 0) & (write_pool_idx >= 0):
            _do_decode()

    @flyc.jit
    def launch_gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_decode_kernel._func.__name__ = KERNEL_NAME
        gdr_decode_kernel(
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            read_indices,
            write_indices,
            state,
            out,
            batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gdr_decode_kernel
