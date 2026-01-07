from math import e
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

from aiter.ops.triton._triton_kernels.gdr_sglang.utils import input_guard

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel1(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    b_dt_bias = gl.load(p_dt_bias) # f32
    b_A_log = gl.load(p_A_log).to(gl.float32) # f32

    # Gating computation pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv

    b_a = gl.load(p_a) # f32
    b_b = gl.load(p_b) # f32

    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 2],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blocked_linear: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1]],
        lane_bases=[[2], [4], [8], [16], [32], [64]],
        warp_bases=[[0], [0]],
        block_bases=[],
        shape=[128],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0]
    )

    o_k_blocked = i_k * BK + gl.arange(0, BK, layout=blocked1)
    o_v_blocked = i_v * BV + gl.arange(0, BV, layout=blocked1)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)

    p_q = q + (bos * H + i_h) * K + o_k_blocked
    p_k = k + (bos * H + i_h) * K + o_k_blocked
    p_v = v + (bos * HV + i_hv) * V + o_v_blocked
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v_blocked



    mask_k_blocked = o_k_blocked < K
    mask_v_blocked = o_v_blocked < V
    mask_k_slice = o_k_slice < K
    mask_v_slice = o_v_slice < V
    mask_h = mask_k_slice[:, None] & mask_v_slice[None, :]

    b_k0 = gl.load(p_k, mask=mask_k_blocked, other=0)  # BKxf32
    b_q0 = gl.load(p_q, mask=mask_k_blocked, other=0)  # BKxf32
    b_v0 = gl.load(p_v, mask=mask_v_blocked, other=0)  # BVxf32
    
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            b_h = gl.load(p_h0).to(gl.float32)  # BKxBVxf32

            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_b += HV
            p_a += HV

            softplus_beta_inv = 1.0 / softplus_beta
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
            beta_x = softplus_beta * x # f32
            # Apply softplus with numerical stability
            softplus_x = gl.where(
                beta_x <= softplus_threshold,
                softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
                x,
            )
            b_A = -gl.exp(b_A_log)
            b_g = b_A * softplus_x # f32

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

            shared_b_k = gl.allocate_shared_memory(b_k0.dtype, [BK], shared_layout, b_k0)
            shared_b_q = gl.allocate_shared_memory(b_q0.dtype, [BK], shared_layout, b_q0) # BKxf32

            gl.amd.cdna3.sched_barrier(0)

            b_k = shared_b_k.load(slice1).to(gl.float32) # BKxf32
            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                k_norm_rev = 1 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
                tl.assume(k_norm_rev > 0)
                b_k = b_k * k_norm_rev # BKxf32
            # b_k = gl.convert_layout(b_k, layout=slice1)

            shared_b_v = gl.allocate_shared_memory(b_v0.dtype, [BV], shared_layout, b_v0) # BVxf32
            # Apply L2 normalization to q
            b_q = shared_b_q.load(slice1).to(gl.float32) # BKxf32
            if USE_QK_L2NORM_IN_KERNEL:
                q_norm_rev = 1 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
                tl.assume(q_norm_rev > 0)
                b_q = b_q * q_norm_rev # BKxf32
            b_q = b_q * scale # BKxf32
            # b_q = gl.convert_layout(b_q, layout=slice1)


            # Apply gating to hidden state: h *= exp(g)
            b_h *= gl.exp(b_g) # BKxBVxf32

            b_v = shared_b_v.load(slice4)
            b_v = b_v.to(gl.float32)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

            # Apply beta gating: v *= beta
            b_v *= b_beta # BVxf32

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

            # Compute output: o = sum(h * q, dim=0)
            b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
            b_o = gl.convert_layout(b_o, layout=blocked1)
            gl.store(p_o, b_o)

            # Store final state back to h0_source with bounds checking
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

        else:
            b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_b += HV
            p_a += HV

            softplus_beta_inv = 1.0 / softplus_beta
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
            beta_x = softplus_beta * x # f32
            # Apply softplus with numerical stability
            softplus_x = gl.where(
                beta_x <= softplus_threshold,
                softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
                x,
            )
            b_A = -gl.exp(b_A_log)
            b_g = b_A * softplus_x # f32

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

            shared_b_k = gl.allocate_shared_memory(b_k0.dtype, [BK], shared_layout, b_k0)
            shared_b_q = gl.allocate_shared_memory(b_q0.dtype, [BK], shared_layout, b_q0) # BKxf32
            gl.amd.cdna3.sched_barrier(0)

            b_k = shared_b_k.load(slice1).to(gl.float32)
            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                k_norm_rev = 1 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
                tl.assume(k_norm_rev > 0)
                b_k = b_k * k_norm_rev # BKxf32
            # b_k = gl.convert_layout(b_k, layout=slice1)

            shared_b_v = gl.allocate_shared_memory(b_v0.dtype, [BV], shared_layout, b_v0) # BVxf32
            # Apply L2 normalization to q
            b_q = shared_b_q.load(slice1).to(gl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                q_norm_rev = 1 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
                tl.assume(q_norm_rev > 0)
                b_q = b_q * q_norm_rev # BKxf32
            b_q = b_q * scale # BKxf32
            # b_q = gl.convert_layout(b_q, layout=slice1)

            # Apply gating to hidden state: h *= exp(g)
            b_h *= gl.exp(b_g) # BKxBVxf32

            b_v = shared_b_v.load(slice4).to(gl.float32)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

            # Apply beta gating: v *= beta
            b_v *= b_beta # BVxf32

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma


            # Compute output: o = sum(h * q, dim=0)
            b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
            b_o = gl.convert_layout(b_o, layout=blocked1)
            gl.store(p_o, b_o)


@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
    #blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )


    o_k_blocked = i_k * BK + gl.arange(0, BK, layout=blocked2)
    o_v_blocked = i_v * BV + gl.arange(0, BV, layout=blocked1)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)

    p_q = q + (bos * H + i_h) * K + o_k_blocked
    p_k = k + (bos * H + i_h) * K + o_k_blocked
    p_v = v + (bos * HV + i_hv) * V + o_v_slice
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v_blocked

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k_blocked = o_k_blocked < K
    mask_v_blocked = o_v_blocked < V
    mask_k_slice = o_k_slice < K
    mask_v_slice = o_v_slice < V
    mask_h = mask_k_slice[:, None] & mask_v_slice[None, :]

    b_A_log = gl.load(p_A_log).to(gl.float32) # f32
    b_dt_bias = gl.load(p_dt_bias) # f32
    softplus_beta_inv = 1.0 / softplus_beta

    b_k = gl.load(p_k, mask=mask_k_blocked, other=0)  # BKxf32

    b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            b_h = gl.load(p_h0).to(gl.float32)  # BKxBVxf32


    b_a = gl.load(p_a) # f32
    b_b = gl.load(p_b) # f32

    b_v = gl.load(p_v, mask=mask_v_slice, other=0)  # BVxf32
    b_q = gl.load(p_q, mask=mask_k_blocked, other=0)  # BKxf32

    p_q += H * K
    p_k += H * K
    p_v += HV * V
    p_b += HV
    p_a += HV

    b_A = -gl.exp(b_A_log)

    # gl.assume(T>1)

    for _ in range(1, T):
        # Compute sigmoid gating
        # Load gating parameters
        b_a0 = gl.load(p_a) # f32
        b_b0 = gl.load(p_b) # f32

         # Load inputs
        b_k0 = gl.load(p_k)  # BKxf32
        b_v0 = gl.load(p_v)  # BVxf32
        b_q0 = gl.load(p_q)  # BKxf32

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_g = b_A * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

        b_k = gl.convert_layout(b_k, layout=slice1)
        b_k = b_k.to(gl.float32)

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)) # BKxf32



        # Apply gating to hidden state: h *= exp(g)
        b_h *= gl.exp(b_g) # BKxBVxf32

        # Delta rule: v -= sum(h * k, dim=0)
        b_v_sum = gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_b += HV
        p_a += HV

        b_v = b_v.to(gl.float32)
        b_v -= b_v_sum
        # Apply beta gating: v *= beta
        b_v *= b_beta # BVxf32

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

        b_q = gl.convert_layout(b_q, layout=slice1)
        b_q = b_q.to(gl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)) # BKxf32
        b_q = b_q * scale # BKxf32
        # Compute output: o = sum(h * q, dim=0)
        b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
        b_o = gl.convert_layout(b_o, layout=blocked1)
        gl.store(p_o, b_o)

        p_o += HV * V

        b_a = b_a0
        b_b = b_b0
        b_q = b_q0
        b_k = b_k0
        b_v = b_v0

    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
    beta_x = softplus_beta * x # f32
    # Apply softplus with numerical stability
    softplus_x = gl.where(
        beta_x <= softplus_threshold,
        softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
        x,
    )
    b_g = b_A * softplus_x # f32

    # Compute beta = sigmoid(b)
    b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

    b_k = gl.convert_layout(b_k, layout=slice1)
    b_k = b_k.to(gl.float32)
    # Apply L2 normalization if enabled
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)) # BKxf32

    # Apply gating to hidden state: h *= exp(g)
    b_h *= gl.exp(b_g) # BKxBVxf32

    b_v = b_v.to(gl.float32)
    # Delta rule: v -= sum(h * k, dim=0)
    b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

    # Apply beta gating: v *= beta
    b_v *= b_beta # BVxf32

    # Update hidden state: h += k[:, None] * v[None, :]
    b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

    b_q = gl.convert_layout(b_q, layout=slice1)
    b_q = b_q.to(gl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / (gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)) # BKxf32
    b_q = b_q * scale # BKxf32
    # Compute output: o = sum(h * q, dim=0)
    b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
    b_o = gl.convert_layout(b_o, layout=blocked1)
    gl.store(p_o, b_o)

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        # idx = gl.load(h0_indices + i_n)
        # if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel0(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
    #blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[32, 2],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )


    o_k_blocked = i_k * BK + gl.arange(0, BK, layout=blocked2)
    o_v_blocked = i_v * BV + gl.arange(0, BV, layout=blocked1)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)

    p_q = q + (bos * H + i_h) * K + o_k_blocked
    p_k = k + (bos * H + i_h) * K + o_k_blocked
    p_v = v + (bos * HV + i_hv) * V + o_v_slice
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v_blocked

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k_blocked = o_k_blocked < K
    mask_v_blocked = o_v_blocked < V
    mask_k_slice = o_k_slice < K
    mask_v_slice = o_v_slice < V
    mask_h = mask_k_slice[:, None] & mask_v_slice[None, :]

    b_A_log = gl.load(p_A_log).to(gl.float32) # f32
    b_dt_bias = gl.load(p_dt_bias) # f32
    softplus_beta_inv = 1.0 / softplus_beta

    b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            b_h = gl.load(p_h0, mask=mask_h, other=0).to(gl.float32)  # BKxBVxf32


    b_A = -gl.exp(b_A_log)
    b_dt_bias = b_dt_bias.to(gl.float32)

    # for t in range(0, T):
    for _ in range(0, T//4):
        # b_q = gl.load(p_q[:,None] + gl.arange(0, 4)[None,:] * H * K, mask=mask_k_blocked, other=0).to(gl.float32)  # BKx4xf32
        # b_q = gl.reshape(b_q, [BK, 2, 2],layout=blocked2d)
        # b_q01_0, b_q01_1 = gl.split(b_q)
        # b_q0_0, b_q1_0 = gl.split(b_q01_0)
        # b_q0_1, b_q1_1 = gl.split(b_q01_1)
        # b_q = (b_q0_0, b_q0_1, b_q1_0, b_q1_1)
        # b_k = gl.load(p_k + gl.arange(0, 4) * H * K, mask=mask_k_blocked, other=0).to(gl.float32)  # BKx4xf32
        # b_v = gl.load(p_v + gl.arange(0, 4) * HV * V, mask=mask_v_slice, other=0).to(gl.float32)  # BVx4xf32
        for i in gl.static_range(4):
            # Load inputs
            # b_q = b_q[i]
            b_q = gl.load(p_q + i * H * K, mask=mask_k_blocked, other=0).to(gl.float32)  # BKxf32
            b_k = gl.load(p_k + i * H * K, mask=mask_k_blocked, other=0).to(gl.float32)  # BKxf32
            b_v = gl.load(p_v + i * HV * V, mask=mask_v_slice, other=0).to(gl.float32)  # BVxf32
            # zeros_BK = gl.zeros([BK], dtype=p_q.dtype.element_ty, layout=blocked2)
            # zeros_BV = gl.zeros([BV], dtype=p_v.dtype.element_ty, layout=slice4)
            # b_q = gl.amd.cdna3.buffer_load(p_q, offsets=o_k_blocked + i * H * K, mask=mask_k_blocked, other=zeros_BK).to(gl.float32)  # BKxf32
            # b_k = gl.amd.cdna3.buffer_load(p_k, offsets=o_k_blocked + i * H * K, mask=mask_k_blocked, other=zeros_BK).to(gl.float32)  # BKxf32
            # b_v = gl.amd.cdna3.buffer_load(p_v, offsets=o_v_slice + i * HV * V, mask=mask_v_slice, other=zeros_BV).to(gl.float32)  # BVxf32


            # Compute sigmoid gating
            # Load gating parameters
            b_a = gl.load(p_a + i * HV).to(gl.float32) # f32
            b_b = gl.load(p_b + i * HV).to(gl.float32) # f32

            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a + b_dt_bias # f32
            beta_x = softplus_beta * x # f32
            # Apply softplus with numerical stability
            softplus_x = gl.where(
                beta_x <= softplus_threshold,
                softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
                x,
            )
            b_g = b_A * softplus_x # f32

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + gl.exp(-b_b)) # f32

            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)) # BKxf32
                b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)) # BKxf32

            b_q = gl.convert_layout(b_q, layout=slice1)
            b_k = gl.convert_layout(b_k, layout=slice1)

            b_q = b_q * scale # BKxf32

            # Apply gating to hidden state: h *= exp(g)
            b_h *= gl.exp(b_g) # BKxBVxf32

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

            # Apply beta gating: v *= beta
            b_v *= b_beta # BVxf32

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

            # Compute output: o = sum(h * q, dim=0)
            b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
            b_o = gl.convert_layout(b_o, layout=blocked1)
            gl.store(p_o + i * HV * V, b_o, mask=mask_v_blocked)
            # gl.amd.cdna3.buffer_store(b_o, p_o, offsets=o_v_blocked + i * HV * V, mask=mask_v_blocked)

            # Update pointers for next timestep


        p_q += H * K * 4
        p_k += H * K * 4
        p_o += HV * V * 4
        p_v += HV * V * 4
        p_b += HV * 4
        p_a += HV * 4

    for _ in range(0, T-T//4):
         # Load inputs
        b_q = gl.load(p_q, mask=mask_k_blocked, other=0).to(gl.float32)  # BKxf32
        b_k = gl.load(p_k, mask=mask_k_blocked, other=0).to(gl.float32)  # BKxf32
        b_v = gl.load(p_v, mask=mask_v_slice, other=0).to(gl.float32)  # BVxf32

        # Compute sigmoid gating
        # Load gating parameters
        b_a = gl.load(p_a).to(gl.float32) # f32
        b_b = gl.load(p_b).to(gl.float32) # f32

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_g = b_A * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + gl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)) # BKxf32
            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)) # BKxf32

        b_q = gl.convert_layout(b_q, layout=slice1)
        b_k = gl.convert_layout(b_k, layout=slice1)

        b_q = b_q * scale # BKxf32

        # Apply gating to hidden state: h *= exp(g)
        b_h *= gl.exp(b_g) # BKxBVxf32

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

        # Apply beta gating: v *= beta
        b_v *= b_beta # BVxf32

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

        # Compute output: o = sum(h * q, dim=0)
        b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
        b_o = gl.convert_layout(b_o, layout=blocked1)
        gl.store(p_o, b_o, mask=mask_v_blocked)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # BKxBVxf32

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # BVxf32
        b_b = tl.load(p_b).to(tl.float32) # f32

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32) # f32
        b_a = tl.load(p_a).to(tl.float32) # f32
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32) # f32

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6)) # BKxf32
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6)) # BKxf32

        b_q = b_q * scale # BKxf32

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g) # BKxBVxf32

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32

        # Apply beta gating: v *= beta
        b_v *= b_beta # BVxf32

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    gluon: bool = False,
    min_hdim: int = 8,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), min_hdim)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)

    if gluon:
        gluon_fused_sigmoid_gating_delta_rule_update_kernel1[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o, # write
            h0_source=initial_state_source, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
        )
    else:
        fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o, # write
            h0_source=initial_state_source, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
        )
    o = o.squeeze(0)
    return o
