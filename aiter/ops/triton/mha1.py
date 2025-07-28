#!/usr/bin/env python
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd with causal masking
2) Any sequence lengths without padding (currently fwd kernel only)
3) Support for different sequence lengths for q and k
4) Nested tensor API currently does not support dropout or bias.

Not currently supported:

1) Non power of two head dims

"""
from typing import Sequence, List
import os
import math
import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from torchao.float8.float8_tensor import Float8Tensor
from .triton_utils import modify_mem_barrier

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.float32
# Seed the RNG so we get reproducible results for testing.
philox_seed = 0x1BF52
philox_offset = 0x1D4B42
max_scale: tl.constexpr = 1e10

# # NOTE: This is a temporary solution to remove mem barriers between atomic add for MI300X.
# JITFunction.compiled_hook = modify_mem_barrier

AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE',
                          '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG',
                       '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF',
                      '0').lower() in ('1', 'true', 'yes')


def get_shape_from_layout(q,
                          k,
                          layout,
                          cu_seqlens_q=None,
                          cu_seqlens_k=None,
                          max_seqlen_q=None,
                          max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(
            cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(
            cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2]
    else:
        assert False, "Got unsupported layout."

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k


def get_strides_from_layout(q, k, v, o, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


def get_input_shapes():
    cases = [
        (max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128) for i in range(8, 18)
    ] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]
    return cases


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target(
    ).arch in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target(
    ).arch in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200",
               "gfx1201")


def get_e4m3_dtype():
    if is_hip():
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn


# def get_e5m2_dtype():
#     if is_hip():
#         return torch.float8_e5m2fnuz
#     else:
#         return torch.float8_e5m2


E4M3_MAX: tl.constexpr = torch.finfo(get_e4m3_dtype()).max
TL_FLOAT8_E4M3: tl.constexpr = tl.float8e4b8 if is_hip() else tl.float8e4nv


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n,
                                  stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n,
                             stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first,
            boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope,
                        seqlen_q,
                        seqlen_k,
                        offs_m,
                        offs_n,
                        transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:,
                                None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _attn_fwd_inner(
        acc, l_i, m_i, q, q_descale, k_descale, p_scale: tl.constexpr,
        USE_FP8: tl.constexpr, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk,
        stride_bn, start_m, actual_seqlen_k, actual_seqlen_q, dropout_p,
        philox_seed, batch_philox_offset, exp_scores_ptrs, block_min,
        block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope,
        score_ptrs, scores_scaled_shifted_ptrs, IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, OFFS_M: tl.constexpr, OFFS_N: tl.constexpr,
        PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
        ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
        ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr, RETURN_SCORES: tl.constexpr):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if USE_FP8:
            idx_block_n = tl.full([1], start_n // BLOCK_N, dtype=tl.int32)
            blk_k_descale = k_descale.gather(index=idx_block_n, axis=0)
        else:
            blk_k_descale = 1.

        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL,
                    actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                        ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M],
                                     actual_seqlen_k,
                                     dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk += tl.dot(q, k)

        if USE_FP8:
            qk_scaled = qk * q_descale * blk_k_descale * SM_SCALE
        else:
            qk_scaled = qk * SM_SCALE

        if RETURN_SCORES:
            score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(score_ptrs, qk_scaled, mask=score_mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0,
                                              BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q,
                           actual_seqlen_k)
            qk_scaled += bias

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q,
                                              actual_seqlen_k,
                                              global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        if RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            scores_scaled_shifted_mask = (
                OFFS_M[:, None] < actual_seqlen_q) & (
                    (start_n + tl.arange(0, BLOCK_N))[None, :]
                    < actual_seqlen_k)
            tl.store(scores_scaled_shifted_ptrs,
                     q_shifted,
                     mask=scores_scaled_shifted_mask)

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M,
                                BLOCK_N, actual_seqlen_k)
            if RETURN_SCORES:
                # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                    (start_n + tl.arange(0, BLOCK_N))[None, :]
                    < actual_seqlen_k)
                tl.store(exp_scores_ptrs,
                         tl.where(keep, p, -p),
                         mask=exp_score_mask)
            p = tl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(exp_scores_ptrs, p, mask=exp_score_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                        ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        if USE_FP8:
            p = tl.clamp(p * p_scale, -240., 240.)

        acc = tl.dot(p.to(v.dtype),
                     v,
                     acc=acc,
                     allow_tf32=False,
                     out_dtype=tl.float32)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += BLOCK_N
            scores_scaled_shifted_ptrs += BLOCK_N
            exp_scores_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_cdna_autotune_fwd_configs():
    return [
        triton.Config(
            {
                'BLOCK_M': 256,
                'BLOCK_N': 64,
                'waves_per_eu': 1,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 256,
                'BLOCK_N': 64,
                'waves_per_eu': 2,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 256,
                'BLOCK_N': 128,
                'waves_per_eu': 2,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 256,
                'BLOCK_N': 128,
                'waves_per_eu': 1,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'waves_per_eu': 2,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'waves_per_eu': 1,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'waves_per_eu': 3,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'waves_per_eu': 2,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=4),
        # Fall-back config.
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 64,
                'waves_per_eu': 2,
                'PRE_LOAD_V': False
            },
            num_stages=1,
            num_warps=4),
    ], [
        'IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K',
        'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK', "USE_FP8"
    ]


def get_autotune_fwd_configs():
    if AUTOTUNE:
        if is_cdna():
            return get_cdna_autotune_fwd_configs()
        else:
            raise ValueError("Unknown Device Type")
    else:
        return [
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    'waves_per_eu': 2,
                    "PRE_LOAD_V": False
                },
                num_stages=1,
                num_warps=4,
            ),
        ], [
            "IS_CAUSAL", "dropout_p", "MAX_SEQLENS_Q", "MAX_SEQLENS_K",
            "ACTUAL_BLOCK_DMODEL", "VARLEN", "HQ", "HK", "USE_FP8"
        ]


autotune_fwd_configs, autotune_fwd_keys = get_autotune_fwd_configs()


@triton.autotune(
    configs=autotune_fwd_configs,
    key=autotune_fwd_keys,
)
@triton.jit
def attn_fwd(Q, K, V, bias, p_scale: tl.constexpr, q_descale_ptr,
             k_descale_ptr, v_scale_ptr, o_scale: tl.constexpr, USE_FP8: tl.constexpr,
             SM_SCALE: tl.constexpr, LSE, Out, stride_qz, stride_qh, stride_qm,
             stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
             stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
             stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az,
             stride_ah, stride_sz, stride_sh, stride_sm, stride_sn,
             stride_lse_z, stride_lse_h, stride_lse_m, stride_qdescale_z,
             stride_qdescale_h, stride_qdescale_m, stride_kdescale_z,
             stride_kdescale_h, stride_kdescale_m, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, scores,
             scores_scaled_shifted, exp_scores, alibi_slopes, HQ: tl.constexpr,
             HK: tl.constexpr, num_block_n: tl.constexpr,
             ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr,
             IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
             PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_SCORES: tl.constexpr,
             USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    # we assume q and k has the same length
    if USE_FP8:
        k_descale_offset = k_descale_ptr + stride_kdescale_z * off_z + stride_kdescale_h * off_h_k + tl.arange(
            0, num_block_n)
        q_descale_offset = q_descale_ptr + stride_qdescale_z * off_z + stride_qdescale_h * off_h_q + start_m  #  + stride_qdescale_m * cu_seqlens_q

        k_descale = tl.load(k_descale_offset)
        q_descale = tl.load(q_descale_offset)
        v_scale = tl.load(v_scale_ptr)
    else:
        k_descale = 1.
        q_descale = 1.
        v_scale = 1.
        o_scale = 1.

    acc_descale = 1. / (p_scale * v_scale)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        # print("cu_seqlens_q_start:", cu_seqlens_q_start)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[
                None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
            l_ptrs = l_offset + offs_m * stride_lse_m

            l = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)

            # mask_m_offsets = start_m + tl.arange(0, BLOCK_M)
            # lse_mask = mask_m_offsets < causal_start_idx
            # softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    n_extra_tokens = 0
    # print("n_extra_tokens:", n_extra_tokens)
    # print("seqlen_k:", seqlen_k)
    # print("BLOCK_N:", BLOCK_N)
    # return
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:,
                               None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:,
                               None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:,
                               None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[
            None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        scores_offset = scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        score_ptrs = scores_offset + offs_m[:, None] * stride_sm + offs_n[
            None, :] * stride_sn

        scores_scaled_shifted_offset = scores_scaled_shifted + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        scores_scaled_shifted_ptrs = scores_scaled_shifted_offset + offs_m[:, None] * stride_sm + offs_n[
            None, :] * stride_sn

        exp_scores_offset = exp_scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        exp_scores_ptrs = exp_scores_offset + offs_m[:,
                                                     None] * stride_sm + offs_n[
                                                         None, :] * stride_sn
    else:
        score_ptrs = None
        scores_scaled_shifted_ptrs = None
        exp_scores_ptrs = None

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)

    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.

    # experimental
    log_p_scale = 5.480638923341991  # log_e_(240) for AMD platform
    # log_p_scale = 6.104793232414985  # log_e_(448) for NVIDIA platform

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.

    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_descale,
            k_descale,
            p_scale,
            USE_FP8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min,
            block_max,
            0,
            0,
            0,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            # IS_CAUSAL, ....
            False,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += n_full_blocks * BLOCK_N
            scores_scaled_shifted_ptrs += n_full_blocks * BLOCK_N
            exp_scores_ptrs += n_full_blocks * BLOCK_N

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_descale,
            k_descale,
            p_scale,
            USE_FP8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES)
    if USE_FP8:
        # FP8 -> FP32
        acc *= acc_descale

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k

    if USE_FP8:
        # FP32 -> FP8
        acc *= o_scale

    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ),
                                        causal_start_idx,
                                        dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:,
                                           None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.dtype))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    l_ptrs = l_offset + offs_m * stride_lse_m
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL:
        # zero out nans caused by -infs when doing causal
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ),
                           BLOCK_M - overflow_size,
                           dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse,
                 mask=l_ptrs_mask)  # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse)  # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:,
                               None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_ptrs_mask)


def attention_forward_triton_impl(q, k, v, o, p_scale, q_descale, k_descale,
                                  v_scale, o_scale, sm_scale, alibi_slopes,
                                  causal, bias, dropout_p, layout,
                                  cu_seqlens_q, cu_seqlens_k, max_seqlens_q,
                                  max_seqlens_k, return_scores, use_exp2,
                                  use_fp8):

    if DEBUG:
        print()
        print("attention_forward_triton_impl")
        print("use_fp8:", use_fp8)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("v_scale:", v_scale)
        print("o_scale:", o_scale)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(
        q, k, v, o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    grid = lambda META: (triton.cdiv(max_seqlens_q, META['BLOCK_M']), nheads_q,
                         batch)

    if return_scores:
        scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k),
                             device=q.device,
                             dtype=torch.float32)
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32)
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2),
                          scores.stride(3))
    else:
        scores = None
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q),
                                  device=q.device,
                                  dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q),
                                  device=q.device,
                                  dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # Seed the RNG so we get reproducible results for testing.
    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2),
                        bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_fp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = q_descale.stride(
            0), q_descale.stride(1), q_descale.stride(2)
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = k_descale.stride(
            0), k_descale.stride(1), k_descale.stride(2)

    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = None, None, None
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = None, None, None

    attn_fwd[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_descale,
        k_descale,
        v_scale,
        o_scale,
        use_fp8,
        sm_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        num_block_n=triton.cdiv(max_seqlens_k,
                                64),  # FIXME: BLOCK_N is 64 for test.
        BLOCK_DMODEL=padded_d_model,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=False if alibi_slopes is None else True,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores)
    return o, softmax_lse, exp_scores, grid, head_size, philox_seed, philox_offset, scores, scores_scaled_shifted


@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    DO_FP8,
    do_scale_ptr,
    Delta,
    USE_FP8: tl.constexpr,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    stride_doscalez,
    stride_doscaleh,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    load o, do and compute delta
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Compute batch and head indices
    off_z = pid_bh // HQ
    off_h = pid_bh % HQ

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # compute offsets
    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_oz + off_h * stride_oh + q_start * stride_om

    # compute pointers
    out_ptrs = o_offset + off_m[:,
                                None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[
        None, :] * stride_dok

    # load
    o = tl.load(out_ptrs).to(tl.float32)
    do = tl.load(do_ptrs).to(tl.float32)

    # compute delta
    # TODO: fine-grained scaling factor
    delta = tl.sum(o * do, axis=1)

    # write-back delta
    delta_offset = Delta + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta)

    if USE_FP8:
        do_scale = 240. / (tl.max(tl.abs(do)) + 1e-7)
        do_fp8 = tl.clamp(do * do_scale, -240., 240.).to(TL_FLOAT8_E4M3)

        do_fp8_offset = DO_FP8 + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
        do_fp8_ptrs = do_fp8_offset + off_m[:, None] * stride_dom + off_d[
            None, :] * stride_dok

        tl.store(do_fp8_ptrs, do_fp8)

        do_scale_offset = do_scale_ptr + off_z * stride_doscalez + off_h * stride_doscaleh + pid_m  #  + q_start * stride_om
        tl.store(do_scale_offset, 1. / do_scale)


@triton.jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    q_descale,
    k_descale,
    v_scale,
    p_scale: tl.constexpr,
    o_scale: tl.constexpr,
    do_descale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    q_offset,
    k_offset,
    v_offset,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    d_offset,
    l_offset,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    Z,
    HQ,
    HK,
    N_CTX_Q,
    N_CTX_K,
    off_h,
    off_z,
    off_hz,
    start_n,
    num_block_m,
    num_block_n,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_FP8: tl.constexpr,
    log_p_scale: tl.constexpr,
    E4M3_MAX: tl.constexpr,
):
    if CAUSAL:
        causal_boundary = start_n * BLOCK_N - BLOCK_M
        lo = (causal_boundary + 1) // BLOCK_M * BLOCK_M
    else:
        lo = 0
    # load scale
    # TODO: fine-grained scaling factor

    # initialize col and head offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    OFFSET_DMODEL = BLOCK_DMODEL // 2

    offs_d = tl.arange(0, BLOCK_DMODEL // 2)
    k_ptrs = k_offset + offs_n[None, :] * stride_kn + offs_d[:,
                                                             None] * stride_kk
    k0 = tl.load(k_ptrs)

    v_ptrs = v_offset + offs_n[None, :] * stride_vn + offs_d[:,
                                                             None] * stride_vk
    v0 = tl.load(v_ptrs)

    k_ptrs = k_ptrs + OFFSET_DMODEL * stride_kk
    k1 = tl.load(k_ptrs)

    v_ptrs = v_ptrs + OFFSET_DMODEL * stride_vk
    v1 = tl.load(v_ptrs)

    dk0 = tl.full([BLOCK_DMODEL // 2, BLOCK_N], 0, dtype=tl.float32)
    dk1 = tl.full([BLOCK_DMODEL // 2, BLOCK_N], 0, dtype=tl.float32)

    dv0 = tl.full([BLOCK_N, BLOCK_DMODEL // 2], 0, dtype=tl.float32)
    dv1 = tl.full([BLOCK_N, BLOCK_DMODEL // 2], 0, dtype=tl.float32)

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        can_skip_causal_block = start_m < causal_boundary
        offs_m = start_m + tl.arange(0, BLOCK_M)

        if USE_FP8:
            idx_block_m = tl.full([1], start_m // BLOCK_M, dtype=tl.int32)
            blk_q_descale = q_descale.gather(index=idx_block_m, axis=0)
            blk_do_descale = do_descale.gather(index=idx_block_m, axis=0)
        else:
            blk_q_descale = 1.
            blk_do_descale = 1.

        # recompute p = softmax(qk, dim=-1).T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # load q, k, v, do on-chip
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[
            None, :] * stride_qk
        q0 = tl.load(q_ptrs)
        qk = tl.dot(q0, k0, acc=qk, out_dtype=tl.float32)

        q_ptrs = q_ptrs + OFFSET_DMODEL * stride_qk
        q1 = tl.load(q_ptrs)
        qk = tl.dot(q1, k1, acc=qk, out_dtype=tl.float32)

        if USE_FP8:
            # can fuse with sm_scale
            qk_descale = blk_q_descale * k_descale
            qk = qk * qk_descale  # we fused sm_scale into blk_q_descale so we do not need one more mul here

        if CAUSAL:
            if not can_skip_causal_block:
                col_offset = N_CTX_Q - N_CTX_K
                causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
                qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = l_offset + offs_m * stride_deltam
        l_i = tl.load(l_ptrs)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] + log_p_scale * RCP_LN2)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None] + log_p_scale)

        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[
            None, :] * stride_qk
        do0 = tl.load(do_ptrs)

        do_ptrs = do_ptrs + OFFSET_DMODEL * stride_qk
        do1 = tl.load(do_ptrs)

        # compute dp
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do0, v0, acc=dp)
        dp = tl.dot(do1, v1, acc=dp)

        if USE_FP8:
            dp_descale = blk_do_descale / v_scale
            dp = dp * dp_descale

        d_ptrs = d_offset + offs_m * stride_deltam
        Di = tl.load(d_ptrs)  # D stored in fp32
        #   p_scale * (v_scale - o_scale) * sm_scale * do_scale

        # we can set ds a static scale here
        # this hacks but works well
        ds = ((p * (dp - Di[:, None])) * sm_scale / p_scale)
        p = tl.trans(p)
        if USE_FP8:
            ds_scale = E4M3_MAX / tl.max(
                tl.abs(ds) + 1e-7)
            ds = tl.clamp(ds * ds_scale, -E4M3_MAX, E4M3_MAX)
            p = tl.clamp(p, -E4M3_MAX, E4M3_MAX).to(k0.dtype)
        else:
            ds_scale = 1.

        # compute dv
        _dv0 = tl.dot(p,
                      do0,
                      out_dtype=tl.float32,
                      allow_tf32=False)
        if USE_FP8:
            dv0 = tl.fma(_dv0, blk_do_descale, dv0)
        else:
            dv0 += _dv0

        _dv1 = tl.dot(p,
                      do1,
                      out_dtype=tl.float32,
                      allow_tf32=False)
        if USE_FP8:
            dv1 = tl.fma(_dv1, blk_do_descale, dv1)
        else:
            dv1 += _dv1

        # compute dk = dot(ds.T, q)
        ds = ds.to(q0.dtype)
        _dk0 = tl.dot(tl.trans(q0), ds) / ds_scale
        if USE_FP8:
            dk0 = tl.fma(_dk0, blk_q_descale, dk0)
        else:
            dk0 += _dk0

        _dk1 = tl.dot(tl.trans(q1), ds) / ds_scale
        if USE_FP8:
            dk1 = tl.fma(_dk1, blk_q_descale, dk1)
        else:
            dk1 += _dk1

        # compute dq
        if SEQUENCE_PARALLEL:
            dq = tl.dot(k0, tl.trans(ds), allow_tf32=False)
            if USE_FP8:
                # dq = dq * ds_s_descale
                dq_descale = k_descale / ds_scale  # ds_scale # 1. / k_scale
                dq = dq * dq_descale
            dq_ptrs = dq_offset + offs_m[
                None, :] * stride_qm + offs_d[:, None] * stride_qk
            tl.atomic_add(dq_ptrs, dq, sem='relaxed')

            dq = tl.dot(k1, tl.trans(ds), allow_tf32=False)
            if USE_FP8:
                # dq = dq * ds_s_descale
                dq_descale = k_descale / ds_scale  # ds_scale # 1. / k_scale
                dq = dq * dq_descale
            dq_ptrs = dq_ptrs + OFFSET_DMODEL * stride_qk
            tl.atomic_add(dq_ptrs, dq, sem='relaxed')

        else:
            dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[
                None, :] * stride_qk
            dq = tl.load(dq_ptrs)
            _dq = tl.dot(k0, tl.trans(ds), allow_tf32=False)
            if USE_FP8:
                # _dq = _dq * ds_s_descale
                dq_descale = k_descale / ds_scale  # 1. / k_scale
                # TODO: we can use mfa here
                _dq = _dq * dq_descale

            dq += _dq
            tl.store(dq_ptrs, dq)

            dq_ptrs = dq_ptrs + OFFSET_DMODEL * stride_qk
            dq = tl.load(dq_ptrs)
            _dq = tl.dot(k1, tl.trans(ds), allow_tf32=False)
            if USE_FP8:
                # _dq = _dq * ds_s_descale
                dq_descale = k_descale / ds_scale  # 1. / k_scale
                # TODO: we can use mfa here
                _dq = _dq * dq_descale

            dq += _dq
            tl.store(dq_ptrs, dq)

    if USE_FP8:
        dv_descale = 1. / (p_scale)
        dv0 *= dv_descale
        dv1 *= dv_descale

    # write-back
    dk_ptrs = dk_offset + offs_n[
        None, :] * stride_kn + offs_d[:, None] * stride_kk
    tl.atomic_add(dk_ptrs, dk0)
    dk_ptrs = dk_ptrs + OFFSET_DMODEL * stride_kk
    tl.atomic_add(dk_ptrs, dk1)

    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[
        None, :] * stride_vk
    tl.atomic_add(dv_ptrs, dv0)
    dv_ptrs = dv_ptrs + OFFSET_DMODEL * stride_vk
    tl.atomic_add(dv_ptrs, dv1)


def get_cdna_autotune_bwd_configs():
    return [
        # fp8 kernel enable us to load bigger block
        # according to: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-autotunable-kernel-config
        # we set matrix_instr_nonkdim 16 here as constant on MI300X
        triton.Config(
            {
                'BLOCK_N': 128,
                'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_N': 128,
                'waves_per_eu': 1,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_N': 64,
                'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_N': 64,
                'waves_per_eu': 1,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_N': 32,
                'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
        # Fall-back config.
        triton.Config(
            {
                'BLOCK_N': 32,
                'waves_per_eu': 1,
                'matrix_instr_nonkdim': 16
            },
            num_stages=1,
            num_warps=4),
    ], [
        "BLOCK_DMODEL",
        "ACTUAL_BLOCK_DMODEL",
    ]


def get_autotune_bwd_configs():
    if AUTOTUNE:
        if is_cdna():
            return get_cdna_autotune_bwd_configs()
        else:
            raise ValueError("Unknown Device Type")
    else:
        return [
            triton.Config(
                {
                    "BLOCK_N": 64,
                },
                num_stages=1,
                num_warps=4,
            ),
        ], [
            "BLOCK_DMODEL", "ACTUAL_BLOCK_DMODEL", "SEQUENCE_PARALLEL",
            "CAUSAL", "USE_FP8"
        ]


autotune_bwd_configs, autotune_bwd_keys = get_autotune_bwd_configs()


@triton.autotune(
    configs=autotune_bwd_configs,
    key=autotune_bwd_keys,
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    q_scale_ptr,
    k_descale_ptr,
    v_scale_ptr,
    p_scale: tl.constexpr,
    o_scale: tl.constexpr,
    do_descale_ptr,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    stride_doscalez,
    stride_doscaleh,
    stride_doscalem,
    stride_qscalez,
    stride_qscaleh,
    stride_qscalem,
    stride_kscalez,
    stride_kscaleh,
    stride_kscalem,
    Z,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FP8: tl.constexpr,
    log_p_scale: tl.constexpr,
    E4M3_MAX: tl.constexpr,
):
    # program ids
    off_hz = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1).to(tl.int64)
    off_z = off_hz // HQ
    off_h_q = off_hz % HQ

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    l_offset = L + off_z * stride_deltaz + off_h_q * stride_deltah + q_start * stride_deltam
    d_offset = D + off_z * stride_deltaz + off_h_q * stride_deltah + q_start * stride_deltam

    # output tensor offsets
    # sume dk and dv
    dk_offset = DK + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    dq_offset = DQ + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm

    if USE_FP8:
        # we keep v in per-tensor scaling
        # while q, k, do in per-block scaling
        v_scale = tl.load(v_scale_ptr)  # + tl.arange(0, num_block_n)

        # test here
        do_descale_offset = do_descale_ptr + off_z * stride_doscalez + off_h_q * stride_doscaleh + tl.arange(
            0, num_block_m)  #  + q_start * stride_qm
        do_descale = tl.load(do_descale_offset)

        q_descale_offset = q_scale_ptr + off_z * stride_qscalez + off_h_q * stride_qscaleh + tl.arange(
            0, num_block_m)  #  + q_start * stride_qm
        q_descale = tl.load(q_descale_offset)

        k_descale_offset = k_descale_ptr + off_z * stride_kscalez + off_h_k * stride_kscaleh + start_n  #  + q_start * stride_qm
        k_descale = tl.load(k_descale_offset)

    else:
        q_descale = 1.
        k_descale = 1.
        v_scale = 1.
        o_scale = 1.
        do_descale = 1.

    # inner loop
    if SEQUENCE_PARALLEL:
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            q_descale,
            k_descale,
            v_scale,
            p_scale,
            o_scale,
            do_descale,
            Out,
            DO,
            DQ,
            DK,
            DV,
            L,
            D,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            d_offset,
            l_offset,
            stride_dq_all,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_deltaz,
            stride_deltah,
            stride_deltam,
            Z,
            HQ,
            HK,
            N_CTX_Q,
            N_CTX_K,
            off_h_q,
            off_z,
            off_hz,
            start_n,
            num_block_m,
            None,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2,
            USE_FP8=USE_FP8,
            log_p_scale=log_p_scale,
            E4M3_MAX=E4M3_MAX,
        )
    else:
        num_block_n = tl.cdiv(N_CTX_K, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                q_descale,
                k_descale,
                v_scale,
                p_scale,
                o_scale,
                do_descale,
                Out,
                DO,
                DQ,
                DK,
                DV,
                L,
                D,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                d_offset,
                l_offset,
                stride_dq_all,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                stride_deltaz,
                stride_deltah,
                stride_deltam,
                Z,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                off_h_q,
                off_z,
                off_hz,
                start_n,
                num_block_m,
                None,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2,
                USE_FP8=USE_FP8,
                log_p_scale=log_p_scale,
                E4M3_MAX=E4M3_MAX,
            )


@torch.library.custom_op("amd::attention_block_backward_triton_impl",
                         mutates_args=())
def attention_block_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    o_scale: float,
    do_scale: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    sm_scale: float,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    layout: str,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    use_exp2: bool,
    use_fp8: bool,
    sequence_parallel: bool = True,
) -> List[torch.Tensor]:
    if DEBUG:
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("sequence_parallel:", sequence_parallel)

    # make contigious
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(
        q, k, v, o, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    batch_headsize = batch * nheads_q
    is_varlen = layout == "thd"

    # get closest power of 2 over or equal to 32.
    # this assumption can help reduce vgpr usage
    assert head_size > 32
    assert head_size % 2 == 0
    # padded_d_model = 1 << (head_size - 1).bit_length()
    # padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = head_size
    # ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()
    # NOTE: we might need to copy the output tensor if they are not continuous or have other issues
    copy_back = {"dq": False, "dk": False, "dv": False}

    # deal with dq
    if dq is None:
        dq = torch.zeros(q.shape, device=q.device, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if (not dq.is_contiguous()):
            dq = dq.contiguous()
            copy_back["dq"] = True

        dq.zero_()
    stride_dq_all = dq.stride()[0]

    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if (not dk.is_contiguous()):
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True

        if (not dv.is_contiguous()):
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()
    # init delta
    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_deltam, stride_deltah = delta.stride()
        stride_deltaz = 0
    else:
        stride_deltaz, stride_deltah, stride_deltam = delta.stride()

    # we set BLOCK_M out of kernel function, caz we need BLOCK_M to be same in both _bwd_preprocess_use_o and _bwd_kernel
    BLOCK_M = 64

    if use_fp8:
        do_fp8 = torch.empty(do.shape, dtype=get_e4m3_dtype(), device=q.device)
        _shape = (batch, nheads_q, triton.cdiv(max_seqlen_q, BLOCK_M))
        do_scale = torch.empty(_shape, dtype=torch.float32, device=q.device)
        stride_descalez, stride_descaleh, stride_descalem = do_scale.stride()
        stride_qscalez, stride_qscaleh, stride_qscalem = q_scale.stride()
        stride_kscalez, stride_kscaleh, stride_kscalem = k_scale.stride()

    else:
        do_fp8 = None
        do_scale = None
        stride_descalez, stride_descaleh, stride_descalem = None, None, None
        stride_qscalez, stride_qscaleh, stride_qscalem = None, None, None
        stride_kscalez, stride_kscaleh, stride_kscalem = None, None, None

    grid_prebwd = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_headsize)
    _bwd_preprocess_use_o[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        delta,
        use_fp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_deltaz,
        stride_deltah,
        stride_deltam,
        stride_descalez,
        stride_descaleh,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_M=BLOCK_M,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:", stride_qz,
              stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:", stride_kz,
              stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:", stride_vz,
              stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL:", BLOCK_DMODEL)
        print("SEQUENCE_PARALLEL:", sequence_parallel)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    log_p_scale = math.log(p_scale)
    num_block_m = triton.cdiv(max_seqlen_q, BLOCK_M)
    grid_bwd = lambda META: (
        batch_headsize,
        triton.cdiv(max_seqlen_k, META['BLOCK_N']) if sequence_parallel else 1,
    )
    _bwd_kernel[grid_bwd](
        q,
        k,
        v,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        o_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_deltaz,
        stride_deltah,
        stride_deltam,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        E4M3_MAX=E4M3_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("delta:", delta, delta.shape)

    if DEBUG:
        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("delta:", delta, delta.shape)
        print("copy_back:", copy_back)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return [
        dq.to(fwd_torch_dtype),
        dk.to(fwd_torch_dtype),
        dv.to(fwd_torch_dtype)
    ]


def block_scaling_node(tensor, BLOCK_M=64, float8_dtype=get_e4m3_dtype()):
    # this funciton help scale tensor in per-block mode
    # block size: [BLOCK_M, D]
    # [B, L, H, D]
    # scale should be [B, H, L//BLOCK_M]
    tensor = tensor.permute(0, 2, 1, 3)  #[B, H, L, D]
    B, H, L, D = tensor.shape
    tensor = tensor.reshape(B, H, L // BLOCK_M, BLOCK_M,
                            D).reshape(B, H, L // BLOCK_M,
                                        BLOCK_M * D)
    MAX_E4M3 = torch.finfo(float8_dtype).max
    scale = MAX_E4M3 / tensor.abs().max(dim=-1)[0]
    tensor = tensor * scale.reshape(scale.shape + (1, ))
    tensor = tensor.clamp(-MAX_E4M3, MAX_E4M3)
    tensor = tensor.to(float8_dtype)
    tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1,
                                                3).contiguous()
    # [B, L, H, D]
    return tensor, 1. / scale.to(torch.float32)

@torch.library.custom_op("amd::triton_attention_block", mutates_args=())
def triton_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    bias: torch.Tensor | None,
    sm_scale: float,
    dropout_p: float,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool,
    return_scores: bool,
    use_exp2: bool,
    layout: str,
    fp8_scales: Sequence[torch.Tensor | None],
    q_hp: torch.Tensor | None,
    k_hp: torch.Tensor | None,
    v_hp: torch.Tensor | None,
) -> List[torch.Tensor]:
    use_fp8 = None
    (q_scale, k_scale, v_scale) = fp8_scales
    if v_scale is not None:
        # print("Enabling FP8")
        use_fp8 = True

        float8_fw = get_e4m3_dtype()
        p_scale = torch.finfo(float8_fw).max
        o_scale = 1.

        # float32 * scale
        def check_and_convert(t, scale):
            finfo = torch.finfo(float8_fw)
            return ((t * scale).clamp(min=finfo.min, max=finfo.max).to(
                dtype=float8_fw) if t.dtype != float8_fw else t)

        # convert to fp8
        if q.dtype != float8_fw:
            q, q_scale = block_scaling_node(q)
        if k.dtype != float8_fw:
            k, k_scale = block_scaling_node(k)
        v = check_and_convert(v, v_scale)
    else:
        use_fp8 = False
        q_scale = torch.tensor([1.], device=q.device)
        k_scale = torch.tensor([1.], device=q.device)
        v_scale = torch.tensor([1.], device=q.device)
        o_scale = 1.
        p_scale = 1.

    if o is None:
        o = torch.empty_like(q,
                             dtype=fwd_torch_dtype if use_fp8 else v.dtype,
                             requires_grad=True)

    (output, softmax_lse, exp_scores, grid, head_size, philox_seed,
        philox_offset, _, _) = attention_forward_triton_impl(
            q,
            k,
            v,
            o,
            p_scale,
            q_scale,
            k_scale,
            v_scale,
            o_scale,
            sm_scale,
            alibi_slopes,
            causal,
            bias,
            dropout_p,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlens_q,
            max_seqlens_k,
            return_scores,
            use_exp2,
            use_fp8)

    return [output, softmax_lse, exp_scores]


def setup_context(ctx, inputs, output):
    q, k, v, _, alibi_slopes, bias, sm_scale, dropout_p, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k, causal, _, use_exp2, layout, fp8_scales, _, _, _ = inputs
    o, softmax_lse, _ = output
    (q_scale, k_scale, v_scale) = fp8_scales
    ctx.use_fp8 = v_scale is not None
    q_k_as_fp8 = ctx.use_fp8
    if ctx.use_fp8:
        float8_fw = get_e4m3_dtype()

        def check_and_convert(t, scale):
            finfo = torch.finfo(float8_fw)
            return ((t * scale).clamp(min=finfo.min, max=finfo.max).to(
                dtype=float8_fw) if t.dtype != float8_fw else t)

        if q.dtype != float8_fw:
            q_k_as_fp8 = False
            q, q_scale = block_scaling_node(q)
        if k.dtype != float8_fw:
            q_k_as_fp8 = False
            k, k_scale = block_scaling_node(k)
        v = check_and_convert(v, v_scale)

    ctx.save_for_backward(q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale,
                          k_scale, v_scale)
    ctx.q_k_as_fp8 = q_k_as_fp8
    ctx.sm_scale = sm_scale
    ctx.causal = causal
    ctx.dropout_p = dropout_p
    ctx.layout = layout
    ctx.use_exp2 = use_exp2
    ctx.cu_seqlens_q = cu_seqlens_q
    ctx.cu_seqlens_k = cu_seqlens_k
    ctx.max_seqlens_q = max_seqlens_q
    ctx.max_seqlens_k = max_seqlens_k


def backward(ctx, grad_output):
    do = grad_output[0]
    q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale = ctx.saved_tensors
    assert bias is None, "Currently bias is not supported by fa backward function."
    assert do.dtype is torch.bfloat16, f"do should be bfloat16 but get {do.dtype}"

    if ctx.use_fp8:
        float8_fw = get_e4m3_dtype()

        do_scale = torch.tensor([1.], device=do.device)
        p_scale = torch.finfo(float8_fw).max
        o_scale = 1.

        # expect do in high-precision format
        assert not isinstance(do, Float8Tensor)

    else:
        do_scale = torch.tensor([1.], device=do.device)
        q_scale = torch.tensor([1.], device=q.device)
        k_scale = torch.tensor([1.], device=k.device)
        v_scale = torch.tensor([1.], device=v.device)
        o_scale = 1.
        p_scale = 1.

    # experiment
    # here we introduce a prescale method:
    # we can prescale the do to a suitable scale for numerical stability
    # and scale the din back out of the fp8 op
    # NOTE: this cause heavy overhead
    _rescale = 0.1 / do.abs().max()
    do = do * _rescale
    #_rescale = 1.

    dq, dk, dv = attention_block_backward_triton_impl(
        do,
        q,
        k,
        v,
        o,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        o_scale,
        do_scale,
        softmax_lse,
        None,
        None,
        None,
        ctx.sm_scale,
        alibi_slopes,
        ctx.causal,
        ctx.layout,
        ctx.cu_seqlens_q,
        ctx.cu_seqlens_k,
        ctx.max_seqlens_q,
        ctx.max_seqlens_k,
        ctx.use_exp2,
        ctx.use_fp8,
        sequence_parallel=True,
    )
    scale_grads = [None, None, None] if ctx.q_k_as_fp8 else None
    # scale_grads = None
    return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, scale_grads, dq / _rescale, dk / _rescale, dv / _rescale


@triton_attention_block.register_fake
def _fake_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    bias: torch.Tensor | None,
    sm_scale: float,
    dropout_p: float,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool,
    return_scores: bool,
    use_exp2: bool,
    layout: str,
    fp8_scales: Sequence[torch.Tensor | None],
    q_hp: torch.Tensor | None,
    k_hp: torch.Tensor | None,
    v_hp: torch.Tensor | None,
) -> List[torch.Tensor]:
    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)

    (q_scale, k_scale, v_scale) = fp8_scales
    use_fp8 = v_scale is not None
    is_varlen = layout == "thd"

    if o is None:
        o = torch.empty_like(q,
                             dtype=fwd_torch_dtype if use_fp8 else v.dtype,
                             requires_grad=True)

    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q),
                                  device=q.device,
                                  dtype=torch.float32)
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q),
                                  device=q.device,
                                  dtype=torch.float32)

    if return_scores:
        exp_scores = torch.empty(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    return [o, softmax_lse, exp_scores]


@attention_block_backward_triton_impl.register_fake
def _fake_attention_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: torch.Tensor,
    o_scale: torch.Tensor,
    do_scale: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    sm_scale: float,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    layout: str,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    use_exp2: bool,
    use_fp8: bool,
    sequence_parallel: bool = True,
) -> List[torch.Tensor]:
    return [
        torch.empty_like(q, dtype=fwd_torch_dtype),
        torch.empty_like(k, dtype=fwd_torch_dtype),
        torch.empty_like(v, dtype=fwd_torch_dtype),
    ]


triton_attention_block.register_autograd(backward, setup_context=setup_context)
