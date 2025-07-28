import triton
import torch
import triton.language as tl
import pytest
import logging
from typing import Any, Dict, Optional
import numpy as np


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE =  False
ATOL_fp8 = 2.5e-1
RTOL_fp8 =  2.5e-1

from aiter.ops.triton.mha import flash_attn_func, flash_attn_fp8_func, flash_attn_varlen_func, flash_attn_varlen_fp8_func
from aiter.test_mha_common import construct_local_mask, attention_ref, generate_random_padding_mask, generate_qkv, pad_rearrange_dropout_mask_hts_to_bhss 



def pad_rearrange_dropout_mask(S_dmask, cu_seqlens_q, cu_seqlens_k,  max_seqlen_q, max_seqlen_k, seqlen_q, seqlen_k, num_q_heads):
    batch_size = cu_seqlens_q.numel() - 1
    
    padded_dropout_mask = torch.ones((batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda")
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
                padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[b, h, : ,:]
    
    
    return padded_dropout_mask



def fp8_assert_close(tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))
    
    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)
    
    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100
    
    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True
    
    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()
    
    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)
    
    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)
    
    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()
    
    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )

from aiter.ops.triton.mha import _flash_attn_forward, cast_to_fp8, cast_per_block_to_fp8, cast_per_token_to_fp8, cast_scores_per_block_to_fp8, Tuple
from einops import rearrange, repeat
import os
os.environ["TRITON_CACHE_DIR"] = "/home/sijieli2/triton_cache"
torch.cuda.set_device(0)
torch.cuda.manual_seed(0)
import aiter.ops.triton.mha as amha
descale_types = (
    # amha.FP8_DESCALE_None,
    # amha.FP8_DESCALE_QhKVh,
    amha.FP8_DESCALE_QtKVb,
    # amha.FP8_DESCALE_QtKVt,
    # amha.FP8_DESCALE_QbKVb,
)

def _multi_head_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    dropout_mask,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    FP8_DESCALE_TYPE = amha.FP8_DESCALE_None,
    SCALE_BLK_M = 32, SCALE_BLK_N = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # print(f"@@@@@@ {q.shape=}, {k.shape=}, {v.shape=}")
    # print(f"@@@@@@ {descale_q.shape=}, {descale_k.shape=}, {descale_v.shape=}")
    # batch, seqlen_q, nheads_q, headdim = q.shape
    # nheads_k = k.shape[2]

    # out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    # for b in range(batch):
    #     for n in range(nheads_q):
    #         n1 = n *nheads_k // nheads_q
    #         m_i = torch.full((SCALE_BLK_M), float("-inf"), dtype=tl.float32)
    #         l_i = torch.full((SCALE_BLK_M), 1.0, dtype=tl.float32)
    #         acc = torch.empty((SCALE_BLK_M, headdim), dtype=torch.float32, device=q.device)
    #         for sq in range(seqlen_q//SCALE_BLK_M):
    #             start_m = sq * SCALE_BLK_M
    #             end_m = start_m + SCALE_BLK_M
    #             q = q[b, start_m:end_m, n, :]
    #             descale_q = descale_q[b, sq, n, 0]
    #             for sk in range(seqlen_k//SCALE_BLK_N):
    #                 start_n = sk * SCALE_BLK_N
    #                 end_n = start_n + SCALE_BLK_N
    #                 k = q[b, start_n:end_n, n1, :]
    #                 v = v[b, start_n:end_n, n1, :]
    #                 descale_k = descale_k[b, sk, n, 0]
    #                 descale_v = descale_v[b, sk, n, 0]
                    
    #                 qk = torch.matmul(q, k)
    #                 qk_scaled = qk * descale_q * descale_k * softmax_scale
    #                 m_ij = torch.maximum(m_i, torch.max(qk_scaled, 1))
    #                 q_shifted = qk_scaled - m_ij[:, None]
    #                 p = tl.math.exp(q_shifted)
    #                 l_ij = tl.sum(p, 1)

    #                 m_diff = m_i - m_ij
    #                 alpha = tl.math.exp2(m_diff * RCP_LN2)
    #                 acc = acc * alpha[:, None]
    #                 l_i = l_i * alpha + l_ij
    #                 m_i = m_ij

    #                 acc += torch.matmul(p.to(v.dtype), v) * descale_v
    # for b in range(batch):
    #     for n0 in range(nheads_q):
    #         n1 = n0 *nheads_k // nheads_q
    #         q = q[b, :, n0, :]
    #         k = k[b, :, n1, :]
    #         v = v[b, :, n1, :]
    #         descale_q = descale_q[b, n0, :, :]  # s d
    #         descale_k = descale_k[b, n1, :, :]  # d s
    #         descale_v = descale_v[b, :, n1, :]  # s d
    #         qk = torch.matmul(q, k.transpose(1,0)).float() * descale_q * descale_k * softmax_scale

    #         p = torch.softmax(qk).to(torch.float8_e4m3fnuz)

    #         out[b, :, n0, :] = torch.matmul(p, v) * descale_v

    # return out,

    window_size = (window_size_left, window_size_right)
    if causal:
        window_size = (window_size_left, 0)
    dtype_og = q.dtype
    q,k,v = q.float(), k.float(), v.float()

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1] ** (-0.5)

    scores = torch.einsum("bthd,bshd->bhts", q, k) 
    print(f"@@@@@@ {scores.shape=}, {descale_q.shape=}, {descale_k.shape=}")
    scores = scores * descale_q * descale_k * d

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            None,
            None,
            q.device,
            key_leftpad=None,
        )
        scores.masked_fill_(local_mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)

    if dropout_p != 0.:
        dropout_scaling = 1.0 / (1 - dropout_p)
        v = v * dropout_scaling
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    # output = torch.einsum("bhts,bshd->bthd", attention_drop, v) * descale_v
    attention_descale, attention_scale = cast_scores_per_block_to_fp8(attention_drop, torch.float8_e4m3fnuz, "bhts", SCALE_BLK_M) 
    attention_scale = rearrange(repeat(attention_scale, "b h t s -> b h (t rT) s", rT=SCALE_BLK_M),
                                "b h t s -> b t h s")
    attention_descale = attention_descale.float()
    output = torch.einsum("bhts,bshd->bthd", attention_descale, v) * attention_scale * descale_v
    
    return output, attention
    # return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

@pytest.mark.parametrize('BATCH', [128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(256,256)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8])
@pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.0, False, False)])
@pytest.mark.parametrize('CAUSAL', [(False)])
# @pytest.mark.parametrize('BATCH', [1,4,57,128])
# @pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
# @pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
# @pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
# @pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.2, True, True), (0.0, False, False)])
# @pytest.mark.parametrize('CAUSAL', [(True), (False)])
@pytest.mark.parametrize('FP8_DESCALE_TYPE', [*descale_types])
@pytest.mark.parametrize('SCALE_BLK_M', [16])
@pytest.mark.parametrize('SCALE_BLK_N', [128])
def test_mha(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, RETURN_LSE: bool, RETURN_SOFTMAX: bool, CAUSAL: bool, FP8_DESCALE_TYPE: int, SCALE_BLK_M: int, SCALE_BLK_N: int, dtype=torch.float16):
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
                
    dropout_mask = None
    fp8_dtype = torch.float8_e4m3fnuz
    if FP8_DESCALE_TYPE & amha.FP8_DESCALE_Qhead:
        q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, "bshd")
    elif FP8_DESCALE_TYPE & amha.FP8_DESCALE_Qblock:
        q_fp8, descale_q = cast_per_block_to_fp8(q, fp8_dtype, "bshd", SCALE_BLK_M)
    elif FP8_DESCALE_TYPE & amha.FP8_DESCALE_Qtoken:
        q_fp8, descale_q = cast_per_token_to_fp8(q, fp8_dtype, "bshd")
    
    if FP8_DESCALE_TYPE & amha.FP8_DESCALE_KVhead:
        k_fp8, descale_k = cast_to_fp8(k, fp8_dtype, "bshd")
        v_fp8, descale_v = cast_to_fp8(v, fp8_dtype, "bshd")
    elif FP8_DESCALE_TYPE & amha.FP8_DESCALE_KVblock:
        k_fp8, descale_k = cast_per_block_to_fp8(k, fp8_dtype, "bshd", SCALE_BLK_N)
        v_fp8, descale_v = cast_per_block_to_fp8(v, fp8_dtype, "bshd", SCALE_BLK_N)
    elif FP8_DESCALE_TYPE & amha.FP8_DESCALE_Qtoken:
        k_fp8, descale_k = cast_per_token_to_fp8(k, fp8_dtype, "bshd")
        v_fp8, descale_v = cast_per_token_to_fp8(v, fp8_dtype, "bshd")
    descale_q = torch.ones_like(descale_q)
    descale_k = torch.ones_like(descale_k)
    descale_v = torch.ones_like(descale_v)
    if FP8_DESCALE_TYPE:
        # triton_out = flash_attn_fp8_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX, fp8_descale_type=FP8_DESCALE_TYPE, SCALE_BLK_M=SCALE_BLK_M, SCALE_BLK_N=SCALE_BLK_N)
        triton_out = _flash_attn_forward(
            q_fp8,
            k_fp8,
            v_fp8,
            DROPOUT,
            q.shape[-1] ** (-0.5),
            causal=CAUSAL,
            window_size_left=-1,
            window_size_right=-1,
            alibi_slopes=None,
            return_lse=False,
            return_softmax=False and dropout_p > 0,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            FP8_DESCALE_TYPE=FP8_DESCALE_TYPE,
            SCALE_BLK_M=SCALE_BLK_M,
            SCALE_BLK_N=SCALE_BLK_N
        )
    else:
        triton_out = flash_attn_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}")

    triton_out = triton_out[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    print(f"@@@@@@ {descale_q.shape=}, {descale_k.shape=}, {descale_v.shape=}")
    
    repeat_heads = descale_q.shape[2]//descale_k.shape[2]

    # descale_q = rearrange(repeat(descale_q, "b s h d -> b (s rS) h d", rS=SCALE_BLK_M),
    #                       "b s h d -> b h s d")
    descale_q = rearrange(descale_q, "b s h d -> b h s d")

    descale_k = rearrange(repeat(descale_k, "b s h d -> b (s rS) (h rH) d", rS=SCALE_BLK_N, rH=repeat_heads),
                          "b s h d -> b h d s")
    descale_v = repeat(descale_v, "b s h d -> b (s rS) (h rH) d ", rS=SCALE_BLK_N, rH=repeat_heads)

    
    torch_out = _multi_head_attn_forward(
        q_fp8, k_fp8, v_fp8, dropout_p=DROPOUT, softmax_scale=q.shape[-1] ** (-0.5), dropout_mask=dropout_mask, causal=CAUSAL,
        window_size_left=-1, window_size_right=-1, alibi_slopes=None, return_lse=None, return_softmax=None,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1], cu_seqlens_q=None, cu_seqlens_k=None,
        descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, FP8_DESCALE_TYPE=FP8_DESCALE_TYPE,
        SCALE_BLK_M=SCALE_BLK_M, SCALE_BLK_N=SCALE_BLK_N
        )
    torch_out, attention_scores = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}")

    triton_out, torch_out = triton_out[0,200:204,28,:], torch_out[0,200:204,28,:]
    with open("/home/sijieli2/mha_test.out", "w") as f:
        print(f"{triton_out=}\n{torch_out=}", file=f)

    torch.testing.assert_close(triton_out, torch_out, atol=2.5e-1, rtol=2.5e-1)

@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.0, False, False), (0.2, True, True)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
@pytest.mark.parametrize('CAUSAL', [(True), (False)])
@pytest.mark.parametrize('FP8',[(False), (True)])
def test_mha_varlen(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, RETURN_LSE: bool, RETURN_SOFTMAX: bool, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.set_printoptions(threshold=10000)
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, "cuda", mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    if DEBUG_MODE:
        print(f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}")
        print(f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}")

        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"q_unpad.shape={q_unpad.shape} q_unpad={q_unpad}")
        print(f"k_unpad.shape={k_unpad.shape} k_unpad={k_unpad}")
        print(f"v_unpad.shape={v_unpad.shape} v_unpad={v_unpad}")
        print(f"max_seqlens_q={max_seqlen_q }")
        print(f"max_seqlens_k={max_seqlen_k }")
        print(f"cu_seqlens_q={cu_seqlens_q }")
        print(f"cu_seqlens_k={cu_seqlens_k }")
    if FP8:
        triton_out = flash_attn_varlen_fp8_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT,causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)
    else:
        triton_out = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT,causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    dropout_mask = None
    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        dropout_mask = pad_rearrange_dropout_mask(dropout_mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS)
        dropout_mask = dropout_mask > 0
        if DEBUG_MODE:
            #print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}")

    triton_out = output_pad_fn(triton_out[0])
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(q, k, v, query_padding_mask=query_padding_mask, key_padding_mask=key_padding_mask, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}")

    if FP8: 
        torch.testing.assert_close(triton_out, torch_out.to(triton_out.dtype),atol=0.25, rtol=10) #Lower tolerance for FP8 
    else:
        torch.testing.assert_close(triton_out, torch_out.to(triton_out.dtype),atol=1e-1, rtol=1e-1)  


@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False)])
#@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. fails for seq >= 64
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
@pytest.mark.parametrize('FP8',[(False)])
#@pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_backward(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")

    with torch.enable_grad():
        if FP8:
            triton_out = flash_attn_fp8_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True)
        else:
            triton_out = flash_attn_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True)

    assert len(triton_out) == 3
    triton_out, lse, sd_mask= triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
        dropout_mask = sd_mask >= 0
    else:
        dropout_mask = None

    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do.clone())

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_lse={lse}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")
        print(f"dropout_mask={dropout_mask}")

    if DEBUG_MODE:
        print("--------------Torch----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")
    with torch.enable_grad():
        torch_out = attention_ref(q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")
        print(f"torch_attn_scores={attention_scores}")
        print(f"torch_dq.shape={torch_dq.shape} torch_dq={torch_dq}")
        print(f"torch_dk.shape={torch_dk.shape} torch_dk={torch_dk}")
        print(f"torch_dv.shape={torch_dv.shape} torch_dv={torch_dv}")

    if FP8:
        fp8_assert_close(triton_dq, torch_dq.to(triton_dq.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
        fp8_assert_close(triton_dk, torch_dk.to(triton_dk.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
        fp8_assert_close(triton_dv, torch_dv.to(triton_dv.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
    else:
        torch.testing.assert_close(triton_dv, torch_dv.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
        torch.testing.assert_close(triton_dk, torch_dk.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
        torch.testing.assert_close(triton_dq, torch_dq.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  



@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False), (0.0, True)])
#@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. Fails for seq >=64
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
@pytest.mark.parametrize('FP8',[(False)])
#@pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_backward_varlen(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, "cuda", mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    if DEBUG_MODE:
        print(f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}")
        print(f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}")

        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"q_unpad.shape={q_unpad.shape} q_unpad={q_unpad}")
        print(f"k_unpad.shape={k_unpad.shape} k_unpad={k_unpad}")
        print(f"v_unpad.shape={v_unpad.shape} v_unpad={v_unpad}")
        print(f"max_seqlens_q={max_seqlen_q }")
        print(f"max_seqlens_k={max_seqlen_k }")
        print(f"cu_seqlens_q={cu_seqlens_q }")
        print(f"cu_seqlens_k={cu_seqlens_k }")
    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"do.shape={do.shape} do={do}")

    with torch.enable_grad():
        triton_out = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True)

    assert len(triton_out) == 3
    triton_out, lse, sd_mask= triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
        dropout_mask = sd_mask >= 0
        dropout_mask = pad_rearrange_dropout_mask(dropout_mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS)
        dropout_mask = dropout_mask > 0
    else:
        dropout_mask = None

    triton_out = output_pad_fn(triton_out)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q_unpad, k_unpad, v_unpad), do.clone())

    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)
    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_lse.shape={lse.shape} triton_lse={lse}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")
        print(f"dropout_mask={dropout_mask}")

    if DEBUG_MODE:
        print("--------------Torch----------------")
        print(f"do.shape={do.shape} do={do}")
    with torch.enable_grad():
        torch_out = attention_ref(q, k, v, query_padding_mask=query_padding_mask, key_padding_mask=key_padding_mask, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")
        print(f"torch_attn_scores={attention_scores}")
        print(f"torch_dq.shape={torch_dq.shape} torch_dq={torch_dq}")
        print(f"torch_dk.shape={torch_dk.shape} torch_dk={torch_dk}")
        print(f"torch_dv.shape={torch_dv.shape} torch_dv={torch_dv}")

    torch.testing.assert_close(triton_dv, torch_dv.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
    torch.testing.assert_close(triton_dk, torch_dk.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
    torch.testing.assert_close(triton_dq, torch_dq.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  