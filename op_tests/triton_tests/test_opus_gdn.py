# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Accuracy comparison: opus_gdn HIP kernel vs Triton chunk_gated_delta_rule vs recurrent ref

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.opus_gdn_prefill import opus_gdn_prefill_fwd
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    assert_close,
    device,
)


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.float()
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 64, 4, 64),
            (2, 128, 4, 64),
            (2, 256, 8, 64),
            (1, 512, 4, 64),
            (2, 128, 4, 16),
            (1, 256, 8, 16),
        ]
    ],
)
def test_opus_vs_triton(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    opus_o, _ = opus_gdn_prefill_fwd(q, k, v, g, beta.to(torch.bfloat16), BT=BT)

    tri_o, _ = chunk_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g=g.clone(), beta=beta.to(torch.bfloat16).clone(),
    )

    assert_close("opus_vs_tri_o", tri_o, opus_o, 0.01)


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 64, 4, 64),
            (2, 128, 4, 64),
            (2, 256, 8, 64),
            (2, 128, 4, 16),
        ]
    ],
)
def test_opus_vs_ref(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    opus_o, _ = opus_gdn_prefill_fwd(q, k, v, g, beta.to(torch.bfloat16), BT=BT)

    ref_o, _ = recurrent_gated_delta_rule_ref(
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta.to(torch.bfloat16).clone(), g=g.clone(),
    )

    assert_close("opus_vs_ref_o", ref_o, opus_o, 0.01)


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(*t, id=f"B{t[0]}-T{t[1]}-H{t[2]}-BT{t[3]}")
        for t in [
            (1, 128, 4, 64),
            (2, 256, 4, 64),
            (1, 128, 4, 16),
        ]
    ],
)
def test_opus_with_initial_state(B: int, T: int, H: int, BT: int):
    torch.manual_seed(42)
    D = 128

    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.1

    opus_o, opus_ht = opus_gdn_prefill_fwd(
        q, k, v, g, beta.to(torch.bfloat16),
        initial_state=h0.clone(), output_final_state=True, BT=BT,
    )

    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta.to(torch.bfloat16).clone(), g=g.clone(),
        initial_state=h0.clone(), output_final_state=True,
    )

    assert_close("init_o", ref_o, opus_o, 0.01)
    assert_close("init_ht", ref_ht, opus_ht, 0.01)
