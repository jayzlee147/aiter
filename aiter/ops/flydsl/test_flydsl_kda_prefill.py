# SPDX-License-Identifier: MIT

import pytest
import torch

from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    flydsl_kda_prefill,
    kda_prepare_fwd_flydsl,
)
from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available() or not is_flydsl_available():
    pytest.skip("ROCm/FlyDSL required", allow_module_level=True)


def _reference(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens):
    output = torch.empty_like(v)
    final = state.clone()
    d = q.shape[-1]
    for seq in range(len(cu_seqlens) - 1):
        begin = int(cu_seqlens[seq])
        end = int(cu_seqlens[seq + 1])
        h = final[seq]
        for token in range(begin, end):
            qf = q[0, token].float()
            kf = k[0, token].float()
            vf = v[0, token].float()
            gf = g[0, token].float()
            qf *= torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) / d**0.5
            kf *= torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            decay = torch.exp(
                -5.0
                * torch.sigmoid(
                    torch.exp(A_log)[:, None] * (gf + dt_bias.float())
                )
            )
            h *= decay[:, None, :]
            residual = vf - torch.einsum("hvk,hk->hv", h, kf)
            update = residual * torch.sigmoid(beta[0, token])[:, None]
            h += update[:, :, None] * kf[:, None, :]
            output[0, token] = torch.einsum("hvk,hk->hv", h, qf).to(output.dtype)
    return output, final


@pytest.mark.parametrize("lengths", [(8,), (3, 5), (1, 4, 2)])
def test_flydsl_kda_prefill_matches_recurrence(lengths):
    torch.manual_seed(7)
    h, d, total = 12, 128, sum(lengths)
    q = torch.randn(1, total, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, total, h, device="cuda", dtype=torch.float32)
    A_log = torch.randn(h, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(h, d, device="cuda", dtype=torch.float32)
    state = torch.randn(len(lengths), h, d, d, device="cuda") * 0.01
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int64)

    expected_out, expected_state = _reference(
        q, k, v, g, beta, A_log, dt_bias, state, cu.cpu()
    )
    actual_out, actual_state = flydsl_kda_prefill(
        q, k, v, g, beta, A_log, dt_bias, state, cu
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("t", [4, 16])
def test_flydsl_kda_prepare_vector_gate_c(t):
    torch.manual_seed(11)
    b, h, d = 1, 2, 128
    k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(k)
    beta = torch.randn(b, t, h, device="cuda", dtype=torch.float32)
    a_log = torch.randn(h, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(h, d, device="cuda", dtype=torch.float32)

    actual = kda_prepare_fwd_flydsl(k, g, beta, a_log, dt_bias)
    torch.cuda.synchronize()
    for head in range(h):
        kn = torch.nn.functional.normalize(k[0, :, head].float(), dim=-1, eps=1e-6)
        gate = -5.0 * torch.sigmoid(
            torch.exp(a_log[head]) * (g[0, :, head].float() + dt_bias[head])
        )
        # The kernel publishes cumulative gates to BF16 LDS before forming A.
        gc_out = gate.cumsum(0)
        gc = gc_out.to(torch.bfloat16).float()
        decay = torch.exp(gc[:, None, :] - gc[None, :, :])
        a = torch.einsum("mk,nk,mnk->mn", kn, kn, decay)
        a = torch.tril(a * torch.sigmoid(beta[0, :, head])[:, None], diagonal=-1)
        expected = torch.linalg.inv(torch.eye(t, device="cuda") + a)
        torch.testing.assert_close(
            actual[0, head, :, :t], expected, rtol=1e-3, atol=1e-3
        )


def test_flydsl_kda_prepare_vector_gate_wu():
    torch.manual_seed(13)
    b, t, h, d = 1, 8, 2, 128
    k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    g = torch.randn_like(k)
    beta = torch.randn(b, t, h, device="cuda", dtype=torch.float32)
    a_log = torch.randn(h, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(h, d, device="cuda", dtype=torch.float32)

    result = kda_prepare_fwd_flydsl(
        k,
        g,
        beta,
        a_log,
        dt_bias,
        v=v,
        output_c=False,
    )
    w_bar, u_bar, g_cumsum = result
    torch.cuda.synchronize()
    for head in range(h):
        kn = torch.nn.functional.normalize(k[0, :, head].float(), dim=-1, eps=1e-6)
        gate = -5.0 * torch.sigmoid(
            torch.exp(a_log[head]) * (g[0, :, head].float() + dt_bias[head])
        )
        gc_out = gate.cumsum(0)
        gc = gc_out.to(torch.bfloat16).float()
        decay = torch.exp(gc[:, None, :] - gc[None, :, :])
        a = torch.einsum("mk,nk,mnk->mn", kn, kn, decay)
        beta_act = torch.sigmoid(beta[0, :, head])
        a = torch.tril(a * beta_act[:, None], diagonal=-1)
        c = torch.linalg.inv(torch.eye(t, device="cuda") + a)
        expected_w = c @ (kn * beta_act[:, None] * torch.exp(gc))
        expected_u = c @ (v[0, :, head].float() * beta_act[:, None])
        torch.testing.assert_close(
            w_bar[0, head].float(), expected_w, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            u_bar[0, head].float(), expected_u, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            g_cumsum[0, head], gc_out, rtol=1e-5, atol=1e-5
        )
