# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the rectangular HIP mixer against the upstream FP32 KDA recurrence."""

from itertools import accumulate
import os

import pytest
import torch
import torch.nn.functional as F

H, K, V = 64, 64, 128
D = H * (2 * K + V)
EPS = 1e-5


@pytest.fixture(scope="module")
def kda():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("rectangular HIP KDA requires a ROCm gfx950 GPU")
    if os.getenv("AITER_TRITON_ONLY") == "1":
        pytest.skip("AITER_TRITON_ONLY disables HIP operators")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    if device.gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("rectangular HIP KDA requires gfx950")
    from aiter.ops import kda_rectangular

    return kda_rectangular


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype).mul_(0.3)


def inputs(total):
    return dict(
        mixed_qkv=rand(total, D),
        conv_weight=rand(D, 4),
        gate=rand(1, total, H, K),
        beta=rand(1, total, H),
        out_gate=rand(1, total, H, V),
        A_log=rand(H, dtype=torch.float32),
        dt_bias=rand(H * K, dtype=torch.float32),
        norm_weight=rand(V),
        norm_eps=EPS,
        lower_bound=-5.0,
    )


def reference(args, start, stop, history, state, prefill):
    from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

    raw = args["mixed_qkv"][start:stop]
    padded = torch.cat((history.T, raw))
    taps = padded.float().unfold(0, 4, 1)
    conv = F.silu((taps * args["conv_weight"].float()).sum(-1))
    # The prefill convolution stores BF16. Fused decode keeps FP32 through Q/K norm.
    if prefill:
        conv = conv.to(raw.dtype)
    n = stop - start
    q, k, v = conv.split((H * K, H * K, H * V), dim=1)
    out, final = chunk_kda_ref(
        q.reshape(1, n, H, K),
        k.reshape(1, n, H, K),
        v.reshape(1, n, H, V),
        args["gate"][:, start:stop],
        args["beta"][:, start:stop],
        A_log=args["A_log"],
        dt_bias=args["dt_bias"],
        initial_state=state.unsqueeze(0),
        output_final_state=True,
        state_v_first=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=args["lower_bound"],
    )
    out = out.to(torch.bfloat16).float()
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + EPS)
    gate = args["out_gate"].reshape(1, -1, H, V)[:, start:stop]
    out = out * args["norm_weight"].float() * gate.float().sigmoid()
    return out.to(torch.bfloat16), final[0], padded[-3:].T


@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
def test_packed_prefill(kda, resumed):
    torch.manual_seed(7164)
    lengths = [0, 2, 19]
    bounds = list(accumulate(lengths, initial=0))
    args = inputs(bounds[-1])
    cache = rand(len(lengths), D, 3)
    expected_cache = cache.clone()
    states = rand(len(lengths), H, V, K, dtype=torch.float32)
    has_history = [resumed, False, resumed]
    expected_state = states.clone() if resumed else torch.zeros_like(states)
    expected_output = []
    for i, n in enumerate(lengths):
        if n:
            history = cache[i] if has_history[i] else torch.zeros_like(cache[i])
            o, s, tail = reference(
                args, bounds[i], bounds[i + 1], history, expected_state[i], True
            )
            expected_output.append(o)
            expected_state[i] = s
            expected_cache[i] = tail

    ci, co = kda.prepare_metadata(lengths, device="cuda")
    assert ci.tolist() == [[1, 0], [2, 0], [2, 1]]
    assert co.tolist() == [0, 0, 1, 3]
    y, final = kda.prefill(
        **args,
        cu_seqlens=torch.tensor(bounds, device="cuda", dtype=torch.int64),
        seq_lens=lengths,
        chunk_indices=ci,
        chunk_offsets=co,
        initial_state=states if resumed else None,
        conv_state=cache,
        has_initial_state=torch.tensor(has_history, device="cuda"),
        output_final_state=True,
    )
    torch.testing.assert_close(y, torch.cat(expected_output, 1), atol=0.012, rtol=0.05)
    torch.testing.assert_close(final, expected_state, atol=0.004, rtol=0.05)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)


def test_chunk_kda_output_aliases_gate(kda):
    torch.manual_seed(1807)
    length = 19
    gate = rand(1, length, H, V)
    original_gate = gate.clone()
    args = dict(
        q=rand(1, length, H, K),
        k=rand(1, length, H, K),
        v=rand(1, length, H, V),
        g=rand(1, length, H, K),
        beta=rand(1, length, H),
        A_log=rand(H, dtype=torch.float32),
        dt_bias=rand(H * K, dtype=torch.float32),
        lower_bound=-5.0,
        cu_seqlens=torch.tensor([0, length], device="cuda"),
        seq_lens=[length],
        output_final_state=True,
        norm_weight=rand(V),
    )
    expected, expected_state = kda.chunk_kda(**args, out_gate=gate)
    assert expected.data_ptr() != gate.data_ptr()
    assert torch.equal(gate, original_gate)
    assert not torch.equal(expected, original_gate)

    actual, final = kda.chunk_kda(**args, out_gate=gate, out=gate)
    assert actual.data_ptr() == gate.data_ptr()
    assert torch.equal(gate, expected)
    assert torch.equal(final, expected_state)


@pytest.mark.parametrize("mtp", [0, 1, 3, 7])
def test_decode_checkpoints_and_rollback(kda, mtp):
    torch.manual_seed(8614 + mtp)
    batch, steps = 3, mtp + 1
    # Reversed recurrent slots and independent convolution slots catch index mixups.
    slots = torch.arange(batch * steps, 0, -1, dtype=torch.int32, device="cuda")
    slots = slots.view(batch, steps)
    conv_slots = [2, 3, 1] if mtp else slots[:, 0].tolist()
    states = rand(batch * steps + 1, H, V, K, dtype=torch.float32)
    cache = rand(batch + 1, D, steps + 2)
    expected_states, expected_cache = states.clone(), cache.clone()
    turns = (
        [([steps] * batch, [1] * batch), ([1, 0, steps], [steps, 1, 2])]
        if mtp
        else [([1, 1, 1], [1] * batch), ([4, 0, 2], [1] * batch)]
    )
    for lengths, accepted in turns:
        bounds = list(accumulate(lengths, initial=0))
        args = inputs(bounds[-1])
        args["out_gate"] = args["out_gate"].view(bounds[-1], H * V)
        expected_output = []
        written = set()
        before = states.clone()
        for seq, n in enumerate(lengths):
            if not n:
                continue
            offset = accepted[seq] - 1
            state = expected_states[slots[seq, offset]].clone()
            row = expected_cache[conv_slots[seq]]
            history = row[:, offset : offset + 3].clone()
            if mtp:
                row[:, :2] = history[:, 1:]
            for i in range(n):
                t = bounds[seq] + i
                out, state, history = reference(args, t, t + 1, history, state, False)
                dest = int(slots[seq, i if mtp else 0])
                expected_states[dest] = state
                written.add(dest)
                expected_output.append(out.reshape(1, H * V))
                if mtp:
                    row[:, 2 + i] = args["mixed_qkv"][t]
                else:
                    row.copy_(history)

        actual = kda.fused_kda_decode(
            **args,
            ssm_state=states,
            conv_state=cache,
            ssm_state_indices=slots if mtp else slots[:, 0],
            cu_seqlens=torch.tensor(bounds, device="cuda", dtype=torch.int32),
            num_accepted_tokens=torch.tensor(accepted, device="cuda") if mtp else None,
            conv_state_indices=torch.tensor(conv_slots, device="cuda") if mtp else None,
        )
        torch.testing.assert_close(
            actual, torch.cat(expected_output), atol=0.008, rtol=0.02
        )
        torch.testing.assert_close(states, expected_states, atol=2e-5, rtol=3e-4)
        torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
        untouched = sorted(set(range(len(states))) - written)
        torch.testing.assert_close(states[untouched], before[untouched], atol=0, rtol=0)


@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
def test_32k_prefill_graph(kda, resumed):
    torch.manual_seed(32581)
    length = 32768
    args = inputs(length)
    ci, co = kda.prepare_metadata([length], device="cuda")
    args.update(
        cu_seqlens=torch.tensor([0, length], device="cuda"),
        seq_lens=[length],
        chunk_indices=ci,
        chunk_offsets=co,
        has_initial_state=torch.tensor([resumed], device="cuda"),
        initial_state=rand(1, H, V, K, dtype=torch.float32) if resumed else None,
        output_final_state=True,
    )
    initial_cache = rand(1, D, 3)
    eager_cache = initial_cache.clone()
    graph_cache = initial_cache.clone()
    expected, final = kda.prefill(**args, conv_state=eager_cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, graph_final = kda.prefill(**args, conv_state=graph_cache)
    graph_cache.copy_(initial_cache)
    graph.replay()
    assert torch.equal(actual, expected)
    assert torch.equal(graph_final, final)
    assert torch.equal(graph_cache, eager_cache)


def test_decode_graph_replays_device_indices(kda):
    torch.manual_seed(572891)
    steps = 4
    args = inputs(8)
    slots = torch.arange(1, 9, device="cuda", dtype=torch.int32).view(2, steps)
    args.update(
        out_gate=args["out_gate"].view(8, H * V),
        cu_seqlens=torch.tensor([0, 4, 8], device="cuda"),
        ssm_state_indices=slots.clone(),
        num_accepted_tokens=torch.ones(2, device="cuda", dtype=torch.int32),
        conv_state_indices=torch.tensor([2, 1], device="cuda", dtype=torch.int32),
    )
    cache = rand(3, D, steps + 2)
    state = rand(9, H, V, K, dtype=torch.float32)
    out = torch.empty((8, H * V), device="cuda", dtype=torch.bfloat16)
    expected_cache, expected_state = cache.clone(), state.clone()
    expected_out = torch.empty_like(out)
    # Warm the eager entry point before capture, then restore both state pools.
    kda.fused_kda_decode(**args, conv_state=cache, ssm_state=state, out=out)
    cache.copy_(expected_cache)
    state.copy_(expected_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kda.fused_kda_decode(**args, conv_state=cache, ssm_state=state, out=out)
    for bounds, accepted, slot_map, conv_map in (
        ([0, 4, 8], [1, 1], slots, [2, 1]),
        ([1, 2, 6], [3, 2], slots.flip((0, 1)), [1, 2]),
        ([0, 0, 2], [1, 4], slots.roll(1, 1), [2, 1]),
    ):
        args["cu_seqlens"].copy_(torch.tensor(bounds, device="cuda"))
        args["num_accepted_tokens"].copy_(torch.tensor(accepted, device="cuda"))
        args["ssm_state_indices"].copy_(slot_map)
        args["conv_state_indices"].copy_(torch.tensor(conv_map, device="cuda"))
        out.fill_(-11)
        expected_out.fill_(-11)
        kda.fused_kda_decode(
            **args,
            conv_state=expected_cache,
            ssm_state=expected_state,
            out=expected_out,
        )
        graph.replay()
        assert torch.equal(out, expected_out)
        assert torch.equal(state, expected_state)
        assert torch.equal(cache, expected_cache)
        assert (out[: bounds[0]] == -11).all() and (out[bounds[-1] :] == -11).all()
