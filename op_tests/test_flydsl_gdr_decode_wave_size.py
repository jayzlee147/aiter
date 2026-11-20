# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Static and arithmetic checks for plain GDR decode wave sizing.

The plain decode kernel now serves both 32-lane and 64-lane targets. The MTP
factory remains wave64-only until it has its own gfx1250 sweep and validation.
These checks deliberately avoid importing torch, FlyDSL, or a live device.
"""

from __future__ import annotations

import ast
from pathlib import Path


KERNEL = "gdr_decode.py"
WAVE_SIZES = (32, 64)
WARP_THREADS_K_VALUES = (4, 8, 16)
VALUES_PER_THREAD_K = {"f32": 4, "bf16": 8}
_SHUFFLE_NAMES = {"shuffle_xor", "shuffle_idx", "shuffle_up", "shuffle_down"}


def _kernel_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "aiter" / "ops" / "flydsl" / "kernels" / KERNEL
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{KERNEL} not found above {here}")


def _tree() -> ast.Module:
    path = _kernel_path()
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _default_int(func: ast.FunctionDef, arg_name: str) -> int | None:
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    first = len(positional) - len(args.defaults)
    for offset, default in enumerate(args.defaults):
        if positional[first + offset].arg == arg_name:
            return ast.literal_eval(default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == arg_name and default is not None:
            return ast.literal_eval(default)
    return None


def _hardcoded_wave_widths(func: ast.FunctionDef) -> list[str]:
    hits = []
    for node in ast.walk(func):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            if (
                isinstance(node.left, ast.Constant)
                and not isinstance(node.left.value, bool)
                and node.left.value == 64
                and isinstance(node.right, ast.Name)
            ):
                hits.append(f"line {node.lineno}: 64 // {node.right.id}")
        if isinstance(node, ast.Assert):
            for sub in ast.walk(node.test):
                if not isinstance(sub, ast.Compare):
                    continue
                for side in [sub.left, *sub.comparators]:
                    if (
                        isinstance(side, ast.Constant)
                        and not isinstance(side.value, bool)
                        and side.value == 64
                    ):
                        hits.append(f"line {node.lineno}: assert against literal 64")
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in _SHUFFLE_NAMES:
                continue
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and not isinstance(arg.value, bool)
                    and arg.value == 64
                ):
                    hits.append(f"line {node.lineno}: {name} uses literal width 64")
    return hits


def _shuffle_offsets(width: int) -> list[int]:
    offsets = []
    value = width // 2
    while value >= 1:
        offsets.append(value)
        value //= 2
    return offsets


def test_plain_decode_factory_has_a_wave_size_cache_key() -> None:
    decode = _function(_tree(), "create_vk_gdr_decode_kernel")
    assert _default_int(decode, "WAVE_SIZE") == 64
    assert not _hardcoded_wave_widths(decode)


def test_mtp_factory_is_not_opened_on_wave32() -> None:
    mtp = _function(_tree(), "create_vk_gdr_mtp_kernel")
    assert _default_int(mtp, "WAVE_SIZE") is None


def test_k_group_reduction_stays_group_local() -> None:
    for wave in WAVE_SIZES:
        for warp_threads_k in WARP_THREADS_K_VALUES:
            if warp_threads_k > wave:
                continue
            assert wave % warp_threads_k == 0
            for lane in range(wave):
                group = lane // warp_threads_k
                for offset in _shuffle_offsets(warp_threads_k):
                    peer = lane ^ offset
                    assert peer // warp_threads_k == group
                    assert peer // wave == lane // wave


def test_value_columns_partition_each_tile() -> None:
    for wave in WAVE_SIZES:
        for warp_threads_k in WARP_THREADS_K_VALUES:
            if warp_threads_k > wave:
                continue
            for values_per_thread_k in VALUES_PER_THREAD_K.values():
                if 128 % (warp_threads_k * values_per_thread_k):
                    continue
                warp_tile_v = wave // warp_threads_k
                for num_warps in (2, 4, 8):
                    group_v = num_warps * warp_tile_v
                    for num_blocks in (1, 2, 4, 8):
                        tile_v = 128 // num_blocks
                        if group_v > tile_v or tile_v % group_v:
                            continue
                        iterations = tile_v // group_v
                        owners: dict[tuple[int, int], int] = {}
                        for warp in range(num_warps):
                            for lane in range(wave):
                                k_lane = lane % warp_threads_k
                                start = warp * warp_tile_v + lane // warp_threads_k
                                for iteration in range(iterations):
                                    key = (k_lane, start + iteration * group_v)
                                    owners[key] = owners.get(key, 0) + 1
                        expected = {
                            (k_lane, column)
                            for k_lane in range(warp_threads_k)
                            for column in range(tile_v)
                        }
                        assert set(owners) == expected
                        assert set(owners.values()) == {1}


def test_k_columns_are_covered() -> None:
    for wave in WAVE_SIZES:
        for warp_threads_k in WARP_THREADS_K_VALUES:
            if warp_threads_k > wave:
                continue
            for values_per_thread_k in VALUES_PER_THREAD_K.values():
                warp_tile_k = warp_threads_k * values_per_thread_k
                for head_k_dim in (64, 128):
                    if head_k_dim % warp_tile_k:
                        continue
                    columns = set()
                    for lane in range(wave):
                        base = (lane % warp_threads_k) * values_per_thread_k
                        for iteration in range(head_k_dim // warp_tile_k):
                            for element in range(values_per_thread_k):
                                columns.add(base + iteration * warp_tile_k + element)
                    assert columns == set(range(head_k_dim))


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
