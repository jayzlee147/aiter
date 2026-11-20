# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AST contracts for the gfx1250 plain GDR decode policy."""

from __future__ import annotations

import ast
from pathlib import Path


MODULE = Path("aiter") / "ops" / "flydsl" / "linear_attention_kernels.py"
NEW_ARCH = "gfx1250"
WAVE64_ARCHS = {"gfx942", "gfx950"}


def _path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / MODULE
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{MODULE} not found above {here}")


def _tree() -> ast.Module:
    path = _path()
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_value(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return node.value
    raise AssertionError(f"module value {name} not found")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _called_names(func: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name:
            names.add(name)
    return names


def _return_dicts(func: ast.FunctionDef) -> list[ast.Dict]:
    return [
        node.value
        for node in ast.walk(func)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]


def _dict_keys(node: ast.Dict) -> set[str]:
    return {
        str(key.value)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _policy_namespace(arch: str) -> dict[str, object]:
    tree = _tree()
    assignments = {
        "_DECODE_GRID_PER_CU",
        "_DECODE_GRID_PER_CU_F32_STATE",
        "_DECODE_V_SPLITS",
        "_DECODE_V_SPLITS_F32_STATE",
        "_WAVE_SIZE_BY_ARCH",
        "_DEFAULT_WAVE_SIZE",
        "_DECODE_WARP_SHAPE",
        "_DECODE_WARP_SHAPE_BY_ARCH",
    }
    functions = {"_tile_warps", "_wave_size", "_decode_warp_shape", "_decode_tiling"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)

    namespace: dict[str, object] = {
        "GDR_GPU_ARCH": arch,
        "get_num_sms": lambda: 256,
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, filename=str(_path()), mode="exec"), namespace)
    return namespace


def test_gfx1250_maps_to_wave32() -> None:
    tree = _tree()
    waves = ast.literal_eval(_module_value(tree, "_WAVE_SIZE_BY_ARCH"))
    assert waves == {NEW_ARCH: 32}
    assert ast.literal_eval(_module_value(tree, "_DEFAULT_WAVE_SIZE")) == 64


def test_decode_and_mtp_arch_admission_are_separate() -> None:
    tree = _tree()
    mtp_archs = set(ast.literal_eval(_module_value(tree, "_SUPPORTED_ARCHS")))
    decode_archs = set(
        ast.literal_eval(_module_value(tree, "_DECODE_SUPPORTED_ARCHS"))
    )
    assert mtp_archs == WAVE64_ARCHS
    assert decode_archs == WAVE64_ARCHS | {NEW_ARCH}

    mtp_policy = _module_value(tree, "_MTP_BY_ARCH")
    assert isinstance(mtp_policy, ast.Dict)
    assert NEW_ARCH not in _dict_keys(mtp_policy)


def test_decode_shape_uses_the_selected_wave() -> None:
    func = _function(_tree(), "_decode_warp_shape")
    params = [arg.arg for arg in func.args.args]
    assert "wave" in params
    for node in ast.walk(func):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.FloorDiv):
            continue
        assert not (
            isinstance(node.left, ast.Constant)
            and not isinstance(node.left.value, bool)
            and node.left.value == 64
        ), f"literal wave width at line {node.lineno}"


def test_every_decode_tiling_return_carries_wave_size() -> None:
    returns = _return_dicts(_function(_tree(), "_decode_tiling"))
    assert len(returns) == 3
    assert all("WAVE_SIZE" in _dict_keys(result) for result in returns)


def test_gfx1250_unsplit_fp32_shape_matches_the_sweep() -> None:
    table = ast.literal_eval(_module_value(_tree(), "_DECODE_WARP_SHAPE_BY_ARCH"))
    assert table[NEW_ARCH][(128, True)] == (8, 8)


def test_decode_tiling_preserves_wave64_and_selects_measured_wave32_shape() -> None:
    gfx1250 = _policy_namespace("gfx1250")
    gfx950 = _policy_namespace("gfx950")

    wave32 = gfx1250["_decode_tiling"](512, 1, 128, 128, "torch.float32")
    wave64 = gfx950["_decode_tiling"](512, 1, 128, 128, "torch.float32")

    assert wave32 == {
        "NUM_BLOCKS_PER_V_DIM": 1,
        "NUM_WARPS": 8,
        "WARP_THREADS_K": 8,
        "WAVE_SIZE": 32,
    }
    assert wave64 == {
        "NUM_BLOCKS_PER_V_DIM": 1,
        "NUM_WARPS": 4,
        "WARP_THREADS_K": 8,
        "WAVE_SIZE": 64,
    }


def test_plain_decode_fails_closed_with_its_own_arch_gate() -> None:
    decode = _function(_tree(), "flydsl_gdr_decode")
    assert "_is_decode_supported_arch" in _called_names(decode)
    assert any(isinstance(node, ast.Raise) for node in ast.walk(decode))


def test_mtp_support_keeps_the_wave64_gate() -> None:
    mtp = _function(_tree(), "_mtp_shapes_supported")
    calls = _called_names(mtp)
    assert "_is_supported_arch" in calls
    assert "_is_decode_supported_arch" not in calls


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
