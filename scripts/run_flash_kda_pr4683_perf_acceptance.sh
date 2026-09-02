#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command coverage and acceptance for the shapes/input contract in
# ROCm/aiter PR #4683's original six-row gfx950 dense table. The canonical
# Python entry point owns the matrix, correctness checks, paired statistics,
# and fail-closed performance decision; this wrapper owns clean-checkout and
# isolated-build provenance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
readonly EXPECTED_SEED_CELLS=18
readonly EXPECTED_LOGICAL_CELLS=6
readonly WARMUP=20
readonly REPEAT=120
readonly BOOTSTRAP_RESAMPLES=10000
readonly MIN_NATIVE_SPEEDUP=1.03

usage() {
    cat <<'EOF'
Usage: scripts/run_flash_kda_pr4683_perf_acceptance.sh [OUTPUT_DIR]

Run the canonical native-HIP versus forced-PR#4683-Triton dense eager matrix
on exactly one visible 256-CU gfx950 GPU. Select the GPU with
ROCR_VISIBLE_DEVICES (or HIP_VISIBLE_DEVICES) before invoking the script.

The checkout must be clean and OUTPUT_DIR must be a new directory outside the
checkout. The default is a new /tmp/flash-kda-pr4683-* directory. The fixed
matrix is B=1, T={8192,16384}, H={32,64,96}, K=V=128 crossed with seeds
42/43/44. Each seed cell requires >=1.03x native p50 speedup, >50% paired
native wins, negative upper bounds for paired p50/mean bootstrap confidence
intervals, and output rRMS <=0.04.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if (($# > 1)); then
    usage >&2
    exit 2
fi

cd "$REPO_ROOT"
repo_is_clean() {
    git -c safe.directory="$REPO_ROOT" diff --quiet -- &&
        git -c safe.directory="$REPO_ROOT" diff --cached --quiet -- &&
        [[ -z $(git -c safe.directory="$REPO_ROOT" \
            ls-files --others --exclude-standard) ]] &&
        ! git -c safe.directory="$REPO_ROOT" submodule status --recursive |
            grep -Eq '^[+U]' &&
        git -c safe.directory="$REPO_ROOT" submodule foreach --recursive \
            --quiet 'test -z "$(git status --porcelain --untracked-files=all)"'
}

if ! repo_is_clean; then
    echo "ERROR: acceptance requires a clean checkout" >&2
    git -c safe.directory="$REPO_ROOT" status --short --untracked-files=all >&2
    exit 2
fi

INITIAL_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
INITIAL_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
if (($# == 1)); then
    OUTPUT_DIR="$(realpath -m -- "$1")"
    case "$OUTPUT_DIR" in
        "$REPO_ROOT"|"$REPO_ROOT"/*)
            echo "ERROR: output directory must be outside the checkout" >&2
            exit 2
            ;;
    esac
    [[ ! -e $OUTPUT_DIR ]] || {
        echo "ERROR: output path already exists: $OUTPUT_DIR" >&2
        exit 2
    }
    mkdir -p "$OUTPUT_DIR"
else
    OUTPUT_PARENT="$(realpath -m -- "${TMPDIR:-/tmp}")"
    case "$OUTPUT_PARENT" in
        "$REPO_ROOT"|"$REPO_ROOT"/*)
            echo "ERROR: TMPDIR must be outside the checkout" >&2
            exit 2
            ;;
    esac
    mkdir -p "$OUTPUT_PARENT"
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-pr4683-XXXXXXXX")"
fi

ACCEPTANCE_STAGE=setup
acceptance_exit() {
    local exit_code=$?
    trap - EXIT
    set +e
    local final_head final_tree repository_ok
    final_head="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
    final_tree="$(
        git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree} 2>/dev/null
    )"
    repository_ok=1
    if [[ $final_head != "$INITIAL_HEAD" || $final_tree != "$INITIAL_TREE" ]] ||
        ! repo_is_clean; then
        repository_ok=0
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=repository-changed
        fi
    fi

    {
        printf 'initial_head=%s\n' "$INITIAL_HEAD"
        printf 'final_head=%s\n' "$final_head"
        printf 'initial_tree=%s\n' "$INITIAL_TREE"
        printf 'final_tree=%s\n' "$final_tree"
        printf 'clean_and_unchanged=%s\n' "$repository_ok"
    } >"$OUTPUT_DIR/repository-after.txt"

    local status=failed
    if ((exit_code == 0)); then
        status=passed
    fi
    {
        printf 'status=%s\n' "$status"
        printf 'exit_code=%s\n' "$exit_code"
        printf 'stage=%s\n' "$ACCEPTANCE_STAGE"
        printf 'git_head=%s\n' "$INITIAL_HEAD"
        printf 'git_tree=%s\n' "$INITIAL_TREE"
        printf 'logical_cells=%s\n' "$EXPECTED_LOGICAL_CELLS"
        printf 'seed_cells=%s\n' "$EXPECTED_SEED_CELLS"
        printf 'seeds=42,43,44\n'
        printf 'warmup=%s\n' "$WARMUP"
        printf 'repeat=%s\n' "$REPEAT"
        printf 'bootstrap_resamples=%s\n' "$BOOTSTRAP_RESAMPLES"
        printf 'minimum_native_speedup=%s\n' "$MIN_NATIVE_SPEEDUP"
        printf 'execution=eager\n'
        printf 'python=%s\n' "$PYTHON_BIN"
        printf 'benchmark_exit=%s\n' "${BENCHMARK_EXIT:-not-run}"
    } >"$OUTPUT_DIR/status.txt"

    local candidate
    local -a artifacts=()
    for candidate in \
        status.txt repository-after.txt git-head.txt git-tree.txt \
        git-provenance.log static-self-test.json static-self-test.log \
        plan.json plan.log environment.log benchmark.log benchmark-status.txt \
        result.json partial-results.jsonl artifact-validation.log \
        jit/module_flash_kda_hip.so; do
        [[ -s $OUTPUT_DIR/$candidate ]] && artifacts+=("$candidate")
    done
    if ((${#artifacts[@]} > 0)); then
        (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0)); then
        echo "FlashKDA PR #4683 performance acceptance passed: $OUTPUT_DIR"
    else
        printf \
            'FlashKDA PR #4683 performance acceptance failed during %s: %s\n' \
            "$ACCEPTANCE_STAGE" "$OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT

printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
{
    git -c safe.directory="$REPO_ROOT" show --no-patch \
        --format=fuller "$INITIAL_HEAD"
    git -c safe.directory="$REPO_ROOT" branch --show-current
    git -c safe.directory="$REPO_ROOT" remote -v
    git -c safe.directory="$REPO_ROOT" submodule status --recursive
} >"$OUTPUT_DIR/git-provenance.log"
echo "FlashKDA PR #4683 acceptance artifacts: $OUTPUT_DIR"

# Route/tuning environment must not silently change either implementation.
# Clear it before the Python process imports AITER or Triton.
while IFS='=' read -r name _; do
    case "$name" in
        AITER_*|FLASH_KDA_*|CHUNK_DELTA_ATTN_*|KDA_*|FLA_*)
            unset "$name"
            ;;
    esac
done < <(env)
unset CK_DIR HIP_KITTENS_DIR OPUS_GEN_CO_DIR PYTHONOPTIMIZE

export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
export AITER_META_DIR="$REPO_ROOT"
export GPU_ARCHS=gfx950
export MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT"
export TRITON_CACHE_DIR="$OUTPUT_DIR/triton-cache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO_ROOT"
mkdir -p "$AITER_JIT_DIR" "$TRITON_CACHE_DIR"

BENCHMARK_ENTRY=op_tests/op_benchmarks/triton/bench_flash_kda_pr4683.py

ACCEPTANCE_STAGE=static-self-test
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --static-self-test \
    --output "$OUTPUT_DIR/static-self-test.json" \
    2>&1 | tee "$OUTPUT_DIR/static-self-test.log"

ACCEPTANCE_STAGE=plan
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --print-plan \
    --output "$OUTPUT_DIR/plan.json" \
    2>&1 | tee "$OUTPUT_DIR/plan.log"

ACCEPTANCE_STAGE=environment
"$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUTPUT_DIR/environment.log"
import hashlib
import os
import platform
import shutil
import subprocess
import sys

import torch
import triton


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("a visible ROCm GPU is required")
if torch.cuda.device_count() != 1:
    raise SystemExit("select exactly one visible ROCm GPU")
properties = torch.cuda.get_device_properties(0)
arch_detail = getattr(properties, "gcnArchName", "unknown")
arch = arch_detail.split(":", 1)[0]
if arch != "gfx950" or properties.multi_processor_count != 256:
    raise SystemExit(
        "acceptance requires the 256-CU gfx950 reference target; "
        f"got {arch}/{properties.multi_processor_count} CU"
    )
print(
    f"GPU: {torch.cuda.get_device_name(0)} "
    f"({arch_detail}, {properties.multi_processor_count} CU)"
)
print(f"PyTorch: {torch.__version__}; ROCm: {torch.version.hip}; {torch.__file__}")
print(f"Triton: {triton.__version__}; {triton.__file__}")
print(
    f"Python: {sys.executable}; {platform.python_version()}; "
    f"sha256={sha256(sys.executable)}"
)
hipcc = shutil.which("hipcc")
print(f"hipcc: {hipcc}")
if hipcc is not None:
    version = subprocess.run(
        [hipcc, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(version)
for name in (
    "AITER_AOT_IMPORT",
    "AITER_JIT_DIR",
    "AITER_META_DIR",
    "CK_DIR",
    "GPU_ARCHS",
    "HIP_KITTENS_DIR",
    "MAX_JOBS",
    "OPUS_GEN_CO_DIR",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "TRITON_CACHE_DIR",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
):
    print(f"{name}={os.environ.get(name)}")
PY

ACCEPTANCE_STAGE=benchmark
set +e
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --output "$OUTPUT_DIR/result.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"
BENCHMARK_EXIT=${PIPESTATUS[0]}
set -e
printf 'benchmark_exit=%s\n' "$BENCHMARK_EXIT" \
    >"$OUTPUT_DIR/benchmark-status.txt"

if ((BENCHMARK_EXIT != 0)); then
    ACCEPTANCE_STAGE=benchmark-failed
    exit "$BENCHMARK_EXIT"
fi

ACCEPTANCE_STAGE=artifact-validation
for artifact in result.json partial-results.jsonl \
    jit/module_flash_kda_hip.so; do
    [[ -s $OUTPUT_DIR/$artifact ]] || {
        echo "ERROR: successful benchmark did not produce $artifact" >&2
        exit 1
    }
done
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY' \
    2>&1 | tee "$OUTPUT_DIR/artifact-validation.log"
import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
with (root / "result.json").open(encoding="utf-8") as handle:
    result = json.load(handle)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(result["schema"] == "flash-kda-pr4683-dense-gfx950-v1", "schema")
require(result["capture_complete"] is True, "capture is incomplete")
require(result["performance_gate_evaluated"] is True, "gate not evaluated")
require(result["performance_gate_passed"] is True, "global gate failed")
require(result["performance_gate"]["failed_logical_cells"] == [], "failed cells")
require(len(result["results"]) == 18, "seed-cell count")
require(result["cross_seed_summary"]["logical_cells"] == 6, "logical-cell count")
require(result["cross_seed_summary"]["seed_cells"] == 18, "summary seed count")
require(
    all(row["performance_gate_passed"] is True for row in result["results"]),
    "at least one seed-cell gate failed",
)
require(result["environment"]["gpu"]["arch"] == "gfx950", "GPU arch")
require(result["environment"]["gpu"]["compute_units"] == 256, "GPU CU count")
require(
    result["environment"]["modules"]["native_jit"][
        "matches_expected_jit_path"
    ]
    is True,
    "native JIT provenance",
)
require(
    result["environment"]["modules"]["native_jit"]["raw_abi_version"] == 3,
    "native raw ABI",
)
require(
    result["environment"]["modules"]["native_build_roots"][
        "both_match_checkout"
    ]
    is True,
    "native build roots",
)
require(
    result["environment"]["modules"]["pr4683_triton_python"][
        "matches_audited_pr_source"
    ]
    is True,
    "PR #4683 Triton source identity",
)
require(
    result["environment"]["active_route_control_environment"] == {},
    "route-control environment",
)
with (root / "partial-results.jsonl").open(encoding="utf-8") as handle:
    events = [json.loads(line) for line in handle if line.strip()]
require(events and events[-1]["event"] == "run-complete", "checkpoint terminal event")
require(events[-1]["complete"] is True, "checkpoint incomplete")
require(events[-1]["performance_gate_passed"] is True, "checkpoint gate failed")
print("artifact validation: PASS (6/6 logical cells, 18/18 seed cells)")
PY

ACCEPTANCE_STAGE=repository-validation
CURRENT_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
CURRENT_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
if [[ $CURRENT_HEAD != "$INITIAL_HEAD" || $CURRENT_TREE != "$INITIAL_TREE" ]] ||
    ! repo_is_clean; then
    echo "ERROR: acceptance changed HEAD or the source checkout" >&2
    exit 1
fi

ACCEPTANCE_STAGE=complete
