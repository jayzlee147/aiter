#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 GVA acceptance. The coverage remains in the canonical
# pytest and benchmark files; this script only provides a reproducible runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
WARMUP="${KDA_ACCEPTANCE_WARMUP:-10}"
REPEAT="${KDA_ACCEPTANCE_REPEAT:-50}"
MIN_SPEEDUP="${KDA_ACCEPTANCE_MIN_SPEEDUP:-1.0}"
MIN_GEOMEAN="${KDA_ACCEPTANCE_MIN_GEOMEAN:-1.05}"
MIN_WIN_FRACTION="${KDA_ACCEPTANCE_MIN_WIN_FRACTION:-0.75}"
SEEDS_INPUT="${KDA_ACCEPTANCE_SEEDS:-42,43,44}"
read -r -a SEEDS <<<"${SEEDS_INPUT//,/ }"
if ((${#SEEDS[@]} == 0)); then
    echo "ERROR: KDA_ACCEPTANCE_SEEDS must contain at least one seed" >&2
    exit 2
fi
for seed in "${SEEDS[@]}"; do
    if [[ ! $seed =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed in KDA_ACCEPTANCE_SEEDS: $seed" >&2
        exit 2
    fi
done
SEEDS_CSV="$(IFS=,; printf '%s' "${SEEDS[*]}")"
readonly BENCHMARKS_PER_SEED=6
EXPECTED_BENCHMARK_RUNS=$((${#SEEDS[@]} * BENCHMARKS_PER_SEED))

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/run_flash_kda_gva_acceptance.sh [OUTPUT_DIR]

Run the FlashKDA Python API/native tests and the raw-ABI route/graph validator,
then run the six gfx950/256-CU GVA HIP-versus-Triton performance suites for
every seed. Every benchmark omits max_seqlen_upper_bound, matching the public
no-hint call contract. Select one GPU with ROCR_VISIBLE_DEVICES (or
HIP_VISIBLE_DEVICES) before invoking the script.

The output directory must be outside the source checkout. The default is a
new /tmp/flash-kda-gva-* directory.
The default seed set is 42, 43, and 44. Override it with
KDA_ACCEPTANCE_SEEDS using commas and/or spaces, for example
KDA_ACCEPTANCE_SEEDS="42,43 44". Override other benchmark settings with
KDA_ACCEPTANCE_WARMUP, KDA_ACCEPTANCE_REPEAT, KDA_ACCEPTANCE_MIN_SPEEDUP,
KDA_ACCEPTANCE_MIN_GEOMEAN, and KDA_ACCEPTANCE_MIN_WIN_FRACTION.
EOF
    exit 0
fi
if (($# > 1)); then
    echo "Usage: scripts/run_flash_kda_gva_acceptance.sh [OUTPUT_DIR]" >&2
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
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-gva-XXXXXXXX")"
fi

ACCEPTANCE_STAGE=setup
CORRECTNESS_RUNS_COMPLETED=0
BENCHMARK_RUNS_COMPLETED=0
acceptance_exit() {
    local exit_code=$?
    trap - EXIT
    set +e
    local final_head final_tree repository_ok
    final_head="$(
        git -c safe.directory="$REPO_ROOT" rev-parse HEAD 2>/dev/null
    )"
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
        printf 'git_head=%s\n' "$final_head"
        printf 'initial_git_head=%s\n' "$INITIAL_HEAD"
        printf 'initial_git_tree=%s\n' "$INITIAL_TREE"
        printf 'clean_and_unchanged=%s\n' "$repository_ok"
        printf 'correctness_runs_completed=%s\n' \
            "$CORRECTNESS_RUNS_COMPLETED"
        printf 'benchmark_runs_completed=%s\n' "$BENCHMARK_RUNS_COMPLETED"
        printf 'benchmark_runs_expected=%s\n' "$EXPECTED_BENCHMARK_RUNS"
        printf 'warmup=%s\n' "$WARMUP"
        printf 'repeat=%s\n' "$REPEAT"
        printf 'seeds=%s\n' "$SEEDS_CSV"
        printf 'max_seqlen_upper_bound=none\n'
        printf 'execution=graph\n'
        printf 'backends=native,triton\n'
        printf 'python=%s\n' "$PYTHON_BIN"
    } >"$OUTPUT_DIR/status.txt"

    local candidate relative
    local -a artifacts=(
        status.txt repository-after.txt provenance.txt git-head.txt
        git-tree.txt environment.log correctness.log raw-validation.log
        module.sha256
        jit/module_flash_kda_hip.so
    )
    for candidate in \
        "$OUTPUT_DIR"/gva-*-seed-*.log \
        "$OUTPUT_DIR"/gva-*-seed-*.csv \
        "$OUTPUT_DIR"/gva-*-seed-*.json; do
        [[ -s $candidate ]] || continue
        relative="${candidate#"$OUTPUT_DIR"/}"
        artifacts+=("$relative")
    done
    local -a present_artifacts=()
    for candidate in "${artifacts[@]}"; do
        [[ -s $OUTPUT_DIR/$candidate ]] && present_artifacts+=("$candidate")
    done
    if ((${#present_artifacts[@]} > 0)); then
        (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${present_artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0)); then
        echo "FlashKDA GVA acceptance passed: $OUTPUT_DIR"
    else
        echo "FlashKDA GVA acceptance failed during $ACCEPTANCE_STAGE: $OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT
printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
{
    printf 'runner=scripts/run_flash_kda_gva_acceptance.sh\n'
    printf 'git_head=%s\n' "$INITIAL_HEAD"
    printf 'git_tree=%s\n' "$INITIAL_TREE"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'benchmark_entry=%s\n' \
        'op_tests/op_benchmarks/triton/bench_flash_kda_native.py'
    printf 'correctness_runs_expected=2\n'
    printf 'benchmark_suites_per_seed=%s\n' "$BENCHMARKS_PER_SEED"
    printf 'benchmark_runs_expected=%s\n' "$EXPECTED_BENCHMARK_RUNS"
    printf 'warmup=%s\n' "$WARMUP"
    printf 'repeat=%s\n' "$REPEAT"
    printf 'seeds=%s\n' "$SEEDS_CSV"
    printf 'min_speedup=%s\n' "$MIN_SPEEDUP"
    printf 'min_geomean_speedup=%s\n' "$MIN_GEOMEAN"
    printf 'min_paired_win_fraction=%s\n' "$MIN_WIN_FRACTION"
    printf 'max_seqlen_hint_contract=absent\n'
    printf 'max_seqlen_upper_bound=none\n'
    printf 'execution=graph\n'
    printf 'backends=native,triton\n'
    printf 'public_k3_routing=1\n'
} >"$OUTPUT_DIR/provenance.txt"
echo "FlashKDA GVA acceptance artifacts: $OUTPUT_DIR"

for prefix in AITER_ KDA_ FLASH_KDA_ CHUNK_DELTA_ATTN_ FLA_; do
    for name in $(compgen -e "$prefix" || true); do
        unset "$name"
    done
done
unset AITER_KDA_BACKEND AITER_TRITON_ONLY AITER_REBUILD
unset AITER_META_DIR CK_DIR HIP_KITTENS_DIR
export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
export GPU_ARCHS=gfx950
export MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1
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
arch = getattr(properties, "gcnArchName", "unknown").split(":", 1)[0]
if arch != "gfx950" or properties.multi_processor_count != 256:
    raise SystemExit(
        "acceptance requires the 256-CU gfx950 reference target; "
        f"got {arch}/{properties.multi_processor_count} CU"
    )
print(
    f"GPU: {torch.cuda.get_device_name(0)} "
    f"({arch}, {properties.multi_processor_count} CU); "
    f"PyTorch {torch.__version__}; ROCm {torch.version.hip}"
)
print(
    f"Python: {sys.executable}; {platform.python_version()}; "
    f"sha256={sha256(sys.executable)}"
)
print(f"Triton: {triton.__version__}; {triton.__file__}")
hipcc = shutil.which("hipcc")
print(f"hipcc: {hipcc}")
if hipcc is not None:
    version = subprocess.run(
        [hipcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(version)
print(f"AITER_JIT_DIR: {os.environ['AITER_JIT_DIR']}")
print(f"AITER_AOT_IMPORT: {os.environ.get('AITER_AOT_IMPORT')}")
print(f"TRITON_CACHE_DIR: {os.environ['TRITON_CACHE_DIR']}")
print(f"MAX_JOBS: {os.environ['MAX_JOBS']}")
PY

run_logged() {
    local label=$1
    shift
    ACCEPTANCE_STAGE=$label
    echo "===== $label ====="
    "$@" 2>&1 | tee "$OUTPUT_DIR/$label.log"
}

run_logged correctness env AITER_REBUILD=1 "$PYTHON_BIN" -m pytest -q \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native_python_api.py \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native.py
CORRECTNESS_RUNS_COMPLETED=1

# This owns the lower-level contract that pytest cannot observe: raw-v3 ABI,
# no-hint/exact/over-hint numerical equivalence, adversarial packed prefixes,
# captured topology, changed-prefix graph replay, and input-state immutability.
run_logged raw-validation env AITER_REBUILD=0 "$PYTHON_BIN" -u \
    op_tests/op_benchmarks/triton/validate_flash_kda_raw_path.py \
    --tokens 2048 --heads 12
CORRECTNESS_RUNS_COMPLETED=2

BENCHMARK=(
    "$PYTHON_BIN" -u op_tests/op_benchmarks/triton/bench_flash_kda_native.py
    --backend native --backend triton --execution graph --public-k3
    --omit-max-seqlen-hint
    --warmup "$WARMUP" --repeat "$REPEAT" --tolerance 0.04
    --require-arch gfx950 --require-compute-units 256
    --min-speedup "$MIN_SPEEDUP"
    --min-geomean-speedup "$MIN_GEOMEAN"
    --min-paired-win-fraction "$MIN_WIN_FRACTION"
)

run_benchmark() {
    local label=$1
    local seed=$2
    shift 2
    run_logged "$label" "${BENCHMARK[@]}" --seed "$seed" "$@" \
        --csv "$OUTPUT_DIR/$label.csv" \
        --raw-csv "$OUTPUT_DIR/$label.raw.csv" \
        --json "$OUTPUT_DIR/$label.json"
    for artifact in \
        "$OUTPUT_DIR/$label.log" \
        "$OUTPUT_DIR/$label.csv" \
        "$OUTPUT_DIR/$label.raw.csv" \
        "$OUTPUT_DIR/$label.json"; do
        [[ -s $artifact ]] || {
            echo "ERROR: successful benchmark did not produce $artifact" >&2
            exit 1
        }
    done
    BENCHMARK_RUNS_COMPLETED=$((BENCHMARK_RUNS_COMPLETED + 1))
}

# GVA cross-coverage includes ratio-2 and ratio-4 single/batched/ragged,
# fresh/resume, actual mixed decode+prefill batches, and the direct/hybrid
# 1024/1025 edge.
# Correctness and the initial HIP compile run once above; all seeds reuse that
# JIT directory and each preserve their own logs and machine-readable results.
for seed in "${SEEDS[@]}"; do
    run_benchmark "gva-hq2-hv4-core-seed-$seed" "$seed" \
        --suite core --heads 2 --value-heads 4
    run_benchmark "gva-hq2-hv8-stress-seed-$seed" "$seed" \
        --case single-2k --case ragged-16k --case resume-4x4k \
        --heads 2 --value-heads 8
    run_benchmark "gva-hq2-hv4-mixed-production-seed-$seed" "$seed" \
        --suite mixed-production --heads 2 --value-heads 4
    run_benchmark "gva-hq2-hv4-mixed-boundary-seed-$seed" "$seed" \
        --suite mixed-boundary --heads 2 --value-heads 4
    run_benchmark "gva-hq2-hv8-mixed-production-seed-$seed" "$seed" \
        --suite mixed-production --heads 2 --value-heads 8
    run_benchmark "gva-hq2-hv8-mixed-boundary-seed-$seed" "$seed" \
        --suite mixed-boundary --heads 2 --value-heads 8
done

ACCEPTANCE_STAGE=artifact-verification
[[ -s $AITER_JIT_DIR/module_flash_kda_hip.so ]] || {
    echo "ERROR: acceptance did not produce the native HIP module" >&2
    exit 1
}
(
    cd "$OUTPUT_DIR"
    sha256sum jit/module_flash_kda_hip.so
) >"$OUTPUT_DIR/module.sha256"
CURRENT_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
CURRENT_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
if [[ $CURRENT_HEAD != "$INITIAL_HEAD" || $CURRENT_TREE != "$INITIAL_TREE" ]] ||
    ! repo_is_clean; then
    echo "ERROR: acceptance changed HEAD or the source checkout" >&2
    exit 1
fi
ACCEPTANCE_STAGE=complete
