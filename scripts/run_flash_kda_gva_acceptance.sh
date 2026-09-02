#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 GVA acceptance. Coverage remains in the canonical pytest
# and benchmark files; this wrapper pins the formal invocation and independently
# verifies the resulting matrix and performance evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
readonly FORMAL_WARMUP=10
readonly FORMAL_REPEAT=50
readonly FORMAL_MIN_SPEEDUP=1.05
readonly FORMAL_MIN_GEOMEAN=1.05
readonly FORMAL_MIN_WIN_FRACTION=0.75
readonly FORMAL_HV4_LOGICAL_CASES=17
readonly FORMAL_HV8_LOGICAL_CASES=9
readonly FORMAL_LOGICAL_CASES=26
readonly FORMAL_SEED_CELLS=78
FORMAL_PLAN_SHA256=72c12f1de2926c6c91adc58902b47ea267fee4fded16e580fe2f7654152a0287
readonly FORMAL_PLAN_SHA256
readonly BENCHMARKS_PER_SEED=6

usage() {
    cat <<'EOF'
Usage: scripts/run_flash_kda_gva_acceptance.sh [--quick] [OUTPUT_DIR]

By default, run the formal FlashKDA GVA acceptance on exactly one visible
gfx950/256-CU GPU. The formal matrix is fixed at Hq=2 with 17 HV=4 and 9 HV=8
logical cases, crossed with seeds 42/43/44 (78 seed-cells). Timing and gates
are fixed at warmup=10, repeat=50, per-cell speedup >=1.05x, per-suite
geometric-mean speedup >=1.05x, and per-cell paired wins >=75%. Every case
uses graph replay, audits public/native/Triton graph routes and final replay,
and omits max_seqlen_upper_bound.

The output directory must be outside the source checkout. The default is a
new /tmp/flash-kda-gva-* directory.

--quick is explicitly non-formal. It permits KDA_ACCEPTANCE_SEEDS,
KDA_ACCEPTANCE_WARMUP, KDA_ACCEPTANCE_REPEAT, KDA_ACCEPTANCE_MIN_SPEEDUP,
KDA_ACCEPTANCE_MIN_GEOMEAN, and KDA_ACCEPTANCE_MIN_WIN_FRACTION overrides,
but it can never report a formal acceptance PASS.
EOF
}

MODE=formal
OUTPUT_ARGUMENT=
while (($# > 0)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --quick|--non-formal)
            if [[ $MODE == quick ]]; then
                echo "ERROR: --quick/--non-formal specified more than once" >&2
                exit 2
            fi
            MODE=quick
            ;;
        --)
            shift
            while (($# > 0)); do
                if [[ -n $OUTPUT_ARGUMENT ]]; then
                    usage >&2
                    exit 2
                fi
                OUTPUT_ARGUMENT=$1
                shift
            done
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n $OUTPUT_ARGUMENT ]]; then
                usage >&2
                exit 2
            fi
            OUTPUT_ARGUMENT=$1
            ;;
    esac
    shift
done

FORMAL=0
if [[ $MODE == formal ]]; then
    FORMAL=1
    for name in \
        KDA_ACCEPTANCE_SEEDS \
        KDA_ACCEPTANCE_WARMUP \
        KDA_ACCEPTANCE_REPEAT \
        KDA_ACCEPTANCE_MIN_SPEEDUP \
        KDA_ACCEPTANCE_MIN_GEOMEAN \
        KDA_ACCEPTANCE_MIN_WIN_FRACTION; do
        if [[ -v $name ]]; then
            echo "ERROR: $name cannot override formal acceptance; " \
                "unset it or explicitly use --quick" >&2
            exit 2
        fi
    done
    WARMUP=$FORMAL_WARMUP
    REPEAT=$FORMAL_REPEAT
    MIN_SPEEDUP=$FORMAL_MIN_SPEEDUP
    MIN_GEOMEAN=$FORMAL_MIN_GEOMEAN
    MIN_WIN_FRACTION=$FORMAL_MIN_WIN_FRACTION
    SEEDS=(42 43 44)
else
    WARMUP="${KDA_ACCEPTANCE_WARMUP:-2}"
    REPEAT="${KDA_ACCEPTANCE_REPEAT:-10}"
    MIN_SPEEDUP="${KDA_ACCEPTANCE_MIN_SPEEDUP:-1.0}"
    MIN_GEOMEAN="${KDA_ACCEPTANCE_MIN_GEOMEAN:-1.0}"
    MIN_WIN_FRACTION="${KDA_ACCEPTANCE_MIN_WIN_FRACTION:-0.5}"
    SEEDS_INPUT="${KDA_ACCEPTANCE_SEEDS:-42}"
    read -r -a SEEDS <<<"${SEEDS_INPUT//,/ }"
fi
if ((${#SEEDS[@]} == 0)); then
    echo "ERROR: seed list must contain at least one seed" >&2
    exit 2
fi
for seed in "${SEEDS[@]}"; do
    if [[ ! $seed =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed: $seed" >&2
        exit 2
    fi
done
readonly MODE FORMAL WARMUP REPEAT MIN_SPEEDUP MIN_GEOMEAN MIN_WIN_FRACTION
readonly -a SEEDS
SEEDS_CSV="$(IFS=,; printf '%s' "${SEEDS[*]}")"
readonly SEEDS_CSV
EXPECTED_BENCHMARK_RUNS=$((${#SEEDS[@]} * BENCHMARKS_PER_SEED))
readonly EXPECTED_BENCHMARK_RUNS

cd "$REPO_ROOT"
repo_is_clean() {
    local submodule_status untracked
    git -c safe.directory="$REPO_ROOT" diff --quiet -- || return 1
    git -c safe.directory="$REPO_ROOT" diff --cached --quiet -- || return 1
    untracked="$(git -c safe.directory="$REPO_ROOT" \
        ls-files --others --exclude-standard)" || return 1
    [[ -z $untracked ]] || return 1
    submodule_status="$(git -c safe.directory="$REPO_ROOT" \
        submodule status --recursive)" || return 1
    if grep -Eq '^[-+U]' <<<"$submodule_status"; then
        return 1
    fi
    git -c safe.directory="$REPO_ROOT" submodule foreach --recursive \
        --quiet 'submodule_tree="$(git status --porcelain --untracked-files=all)" || exit $?; test -z "$submodule_tree"'
}

if ! repo_is_clean; then
    echo "ERROR: acceptance requires a clean checkout" >&2
    exit 2
fi
INITIAL_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
INITIAL_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"

if [[ -n $OUTPUT_ARGUMENT ]]; then
    OUTPUT_DIR="$(realpath -m -- "$OUTPUT_ARGUMENT")"
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
    if ! {
        printf 'initial_head=%s\n' "$INITIAL_HEAD"
        printf 'final_head=%s\n' "$final_head"
        printf 'initial_tree=%s\n' "$INITIAL_TREE"
        printf 'final_tree=%s\n' "$final_tree"
        printf 'clean_and_unchanged=%s\n' "$repository_ok"
    } >"$OUTPUT_DIR/repository-after.txt"; then
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=repository-evidence-finalization
        fi
    fi

    write_acceptance_status() {
        local status=failed
        local formal_pass=0
        if ((exit_code == 0 && FORMAL == 1)); then
            status=passed
            formal_pass=1
        elif ((exit_code == 0)); then
            status=non-formal-completed
        fi
        {
            printf 'status=%s\n' "$status"
            printf 'mode=%s\n' "$MODE"
            printf 'formal=%s\n' "$FORMAL"
            printf 'formal_pass=%s\n' "$formal_pass"
            printf 'formal_public_witness=%s\n' "$FORMAL"
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
            printf 'minimum_speedup=%s\n' "$MIN_SPEEDUP"
            printf 'minimum_geomean_speedup=%s\n' "$MIN_GEOMEAN"
            printf 'minimum_paired_win_fraction=%s\n' "$MIN_WIN_FRACTION"
            printf 'formal_hv4_logical_cases=%s\n' \
                "$FORMAL_HV4_LOGICAL_CASES"
            printf 'formal_hv8_logical_cases=%s\n' \
                "$FORMAL_HV8_LOGICAL_CASES"
            printf 'formal_logical_cases_per_seed=%s\n' \
                "$FORMAL_LOGICAL_CASES"
            printf 'formal_seed_cells=%s\n' "$FORMAL_SEED_CELLS"
            printf 'formal_plan_sha256=%s\n' "$FORMAL_PLAN_SHA256"
            printf 'max_seqlen_upper_bound=none\n'
            printf 'execution=graph\n'
            printf 'backends=native,triton\n'
            printf 'python=%s\n' "$PYTHON_BIN"
        } >"$OUTPUT_DIR/status.txt"
    }
    if ! write_acceptance_status; then
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=status-finalization
        fi
    fi

    local candidate relative
    local -a artifacts=(
        status.txt repository-after.txt provenance.txt git-head.txt
        git-tree.txt environment.log correctness.log raw-validation.log
        artifact-validation.log formal-matrix.json formal-summary.json
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
        if ! (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${present_artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"; then
            rm -f -- "$OUTPUT_DIR/artifacts.sha256"
            if ((exit_code == 0)); then
                exit_code=1
                ACCEPTANCE_STAGE=artifact-checksum-finalization
            fi
        fi
    fi

    # The checksum or another finalization step may have changed the final
    # outcome after the first status write.  Rewrite status last so quick runs
    # cannot look formal and finalization failures cannot retain a PASS marker.
    if ! write_acceptance_status && ((exit_code == 0)); then
        exit_code=1
        ACCEPTANCE_STAGE=status-finalization
        rm -f -- "$OUTPUT_DIR/artifacts.sha256"
        write_acceptance_status || true
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0 && FORMAL == 1)); then
        echo "FlashKDA GVA formal acceptance PASS: $OUTPUT_DIR"
    elif ((exit_code == 0)); then
        echo "FlashKDA GVA non-formal quick run completed " \
            "(not a formal PASS): $OUTPUT_DIR"
    elif ((FORMAL == 1)); then
        echo "FlashKDA GVA formal acceptance failed during " \
            "$ACCEPTANCE_STAGE: $OUTPUT_DIR" >&2
    else
        echo "FlashKDA GVA non-formal quick run failed during " \
            "$ACCEPTANCE_STAGE: $OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT
printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
{
    printf 'runner=scripts/run_flash_kda_gva_acceptance.sh\n'
    printf 'mode=%s\n' "$MODE"
    printf 'formal=%s\n' "$FORMAL"
    printf 'formal_public_witness=%s\n' "$FORMAL"
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
    printf 'formal_hv4_logical_cases=%s\n' "$FORMAL_HV4_LOGICAL_CASES"
    printf 'formal_hv8_logical_cases=%s\n' "$FORMAL_HV8_LOGICAL_CASES"
    printf 'formal_logical_cases_per_seed=%s\n' "$FORMAL_LOGICAL_CASES"
    printf 'formal_seed_cells=%s\n' "$FORMAL_SEED_CELLS"
    printf 'formal_plan_sha256=%s\n' "$FORMAL_PLAN_SHA256"
    printf 'max_seqlen_hint_contract=absent\n'
    printf 'max_seqlen_upper_bound=none\n'
    printf 'execution=graph\n'
    printf 'backends=native,triton\n'
    printf 'public_k3_routing=1\n'
} >"$OUTPUT_DIR/provenance.txt"
if ((FORMAL == 1)); then
    echo "FlashKDA GVA formal acceptance artifacts: $OUTPUT_DIR"
else
    echo "FlashKDA GVA non-formal quick artifacts: $OUTPUT_DIR"
fi

for prefix in AITER_ KDA_ FLASH_KDA_ CHUNK_DELTA_ATTN_ FLA_; do
    for name in $(compgen -e "$prefix" || true); do
        unset "$name"
    done
done
unset AITER_KDA_BACKEND AITER_TRITON_ONLY AITER_REBUILD
unset AITER_META_DIR CK_DIR HIP_KITTENS_DIR PYTHONOPTIMIZE
export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
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
print(f"PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")
print(f"PYTHONOPTIMIZE: {os.environ.get('PYTHONOPTIMIZE')}")
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
if ((FORMAL == 1)); then
    BENCHMARK+=(--formal-public-witness)
fi

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

if ((BENCHMARK_RUNS_COMPLETED != EXPECTED_BENCHMARK_RUNS)); then
    echo "ERROR: completed $BENCHMARK_RUNS_COMPLETED benchmark suites; " \
        "expected $EXPECTED_BENCHMARK_RUNS" >&2
    exit 1
fi

if ((FORMAL == 1)); then
    ACCEPTANCE_STAGE=artifact-validation
    "$PYTHON_BIN" - \
        "$OUTPUT_DIR" \
        "$INITIAL_HEAD" \
        "$FORMAL_PLAN_SHA256" \
        "$FORMAL_MIN_SPEEDUP" \
        "$FORMAL_MIN_GEOMEAN" \
        "$FORMAL_MIN_WIN_FRACTION" <<'PY' \
        2>&1 | tee "$OUTPUT_DIR/artifact-validation.log"
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys


root = pathlib.Path(sys.argv[1]).resolve()
expected_head = sys.argv[2]
expected_plan_sha256 = sys.argv[3]
min_speedup = float(sys.argv[4])
min_geomean = float(sys.argv[5])
min_win_fraction = float(sys.argv[6])
seeds = (42, 43, 44)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


TRITON_COMPACT_PATTERNS = (
    "flash_kda_prepare",
    "flash_kda_segment",
    "flash_kda_seg_scan",
)
TRITON_GVA_STAGE_PATTERNS = (
    "l2norm_fwd_kernel",
    "beta_sigmoid_fwd_kernel",
    "chunk_gate_cumsum_kernel",
    "chunk_delta_attn_fwd_kernel_intra_sub_chunk",
    "chunk_delta_attn_fwd_kernel_inter_solve",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel",
    "chunk_gla_fwd_kernel_o",
)


def canonical_graph_signature(records):
    entries = [
        (
            record.get("node_type"),
            record.get("name"),
            tuple(record.get("grid", [])),
            tuple(record.get("block", [])),
            record.get("shared_mem_bytes"),
        )
        for record in records
    ]
    return tuple(sorted(entries, key=repr))


def validate_gva_route_audit(route_audit, label):
    """Recompute every graph-route claim from the serialized HIP records."""

    for key in ("graph_handles", "stream_handles"):
        handles = route_audit.get(key)
        require(
            isinstance(handles, list) and len(handles) == 3,
            f"{label}: {key} cardinality",
        )
        require(
            all(type(value) is int and value > 0 for value in handles),
            f"{label}: invalid {key}",
        )
        require(len(set(handles)) == 3, f"{label}: aliased {key}")

    audited_backends = route_audit.get("backends")
    require(isinstance(audited_backends, dict), f"{label}: audited backends")
    require(
        set(audited_backends) == {"native", "explicit-native", "triton"},
        f"{label}: route backend set",
    )

    records_by_backend = {}
    for backend in ("native", "explicit-native", "triton"):
        route = audited_backends[backend]
        require(isinstance(route, dict), f"{label}/{backend}: route record")
        records = route.get("records")
        require(isinstance(records, list) and records,
                f"{label}/{backend}: missing graph records")
        require(
            type(route.get("node_count")) is int
            and route["node_count"] == len(records),
            f"{label}/{backend}: node count",
        )

        kernels = []
        for expected_index, record in enumerate(records):
            require(isinstance(record, dict),
                    f"{label}/{backend}: malformed graph record")
            require(type(record.get("index")) is int
                    and record["index"] == expected_index,
                    f"{label}/{backend}: noncanonical node index")
            node_type = record.get("node_type")
            require(type(node_type) is int and node_type >= 0,
                    f"{label}/{backend}: invalid node type")
            if node_type != 0:
                continue
            name = record.get("name")
            require(
                isinstance(name, str) and name not in ("", "<unknown>"),
                f"{label}/{backend}: unresolved kernel name",
            )
            for field in ("grid", "block"):
                dimensions = record.get(field)
                require(
                    isinstance(dimensions, list)
                    and len(dimensions) == 3
                    and all(type(value) is int and value > 0
                            for value in dimensions),
                    f"{label}/{backend}: invalid {field} for {name}",
                )
            shared_mem = record.get("shared_mem_bytes")
            require(type(shared_mem) is int and shared_mem >= 0,
                    f"{label}/{backend}: invalid shared memory for {name}")
            kernels.append(record)

        names = [record["name"] for record in kernels]
        require(
            type(route.get("kernel_node_count")) is int
            and route["kernel_node_count"] == len(kernels)
            and len(kernels) > 0,
            f"{label}/{backend}: kernel node count",
        )
        require(route.get("kernel_names") == names,
                f"{label}/{backend}: kernel name witness mismatch")

        lowered = [name.lower() for name in names]
        has_native_k1 = any("k1_kda_" in name for name in lowered)
        has_native_k2 = any("k2_kda_" in name for name in lowered)
        has_compact_triton = any(
            pattern in name
            for name in lowered
            for pattern in TRITON_COMPACT_PATTERNS
        )
        gva_stage_matches = {
            pattern: [index for index, name in enumerate(lowered)
                      if pattern in name]
            for pattern in TRITON_GVA_STAGE_PATTERNS
        }
        has_any_gva_stage = any(gva_stage_matches.values())

        if backend in ("native", "explicit-native"):
            require(has_native_k1 and has_native_k2,
                    f"{label}/{backend}: missing native K1/K2 anchors")
            require(not has_compact_triton and not has_any_gva_stage,
                    f"{label}/{backend}: mixed native/Triton graph")
            expected_route = "native-hip-k1-k2"
        else:
            require(not has_native_k1 and not has_native_k2,
                    f"{label}/{backend}: native kernels in Triton graph")
            require(not has_compact_triton,
                    f"{label}/{backend}: compact equal-head Triton graph")
            missing = [
                pattern for pattern, matches in gva_stage_matches.items()
                if not matches
            ]
            require(not missing,
                    f"{label}/{backend}: missing GVA stages {missing}")
            matched_indices = [
                index
                for matches in gva_stage_matches.values()
                for index in matches
            ]
            require(len(set(matched_indices)) >= len(TRITON_GVA_STAGE_PATTERNS),
                    f"{label}/{backend}: aliased GVA stage anchors")
            expected_route = "triton-gva-chunk-kda"

        require(route.get("route_verified") is True,
                f"{label}/{backend}: route proof")
        require(route.get("route") == expected_route,
                f"{label}/{backend}: route label")
        records_by_backend[backend] = records

    require(
        canonical_graph_signature(records_by_backend["native"])
        == canonical_graph_signature(records_by_backend["explicit-native"]),
        f"{label}: public/explicit graph signature mismatch",
    )


def make_case(name, seq_lens, *, resume=False, resume_mask=None):
    if resume_mask is None:
        state = "resume" if resume else "fresh"
    elif any(resume_mask):
        state = "resume" if all(resume_mask) else "mixed"
    else:
        state = "fresh"
    return {
        "name": name,
        "seq_lens": list(seq_lens),
        "resume": resume,
        "resume_mask": (
            list(resume_mask) if resume_mask is not None else None
        ),
        "state": state,
    }


core = [
    make_case("single-128", (128,)),
    make_case("single-256", (256,)),
    make_case("single-512", (512,)),
    make_case("single-1k", (1024,)),
    make_case("single-2k", (2048,)),
    make_case("single-8k", (8192,)),
    make_case("single-16k", (16384,)),
    make_case("batch-16x1k", (1024,) * 16),
    make_case("batch-64x256", (256,) * 64),
    make_case(
        "ragged-16k",
        (127, 255, 511, 1023, 2047, 3073, 4095, 5253),
    ),
    make_case("resume-4x4k", (4096,) * 4, resume=True),
]


def mixed_case(decodes):
    return make_case(
        f"mixed-{decodes}d-budget16k",
        (1,) * decodes + (16384 - decodes,),
        resume_mask=(True,) * decodes + (False,),
    )


mixed_production = [mixed_case(decodes) for decodes in (7, 8, 32, 63)]
mixed_boundary = [
    make_case(
        "mixed-15d-prefill-1024",
        (1,) * 15 + (1024,),
        resume_mask=(True,) * 15 + (False,),
    ),
    make_case(
        "mixed-15d-prefill-1025",
        (1,) * 15 + (1025,),
        resume_mask=(True,) * 15 + (False,),
    ),
]
stress = [core[4], core[9], core[10]]
manifest = [
    {
        "run": "gva-hq2-hv4-core",
        "heads": 2,
        "value_heads": 4,
        "selection_mode": "suite",
        "suite": "core",
        "cases": core,
    },
    {
        "run": "gva-hq2-hv4-mixed-production",
        "heads": 2,
        "value_heads": 4,
        "selection_mode": "suite",
        "suite": "mixed-production",
        "cases": mixed_production,
    },
    {
        "run": "gva-hq2-hv4-mixed-boundary",
        "heads": 2,
        "value_heads": 4,
        "selection_mode": "suite",
        "suite": "mixed-boundary",
        "cases": mixed_boundary,
    },
    {
        "run": "gva-hq2-hv8-stress",
        "heads": 2,
        "value_heads": 8,
        "selection_mode": "case",
        "suite": None,
        "cases": stress,
    },
    {
        "run": "gva-hq2-hv8-mixed-production",
        "heads": 2,
        "value_heads": 8,
        "selection_mode": "suite",
        "suite": "mixed-production",
        "cases": mixed_production,
    },
    {
        "run": "gva-hq2-hv8-mixed-boundary",
        "heads": 2,
        "value_heads": 8,
        "selection_mode": "suite",
        "suite": "mixed-boundary",
        "cases": mixed_boundary,
    },
]
formal_plan = {
    "schema": "flash-kda-gva-formal-plan-v1",
    "seeds": list(seeds),
    "warmup": 10,
    "repeat": 50,
    "tolerance": 0.04,
    "thresholds": {
        "minimum_speedup": min_speedup,
        "minimum_geomean_speedup": min_geomean,
        "minimum_paired_win_fraction": min_win_fraction,
    },
    "execution": "graph",
    "backends": ["native", "triton"],
    "head_dim": 128,
    "tp_size": 8,
    "initial_state_contract": "materialized-fp32-v-first",
    "public_k3": True,
    "formal_public_witness": True,
    "omit_max_seqlen_hint": True,
    "max_seqlen_hint_contract": "omitted-none",
    "require_arch": "gfx950",
    "require_compute_units": 256,
    "manifest": manifest,
}
plan_bytes = json.dumps(
    formal_plan, sort_keys=True, separators=(",", ":")
).encode()
plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
require(plan_sha256 == expected_plan_sha256, "embedded formal plan hash drift")

expected_module = (root / "jit/module_flash_kda_hip.so").resolve()
require(expected_module.is_file() and expected_module.stat().st_size > 0,
        "native module is missing before artifact validation")
module_digest = hashlib.sha256()
with expected_module.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        module_digest.update(block)
expected_module_sha256 = module_digest.hexdigest()

expected_paths = {
    root / f"{entry['run']}-seed-{seed}.json"
    for seed in seeds
    for entry in manifest
}
actual_paths = set(root.glob("gva-*-seed-*.json"))
require(actual_paths == expected_paths, "missing or unexpected benchmark JSON")

suite_summaries = []
seed_native_rows = {seed: [] for seed in seeds}
actual_manifest = []
for entry in manifest:
    canonical_cases = None
    for seed in seeds:
        stem = f"{entry['run']}-seed-{seed}"
        json_path = root / f"{stem}.json"
        for suffix in (".log", ".csv", ".raw.csv", ".json"):
            artifact = root / f"{stem}{suffix}"
            require(artifact.is_file() and artifact.stat().st_size > 0,
                    f"missing or empty artifact: {artifact.name}")
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_path = root / f"{stem}.raw.csv"
        with raw_path.open(newline="", encoding="utf-8") as handle:
            raw_reader = csv.DictReader(handle)
            raw_rows = list(raw_reader)
            require(set(raw_reader.fieldnames or ()) == {
                "backend", "case", "execution", "heads", "latency_ms",
                "max_seqlen_upper_bound", "order", "round", "seed",
                "sequences", "state", "tokens", "value_heads", "packed",
                "cu_seqlens_is_none",
                "public_max_seqlen_upper_bound_keyword_omitted",
                "native_policy_effective_max_seqlen_upper_bound",
            }, f"{stem}: raw CSV schema")
        require(len(raw_rows) == len(entry["cases"]) * 2 * 50,
                f"{stem}: raw CSV row count")
        raw_by_key = {}
        expected_case_by_name = {
            case["name"]: case for case in entry["cases"]
        }
        for raw in raw_rows:
            case_name = raw.get("case")
            backend = raw.get("backend")
            require(case_name in expected_case_by_name,
                    f"{stem}: raw CSV case")
            require(backend in ("native", "triton"),
                    f"{stem}: raw CSV backend")
            try:
                round_index = int(raw["round"])
                order = int(raw["order"])
                latency_ms = float(raw["latency_ms"])
                raw_seed = int(raw["seed"])
                raw_heads = int(raw["heads"])
                raw_value_heads = int(raw["value_heads"])
                raw_sequences = int(raw["sequences"])
                raw_tokens = int(raw["tokens"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"{stem}: malformed raw CSV row") from error
            case = expected_case_by_name[case_name]
            require(0 <= round_index < 50, f"{stem}: raw CSV round")
            expected_order = (
                round_index % 2 if backend == "native" else 1 - round_index % 2
            )
            require(order == expected_order, f"{stem}: raw CSV backend order")
            require(math.isfinite(latency_ms) and latency_ms > 0.0,
                    f"{stem}: raw CSV latency")
            require(raw.get("execution") == "graph",
                    f"{stem}: raw CSV execution")
            require(raw.get("max_seqlen_upper_bound") == "",
                    f"{stem}: raw CSV max-seqlen hint")
            require(raw.get("packed") == "True",
                    f"{stem}: raw CSV packed layout")
            require(raw.get("cu_seqlens_is_none") == "False",
                    f"{stem}: raw CSV cu_seqlens contract")
            require(
                raw.get(
                    "public_max_seqlen_upper_bound_keyword_omitted"
                ) == "True",
                f"{stem}: raw CSV public max-seqlen keyword",
            )
            require(
                raw.get(
                    "native_policy_effective_max_seqlen_upper_bound"
                ) == "",
                f"{stem}: raw CSV native effective max-seqlen",
            )
            require(raw_seed == seed, f"{stem}: raw CSV seed")
            require(raw_heads == entry["heads"], f"{stem}: raw CSV Hq")
            require(raw_value_heads == entry["value_heads"],
                    f"{stem}: raw CSV HV")
            require(raw_sequences == len(case["seq_lens"]),
                    f"{stem}: raw CSV sequence count")
            require(raw_tokens == sum(case["seq_lens"]),
                    f"{stem}: raw CSV token count")
            require(raw.get("state") == case["state"],
                    f"{stem}: raw CSV state")
            key = (case_name, backend, round_index)
            require(key not in raw_by_key, f"{stem}: duplicate raw CSV row")
            raw_by_key[key] = latency_ms

        config = payload.get("configuration")
        require(isinstance(config, dict), f"{stem}: configuration")
        expected_config = {
            "warmup": 10,
            "repeat": 50,
            "seed": seed,
            "heads": entry["heads"],
            "value_heads": entry["value_heads"],
            "execution": "graph",
            "selection_mode": entry["selection_mode"],
            "requested_suite": (
                entry["suite"]
                if entry["selection_mode"] == "suite"
                else "core"
            ),
            "suite": entry["suite"],
            "tolerance": 0.04,
            "backends": ["native", "triton"],
            "public_k3": True,
            "formal_public_witness": True,
            "omit_max_seqlen_hint": True,
            "max_seqlen_hint_contract": "omitted-none",
            "min_speedup": min_speedup,
            "min_geomean_speedup": min_geomean,
            "min_paired_win_fraction": min_win_fraction,
            "require_arch": "gfx950",
            "require_compute_units": 256,
        }
        require(config == expected_config, f"{stem}: configuration drift")

        environment = payload.get("environment")
        require(isinstance(environment, dict), f"{stem}: environment")
        require(environment.get("git_head") == expected_head,
                f"{stem}: git HEAD mismatch")
        require(environment.get("git_status_porcelain") == "",
                f"{stem}: benchmark observed a dirty checkout")
        require(environment.get("arch") == "gfx950", f"{stem}: GPU arch")
        require(environment.get("compute_units") == 256,
                f"{stem}: GPU CU count")
        require(environment.get("heads") == entry["heads"],
                f"{stem}: environment Hq")
        require(environment.get("value_heads") == entry["value_heads"],
                f"{stem}: environment HV")
        require(environment.get("head_dim") == 128, f"{stem}: head dim")
        require(environment.get("tp_size") == 8, f"{stem}: TP size")
        module_path = environment.get("module_path")
        require(isinstance(module_path, str), f"{stem}: native module path")
        require(pathlib.Path(module_path).resolve() == expected_module,
                f"{stem}: native module path differs from AITER_JIT_DIR")
        module_sha256 = environment.get("module_sha256")
        require(isinstance(module_sha256, str) and len(module_sha256) == 64,
                f"{stem}: native module SHA256")
        require(module_sha256 == expected_module_sha256,
                f"{stem}: native module differs from acceptance DSO")
        loaded_modules = environment.get("loaded_module_identities")
        require(isinstance(loaded_modules, dict), f"{stem}: loaded modules")
        loaded_native = loaded_modules.get("module_flash_kda_hip")
        require(isinstance(loaded_native, dict),
                f"{stem}: loaded native module identity")
        require(loaded_native.get("matches_expected_jit_path") is True,
                f"{stem}: loaded native module path")
        require(loaded_native.get("path") == str(expected_module),
                f"{stem}: loaded native module pathname")
        require(loaded_native.get("sha256") == module_sha256,
                f"{stem}: loaded native module SHA256 mismatch")
        controlled = environment.get("controlled_environment")
        require(isinstance(controlled, dict), f"{stem}: controlled environment")
        require("AITER_KDA_BACKEND" not in controlled,
                f"{stem}: forced KDA backend")
        require("AITER_TRITON_ONLY" not in controlled,
                f"{stem}: Triton-only environment")
        require(not any(key.startswith("FLASH_KDA_") for key in controlled),
                f"{stem}: active native route override")
        require(controlled.get("PYTHONHASHSEED") == "0",
                f"{stem}: PYTHONHASHSEED is not pinned")
        require("PYTHONOPTIMIZE" not in controlled,
                f"{stem}: Python assertions are disabled")

        actual_cases = payload.get("cases")
        require(isinstance(actual_cases, list), f"{stem}: cases")
        stripped_cases = []
        for case in actual_cases:
            require(case.get("max_seqlen_upper_bound") is None,
                    f"{stem}/{case.get('name')}: max-seqlen hint is present")
            require(case.get("packed") is True,
                    f"{stem}/{case.get('name')}: packed layout")
            require(case.get("cu_seqlens_is_none") is False,
                    f"{stem}/{case.get('name')}: cu_seqlens contract")
            require(case.get("observed_max_seqlen") == max(case["seq_lens"]),
                    f"{stem}/{case.get('name')}: observed maximum")
            stripped_cases.append(
                {
                    key: case.get(key)
                    for key in (
                        "name",
                        "seq_lens",
                        "resume",
                        "resume_mask",
                        "state",
                    )
                }
            )
        require(stripped_cases == entry["cases"], f"{stem}: case manifest")
        if canonical_cases is None:
            canonical_cases = stripped_cases
        else:
            require(stripped_cases == canonical_cases,
                    f"{stem}: cross-seed manifest drift")

        rows = payload.get("results")
        require(isinstance(rows, list), f"{stem}: results")
        expected_row_keys = {
            (case["name"], backend)
            for case in entry["cases"]
            for backend in ("native", "triton")
        }
        actual_row_keys = {
            (row.get("case"), row.get("backend")) for row in rows
        }
        require(len(rows) == len(expected_row_keys), f"{stem}: result count")
        require(actual_row_keys == expected_row_keys,
                f"{stem}: missing, duplicate, or unexpected result row")
        by_key = {(row["case"], row["backend"]): row for row in rows}
        suite_speedups = []
        for case in entry["cases"]:
            native = by_key[(case["name"], "native")]
            triton = by_key[(case["name"], "triton")]
            for backend, row in (("native", native), ("triton", triton)):
                require(row.get("seed") == seed, f"{stem}: row seed")
                require(row.get("heads") == 2, f"{stem}: row Hq")
                require(row.get("value_heads") == entry["value_heads"],
                        f"{stem}: row HV")
                require(row.get("execution") == "graph",
                        f"{stem}: row execution")
                require(row.get("max_seqlen_upper_bound") is None,
                        f"{stem}: row max-seqlen hint")
                require(row.get("packed") is True,
                        f"{stem}: row packed layout")
                require(row.get("cu_seqlens_is_none") is False,
                        f"{stem}: row cu_seqlens contract")
                require(
                    row.get(
                        "public_max_seqlen_upper_bound_keyword_omitted"
                    ) is True,
                    f"{stem}: row public max-seqlen keyword",
                )
                require(
                    row.get(
                        "native_policy_effective_max_seqlen_upper_bound"
                    ) is None,
                    f"{stem}: row native effective max-seqlen",
                )
                require(row.get("samples") == 50, f"{stem}: sample count")
                require(row.get("output_contract_verified") is True,
                        f"{stem}/{case['name']}/{backend}: output contract")
                require(row.get("input_resume_mask_verified") is True,
                        f"{stem}/{case['name']}/{backend}: state mask")
                require(row.get("input_initial_state_unchanged") is True,
                        f"{stem}/{case['name']}/{backend}: state mutation")
                require(row.get("input_state_immutability_checks", 0) > 0,
                        f"{stem}/{case['name']}/{backend}: state checks")
                require(row.get("input_initial_state_literal_none") is False,
                        f"{stem}/{case['name']}/{backend}: state presence")
                require(row.get("input_initial_state_dtype") == "torch.float32",
                        f"{stem}/{case['name']}/{backend}: state dtype")
                require(row.get("graph_eager_output_bitwise_equal") is True,
                        f"{stem}/{case['name']}/{backend}: graph output")
                require(row.get("graph_eager_final_state_bitwise_equal") is True,
                        f"{stem}/{case['name']}/{backend}: graph state")
                require(
                    row.get("final_graph_eager_output_bitwise_equal") is True,
                    f"{stem}/{case['name']}/{backend}: final graph output",
                )
                require(
                    row.get("final_graph_eager_final_state_bitwise_equal") is True,
                    f"{stem}/{case['name']}/{backend}: final graph state",
                )
                require(row.get("error_reference") == "triton",
                        f"{stem}/{case['name']}/{backend}: reference")
                for metric in (
                    "output_relative_rms",
                    "output_max_sequence_relative_rms",
                    "state_relative_rms",
                    "state_max_sequence_relative_rms",
                ):
                    value = row.get(metric)
                    require(finite_number(value),
                            f"{stem}/{case['name']}/{backend}: {metric}")
                    require(0.0 <= value <= 0.04,
                            f"{stem}/{case['name']}/{backend}: {metric} gate")
                if case["state"] == "mixed":
                    value = row.get("decode_output_relative_rms")
                    require(finite_number(value),
                            f"{stem}/{case['name']}/{backend}: decode RMS")
                    require(0.0 <= value <= 0.04,
                            f"{stem}/{case['name']}/{backend}: decode RMS gate")
            require(native.get("public_default_bitwise_native") is True,
                    f"{stem}/{case['name']}: public resolver is not native")
            route_audit = native.get("graph_route_audit")
            require(isinstance(route_audit, dict),
                    f"{stem}/{case['name']}: graph route witness")
            validate_gva_route_audit(
                route_audit, f"{stem}/{case['name']}"
            )
            require(route_audit.get("all_routes_verified") is True,
                    f"{stem}/{case['name']}: graph routes")
            require(route_audit.get("graphs_independent") is True,
                    f"{stem}/{case['name']}: graph independence")
            require(route_audit.get("streams_independent") is True,
                    f"{stem}/{case['name']}: stream independence")
            require(
                route_audit.get("public_explicit_graph_signatures_equal") is True,
                f"{stem}/{case['name']}: public/explicit graph signature",
            )
            audited_backends = route_audit.get("backends")
            require(isinstance(audited_backends, dict),
                    f"{stem}/{case['name']}: audited backends")
            require(set(audited_backends) == {
                "native", "explicit-native", "triton"
            }, f"{stem}/{case['name']}: route backend set")
            for route_backend, expected_route in (
                ("native", "native-hip-k1-k2"),
                ("explicit-native", "native-hip-k1-k2"),
                ("triton", "triton-gva-chunk-kda"),
            ):
                route = audited_backends[route_backend]
                require(route.get("route_verified") is True,
                        f"{stem}/{case['name']}/{route_backend}: route proof")
                require(route.get("route") == expected_route,
                        f"{stem}/{case['name']}/{route_backend}: route")
            native_ms = native.get("latency_median_ms")
            triton_ms = triton.get("latency_median_ms")
            speedup = native.get("speedup_vs_triton")
            win_fraction = native.get("paired_win_fraction")
            require(all(
                finite_number(value)
                for value in (native_ms, triton_ms, speedup, win_fraction)
            ), f"{stem}/{case['name']}: nonfinite performance result")
            require(native_ms > 0.0 and triton_ms > 0.0,
                    f"{stem}/{case['name']}: nonpositive latency")
            require(math.isclose(speedup, triton_ms / native_ms,
                                 rel_tol=1e-12, abs_tol=1e-12),
                    f"{stem}/{case['name']}: inconsistent speedup")
            native_samples = [
                raw_by_key[(case["name"], "native", index)]
                for index in range(50)
            ]
            triton_samples = [
                raw_by_key[(case["name"], "triton", index)]
                for index in range(50)
            ]
            require(math.isclose(native_ms, statistics.median(native_samples),
                                 rel_tol=1e-12, abs_tol=1e-12),
                    f"{stem}/{case['name']}: native median vs raw CSV")
            require(math.isclose(triton_ms, statistics.median(triton_samples),
                                 rel_tol=1e-12, abs_tol=1e-12),
                    f"{stem}/{case['name']}: Triton median vs raw CSV")
            raw_win_fraction = sum(
                native_sample < triton_sample
                for native_sample, triton_sample in zip(
                    native_samples, triton_samples
                )
            ) / 50
            require(math.isclose(win_fraction, raw_win_fraction,
                                 rel_tol=0.0, abs_tol=1e-12),
                    f"{stem}/{case['name']}: paired wins vs raw CSV")
            raw_paired_speedup = statistics.median(
                triton_sample / native_sample
                for native_sample, triton_sample in zip(
                    native_samples, triton_samples
                )
            )
            paired_speedup = native.get("paired_speedup_median")
            require(finite_number(paired_speedup),
                    f"{stem}/{case['name']}: paired speedup")
            require(math.isclose(paired_speedup, raw_paired_speedup,
                                 rel_tol=1e-12, abs_tol=1e-12),
                    f"{stem}/{case['name']}: paired speedup vs raw CSV")
            require(speedup >= min_speedup,
                    f"{stem}/{case['name']}: speedup {speedup:.6f}x")
            require(win_fraction >= min_win_fraction,
                    f"{stem}/{case['name']}: paired wins {win_fraction:.1%}")
            suite_speedups.append(float(speedup))
            seed_native_rows[seed].append(native)

        geomean = math.exp(
            math.fsum(math.log(value) for value in suite_speedups)
            / len(suite_speedups)
        )
        require(geomean >= min_geomean,
                f"{stem}: geomean {geomean:.6f}x")
        suite_summaries.append(
            {
                "run": entry["run"],
                "seed": seed,
                "logical_cases": len(entry["cases"]),
                "minimum_speedup": min(suite_speedups),
                "geomean_speedup": geomean,
                "minimum_paired_win_fraction": min(
                    row["paired_win_fraction"]
                    for row in seed_native_rows[seed]
                    if row["case"] in {case["name"] for case in entry["cases"]}
                    and row["value_heads"] == entry["value_heads"]
                ),
            }
        )

    actual_manifest.append(
        {
            "run": entry["run"],
            "heads": entry["heads"],
            "value_heads": entry["value_heads"],
            "selection_mode": entry["selection_mode"],
            "suite": entry["suite"],
            "cases": canonical_cases,
        }
    )

require(actual_manifest == manifest, "formal manifest differs from expectation")
actual_plan = dict(formal_plan)
actual_plan["manifest"] = actual_manifest
actual_plan_bytes = json.dumps(
    actual_plan, sort_keys=True, separators=(",", ":")
).encode()
require(hashlib.sha256(actual_plan_bytes).hexdigest() == expected_plan_sha256,
        "artifact formal plan hash mismatch")

seed_summaries = []
all_speedups = []
for seed, rows in seed_native_rows.items():
    hv4_count = sum(row["value_heads"] == 4 for row in rows)
    hv8_count = sum(row["value_heads"] == 8 for row in rows)
    require(hv4_count == 17, f"seed {seed}: expected 17 HV=4 cells")
    require(hv8_count == 9, f"seed {seed}: expected 9 HV=8 cells")
    require(len(rows) == 26, f"seed {seed}: expected 26 logical cells")
    speedups = [float(row["speedup_vs_triton"]) for row in rows]
    geomean = math.exp(
        math.fsum(math.log(value) for value in speedups) / len(speedups)
    )
    require(geomean >= min_geomean,
            f"seed {seed}: matrix geomean {geomean:.6f}x")
    seed_summaries.append(
        {
            "seed": seed,
            "hv4_logical_cases": hv4_count,
            "hv8_logical_cases": hv8_count,
            "logical_cases": len(rows),
            "minimum_speedup": min(speedups),
            "geomean_speedup": geomean,
            "minimum_paired_win_fraction": min(
                float(row["paired_win_fraction"]) for row in rows
            ),
        }
    )
    all_speedups.extend(speedups)
require(len(all_speedups) == 78, "expected 78 formal seed-cells")
global_geomean = math.exp(
    math.fsum(math.log(value) for value in all_speedups) / len(all_speedups)
)
require(global_geomean >= min_geomean,
        f"global geomean {global_geomean:.6f}x")

matrix_payload = {
    "schema": "flash-kda-gva-formal-matrix-v2",
    "plan_sha256": expected_plan_sha256,
    "logical_cases_per_seed": 26,
    "hv4_logical_cases_per_seed": 17,
    "hv8_logical_cases_per_seed": 9,
    "plan": formal_plan,
}
with (root / "formal-matrix.json").open("w", encoding="utf-8") as handle:
    json.dump(matrix_payload, handle, indent=2, allow_nan=False)
    handle.write("\n")
summary = {
    "schema": "flash-kda-gva-formal-acceptance-v2",
    "formal": True,
    # Overall formal PASS is written only by the shell EXIT handler after the
    # repository/module final checks.  This file attests the artifact/gate
    # portion without being able to mask a later finalization failure.
    "artifact_validation_pass": True,
    "performance_gate_passed": True,
    "raw_evidence_validated": True,
    "route_witness_validated": True,
    "native_module_identity_validated": True,
    "git_head": expected_head,
    "plan_sha256": expected_plan_sha256,
    "seeds": list(seeds),
    "seed_cells": len(all_speedups),
    "thresholds": {
        "warmup": 10,
        "repeat": 50,
        "minimum_speedup": min_speedup,
        "minimum_geomean_speedup": min_geomean,
        "minimum_paired_win_fraction": min_win_fraction,
    },
    "minimum_speedup": min(all_speedups),
    "geomean_speedup": global_geomean,
    "minimum_paired_win_fraction": min(
        float(row["paired_win_fraction"])
        for rows in seed_native_rows.values()
        for row in rows
    ),
    "seeds_summary": seed_summaries,
    "suites_summary": suite_summaries,
}
with (root / "formal-summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(
    "formal artifact validation: PASS "
    f"(17 HV4 + 9 HV8 = 26 logical cases/seed; "
    f"78/78 seed-cells; plan {expected_plan_sha256})"
)
PY
fi

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
