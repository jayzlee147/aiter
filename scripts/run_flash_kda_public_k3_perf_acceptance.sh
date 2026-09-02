#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 public-K3 performance acceptance. The benchmark logic and
# acceptance criteria remain in the canonical Python entry point; this script
# provides a clean, reproducible runner and preserves the resulting artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
readonly WARMUP=20
readonly REPEAT=120
readonly BOOTSTRAP_RESAMPLES=10000
readonly MIN_SPEEDUP=1.05
readonly EXPECTED_LOGICAL_CELLS=31
readonly EXPECTED_SEED_CELLS=93
readonly FORMAL_PLAN_SHA256=026ab17c1f2808441d488084c19884b62215ea59daedf332833fb853efedb7a7

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/run_flash_kda_public_k3_perf_acceptance.sh [OUTPUT_DIR]

Run the formal public-K3 performance acceptance on exactly one visible
gfx950/256-CU GPU. Select the GPU with ROCR_VISIBLE_DEVICES (or
HIP_VISIBLE_DEVICES) before invoking the script.

The output directory must be outside the source checkout. The default is a
new /tmp/flash-kda-public-k3-* directory. The formal seed set is 42, 43, and
44. Timing and statistics are pinned to warmup 20, repeat 120, and 10,000
bootstrap resamples. Every cell uses Hq=HV=12, materialized FP32 state, and
omits max_seqlen_upper_bound to reproduce the ATOM public call contract.
Every cell must reach at least 1.05x HIP-over-Triton p50 speedup in every seed,
in addition to the paired-win and bootstrap-confidence gates.
EOF
    exit 0
fi
if (($# > 1)); then
    echo "Usage: scripts/run_flash_kda_public_k3_perf_acceptance.sh [OUTPUT_DIR]" >&2
    exit 2
fi

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
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-public-k3-XXXXXXXX")"
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
            printf 'warmup=%s\n' "$WARMUP"
            printf 'repeat=%s\n' "$REPEAT"
            printf 'seeds=42,43,44\n'
            printf 'bootstrap_resamples=%s\n' "$BOOTSTRAP_RESAMPLES"
            printf 'minimum_speedup=%s\n' "$MIN_SPEEDUP"
            printf 'logical_cells=%s\n' "$EXPECTED_LOGICAL_CELLS"
            printf 'seed_cells=%s\n' "$EXPECTED_SEED_CELLS"
            printf 'formal_plan_sha256=%s\n' "$FORMAL_PLAN_SHA256"
            printf 'max_seqlen_upper_bound=none\n'
            printf 'initial_state=materialized-fp32\n'
            printf 'python=%s\n' "$PYTHON_BIN"
            printf 'benchmark_exit=%s\n' "${BENCHMARK_EXIT:-not-run}"
        } >"$OUTPUT_DIR/status.txt"
    }
    if ! write_acceptance_status; then
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=status-finalization
        fi
        # Retry once so a transient write failure cannot leave a stale PASS.
        write_acceptance_status || true
    fi

    local candidate
    local -a artifacts=()
    for candidate in \
        status.txt repository-after.txt git-head.txt git-tree.txt \
        static-self-test.json static-self-test.log plan.json plan.log \
        environment.log benchmark.log benchmark-status.txt result.json \
        partial-results.jsonl artifact-validation.log jit/module_aiter_core.so \
        jit/module_flash_kda_hip.so; do
        [[ -s $OUTPUT_DIR/$candidate ]] && artifacts+=("$candidate")
    done
    if ((${#artifacts[@]} > 0)); then
        if ! (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"; then
            rm -f -- "$OUTPUT_DIR/artifacts.sha256"
            if ((exit_code == 0)); then
                exit_code=1
                ACCEPTANCE_STAGE=artifact-checksum-finalization
            fi
            # The final outcome changed after status.txt was written.  Never
            # leave a PASS marker behind when checksum finalization failed.
            write_acceptance_status || true
        fi
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0)); then
        echo "FlashKDA public-K3 performance acceptance passed: $OUTPUT_DIR"
    else
        printf \
            'FlashKDA public-K3 performance acceptance failed during %s: %s\n' \
            "$ACCEPTANCE_STAGE" "$OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT
printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
echo "FlashKDA public-K3 performance acceptance artifacts: $OUTPUT_DIR"

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
mkdir -p "$AITER_JIT_DIR"

BENCHMARK_ENTRY=op_tests/op_benchmarks/triton/bench_flash_kda_public_k3.py

ACCEPTANCE_STAGE=static-self-test
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --static-self-test \
    --output "$OUTPUT_DIR/static-self-test.json" \
    2>&1 | tee "$OUTPUT_DIR/static-self-test.log"

ACCEPTANCE_STAGE=plan
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --print-plan \
    --seed 42 --seed 43 --seed 44 \
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

ACCEPTANCE_STAGE=benchmark
set +e
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --seed 42 --seed 43 --seed 44 \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
    --output "$OUTPUT_DIR/result.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"
BENCHMARK_EXIT=${PIPESTATUS[0]}
set -e
printf 'benchmark_exit=%s\n' "$BENCHMARK_EXIT" \
    >"$OUTPUT_DIR/benchmark-status.txt"

ACCEPTANCE_STAGE=artifact-finalization
CURRENT_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
CURRENT_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
REPOSITORY_OK=1
if [[ $CURRENT_HEAD != "$INITIAL_HEAD" || $CURRENT_TREE != "$INITIAL_TREE" ]] ||
    ! repo_is_clean; then
    REPOSITORY_OK=0
fi
if ((BENCHMARK_EXIT == 0)); then
    for artifact in result.json partial-results.jsonl \
        jit/module_flash_kda_hip.so; do
        [[ -s $OUTPUT_DIR/$artifact ]] || {
            echo "ERROR: successful benchmark did not produce $artifact" >&2
            exit 1
        }
    done
fi

if ((REPOSITORY_OK == 0)); then
    echo "ERROR: acceptance changed HEAD or the source checkout" >&2
    exit 1
fi
if ((BENCHMARK_EXIT != 0)); then
    ACCEPTANCE_STAGE=benchmark-failed
    exit "$BENCHMARK_EXIT"
fi

ACCEPTANCE_STAGE=artifact-validation
"$PYTHON_BIN" - \
    "$OUTPUT_DIR" "$INITIAL_HEAD" "$FORMAL_PLAN_SHA256" \
    "$EXPECTED_LOGICAL_CELLS" "$EXPECTED_SEED_CELLS" <<'PY' \
    2>&1 | tee "$OUTPUT_DIR/artifact-validation.log"
import hashlib
import json
import math
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
expected_head = sys.argv[2]
expected_plan_sha256 = sys.argv[3]
expected_logical_cells = int(sys.argv[4])
expected_seed_cells = int(sys.argv[5])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


with (root / "result.json").open(encoding="utf-8") as handle:
    result = json.load(handle)
with (root / "plan.json").open(encoding="utf-8") as handle:
    emitted_plan = json.load(handle)

require(result.get("schema") == "flash-kda-public-k3-formal-matrix-v3",
        "result schema")
require(result.get("capture_complete") is True, "capture is incomplete")
require(result.get("all_nonperformance_contract_checks_passed") is True,
        "non-performance contract check failed")
require(result.get("performance_gate_evaluated") is True,
        "performance gate was not evaluated")
require(result.get("performance_gate_passed") is True,
        "global performance gate failed")
require(result.get("plan_sha256") == expected_plan_sha256,
        "result plan hash")
require(result.get("plan") == emitted_plan, "emitted and embedded plans differ")

plan = result["plan"]
require(plan.get("logical_cells_per_seed") == expected_logical_cells,
        "logical-cell count")
require(plan.get("total_seed_cells") == expected_seed_cells,
        "seed-cell count")
require(plan.get("seeds") == [42, 43, 44], "seed set")
require(len(plan.get("cells", ())) == expected_seed_cells, "plan cell count")
route_coverage = plan.get("k3_16k_no_hint_route_coverage", {})
require(route_coverage.get("logical_cells") == 7,
        "K3 N4/N8 route-cell count")
require(
    {cell.get("scenario") for cell in route_coverage.get("cells", ())}
    == {
        "n4-equal", "n4-extreme-tail", "n4-mixed", "n8-equal",
        "n8-extreme-tail", "n8-ragged", "n8-mixed",
    },
    "K3 N4/N8 route scenarios",
)

configuration = result.get("configuration", {})
require(configuration.get("seeds") == [42, 43, 44], "configuration seeds")
require(configuration.get("warmup") == 20, "configuration warmup")
require(configuration.get("repeat") == 120, "configuration repeat")
require(configuration.get("bootstrap_resamples") == 10000,
        "configuration bootstrap resamples")
require(configuration.get("heads") == 12, "configuration Hq")
require(configuration.get("value_heads") == 12, "configuration HV")
require(configuration.get("execution") == "graph", "configuration execution")
require(configuration.get("max_seqlen_upper_bound") is None,
        "configuration max-seqlen hint")
require(configuration.get("max_seqlen_hint_omitted") is True,
        "configuration hint omission")
require(configuration.get("initial_state_literal_none") is False,
        "configuration state presence")
require(configuration.get("initial_state_dtype") == "torch.float32",
        "configuration state dtype")

environment = result.get("environment", {})
require(environment.get("git_head") == expected_head, "benchmark git HEAD")
require(environment.get("git_status_porcelain") == "",
        "benchmark observed a dirty checkout")
require(environment.get("arch") == "gfx950", "GPU arch")
require(environment.get("compute_units") == 256, "GPU CU count")
require(environment.get("active_kda_environment") == {},
        "active KDA route environment")
expected_module = (root / "jit/module_flash_kda_hip.so").resolve()
require(expected_module.is_file() and expected_module.stat().st_size > 0,
        "native module is missing")
module_digest = hashlib.sha256(expected_module.read_bytes()).hexdigest()
require(environment.get("module_sha256") == module_digest,
        "native module SHA256")
loaded = environment.get("loaded_module_identities", {}).get(
    "module_flash_kda_hip", {}
)
require(loaded.get("matches_expected_jit_path") is True,
        "loaded native module path")
require(loaded.get("sha256") == module_digest,
        "loaded native module identity")

rows = result.get("results")
require(isinstance(rows, list) and len(rows) == expected_seed_cells,
        "result seed-cell count")
expected_keys = {
    (cell["logical_name"], cell["seed"]) for cell in plan["cells"]
}
actual_keys = {(row.get("logical_name"), row.get("seed")) for row in rows}
require(len(actual_keys) == expected_seed_cells and actual_keys == expected_keys,
        "missing, duplicate, or unexpected seed cell")
for row in rows:
    label = f"{row.get('logical_name')}/seed-{row.get('seed')}"
    require(row.get("heads") == 12 and row.get("value_heads") == 12,
            f"{label}: heads")
    require(row.get("execution") == "graph", f"{label}: execution")
    require(row.get("max_seqlen_upper_bound") is None,
            f"{label}: max-seqlen hint")
    state = row.get("initial_state_contract", {})
    require(state.get("literal_none") is False,
            f"{label}: state presence")
    require(state.get("materialized_dtype") == "torch.float32",
            f"{label}: state dtype")
    require(row.get("all_contract_checks_passed") is True,
            f"{label}: contract checks")
    require(row.get("all_input_state_immutability_checks_passed") is True,
            f"{label}: state immutability")
    require(row.get("performance_gate_evaluated") is True,
            f"{label}: gate evaluation")
    require(row.get("performance_gate_passed") is True,
            f"{label}: performance gate")
    timing = row.get("timing", {})
    require(timing.get("samples_per_backend") == 120,
            f"{label}: timing samples")
    speedup = timing.get("triton_over_public_speedup_from_p50")
    require(finite_number(speedup) and float(speedup) >= 1.05,
            f"{label}: p50 speedup")
    require(finite_number(timing.get("public_win_fraction")) and
            float(timing["public_win_fraction"]) > 0.5,
            f"{label}: paired win fraction")
    require(len(row.get("raw_timing_samples", ())) == 240,
            f"{label}: raw timing count")
    require(all(sample.get("max_seqlen_upper_bound") is None
                for sample in row["raw_timing_samples"]),
            f"{label}: raw timing hint")
    audit = row.get("graph_route_audit", {})
    require(all(audit.get(field) is True for field in (
                "all_routes_verified", "graphs_independent",
                "streams_independent",
                "public_explicit_graph_signatures_equal")),
            f"{label}: graph route audit")

summary = result.get("cross_seed_summary", {})
require(summary.get("logical_cells") == expected_logical_cells,
        "summary logical-cell count")
require(summary.get("seed_cells") == expected_seed_cells,
        "summary seed-cell count")
require(len(summary.get("cells", ())) == expected_logical_cells,
        "summary cell list")
require(all(cell.get("performance_gate_passed") is True
            for cell in summary["cells"]), "summary cell gate")
gate = result.get("performance_gate", {})
require(gate.get("passed") is True, "global gate")
require(gate.get("failed_logical_cells") == [], "failed logical cells")

with (root / "partial-results.jsonl").open(encoding="utf-8") as handle:
    events = [json.loads(line) for line in handle if line.strip()]
require(sum(event.get("event") == "seed-cell-complete" for event in events)
        == expected_seed_cells, "checkpoint seed-cell count")
require(not any(event.get("event") == "run-failed" for event in events),
        "checkpoint contains a failure")
require(events and events[-1].get("event") == "run-complete",
        "checkpoint terminal event")
require(events[-1].get("complete") is True, "checkpoint incomplete")
require(events[-1].get("performance_gate_passed") is True,
        "checkpoint performance gate")
print("artifact validation: PASS (31/31 logical cells, 93/93 seed cells)")
PY

ACCEPTANCE_STAGE=complete
