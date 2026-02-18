#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
CASE_DIR="$ROOT_DIR/test_example/si-oncv/symmetry-enabled"

PW_BIN="${PW_BIN:-$ROOT_DIR/target/release/pw}"
FORCE_BUILD="${FORCE_BUILD:-0}"

FORCE_SUM_TOL="${FORCE_SUM_TOL:-5.0e-3}"           # eV/A
STRESS_ASYM_TOL="${STRESS_ASYM_TOL:-5.0e-2}"       # kbar

SKIP_RUN=0
if [[ "${1:-}" == "--skip-run" ]]; then
    SKIP_RUN=1
fi

if [[ "$FORCE_BUILD" == "1" || ! -x "$PW_BIN" ]]; then
    echo "Building pw binary at $PW_BIN ..."
    cargo build --release -p pw
fi

run_stage() {
    local stage="$1"
    local stage_dir="$CASE_DIR/$stage"
    local log_name="$2"
    local log_file="$stage_dir/$log_name"

    if [[ "$SKIP_RUN" == "0" ]]; then
        echo "Running $stage ..."
        (
            cd "$stage_dir"
            "$PW_BIN" > "$log_name" 2>&1
        )
    fi

    if [[ ! -f "$log_file" ]]; then
        echo "missing log file: $log_file"
        exit 1
    fi
}

run_stage "scf" "out.scf.log"
run_stage "relax" "out.relax.log"

python3 - \
    "$CASE_DIR/scf/in.kmesh" \
    "$CASE_DIR/scf/out.scf.log" \
    "$CASE_DIR/relax/in.kmesh" \
    "$CASE_DIR/relax/out.relax.log" \
    "$FORCE_SUM_TOL" \
    "$STRESS_ASYM_TOL" <<'PY'
import re
import sys

(
    scf_kmesh_path,
    scf_log_path,
    relax_kmesh_path,
    relax_log_path,
    force_sum_tol,
    stress_asym_tol,
) = sys.argv[1:]

force_sum_tol = float(force_sum_tol)
stress_asym_tol = float(stress_asym_tol)


def read_mesh_total(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    parts = first.split()
    if len(parts) != 3:
        raise RuntimeError(f"invalid kmesh line in {path}: '{first}'")
    n1, n2, n3 = [int(x) for x in parts]
    return n1 * n2 * n3


def parse_nkpt(text: str) -> int:
    m = re.search(r"nkpt\s*\(IR/full\)\s*=\s*(\d+)", text)
    if not m:
        m = re.search(r"nkpt\s*=\s*(\d+)", text)
    if not m:
        raise RuntimeError("failed to parse nkpt from log")
    return int(m.group(1))


def parse_force_blocks(text: str):
    lines = text.splitlines()
    force_line = re.compile(
        r"^\s*\d+\s+\S+\s*:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    )
    blocks = []
    i = 0
    while i < len(lines):
        if "total-force (cartesian)" not in lines[i]:
            i += 1
            continue

        i += 1
        block = []
        while i < len(lines):
            m = force_line.match(lines[i])
            if m:
                block.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
                i += 1
                continue
            if block:
                break
            i += 1
        if block:
            blocks.append(block)
    return blocks


def parse_total_stress_blocks(text: str):
    lines = text.splitlines()
    row = re.compile(r"\|\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\|")
    blocks = []
    i = 0
    while i < len(lines):
        if "stress (kbar)" not in lines[i]:
            i += 1
            continue

        j = i + 1
        while j < len(lines) and "total" not in lines[j]:
            j += 1
        if j + 3 >= len(lines):
            i += 1
            continue

        mat = []
        ok = True
        for r in range(j + 1, j + 4):
            m = row.search(lines[r])
            if not m:
                ok = False
                break
            mat.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if ok:
            blocks.append(mat)
        i = j + 1
    return blocks


def force_block_metric(block):
    sums = [sum(v[d] for v in block) for d in range(3)]
    return max(abs(x) for x in sums)


def stress_block_metrics(mat):
    asym = max(
        abs(mat[0][1] - mat[1][0]),
        abs(mat[0][2] - mat[2][0]),
        abs(mat[1][2] - mat[2][1]),
    )
    offdiag = max(
        abs(mat[0][1]), abs(mat[0][2]), abs(mat[1][0]), abs(mat[1][2]), abs(mat[2][0]), abs(mat[2][1])
    )
    diag = [mat[0][0], mat[1][1], mat[2][2]]
    diag_spread = max(diag) - min(diag)
    return asym, offdiag, diag_spread


def verify_stage(stage_name: str, kmesh_path: str, log_path: str):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    mesh_total = read_mesh_total(kmesh_path)
    nkpt = parse_nkpt(text)
    if nkpt >= mesh_total:
        raise RuntimeError(
            f"{stage_name}: expected symmetry-reduced nkpt < {mesh_total}, got {nkpt}"
        )

    force_blocks = parse_force_blocks(text)
    if not force_blocks:
        raise RuntimeError(f"{stage_name}: no total-force block found")
    worst_force_sum = max(force_block_metric(b) for b in force_blocks)
    if worst_force_sum > force_sum_tol:
        raise RuntimeError(
            f"{stage_name}: force symmetry check failed, max |sum(F)|={worst_force_sum:.6e} eV/A"
        )

    stress_blocks = parse_total_stress_blocks(text)
    if not stress_blocks:
        raise RuntimeError(f"{stage_name}: no total stress block found")
    stress_metrics = [stress_block_metrics(b) for b in stress_blocks]
    worst_asym = max(x[0] for x in stress_metrics)
    worst_offdiag = max(x[1] for x in stress_metrics)
    worst_diag_spread = max(x[2] for x in stress_metrics)

    if worst_asym > stress_asym_tol:
        raise RuntimeError(
            f"{stage_name}: stress asymmetry too large, max |sigma_ij-sigma_ji|={worst_asym:.6e} kbar"
        )

    print(
        f"{stage_name}: nkpt={nkpt}/{mesh_total}, "
        f"max|sum(F)|={worst_force_sum:.3e} eV/A, "
        f"max asym={worst_asym:.3e} kbar, "
        f"max offdiag={worst_offdiag:.3e} kbar, "
        f"max diag spread={worst_diag_spread:.3e} kbar"
    )


verify_stage("scf", scf_kmesh_path, scf_log_path)
verify_stage("relax", relax_kmesh_path, relax_log_path)
print("Si symmetry-enabled SCF/relax verification passed.")
PY
