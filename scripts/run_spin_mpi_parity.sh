#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SCF_TEMPLATE_DIR="$ROOT_DIR/test_example/si-oncv/scf"
PW_BIN="${PW_BIN:-$ROOT_DIR/target/debug/pw}"
FORCE_BUILD="${FORCE_BUILD:-0}"

MPI_LAUNCH="${MPI_LAUNCH:-mpirun}"
MPI_FLAG_N="${MPI_FLAG_N:--n}"
MPI_RANKS_REF="${MPI_RANKS_REF:-1}"
MPI_RANKS_CMP="${MPI_RANKS_CMP:-2}"

ENERGY_TOL_RY="${ENERGY_TOL_RY:-5e-4}"
FORCE_COMPONENT_TOL_EV_A="${FORCE_COMPONENT_TOL_EV_A:-5e-3}"
STRESS_COMPONENT_TOL_KBAR="${STRESS_COMPONENT_TOL_KBAR:-5e-2}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dftworks-spin-mpi-parity.XXXXXX")"
cleanup() {
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

if [[ "$FORCE_BUILD" == "1" || ! -x "$PW_BIN" ]]; then
    echo "Building pw binary at $PW_BIN ..."
    cargo build -p pw
fi

if ! command -v "$MPI_LAUNCH" >/dev/null 2>&1; then
    echo "MPI launcher not found: $MPI_LAUNCH"
    exit 1
fi

prepare_case() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    cp "$SCF_TEMPLATE_DIR/in.ctrl" "$out_dir/in.ctrl"
    cp "$SCF_TEMPLATE_DIR/in.crystal" "$out_dir/in.crystal"
    cp "$SCF_TEMPLATE_DIR/in.kmesh" "$out_dir/in.kmesh"
    cp "$SCF_TEMPLATE_DIR/in.pot" "$out_dir/in.pot"
    cp "$SCF_TEMPLATE_DIR/in.spin" "$out_dir/in.spin"
    cp "$SCF_TEMPLATE_DIR/in.magmom" "$out_dir/in.magmom"
    cp -R "$SCF_TEMPLATE_DIR/pot" "$out_dir/pot"

    perl -i -pe 's/^spin_scheme\s*=\s*\S+/spin_scheme = spin/' "$out_dir/in.ctrl"
    perl -i -pe 's/^xc_scheme\s*=\s*\S+/xc_scheme = pbe/' "$out_dir/in.ctrl"
    if grep -q '^restart\s*=' "$out_dir/in.ctrl"; then
        perl -i -pe 's/^restart\s*=\s*\S+/restart = false/' "$out_dir/in.ctrl"
    else
        echo "restart = false" >> "$out_dir/in.ctrl"
    fi
}

run_case() {
    local nranks="$1"
    local case_dir="$2"
    local log_file="$3"
    if ! (
        cd "$case_dir"
        OMPI_ALLOW_RUN_AS_ROOT=1 \
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
        "$MPI_LAUNCH" "$MPI_FLAG_N" "$nranks" "$PW_BIN" > "$log_file" 2>&1
    ); then
        echo "MPI run failed for nranks=$nranks. See $case_dir/$log_file"
        tail -n 80 "$case_dir/$log_file" || true
        exit 1
    fi
}

ref_dir="$WORKDIR/ref"
cmp_dir="$WORKDIR/cmp"
prepare_case "$ref_dir"
prepare_case "$cmp_dir"

run_case "$MPI_RANKS_REF" "$ref_dir" "out.log"
run_case "$MPI_RANKS_CMP" "$cmp_dir" "out.log"

python3 - \
    "$ref_dir/out.log" \
    "$cmp_dir/out.log" \
    "$ENERGY_TOL_RY" \
    "$FORCE_COMPONENT_TOL_EV_A" \
    "$STRESS_COMPONENT_TOL_KBAR" \
    "$MPI_RANKS_REF" \
    "$MPI_RANKS_CMP" <<'PY'
import math
import re
import sys

(
    ref_log_path,
    cmp_log_path,
    energy_tol_ry,
    force_tol_ev_a,
    stress_tol_kbar,
    ranks_ref,
    ranks_cmp,
) = sys.argv[1:]

energy_tol_ry = float(energy_tol_ry)
force_tol_ev_a = float(force_tol_ev_a)
stress_tol_kbar = float(stress_tol_kbar)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_last_float(pattern: str, text: str, label: str) -> float:
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"failed to parse {label}")
    return float(matches[-1])


def parse_last_scf_energy_ry(text: str) -> float:
    labeled = re.findall(r"scf_energy \(Ry\)\s*=\s*([-+0-9.eE]+)", text)
    if labeled:
        return float(labeled[-1])

    # Root-only spin summary table format:
    # iter: eps Fermi charge Eharris Escf dE
    table = re.findall(
        r"^\s*\d+:\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s*$",
        text,
        flags=re.MULTILINE,
    )
    if table:
        return float(table[-1])

    raise RuntimeError("failed to parse scf_energy")


def parse_force_block(text: str):
    lines = text.splitlines()
    force_line = re.compile(
        r"^\s*\d+\s+\S+\s*:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    )
    block = None
    i = 0
    while i < len(lines):
        if "total-force (cartesian)" not in lines[i]:
            i += 1
            continue
        i += 1
        current = []
        while i < len(lines):
            m = force_line.match(lines[i])
            if m:
                current.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
                i += 1
                continue
            if current:
                break
            i += 1
        if current:
            block = current
    if block is None:
        raise RuntimeError("failed to parse total-force block")
    return block


def parse_stress_block(text: str):
    lines = text.splitlines()
    row = re.compile(r"\|\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\|")
    block = None
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
            block = mat
        i = j + 1
    if block is None:
        raise RuntimeError("failed to parse total stress block")
    return block


def max_force_component_diff(a, b):
    if len(a) != len(b):
        raise RuntimeError(f"force block atom count mismatch: {len(a)} vs {len(b)}")
    dmax = 0.0
    for va, vb in zip(a, b):
        for xa, xb in zip(va, vb):
            dmax = max(dmax, abs(xa - xb))
    return dmax


def max_stress_component_diff(a, b):
    dmax = 0.0
    for i in range(3):
        for j in range(3):
            dmax = max(dmax, abs(a[i][j] - b[i][j]))
    return dmax


ref_text = read_text(ref_log_path)
cmp_text = read_text(cmp_log_path)

if "scf_convergence_success" not in ref_text:
    raise RuntimeError(f"{ranks_ref}-rank run did not converge")
if "scf_convergence_success" not in cmp_text:
    raise RuntimeError(f"{ranks_cmp}-rank run did not converge")

energy_ref = parse_last_scf_energy_ry(ref_text)
energy_cmp = parse_last_scf_energy_ry(cmp_text)
energy_diff = abs(energy_ref - energy_cmp)

force_ref = parse_force_block(ref_text)
force_cmp = parse_force_block(cmp_text)
force_diff = max_force_component_diff(force_ref, force_cmp)

stress_ref = parse_stress_block(ref_text)
stress_cmp = parse_stress_block(cmp_text)
stress_diff = max_stress_component_diff(stress_ref, stress_cmp)

print(
    f"spin_mpi_parity: ranks={ranks_ref} vs {ranks_cmp}, "
    f"dE_ry={energy_diff:.6e}, dF_max_evA={force_diff:.6e}, dS_max_kbar={stress_diff:.6e}"
)

if energy_diff > energy_tol_ry:
    raise RuntimeError(
        f"energy parity failed: dE_ry={energy_diff:.6e} > tol={energy_tol_ry:.6e}"
    )
if force_diff > force_tol_ev_a:
    raise RuntimeError(
        f"force parity failed: dF_max_evA={force_diff:.6e} > tol={force_tol_ev_a:.6e}"
    )
if stress_diff > stress_tol_kbar:
    raise RuntimeError(
        f"stress parity failed: dS_max_kbar={stress_diff:.6e} > tol={stress_tol_kbar:.6e}"
    )

print("Spin MPI parity regression passed.")
PY
