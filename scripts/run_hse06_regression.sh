#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
CASE_TEMPLATE_DIR="$ROOT_DIR/test_example/si-oncv/hse06-gamma"
PW_BIN="${PW_BIN:-$ROOT_DIR/target/debug/pw}"
FORCE_BUILD="${FORCE_BUILD:-0}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

DYLD_FALLBACK_LIBRARY_PATH_DEFAULT="/opt/local/lib/libgcc:/opt/local/lib:/opt/local/lib/mpich-mp:/opt/spglib/lib:/opt/lapack/lib"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-$DYLD_FALLBACK_LIBRARY_PATH_DEFAULT}"

if [[ "$FORCE_BUILD" == "1" || ! -x "$PW_BIN" ]]; then
    echo "Building pw binary at $PW_BIN ..." >&2
    cargo build -p pw
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dftworks-hse06-regression.XXXXXX")"
cleanup() {
    if [[ "$KEEP_WORKDIR" == "1" ]]; then
        echo "Keeping regression workdir: $WORKDIR" >&2
        return
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

case_dir="$WORKDIR/hse06-gamma"
mkdir -p "$case_dir"

cp "$CASE_TEMPLATE_DIR/in.ctrl" "$case_dir/in.ctrl"
cp "$CASE_TEMPLATE_DIR/in.crystal" "$case_dir/in.crystal"
cp "$CASE_TEMPLATE_DIR/in.kmesh" "$case_dir/in.kmesh"
cp "$CASE_TEMPLATE_DIR/in.pot" "$case_dir/in.pot"
# Dereference symlinks so temporary workdir has standalone pseudopotentials.
cp -RL "$CASE_TEMPLATE_DIR/pot" "$case_dir/pot"

if ! (
    cd "$case_dir"
    "$PW_BIN" > out.log 2>&1
); then
    echo "HSE06 case execution failed. See $case_dir/out.log" >&2
    exit 1
fi

if ! grep -q "scf_convergence_success" "$case_dir/out.log"; then
    echo "HSE06 regression failed: SCF did not converge. See $case_dir/out.log" >&2
    exit 1
fi

energy_ry="$(awk '/^[[:space:]]*[0-9]+:/{val=$(NF-1)} END{print val}' "$case_dir/out.log")"
fermi_ev="$(awk '/^[[:space:]]*[0-9]+:/{val=$3} END{print val}' "$case_dir/out.log")"
scf_iters="$(awk '/^[[:space:]]*[0-9]+:/{n++} END{print n+0}' "$case_dir/out.log")"

if [[ -z "$energy_ry" || -z "$fermi_ev" || "$scf_iters" -le 0 ]]; then
    echo "Failed to parse HSE06 SCF metrics. See $case_dir/out.log" >&2
    exit 1
fi

echo "hse06_gamma: energy_ry=$energy_ry fermi_ev=$fermi_ev scf_iters=$scf_iters"
echo "HSE06 regression passed."
