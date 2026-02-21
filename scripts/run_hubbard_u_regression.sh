#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SCF_TEMPLATE_DIR="$ROOT_DIR/test_example/si-oncv/scf"
PW_BIN="${PW_BIN:-$ROOT_DIR/target/debug/pw}"
FORCE_BUILD="${FORCE_BUILD:-0}"
ENERGY_SHIFT_MIN_RY="${ENERGY_SHIFT_MIN_RY:-1e-6}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

DYLD_FALLBACK_LIBRARY_PATH_DEFAULT="/opt/local/lib/libgcc:/opt/local/lib:/opt/local/lib/mpich-mp:/opt/spglib/lib:/opt/lapack/lib"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-$DYLD_FALLBACK_LIBRARY_PATH_DEFAULT}"

if [[ "$FORCE_BUILD" == "1" || ! -x "$PW_BIN" ]]; then
    echo "Building pw binary at $PW_BIN ..." >&2
    cargo build -p pw
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dftworks-hubbard-regression.XXXXXX")"
cleanup() {
    if [[ "$KEEP_WORKDIR" == "1" ]]; then
        echo "Keeping regression workdir: $WORKDIR" >&2
        return
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

copy_case_inputs() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    cp "$SCF_TEMPLATE_DIR/in.ctrl" "$out_dir/in.ctrl"
    cp "$SCF_TEMPLATE_DIR/in.crystal" "$out_dir/in.crystal"
    cp "$SCF_TEMPLATE_DIR/in.kmesh" "$out_dir/in.kmesh"
    cp "$SCF_TEMPLATE_DIR/in.pot" "$out_dir/in.pot"
    cp "$SCF_TEMPLATE_DIR/in.spin" "$out_dir/in.spin"
    cp "$SCF_TEMPLATE_DIR/in.magmom" "$out_dir/in.magmom"
    cp -R "$SCF_TEMPLATE_DIR/pot" "$out_dir/pot"
}

set_ctrl_key() {
    local ctrl_file="$1"
    local key="$2"
    local value="$3"
    perl -i -pe "s/^${key}\\s*=\\s*.*$/${key} = ${value}/" "$ctrl_file"
}

append_ctrl_key_if_missing() {
    local ctrl_file="$1"
    local key="$2"
    local value="$3"
    if ! grep -q "^${key}\\s*=" "$ctrl_file"; then
        printf "%s = %s\n" "$key" "$value" >> "$ctrl_file"
    fi
}

parse_spin_energy_ry() {
    local log_file="$1"
    awk '/scf_energy \(Ry\)/{val=$NF} END{print val}' "$log_file"
}

parse_spin_iters() {
    local log_file="$1"
    awk '/#step: geom-[0-9]+-scf-[0-9]+/{n++} END{print n+0}' "$log_file"
}

run_case() {
    local case_name="$1"
    local enable_hubbard="$2"
    local case_dir="$WORKDIR/$case_name"

    copy_case_inputs "$case_dir"

    local ctrl="$case_dir/in.ctrl"
    set_ctrl_key "$ctrl" "spin_scheme" "spin"
    set_ctrl_key "$ctrl" "xc_scheme" "pbe"
    set_ctrl_key "$ctrl" "scf_max_iter" "40"
    set_ctrl_key "$ctrl" "energy_epsilon" "1E-6"
    set_ctrl_key "$ctrl" "ecut_wfc" "300"

    append_ctrl_key_if_missing "$ctrl" "hubbard_u_enabled" "false"
    append_ctrl_key_if_missing "$ctrl" "hubbard_species" "Si1"
    append_ctrl_key_if_missing "$ctrl" "hubbard_l" "1"
    append_ctrl_key_if_missing "$ctrl" "hubbard_u" "4.0"
    append_ctrl_key_if_missing "$ctrl" "hubbard_j" "0.0"

    set_ctrl_key "$ctrl" "hubbard_u_enabled" "$enable_hubbard"
    set_ctrl_key "$ctrl" "hubbard_species" "Si1"
    set_ctrl_key "$ctrl" "hubbard_l" "1"
    set_ctrl_key "$ctrl" "hubbard_u" "4.0"
    set_ctrl_key "$ctrl" "hubbard_j" "0.0"

    if ! (
        cd "$case_dir"
        "$PW_BIN" > out.log 2>&1
    ); then
        echo "Case '$case_name' execution failed. See $case_dir/out.log" >&2
        exit 1
    fi

    if ! grep -q "scf_convergence_success" "$case_dir/out.log"; then
        echo "Case '$case_name' did not converge. See $case_dir/out.log" >&2
        exit 1
    fi

    local energy_ry
    local n_iter
    energy_ry="$(parse_spin_energy_ry "$case_dir/out.log")"
    n_iter="$(parse_spin_iters "$case_dir/out.log")"

    if [[ -z "$energy_ry" ]]; then
        echo "Failed to parse scf_energy (Ry) for '$case_name'" >&2
        exit 1
    fi

    if [[ "$n_iter" -le 0 ]]; then
        echo "Failed to parse SCF iteration count for '$case_name'" >&2
        exit 1
    fi

    printf "%s\t%s\t%s\n" "$case_name" "$energy_ry" "$n_iter"
}

echo "Running Hubbard +U regression in $WORKDIR"

baseline_result="$(run_case "baseline_spin_pbe" "false")"
hubbard_result="$(run_case "hubbard_spin_pbe_u4" "true")"

baseline_energy="$(echo "$baseline_result" | awk -F'\t' '{print $2}')"
baseline_iter="$(echo "$baseline_result" | awk -F'\t' '{print $3}')"
hubbard_energy="$(echo "$hubbard_result" | awk -F'\t' '{print $2}')"
hubbard_iter="$(echo "$hubbard_result" | awk -F'\t' '{print $3}')"

energy_shift="$(awk -v a="$hubbard_energy" -v b="$baseline_energy" 'BEGIN{d=a-b; if (d<0) d=-d; print d}')"
shift_ok="$(awk -v d="$energy_shift" -v tol="$ENERGY_SHIFT_MIN_RY" 'BEGIN{print (d>=tol)?1:0}')"

echo "baseline: energy_ry=$baseline_energy scf_iters=$baseline_iter"
echo "hubbard : energy_ry=$hubbard_energy scf_iters=$hubbard_iter"
echo "abs energy shift (Ry) = $energy_shift"

if [[ "$shift_ok" != "1" ]]; then
    echo "Hubbard regression failed: expected |E_hubbard - E_baseline| >= $ENERGY_SHIFT_MIN_RY Ry"
    exit 1
fi

echo "Hubbard +U regression passed."
