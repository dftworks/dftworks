#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SCF_TEMPLATE_DIR="$ROOT_DIR/test_example/si-oncv/scf"
REFERENCE_FILE="$ROOT_DIR/test_example/si-oncv/regression/phase12_reference.tsv"

PW_BIN="${PW_BIN:-$ROOT_DIR/target/debug/pw}"
ENERGY_TOL_RY="${ENERGY_TOL_RY:-5e-4}"
FERMI_TOL_EV="${FERMI_TOL_EV:-5e-3}"

UPDATE_MODE=0
if [[ "${1:-}" == "--update" ]]; then
    UPDATE_MODE=1
fi

if [[ ! -x "$PW_BIN" ]]; then
    echo "Building pw binary at $PW_BIN ..."
    cargo build -p pw
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dftworks-phase12.XXXXXX")"
RESULT_FILE="$WORKDIR/results.tsv"
trap 'rm -rf "$WORKDIR"' EXIT

mkdir -p "$ROOT_DIR/test_example/si-oncv/regression"

CASES=(
    "lda_nonspin|nonspin|lda-pz"
    "lsda_spin|spin|lsda-pz"
    "pbe_nonspin|nonspin|pbe"
    "pbe_spin|spin|pbe"
)

patch_ctrl() {
    local ctrl_file="$1"
    local spin_scheme="$2"
    local xc_scheme="$3"

    perl -i -pe "s/^spin_scheme\\s*=\\s*\\S+/spin_scheme = ${spin_scheme}/" "$ctrl_file"
    perl -i -pe "s/^xc_scheme\\s*=\\s*\\S+/xc_scheme = ${xc_scheme}/" "$ctrl_file"
}

parse_nonspin_energy_fermi() {
    local log_file="$1"
    local energy fermi
    energy="$(awk '/^[[:space:]]*[0-9]+:/{val=$(NF-1)} END{print val}' "$log_file")"
    fermi="$(awk '/^[[:space:]]*[0-9]+:/{val=$3} END{print val}' "$log_file")"
    echo "${energy} ${fermi}"
}

parse_spin_energy_fermi() {
    local log_file="$1"
    local energy fermi
    energy="$(awk '/scf_energy \(Ry\)/{val=$NF} END{print val}' "$log_file")"
    fermi="$(awk '/Fermi_level/{val=$NF} END{print val}' "$log_file")"
    echo "${energy} ${fermi}"
}

echo -n "" > "$RESULT_FILE"

for case_spec in "${CASES[@]}"; do
    IFS='|' read -r case_name spin_scheme xc_scheme <<< "$case_spec"

    case_dir="$WORKDIR/$case_name"
    mkdir -p "$case_dir"

    cp "$SCF_TEMPLATE_DIR/in.ctrl" "$case_dir/in.ctrl"
    cp "$SCF_TEMPLATE_DIR/in.crystal" "$case_dir/in.crystal"
    cp "$SCF_TEMPLATE_DIR/in.kmesh" "$case_dir/in.kmesh"
    cp "$SCF_TEMPLATE_DIR/in.pot" "$case_dir/in.pot"
    cp "$SCF_TEMPLATE_DIR/in.spin" "$case_dir/in.spin"
    cp "$SCF_TEMPLATE_DIR/in.magmom" "$case_dir/in.magmom"
    cp -R "$SCF_TEMPLATE_DIR/pot" "$case_dir/pot"

    patch_ctrl "$case_dir/in.ctrl" "$spin_scheme" "$xc_scheme"

    (
        cd "$case_dir"
        "$PW_BIN" > out.log 2>&1
    )

    if ! rg -q "scf_convergence_success" "$case_dir/out.log"; then
        echo "Case '$case_name' did not converge. See $case_dir/out.log"
        exit 1
    fi

    if [[ "$spin_scheme" == "spin" ]]; then
        read -r energy_ry fermi_ev <<< "$(parse_spin_energy_fermi "$case_dir/out.log")"
    else
        read -r energy_ry fermi_ev <<< "$(parse_nonspin_energy_fermi "$case_dir/out.log")"
    fi

    if [[ -z "${energy_ry:-}" || -z "${fermi_ev:-}" ]]; then
        echo "Failed to parse metrics for '$case_name'. See $case_dir/out.log"
        exit 1
    fi

    printf "%s\t%s\t%s\t%s\t%s\n" "$case_name" "$spin_scheme" "$xc_scheme" "$energy_ry" "$fermi_ev" >> "$RESULT_FILE"
    echo "Case $case_name: energy_ry=$energy_ry fermi_ev=$fermi_ev"
done

if [[ "$UPDATE_MODE" -eq 1 ]]; then
    cp "$RESULT_FILE" "$REFERENCE_FILE"
    echo
    echo "Updated reference: $REFERENCE_FILE"
    cat "$REFERENCE_FILE"
    exit 0
fi

if [[ ! -f "$REFERENCE_FILE" ]]; then
    echo "Reference file not found: $REFERENCE_FILE"
    echo "Run with --update first."
    exit 1
fi

declare -A ref_energy
declare -A ref_fermi
declare -A ref_spin
declare -A ref_xc

while IFS=$'\t' read -r case_name spin_scheme xc_scheme energy_ry fermi_ev; do
    [[ -z "${case_name:-}" ]] && continue
    ref_spin["$case_name"]="$spin_scheme"
    ref_xc["$case_name"]="$xc_scheme"
    ref_energy["$case_name"]="$energy_ry"
    ref_fermi["$case_name"]="$fermi_ev"
done < "$REFERENCE_FILE"

all_passed=1
while IFS=$'\t' read -r case_name spin_scheme xc_scheme energy_ry fermi_ev; do
    [[ -z "${case_name:-}" ]] && continue

    if [[ "${ref_spin[$case_name]:-}" != "$spin_scheme" || "${ref_xc[$case_name]:-}" != "$xc_scheme" ]]; then
        echo "Case metadata mismatch for '$case_name'"
        all_passed=0
        continue
    fi

    energy_diff="$(awk -v a="$energy_ry" -v b="${ref_energy[$case_name]}" 'BEGIN{d=a-b; if (d<0) d=-d; print d}')"
    fermi_diff="$(awk -v a="$fermi_ev" -v b="${ref_fermi[$case_name]}" 'BEGIN{d=a-b; if (d<0) d=-d; print d}')"

    energy_ok="$(awk -v d="$energy_diff" -v tol="$ENERGY_TOL_RY" 'BEGIN{print (d<=tol)?1:0}')"
    fermi_ok="$(awk -v d="$fermi_diff" -v tol="$FERMI_TOL_EV" 'BEGIN{print (d<=tol)?1:0}')"

    echo "Case $case_name: dE_ry=$energy_diff dFermi_ev=$fermi_diff"

    if [[ "$energy_ok" != "1" || "$fermi_ok" != "1" ]]; then
        echo "Tolerance check failed for '$case_name' (energy_tol=$ENERGY_TOL_RY, fermi_tol=$FERMI_TOL_EV)"
        all_passed=0
    fi
done < "$RESULT_FILE"

if [[ "$all_passed" != "1" ]]; then
    echo "Phase 1/2 regression failed."
    exit 1
fi

echo "Phase 1/2 regression passed."
