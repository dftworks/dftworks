#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"

cd "$ROOT_DIR"

NONSPIN_CASE="test_example/si-oncv/scf"
SPIN_CASE="test_example/si-oncv/wannier90-spin"

contains() {
    local pattern="$1"
    if command -v rg >/dev/null 2>&1; then
        rg -q "$pattern"
    else
        grep -q "$pattern"
    fi
}

echo "[smoke] memory_estimate nonspin (human-readable)"
nonspin_output="$(cargo run -p pw --bin memory_estimate -- --case "$NONSPIN_CASE")"
echo "$nonspin_output"
echo "$nonspin_output" | contains "estimated_bytes_total"
echo "$nonspin_output" | contains "spin_scheme = nonspin"

echo
echo "[smoke] memory_estimate spin (json)"
spin_json="$(cargo run -p pw --bin memory_estimate -- --case "$SPIN_CASE" --json)"
echo "$spin_json"
echo "$spin_json" | contains "\"status\": \"ok\""
echo "$spin_json" | contains "\"spin_scheme\": \"spin\""
echo "$spin_json" | contains "\"estimated_bytes_total\""

echo
echo "[smoke] memory_estimate smoke checks passed"
