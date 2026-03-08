#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

echo "== cargo test -p control =="
cargo test -p control

echo "== cargo test -p crystal =="
cargo test -p crystal

echo "== cargo test -p pspot =="
cargo test -p pspot

if command -v mpicc >/dev/null 2>&1; then
  echo "== cargo check -p pw (host MPI detected) =="
  cargo check -p pw
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: host MPI is unavailable and docker is not installed; cannot validate pw" >&2
  exit 1
fi

echo "== docker cargo check -p pw =="
docker run --rm -v "$repo_root":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && cargo check -p pw'
