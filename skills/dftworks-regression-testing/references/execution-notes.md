# DFTWorks Regression Execution Notes

## Host Versus Docker

- Prefer host runs when the required toolchain is available, especially for scripts that expect local binaries or MPI launchers.
- In this workspace, `docker` is already approval-allowed. If a regression needs Docker, run it directly instead of pausing for another permission prompt.
- If host MPI is unavailable, use Docker for `cargo check -p pw`:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && cargo check -p pw'
```

- The same container pattern can run non-MPI shell regressions when the host Rust toolchain is not usable:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && bash scripts/run_phase12_regression.sh'
```

## Useful Script Knobs

- `PW_BIN=/path/to/pw`: point a regression script at an already-built binary.
- `FORCE_BUILD=1`: force the script to rebuild before running.
- `CARGO_FEATURES=...`: build `pw` with explicit features when needed.
- `KEEP_WORKDIR=1`: keep temporary regression directories for inspection in scripts that support it.
- `MPI_LAUNCH`, `MPI_FLAG_N`, `MPI_RANKS_REF`, `MPI_RANKS_CMP`: tune the spin/MPI parity runner without editing the script.

## Manual Example Runs

- Wannier examples have concrete run instructions in:
  `test_example/si-oncv/wannier90/README.md`
  `test_example/si-oncv/wannier90-spin/README.md`
  `test_example/si-oncv/wannier90-projected/README.md`

- The workflow YAML pipeline example is documented in:
  `test_example/si-oncv/workflow-pipeline-yaml/README.md`

- The symmetry case has a dedicated runner:
  `test_example/si-oncv/symmetry-enabled/run_and_verify.sh`

## Reporting Expectations

In the close-out, always state:

- the exact commands you ran,
- whether they ran on host or in Docker,
- any environment assumptions such as MPI or external Wannier tools,
- and which relevant regressions you did not run.
