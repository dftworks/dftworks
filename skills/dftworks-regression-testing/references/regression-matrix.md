# DFTWorks Regression Matrix

## Fast Baseline

- `control/src` changes: `cargo test -p control`
- `crystal/src` changes: `cargo test -p crystal`
- `pspot/src` changes: `cargo test -p pspot`
- Any runtime-facing `pw` change: `cargo check -p pw` on host if MPI works, otherwise use the Docker fallback from `execution-notes.md`

## Script-Backed Regressions

- SCF iteration summaries, spin/nonspin output parsing, XC toggles, or energy/Fermi regressions:
  `bash scripts/run_phase12_regression.sh`
  This covers LDA/LSDA/PBE across spin and nonspin cases and compares against `test_example/si-oncv/regression/phase12_reference.tsv`.

- MPI decomposition, spin parity, forces, or stress consistency:
  `bash scripts/run_spin_mpi_parity.sh`
  Use this when a change could alter rank-dependent behavior or spin-path force/stress output.

- Hybrid functional or exact-exchange changes:
  `bash scripts/run_hse06_regression.sh`
  This exercises the `test_example/si-oncv/hse06-gamma` case and checks convergence plus parsed SCF metrics.

- Hubbard +U implementation changes:
  `bash scripts/run_hubbard_u_regression.sh`
  This compares baseline spin-PBE versus enabled Hubbard +U and requires a measurable energy shift.

- Memory estimator changes:
  `bash scripts/run_memory_estimator_smoke.sh`
  Use this for `pw --bin memory_estimate` logic, case loading, or JSON/human-readable output changes.

- Symmetry reduction, force balance, or stress tensor symmetry:
  `bash test_example/si-oncv/symmetry-enabled/run_and_verify.sh`
  This runs the symmetry-enabled Si SCF and relax cases and checks reduced `nkpt`, force sums, and stress symmetry.

## Workflow And Wannier Coverage

- `workflow/src` pipeline, staged run semantics, or status/provenance changes:
  `cargo run -p workflow --bin dwf -- validate test_example/si-oncv/workflow-pipeline-yaml`
  `cargo run -p workflow --bin dwf -- run pipeline test_example/si-oncv/workflow-pipeline-yaml`
  `cargo run -p workflow --bin dwf -- status test_example/si-oncv/workflow-pipeline-yaml`

- Nonspin Wannier export changes in `pw`, `w90-win`, or `w90-amn`:
  Run the case in `test_example/si-oncv/wannier90` and verify `si.eig`, `si.win`, `si.nnkp`, `si.mmn`, and `si.amn` generation.

- Spin-resolved Wannier export changes:
  Run `test_example/si-oncv/wannier90-spin` and verify both `si.up.*` and `si.dn.*` artifact sets, especially `.eig`.

- Projected Wannier or level-3 output changes:
  Run `test_example/si-oncv/wannier90-projected` and, if projector/post-processing logic changed, also verify `w90-proj` outputs under `level3-out/`.

## Selecting The Smallest Sufficient Matrix

- Input-loader or bootstrap hardening:
  Run the affected crate tests plus `cargo check -p pw`. Add a runtime script only if the change reaches execution semantics.

- `pw/src/orchestration/outputs.rs`, restart, checkpoint, or export logic:
  Run `cargo check -p pw` plus the nearest artifact-producing regression, usually a Wannier case or symmetry/script-backed runtime case.

- `workflow/src/main.rs` changes:
  Run the workflow pipeline commands even if lower-level unit tests pass.

- Output-format or parser-only changes:
  Use the narrowest script that exercises that output path, usually `run_phase12_regression.sh`, `run_spin_mpi_parity.sh`, or `run_memory_estimator_smoke.sh`.
