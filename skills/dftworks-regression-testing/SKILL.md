---
name: dftworks-regression-testing
description: "Use for selecting and running DFTWorks validation and regression checks after code changes in pw, workflow, control, crystal, or pspot. Map touched areas to the lightest sufficient matrix: parser/unit tests, host or Docker cargo check for pw, phase12 regression, spin MPI parity, HSE06 regression, Hubbard +U regression, memory_estimate smoke, symmetry verification, workflow pipeline checks, and targeted Wannier example runs."
---

# DFTWorks Regression Testing

## Overview

Use this skill when a DFTWorks change needs validation beyond a single `cargo test`. Choose the smallest set of checks that covers the changed behavior, prefer existing repo scripts over ad hoc harnesses, and report exactly what ran versus what remained blocked.

## Regression Workflow

1. Start with the smallest relevant package tests.
   For parser, control, or bootstrap changes, begin with the affected crate tests before touching heavier runtime suites.

2. Add a `pw` build check for runtime-facing changes.
   Run host `cargo check -p pw` if MPI is available. If host MPI is missing, use the Docker fallback in `references/execution-notes.md`.

3. Escalate to behavior-specific regressions.
   Use `references/regression-matrix.md` to map the changed area to phase12, spin/MPI parity, HSE06, Hubbard +U, memory-estimator, symmetry, workflow, or Wannier coverage.

4. Prefer dedicated repo scripts.
   Use `scripts/*.sh` or case-local runners such as `test_example/si-oncv/symmetry-enabled/run_and_verify.sh` before inventing new glue code.

5. Keep the matrix proportional.
   Do not run every heavy regression by default. Cover the code you changed and the nearest user-visible failure mode.

6. Close with explicit validation notes.
   State the commands you ran, which ones needed Docker or release binaries, and which suites you intentionally skipped.

## Common Triggers

- Changes under `pw/src` that affect SCF, energies, forces, stress, restart, export, finalization, or allocator diagnostics.
- Changes under `control/src`, `crystal/src`, `pspot/src`, or bootstrap code that alter accepted inputs or runtime construction.
- Changes under `workflow/src` or staged Wannier logic that can break `dwf` or multi-step artifact handoff.
- XC-mode work such as spin, phase12, HSE06, or Hubbard +U.
- Symmetry or k-point reduction changes that need example-driven verification.

## References

- Read `references/regression-matrix.md` to choose the right suites by changed area.
- Read `references/execution-notes.md` for host-vs-Docker guidance, environment variables, and reporting expectations.
