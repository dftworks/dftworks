# Phase 1/2 Parity Report

- Run date (UTC): 2026-02-13 02:51:38 UTC
- Command:
  - `FORCE_BUILD=1 bash ./scripts/run_phase12_regression.sh`
- Reference:
  - `test_example/si-oncv/regression/phase12_reference.tsv`

## Result

- Overall status: `PASS`
- Script summary: `Phase 1/2 regression passed.`

## Case Deltas (Current - Reference)

| Case | dE (Ry) | dFermi (eV) |
| --- | ---: | ---: |
| `lda_nonspin` | `5.799e-08` | `0` |
| `lsda_spin` | `4.066e-08` | `1e-05` |
| `pbe_nonspin` | `4.593e-08` | `0` |
| `pbe_spin` | `5.583e-09` | `0` |

## Tolerances

- `ENERGY_TOL_RY = 5e-4`
- `FERMI_TOL_EV = 5e-3`
