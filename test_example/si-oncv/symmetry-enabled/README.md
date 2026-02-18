# Silicon Symmetry-Enabled SCF/Relax Check

This example enables `symmetry = true` for both SCF and relax to validate:

- k-mesh symmetry reduction (`nkpt < nk1*nk2*nk3`),
- force symmetry consistency (`sum_i F_i ~ 0`),
- stress symmetry/shape consistency (symmetric tensor, small off-diagonals, near-isotropic diagonal for Si).

## Layout

- `scf/`: one-step SCF + force/stress output
- `relax/`: short cell relaxation with repeated force/stress output
- `run_and_verify.sh`: run both stages and check symmetry metrics from logs

## Run

From the repository root:

```bash
bash test_example/si-oncv/symmetry-enabled/run_and_verify.sh
```

Optional flags:

- `PW_BIN=/path/to/pw` to override binary path (default: `target/release/pw`)
- `FORCE_BUILD=1` to force rebuild of `pw` before run
- `--skip-run` to validate existing logs only

Tolerance environment variables (optional):

- `FORCE_SUM_TOL` (eV/A, default `5e-3`)
- `STRESS_ASYM_TOL` (kbar, default `5e-2`)
- `STRESS_OFFDIAG_TOL` (kbar, default `5e-1`)
- `STRESS_DIAG_TOL` (kbar, default `1.0`)
