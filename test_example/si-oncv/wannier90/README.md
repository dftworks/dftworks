# Wannier90 Example (Si)

This example runs DFTWorks for silicon and exports Wannier90 interface files.

For a collinear-spin export example, see `../wannier90-spin`.

## Run DFTWorks

From this directory:

```bash
../../../target/release/pw
```

After the run, DFTWorks writes:

- `si.win`
- `si.nnkp`
- `si.mmn`
- `si.amn`
- `si.eig`

## Run Wannier90

Edit `si.win` if needed (for example projections and runtime options), then run:

```bash
wannier90.x si
```

Notes:

- `si.amn` is an identity-gauge initial guess for the first `num_wann` bands.
- If you want a different initial projection gauge, replace `si.amn` before running `wannier90.x`.
