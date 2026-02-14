# Wannier90 Spin Example (Si)

This example runs collinear-spin DFTWorks for silicon and exports
channel-separated Wannier90 interface files.

## Run DFTWorks

From this directory:

```bash
../../../target/release/pw
```

After the run, DFTWorks writes:

- `si.up.win`
- `si.up.nnkp`
- `si.up.mmn`
- `si.up.amn`
- `si.up.eig`
- `si.dn.win`
- `si.dn.nnkp`
- `si.dn.mmn`
- `si.dn.amn`
- `si.dn.eig`

## Run Wannier90

Edit `si.up.win` and/or `si.dn.win` if needed (for example projections and runtime options), then run:

```bash
wannier90.x si.up
wannier90.x si.dn
```

Notes:

- `si.up.amn` and `si.dn.amn` are identity-gauge initial guesses for the first `num_wann` bands.
- If you want different initial projection gauges, replace `si.up.amn` and `si.dn.amn` before running Wannier90.
