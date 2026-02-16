# Wannier90 Spin Example (Si)

This example runs collinear-spin DFTWorks for silicon and then generates
channel-separated Wannier90 inputs in post-SCF steps.

## Run DFTWorks

From this directory:

```bash
../../../target/release/pw
```

After the run, DFTWorks provides spin-resolved wavefunctions
(`out.wfc.up.k.*.hdf5`, `out.wfc.dn.k.*.hdf5`) and, with
`wannier90_export = true`, writes `si.up.eig` / `si.dn.eig`.

## Generate `si.up.win` and `si.dn.win`

```bash
../../../target/release/w90-win
```

## Configure projectors in `si.up.win` / `si.dn.win`

Edit each `begin projections ... end projections` block as needed.

## Generate overlaps (`*.nnkp`, `*.mmn`, `*.amn`)

```bash
../../../target/release/w90-amn
```

## Run Wannier90

```bash
wannier90.x si.up
wannier90.x si.dn
```

Notes:

- `w90-amn` reads projector settings from `si.up.win` / `si.dn.win`.
- If a projections block is empty, that channel falls back to pseudo-atomic trial orbitals.
