# Wannier90 Example (Si)

This example runs DFTWorks for silicon and then generates Wannier90 files in
separate post-SCF steps.

For a collinear-spin export example, see `../wannier90-spin`.
For a non-spin explicit-projector example (`sp3`, `num_wann = 8`), see `../wannier90-projected`.

## Run DFTWorks

From this directory:

```bash
../../../target/release/pw
```

After the run, DFTWorks provides converged wavefunctions (`out.wfc.k.*.hdf5`)
and, with `wannier90_export = true`, writes `si.eig`.

## Generate `si.win`

```bash
../../../target/release/w90-win
```

## Configure projectors in `si.win`

Edit the `begin projections ... end projections` block (for example `Si1:sp3`
for the two-atom Si cell with `num_wann = 8`).

## Generate overlaps (`si.nnkp`, `si.mmn`, `si.amn`)

```bash
../../../target/release/w90-amn
```

## Run Wannier90

```bash
wannier90.x si
```

Notes:

- `w90-amn` reads projections from `si.win` and builds `si.amn` from those choices.
- If the projections block is left empty, `w90-amn` falls back to pseudo-atomic trial orbitals.
