# Wannier90 Projected Example (Si, non-spin)

This example uses explicit `sp3` projectors for silicon (`num_wann = 8`) with
the staged Wannier90 workflow.

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

## Activate projectors in `si.win`

Replace the placeholder projection block with an explicit Si `sp3` block:

```bash
perl -0777 -i -pe 's/begin projections.*?end projections/begin projections\nSi1:sp3\nend projections/s' si.win
```

## Generate overlaps (`si.nnkp`, `si.mmn`, `si.amn`)

```bash
../../../target/release/w90-amn
```

## Run Wannier90

```bash
wannier90.x si
```

Notes:

- This example is configured with `wannier90_num_wann = 8` in `in.ctrl`.
- The `Si1:sp3` block is consistent with this `num_wann` setting for the two-atom Si cell.
