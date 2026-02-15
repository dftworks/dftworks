# Wannier90 Projected Example (Si, non-spin)

This example uses explicit `sp3` projectors for silicon and a matching
`wannier90_num_wann = 8`.

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

## Activate projectors in `si.win`

Replace the placeholder projection block with an explicit Si `sp3` block:

```bash
perl -0777 -i -pe 's/! begin projections\n! <SPECIES>:s;p;d\n! end projections/begin projections\nSi1:sp3\nend projections/s' si.win
```

## Run Wannier90

```bash
wannier90.x si
```

Notes:

- This example is configured with `wannier90_num_wann = 8` in `in.ctrl`.
- The `Si1:sp3` block is consistent with this `num_wann` setting for the two-atom Si cell.
