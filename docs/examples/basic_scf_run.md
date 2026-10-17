# Basic SCF Calculation Example

This guide walks through a basic SCF (self-consistent field) calculation with `pw`,
using the silicon case in `test_example/si-oncv/scf` as the reference. Every input
shown here is taken from that case or accepted by the current `in.ctrl` parser.

## Quick Start

```bash
# 1. Start from the silicon example
cp -r test_example/si-oncv/scf my-si-scf && cd my-si-scf
ls
# in.crystal  in.ctrl  in.kline  in.kmesh  in.magmom  in.pot  in.spin  pot/

# 2. Run the SCF calculation (from the repository root build)
cargo run --release --manifest-path ../Cargo.toml -p pw > out.pw.log 2>&1
# or, with MPI:
mpirun -np 4 ../target/release/pw > out.pw.log 2>&1

# 3. Check the output
grep -A3 scf_convergence out.pw.log
```

## Input Files

`pw` reads these files from the working directory:

| File | Purpose |
| --- | --- |
| `in.ctrl` | Runtime parameters (`key = value`) |
| `in.crystal` | Lattice vectors, relaxation masks, atomic positions |
| `in.pot` | Species → pseudopotential file |
| `in.kmesh` | k-point mesh (when `kpts_scheme = kmesh`, the default) |
| `in.kline` | k-point path (when `kpts_scheme = kline`) |
| `in.spin`, `in.magmom` | Initial magnetic moments (spin runs only) |

---

## Example: Silicon (diamond, primitive cell)

### 1. Crystal Structure (`in.crystal`)

```
5.43 5.43 5.43
0.000000000000 0.499112077875 0.499112082365 F T T
0.499112083364 0.000000000000 0.499112096736 T F T
0.499112097243 0.499112106257 0.000000000000 T T F
Si1 -0.1250004154187434 -0.12499984072271753 -0.125000173733401
Si1 0.12499979505138148 0.12500007176891748 0.1249996165758156
```

**Format:**
- Line 1: scale factors (Å) for the three lattice vectors.
- Lines 2–4: lattice vectors `a`, `b`, `c` (multiplied by the scale factors), each
  followed by three `T`/`F` flags that mark which Cartesian components may change
  during a variable-cell relaxation.
- Remaining lines: `species x y z` in fractional (crystal) coordinates. The species
  label must match an entry in `in.pot`.
- `#` starts a comment.

---

### 2. K-Point Mesh (`in.kmesh`)

```
2 2 2
0 0 0
```

- Line 1: Monkhorst–Pack divisions along `b1`, `b2`, `b3`.
- Line 2: mesh shift.
- Increase the mesh (e.g. `8 8 8`) until the total energy is converged.

---

### 3. Control Parameters (`in.ctrl`)

```
task = scf
xc_scheme = pbe
spin_scheme = nonspin
pot_scheme = upf
kpts_scheme = kmesh

# Basis (eV)
ecut_wfc = 400
# ecut_rho defaults to 4 * ecut_wfc

# Bands and occupations
nband = 8
smearing_scheme = mp2
temperature = 0

# SCF loop and convergence (energies in eV)
scf_max_iter = 60
scf_min_iter = 1
energy_epsilon = 1E-6
eigval_epsilon = 1E-6
eigen_solver = pcg

# Density mixing
scf_rho_mix_scheme = broyden
scf_rho_mix_alpha = 0.8
scf_rho_mix_history_steps = 4

# Output
verbosity = normal
save_rho = true
save_wfc = true
```

**Key parameters:**
- `task = scf` – run mode (`scf` or `band`; both currently run the SCF path).
- `xc_scheme = pbe` – exchange-correlation functional (`lda-pz`, `lsda-pz`, `pbe`, `hse06`).
- `ecut_wfc = 400` – wavefunction cutoff in **eV**; `ecut_rho` defaults to 4× this.
- `nband = 8` – number of bands (4 occupied + 4 empty for Si).
- `smearing_scheme = mp2` with `temperature = 0` – occupations; use a finite
  `temperature` (K) for metals.
- `scf_rho_mix_scheme = broyden` – `pulay` (default) or `broyden`;
  `scf_rho_mix_alpha` is the weight of the new output density.

Every named option is checked when `in.ctrl` is read: an unknown key or value stops
the run with the line number and the accepted values. See the README section
"`in.ctrl` parameter notes" for details.

---

### 4. Pseudopotential Mapping (`in.pot`)

```
Si1 Si-sr.upf
```

- One line per species: `label file`. The file is looked up in `pot/`.
- `pot_scheme = upf` (default) reads UPF files; use `upf-fr` for fully relativistic UPF.

---

## Running the Calculation

### Serial Execution

```bash
./pw > out.pw.log 2>&1
```

### MPI Parallel Execution

```bash
# 4 MPI ranks (k-point parallelism)
mpirun -np 4 ./pw > out.pw.log 2>&1
```

**Note:** `pw` distributes k-points across MPI ranks; use no more ranks than
irreducible k-points. `kpoint_schedule` (`cost_aware` by default, or `contiguous`,
`dynamic`) controls the assignment.

---

## Output

The SCF loop prints one line per iteration with the columns
`eps(eV)`, `Fermi(eV)`, `charge`, `Eharris(Ry)`, `Escf(Ry)` and `dE(eV)`, followed by
`scf_convergence_success` or `scf_convergence_failure`.

Checkpoint files written to the working directory:

| File | When |
| --- | --- |
| `out.scf.rho.hdf5` (`out.scf.rho.up/dn.hdf5` for spin) | `save_rho = true` (default) |
| `out.wfc.k.<ik>.hdf5` (`out.wfc.up/dn.k.<ik>.hdf5` for spin) | `save_wfc = true` (default) |
| `run.provenance.json` | always (name set by `provenance_manifest`) |

For structured outputs (`summary.json`, `energy.csv`, `dos.dat`, ...) run the case
through the workflow wrapper, `dwf run scf <case_dir>`; see the README section
"Workflow wrapper (`dwf`)".

---

## Common Tasks

### Geometry Optimization

There is no separate relax task: set `geom_optim_max_steps` above 1
(see `test_example/si-oncv/relax`).

```
geom_optim_scheme = diis        # bfgs (default) or diis
geom_optim_max_steps = 20
geom_optim_cell = true          # also relax the cell (uses the in.crystal masks)
geom_optim_force_tolerance = 0.01   # eV/Å
geom_optim_stress_tolerance = 0.05  # kbar
```

### Band Structure Calculation

Band structures use a k-point path (`kpts_scheme = kline`) after an SCF run
(see `test_example/si-oncv/bands`, or `dwf run bands <case_dir> --from latest`).

```
# in.ctrl (bands)
kpts_scheme = kline
nband = 16
scf_max_iter = 1
scf_max_iter_wfc = 60
save_rho = false
```

```
# in.kline: "line <points per segment>", then one segment per line (start, end)
line 20
G 0.0 0.0 0.0 X 0.0 0.5 0.5
X 0.0 0.5 0.5 W 0.25 0.75 0.5
W 0.25 0.75 0.5 L 0.5 0.5 0.5
L 0.5 0.5 0.5 G 0.0 0.0 0.0
```

### DOS Calculation

Run an SCF on a denser `in.kmesh`, then post-process the eigenvalues with
`dwf properties <run_dir> nscf --dos-sigma 0.1 --dos-ne 500`. The `in.ctrl` defaults
for the broadening are `dos_sigma = 0.1` (eV) and `dos_ne = 500`.

---

## Restart Calculation

With the defaults `save_rho = true` and `save_wfc = true`, a run leaves
`out.scf.rho.hdf5` and `out.wfc.k.*.hdf5` in the working directory. To start from them:

```
# in.ctrl (restart run)
restart = true
```

Restart is supported for `spin_scheme = nonspin` and `spin`.

---

## Troubleshooting

### SCF Not Converging

The run ends with `scf_convergence_failure`.

1. Reduce the mixing weight of the new density:
   ```
   scf_rho_mix_alpha = 0.3   # default 0.7
   ```
2. Increase the k-point mesh (for metals):
   ```
   # in.kmesh
   8 8 8
   0 0 0
   ```
3. Use smearing with a finite electronic temperature (for metals):
   ```
   smearing_scheme = mv      # fd, gs, mp1, mp2 (default), mv
   temperature = 300         # K
   ```
4. Allow more iterations:
   ```
   scf_max_iter = 100
   ```

### Memory Issues

1. Estimate memory before running:
   `cargo run -p pw --bin memory_estimate -- --case <case_dir>` (see the README
   section on `memory_estimate`).
2. Reduce the cutoff:
   ```
   ecut_wfc = 300   # eV
   ```
3. Reduce bands:
   ```
   nband = 6
   ```
4. Use more MPI ranks to distribute k-points.

### Wrong Results

Check that:
1. The pseudopotential files listed in `in.pot` exist under `pot/`.
2. The `in.crystal` scale factors are in Å.
3. The k-point mesh is converged (e.g. compare 4×4×4 with 8×8×8).
4. `ecut_wfc` is converged (in eV, e.g. compare 400, 500, 600).
5. `temperature` is small for semiconductors.

---

## Performance Tips

1. **K-point parallelism:** use about one MPI rank per irreducible k-point; enable
   `symmetry = true` to reduce the k-point count when the structure allows it.
2. **Less output:** `verbosity = quiet`.
3. **FFT planning:**
   ```
   fft_planner = measure          # slower first run, faster transforms
   fft_wisdom_file = fftw_wisdom.dat
   ```

---

## Further Reading

- `test_example/si-oncv/` – working cases (`scf`, `relax`, `bands`, `hse06-gamma`, `wannier90`, ...)
- `README.md` – workflow wrapper, DFT+U, HSE06, `in.ctrl` parameter notes, capability matrix
- `docs/architecture/scf_pipeline.md` – SCF iteration architecture
- `docs/architecture/checkpoint_abstraction.md` – restart mechanism
- `WORKSPACE_GUIDE.md` – performance patterns
