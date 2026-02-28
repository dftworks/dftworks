# Basic SCF Calculation Example

This guide walks through a basic SCF (Self-Consistent Field) calculation using DFTWorks.

## Quick Start

```bash
# 1. Prepare input files in your working directory
ls
# in.ctrl in.crystal in.kmesh in.pot pot/

# 2. Run SCF calculation
mpirun -np 4 ./pw

# 3. Check output
cat out.log
ls runs/scf/
```

## Input Files

### Required Files

1. **`in.ctrl`** - Runtime control parameters
2. **`in.crystal`** - Crystal structure (lattice + atomic positions)
3. **`in.kmesh`** - K-point sampling mesh
4. **`in.pot`** - Pseudopotential mapping
5. **`pot/`** - Directory containing pseudopotential files

---

## Example: Silicon Diamond Structure

### 1. Crystal Structure (`in.crystal`)

```
# Silicon diamond structure (conventional cubic cell)
# Lattice parameter: 10.26 bohr (5.43 Å)

LATTICE_PARAMETER
10.26

LATTICE_VECTORS
1.0  0.0  0.0
0.0  1.0  0.0
0.0  0.0  1.0

NUMBER_OF_ATOMS
2

ATOMIC_POSITIONS (crystal)
Si  0.00  0.00  0.00
Si  0.25  0.25  0.25
```

**Explanation:**
- Cubic lattice with parameter 10.26 bohr
- 2 atoms in conventional cell (8 atoms in primitive cell reduced by symmetry)
- Positions in crystal (fractional) coordinates

---

### 2. K-Point Mesh (`in.kmesh`)

```
# 4x4x4 Monkhorst-Pack mesh
K_POINTS (mesh)
4 4 4
0 0 0
```

**Explanation:**
- 4×4×4 = 64 k-points (reduced to ~10 by symmetry)
- Monkhorst-Pack scheme with Gamma-centered shift (0,0,0)
- Converge total energy within ~0.01 eV

---

### 3. Control Parameters (`in.ctrl`)

```
# Basic SCF parameters
task = scf
xc_scheme = pbe
spin_scheme = nonspin

# Convergence
max_scf_iter = 50
energy_tol = 1.0e-6
density_tol = 1.0e-5

# Basis sets
cutoff_wfc = 50.0
cutoff_rho = 200.0

# Electronic structure
nband = 8
smearing_scheme = gaussian
smearing_width = 0.01

# Mixing
mixing_mode = pulay
mixing_beta = 0.7
mixing_ndim = 8

# Output
verbosity = normal
task_options = energy,force,stress
```

**Key parameters:**
- `task = scf` - Standard SCF calculation
- `xc_scheme = pbe` - PBE exchange-correlation functional
- `cutoff_wfc = 50 Ry` - Wavefunction cutoff energy (converged for Si)
- `cutoff_rho = 200 Ry` - Density cutoff (4× wavefunction cutoff for GGA)
- `nband = 8` - 8 bands (4 valence + 4 conduction for Si)
- `smearing_scheme = gaussian` - Smearing for metallic/near-gap systems
- `mixing_mode = pulay` - Pulay (DIIS) mixing for fast convergence

---

### 4. Pseudopotential Mapping (`in.pot`)

```
# Pseudopotential files for each element
Si  pot/Si.oncv.upf
```

**Explanation:**
- Maps element symbol to pseudopotential file
- `pot/Si.oncv.upf` must exist (ONCV norm-conserving pseudopotential)

---

### 5. Pseudopotential File

Download or generate `pot/Si.oncv.upf`:

```bash
# Download from PseudoDojo or other database
wget http://www.pseudo-dojo.org/.../Si.upf -O pot/Si.oncv.upf
```

---

## Running the Calculation

### Serial Execution

```bash
./pw > out.log 2>&1
```

### MPI Parallel Execution

```bash
# 4 MPI ranks (k-point parallelism)
mpirun -np 4 ./pw > out.log 2>&1

# 8 MPI ranks
mpirun -np 8 ./pw > out.log 2>&1
```

**Note:** DFTWorks uses MPI for k-point parallelism. Use `N_MPI ≤ N_kpoints` for efficiency.

---

## Output Files

After successful run:

```
runs/scf/
├── properties/
│   ├── summary.json           # Machine-readable summary
│   ├── energies.dat           # Total energies
│   ├── forces.dat             # Atomic forces
│   └── stress.dat             # Stress tensor
├── out.crystal                # Final crystal structure
├── out.scf.rho.hdf5          # Converged density (for restart)
├── out.wfc.k.*.hdf5          # Wavefunctions (if save_wavefunctions=true)
└── run.provenance.json        # Reproducibility metadata
```

**Main output:** `out.log`
```
SCF iteration summary:
Iter    Fermi(eV)  E_Harris(Ry)    ΔE(Ry)      Δρ        Time(s)
   1    6.643      -15.812345      1.2e-1      3.4e-2    2.1
   2    6.643      -15.813456      1.1e-3      8.9e-3    1.8
   3    6.643      -15.813501      4.5e-5      2.1e-3    1.7
   4    6.643      -15.813503      2.1e-6      5.3e-4    1.7
   5    6.643      -15.813503      3.4e-8      9.8e-5    1.7
   6    6.643      -15.813503      1.2e-9      1.5e-5    1.7

SCF converged after 6 iterations
Final energy: -15.813503 Ry
Fermi level: 6.643 eV
```

---

## Common Tasks

### Geometry Optimization

```
# in.ctrl
task = relax

# Convergence criteria
force_tol = 1.0e-3      # Ry/bohr
max_geom_iter = 50

# Optimizer
optimizer = bfgs
```

### Band Structure Calculation

**Step 1:** SCF with dense k-mesh
```
# in.ctrl (scf)
task = scf
# ... (use dense mesh, e.g., 8x8x8)
```

**Step 2:** NSCF along high-symmetry path
```
# in.ctrl (nscf)
task = nscf

# in.kline (high-symmetry path)
K_POINTS (line)
40   # points per segment
L    0.5  0.5  0.5
G    0.0  0.0  0.0
X    0.5  0.0  0.5
```

### DOS Calculation

```
# in.ctrl
task = nscf

# Dense k-mesh for DOS
# in.kmesh
K_POINTS (mesh)
12 12 12
0 0 0

# Post-process
# DOS output in runs/nscf/properties/dos.dat
```

---

## Restart Calculation

**Save checkpoint after SCF:**
```
# in.ctrl (first run)
task = scf
save_checkpoints = true
```

**Restart from checkpoint:**
```
# in.ctrl (restart run)
task = scf
restart = true
```

The code will load `out.scf.rho.hdf5` as initial density for warm start.

---

## Troubleshooting

### SCF Not Converging

**Symptoms:**
```
SCF did not converge after 50 iterations
Final energy delta: 1.2e-3 Ry
Final density delta: 4.5e-3
```

**Solutions:**
1. Reduce mixing parameter:
   ```
   mixing_beta = 0.3   # (default: 0.7)
   ```

2. Increase k-point mesh (for metals):
   ```
   # in.kmesh
   8 8 8  # denser mesh
   ```

3. Increase smearing width (for metals):
   ```
   smearing_width = 0.02  # (default: 0.01 eV)
   ```

4. Try different smearing:
   ```
   smearing_scheme = fermi_dirac  # or methfessel_paxton
   ```

### Memory Issues

**Error:** `Out of memory`

**Solutions:**
1. Reduce cutoffs:
   ```
   cutoff_wfc = 40.0  # (instead of 50)
   ```

2. Reduce bands:
   ```
   nband = 6  # Minimum: number of occupied bands
   ```

3. Use more MPI ranks (distribute k-points):
   ```bash
   mpirun -np 8 ./pw  # More ranks = less memory per rank
   ```

### Wrong Results

**Check:**
1. Pseudopotential files exist and are correct
2. Lattice parameter is in Bohr (not Angstrom): `a_bohr = a_ang × 1.8897`
3. K-point mesh is converged (test 4×4×4 vs 8×8×8)
4. Cutoffs are converged (test 40, 50, 60 Ry)
5. Smearing width is small enough (<0.02 eV for semiconductors)

---

## Performance Tips

1. **K-point parallelism:**
   ```bash
   # Use N_MPI ~ N_kpoints for best scaling
   mpirun -np 10 ./pw  # For 10 irreducible k-points
   ```

2. **Reduce output verbosity:**
   ```
   verbosity = quiet  # Less I/O overhead
   ```

3. **Use restart for multi-stage calculations:**
   - SCF → save checkpoint
   - NSCF/bands/DOS → restart from checkpoint

4. **Optimize FFT planning:**
   ```
   fft_planner = measure     # Slower first run, faster iterations
   fft_wisdom_file = fftw_wisdom.dat
   ```

---

## Next Steps

- [Geometry Optimization Example](geometry_optimization.md)
- [Band Structure and DOS Example](band_structure.md)
- [Restart Workflow Example](restart_workflow.md)
- [HSE06 Hybrid Functional Example](hse06_example.md)
- [DFT+U Example](hubbard_u_example.md)

## Further Reading

- `docs/architecture/scf_pipeline.md` - SCF iteration architecture
- `docs/architecture/checkpoint_abstraction.md` - Restart mechanism
- `WORKSPACE_GUIDE.md` - Performance optimization patterns
