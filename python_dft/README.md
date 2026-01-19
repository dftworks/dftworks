# Educational Plane-Wave DFT in Python

A simplified, educational implementation of plane-wave pseudopotential Density Functional Theory (DFT) in Python. This code is designed to help understand the fundamental algorithms used in production DFT codes like Quantum ESPRESSO, VASP, and dftworks.

## Version 0.2.0

This release includes:
- **K-point sampling** (Monkhorst-Pack mesh)
- **Electronic smearing** (Fermi-Dirac, Gaussian, Methfessel-Paxton)
- **Ewald summation** (ion-ion energy, forces, stress)
- **Pseudopotential framework** (local and non-local KB form)
- **Crystal structure** module
- **Total energy** breakdown

## Purpose

This implementation prioritizes **clarity over performance**. It demonstrates:

- The Self-Consistent Field (SCF) loop
- Plane wave basis set construction
- K-point sampling (Monkhorst-Pack mesh)
- Hartree potential computation
- LDA exchange-correlation (Perdew-Zunger)
- Preconditioned Conjugate Gradient eigensolver
- Broyden density mixing
- Electronic smearing (Fermi-Dirac, Gaussian, MP)
- Ewald summation for ion-ion interactions
- Total energy calculation

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src import *

# Create a simple cubic cell (10 Bohr side)
lattice = Lattice.cubic(10.0)

# Set up SCF solver
solver = SCFSolver(
    lattice=lattice,
    ecut=5.0,       # Energy cutoff in Hartree
    n_bands=4,      # Number of bands to compute
    n_electrons=2   # Number of electrons
)

# Run SCF
energy = solver.run(max_iter=50, tol=1e-6)
print(f"Total energy: {energy:.6f} Hartree")
```

### Crystal Structure Example

```python
from src import Crystal, Lattice

# Create diamond Silicon
crystal = Crystal.diamond_si()
crystal.display()

# Or build custom structure
lattice = Lattice.cubic(10.0)
crystal = Crystal(lattice)
crystal.add_atom('Na', [0.0, 0.0, 0.0], zion=1)
crystal.add_atom('Cl', [0.5, 0.5, 0.5], zion=7)
```

### K-Point Sampling

```python
from src import monkhorst_pack, Lattice

lattice = Lattice.cubic(10.0)
kpts = monkhorst_pack(lattice, 4, 4, 4)
kpts.display()
```

### Ewald Summation

```python
from src.ewald import Ewald

lattice = Lattice.cubic(10.0)
positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
charges = [1.0, -1.0]

ewald = Ewald(lattice, positions, charges)
print(f"Ion-ion energy: {ewald.energy:.6f} Ha")
```

## Project Structure

```
python_dft/
├── README.md           # This file
├── ALGORITHMS.md       # Detailed algorithm documentation
├── requirements.txt    # Python dependencies
└── src/
    ├── __init__.py     # Module exports
    ├── constants.py    # Physical and numerical constants
    ├── lattice.py      # Crystal lattice operations
    ├── crystal.py      # Crystal structure with atoms
    ├── gvector.py      # G-vector generation
    ├── pwbasis.py      # Plane wave basis set
    ├── kpoints.py      # K-point sampling (MP, Gamma)
    ├── xc.py           # Exchange-correlation functionals
    ├── hartree.py      # Hartree potential
    ├── hamiltonian.py  # Hamiltonian application (H*psi)
    ├── eigensolver.py  # PCG eigensolver
    ├── mixing.py       # Density mixing schemes
    ├── smearing.py     # Fermi-Dirac, Gaussian smearing
    ├── density.py      # Electron density computation
    ├── ewald.py        # Ion-ion Ewald summation
    ├── pseudopotential.py  # Pseudopotential handling
    ├── total_energy.py # Energy components
    ├── scf.py          # SCF driver
    └── example.py      # Working examples
```

## Modules

| Module | Description |
|--------|-------------|
| `constants.py` | Physical constants (Hartree, Bohr, etc.) |
| `lattice.py` | Lattice vectors, reciprocal lattice, coordinate conversion |
| `crystal.py` | Crystal structure with atoms and pseudopotentials |
| `gvector.py` | G-vector generation within energy cutoff |
| `pwbasis.py` | Plane wave basis set for given k-point |
| `kpoints.py` | Monkhorst-Pack mesh, Gamma point, band paths |
| `xc.py` | LDA-PZ exchange-correlation functional |
| `hartree.py` | Hartree potential in G-space |
| `hamiltonian.py` | Apply H to wavefunction (kinetic + local potential) |
| `eigensolver.py` | Preconditioned Conjugate Gradient solver |
| `mixing.py` | Broyden and linear density mixing |
| `smearing.py` | Fermi-Dirac, Gaussian, Methfessel-Paxton |
| `density.py` | Compute density from wavefunctions |
| `ewald.py` | Ion-ion Coulomb energy, forces, stress |
| `pseudopotential.py` | Local and non-local (KB) pseudopotentials |
| `total_energy.py` | Compute and display energy components |
| `scf.py` | Main SCF loop driver |

## Key Features vs. Simplifications

| Feature | This Code | Production Codes |
|---------|-----------|------------------|
| k-points | Monkhorst-Pack mesh | + symmetry reduction |
| Spin | Non-spin-polarized | Spin-polarized, SOC |
| Pseudopotentials | Simple model | Norm-conserving, PAW |
| Parallelization | None | MPI, OpenMP, GPU |
| FFT | NumPy | FFTW, cuFFT |
| Symmetry | None | Space group operations |
| Smearing | FD, Gaussian, MP | + cold smearing, etc. |

## Algorithm Overview

The SCF loop follows this workflow:

```
1. Initialize electron density (uniform or atomic)
2. LOOP:
   a. Compute Hartree potential:     V_H(G) = 4π ρ(G) / G²
   b. Compute XC potential:          V_xc = d(ρ·ε_xc)/dρ
   c. Build total potential:         V_eff = V_H + V_xc + V_ext
   d. Solve eigenvalue problem:      H|ψ⟩ = ε|ψ⟩ (PCG)
   e. Compute occupations:           f_i = f(ε_i, μ) (smearing)
   f. Build new density:             ρ_new = Σ_k w_k Σ_n f_n |ψ_{nk}|²
   g. Mix densities:                 ρ_next = mix(ρ_old, ρ_new) (Broyden)
   h. Check convergence
3. Output: Total energy, eigenvalues, forces, stress
```

## Documentation

See [ALGORITHMS.md](ALGORITHMS.md) for detailed mathematical descriptions of:

- Kohn-Sham equations and DFT theory
- Plane wave basis set construction
- K-point sampling and Monkhorst-Pack mesh
- Electronic smearing functions
- Pseudopotentials (local and non-local)
- Ewald summation
- Each algorithm component with formulas
- References to original papers

## Examples

### Run All Examples

```bash
# See src/example.py for a complete working example
cd python_dft
python -m src.example
```

This runs demonstrations of:
1. Lattice construction (cubic, FCC)
2. Crystal structure (diamond Si)
3. G-vector generation
4. K-point sampling
5. LDA exchange-correlation
6. Ewald summation
7. Electronic smearing
8. SCF calculation (harmonic trap)
9. Jellium box calculation

## Dependencies

- **numpy**: Array operations and FFT
- **scipy**: Special functions, integration, linear algebra

## Relationship to dftworks

This Python code is a simplified version of the algorithms in [dftworks](https://github.com/dftworks/dftworks), a production DFT code written in Rust. The Rust implementation includes:

- Full pseudopotential support (UPF format)
- Multiple k-points with symmetry
- Spin polarization (collinear and non-collinear)
- MPI parallelization
- Forces and stress tensors
- Geometry optimization (BFGS, DIIS)
- Phonons and response functions

## Learning Path

1. Start with `ALGORITHMS.md` to understand the theory
2. Read `src/constants.py` for units and constants
3. Explore `src/lattice.py` and `src/crystal.py` for structure
4. Follow the data flow in `src/scf.py`
5. Explore individual components:
   - `gvector.py` → `pwbasis.py` (basis construction)
   - `kpoints.py` (Brillouin zone sampling)
   - `hartree.py` → `xc.py` (potential computation)
   - `hamiltonian.py` → `eigensolver.py` (eigenvalue solution)
   - `smearing.py` (occupation numbers)
   - `ewald.py` (ion-ion interactions)

## License

MIT License - See the main dftworks repository for details.

## References

1. Martin, R. M. "Electronic Structure: Basic Theory and Practical Methods" (Cambridge, 2004)
2. Payne, M. C., et al. "Iterative minimization techniques for ab initio total-energy calculations" Rev. Mod. Phys. 64, 1045 (1992)
3. Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone integrations" Phys. Rev. B 13, 5188 (1976)
4. Perdew, J. P. & Zunger, A. "Self-interaction correction to density-functional approximations" Phys. Rev. B 23, 5048 (1981)
5. Ewald, P. P. "Die Berechnung optischer und elektrostatischer Gitterpotentiale" Ann. Phys. 369, 253 (1921)
