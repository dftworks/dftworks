# Educational Plane-Wave DFT in Python

A simplified, educational implementation of plane-wave pseudopotential Density Functional Theory (DFT) in Python. This code is designed to help understand the fundamental algorithms used in production DFT codes like Quantum ESPRESSO, VASP, and dftworks.

## Purpose

This implementation prioritizes **clarity over performance**. It demonstrates:

- The Self-Consistent Field (SCF) loop
- Plane wave basis set construction
- Hartree potential computation
- LDA exchange-correlation (Perdew-Zunger)
- Preconditioned Conjugate Gradient eigensolver
- Broyden density mixing
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
from src.scf import SCFSolver
from src.lattice import Lattice

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

## Project Structure

```
python_dft/
├── README.md           # This file
├── ALGORITHMS.md       # Detailed algorithm documentation
├── requirements.txt    # Python dependencies
└── src/
    ├── __init__.py
    ├── constants.py    # Physical and numerical constants
    ├── lattice.py      # Crystal lattice operations
    ├── gvector.py      # G-vector generation
    ├── pwbasis.py      # Plane wave basis set
    ├── xc.py           # Exchange-correlation functionals
    ├── hartree.py      # Hartree potential
    ├── hamiltonian.py  # Hamiltonian application (H*psi)
    ├── eigensolver.py  # PCG eigensolver
    ├── mixing.py       # Density mixing schemes
    ├── scf.py          # SCF driver
    └── example.py      # Working examples
```

## Key Simplifications

This educational code makes several simplifications compared to production codes:

| Feature | This Code | Production Codes |
|---------|-----------|------------------|
| k-points | Gamma only | Full BZ sampling |
| Spin | Non-spin-polarized | Spin-polarized |
| Pseudopotentials | Model/jellium | Norm-conserving, PAW |
| Parallelization | None | MPI, OpenMP |
| FFT | NumPy | FFTW, cuFFT |
| Symmetry | None | Space group |

## Algorithm Overview

The SCF loop follows this workflow:

```
1. Initialize electron density (uniform or atomic)
2. LOOP:
   a. Compute Hartree potential:     V_H(G) = 4π ρ(G) / G²
   b. Compute XC potential:          V_xc = d(ρ·ε_xc)/dρ
   c. Build total potential:         V_eff = V_H + V_xc + V_ext
   d. Solve eigenvalue problem:      H|ψ⟩ = ε|ψ⟩
   e. Build new density:             ρ_new = Σ f_i |ψ_i|²
   f. Mix densities:                 ρ_next = mix(ρ_old, ρ_new)
   g. Check convergence
3. Output: Total energy, eigenvalues
```

## Documentation

See [ALGORITHMS.md](ALGORITHMS.md) for detailed mathematical descriptions of:

- Kohn-Sham equations and DFT theory
- Plane wave basis set construction
- Each algorithm component with formulas
- References to original papers

## Examples

### Electron in a Box

```python
# See src/example.py for a complete working example
python -m src.example
```

## Dependencies

- **numpy**: Array operations and FFT
- **scipy**: Linear algebra (optional, for comparison)

## Relationship to dftworks

This Python code is a simplified version of the algorithms in [dftworks](https://github.com/dftworks/dftworks), a production DFT code written in Rust. The Rust implementation includes:

- Full pseudopotential support (UPF format)
- Multiple k-points
- Spin polarization
- MPI parallelization
- Forces and stress tensors
- Geometry optimization

## Learning Path

1. Start with `ALGORITHMS.md` to understand the theory
2. Read `src/constants.py` for units and constants
3. Follow the data flow in `src/scf.py`
4. Explore individual components:
   - `gvector.py` → `pwbasis.py` (basis construction)
   - `hartree.py` → `xc.py` (potential computation)
   - `hamiltonian.py` → `eigensolver.py` (eigenvalue solution)

## License

MIT License - See the main dftworks repository for details.

## References

1. Martin, R. M. "Electronic Structure: Basic Theory and Practical Methods"
2. Payne, M. C., et al. "Iterative minimization techniques for ab initio total-energy calculations" Rev. Mod. Phys. 64, 1045 (1992)
