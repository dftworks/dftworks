---
name: DFT Python Documentation
overview: Document the core DFT algorithms from dftworks and create an educational Python implementation that demonstrates the fundamental concepts of plane-wave pseudopotential DFT calculations.
todos:
  - id: create-folder
    content: Create python_dft/ directory structure
    status: pending
  - id: algorithms-doc
    content: Write ALGORITHMS.md with mathematical foundations and algorithm descriptions
    status: pending
  - id: readme
    content: Write README.md with overview and usage
    status: pending
  - id: constants
    content: Implement constants.py with physical/numerical constants
    status: pending
  - id: lattice
    content: Implement lattice.py for crystal lattice operations
    status: pending
  - id: gvector
    content: Implement gvector.py for G-vector generation
    status: pending
  - id: pwbasis
    content: Implement pwbasis.py for plane wave basis
    status: pending
  - id: xc
    content: Implement xc.py with LDA exchange-correlation
    status: pending
  - id: hartree
    content: Implement hartree.py for Hartree potential
    status: pending
  - id: hamiltonian
    content: Implement hamiltonian.py for H*psi
    status: pending
  - id: eigensolver
    content: Implement eigensolver.py with PCG method
    status: pending
  - id: mixing
    content: Implement mixing.py with Broyden mixing
    status: pending
  - id: scf
    content: Implement scf.py as the main driver
    status: pending
  - id: example
    content: Create example.py with a working demonstration
    status: pending
---

# DFT Documentation and Python Implementation Plan

## Overview

Create comprehensive documentation of the core DFT algorithms in dftworks and develop an educational Python implementation that demonstrates the fundamental concepts of plane-wave pseudopotential DFT.

## Code Analysis Summary

The dftworks codebase implements plane-wave pseudopotential DFT with:

```mermaid
flowchart TD
    subgraph scf_loop [SCF Loop]
        A[Initial Density rho] --> B[Compute Potentials]
        B --> C[Solve Kohn-Sham Equations]
        C --> D[Update Density]
        D --> E{Converged?}
        E -->|No| F[Mix Densities]
        F --> B
        E -->|Yes| G[Compute Forces/Stress]
    end
    
    subgraph potentials [Potential Components]
        B --> B1[Hartree: V_H]
        B --> B2[Exchange-Correlation: V_xc]
        B --> B3[Local Pseudopotential: V_loc]
        B --> B4[Non-local Pseudopotential: V_nl]
    end
    
    subgraph eigensolver [Eigensolver]
        C --> C1[PCG Iterative Solver]
        C1 --> C2[Gram-Schmidt Orthogonalization]
    end
```

## Core Algorithms Identified

### 1. Self-Consistent Field (SCF) Loop
- Location: [scf/src/nonspin.rs](scf/src/nonspin.rs)
- Algorithm: Iterative solution of Kohn-Sham equations until energy converges

### 2. Hartree Potential
- Location: [scf/src/hartree.rs](scf/src/hartree.rs)
- Formula: `V_H(G) = 4*pi * rho(G) / G^2`

### 3. Exchange-Correlation (LDA-PZ)
- Location: [xc/src/ldapz.rs](xc/src/ldapz.rs)
- Slater exchange: `vx = cx * rho^(1/3)`
- Perdew-Zunger correlation with rs-dependent parameterization

### 4. Preconditioned Conjugate Gradient Eigensolver
- Location: [eigensolver/src/pcg.rs](eigensolver/src/pcg.rs)
- Band-by-band optimization with Gram-Schmidt orthogonalization

### 5. Hamiltonian Application (H*psi)
- Location: [hpsi/src/lib.rs](hpsi/src/lib.rs)
- Kinetic: `T*psi = |k+G|^2 * psi(G)`
- Local: FFT-based convolution in real space
- Non-local: Kleinman-Bylander projectors

### 6. Density Mixing (Broyden)
- Location: [mixing/src/broyden.rs](mixing/src/broyden.rs)
- Modified Broyden method for SCF acceleration

### 7. Ewald Summation (Ion-Ion Energy)
- Location: [ewald/src/lib.rs](ewald/src/lib.rs)
- Real-space + G-space decomposition

### 8. Plane Wave Basis
- Location: [pwbasis/src/lib.rs](pwbasis/src/lib.rs)
- G-vectors with |k+G|^2 < E_cutoff

## Documentation Structure

Create `python_dft/` folder with:

1. **ALGORITHMS.md** - Detailed algorithm documentation
   - Mathematical foundations
   - Kohn-Sham equations
   - Each component algorithm with formulas
   - SCF workflow diagram

2. **README.md** - Overview and usage instructions

## Python Implementation Structure

Educational Python code in `python_dft/src/`:

```
python_dft/
├── README.md
├── ALGORITHMS.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── constants.py       # Physical constants
    ├── lattice.py         # Crystal lattice operations
    ├── gvector.py         # G-vector generation
    ├── pwbasis.py         # Plane wave basis set
    ├── xc.py              # LDA exchange-correlation
    ├── hartree.py         # Hartree potential
    ├── eigensolver.py     # PCG eigensolver
    ├── hamiltonian.py     # H*psi application
    ├── mixing.py          # Density mixing
    ├── scf.py             # SCF driver
    └── example.py         # Simple example (hydrogen atom in a box)
```

## Python Implementation Scope

Focus on clarity over performance:
- Use numpy for arrays and FFT
- Simple cubic lattice for demonstration
- Local potential only (no pseudopotentials)
- LDA exchange-correlation
- PCG eigensolver (simplified)
- Broyden mixing

## Key Simplifications

- Single k-point (Gamma only)
- No spin polarization
- No pseudopotentials (jellium or simple model potential)
- No symmetry
- No parallelization
