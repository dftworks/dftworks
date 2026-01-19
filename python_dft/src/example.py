#!/usr/bin/env python3
"""
Example DFT calculations demonstrating the educational plane-wave code.

This script shows:
1. Basic lattice and G-vector setup
2. K-point sampling with Monkhorst-Pack mesh
3. LDA exchange-correlation functional
4. Ewald summation for ion-ion energy
5. Fermi-Dirac smearing
6. Full SCF calculation for electrons in a box
7. Jellium model calculation
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import HA_TO_EV, BOHR_TO_ANG
from src.lattice import Lattice
from src.crystal import Crystal
from src.gvector import GVector
from src.kpoints import KPoints, monkhorst_pack, gamma_only
from src.xc import lda_xc
from src.ewald import Ewald
from src.smearing import create_smearing, find_fermi_level
from src.scf import SCFSolver, create_jellium_box


def example_lattice():
    """Demonstrate lattice operations."""
    print("\n" + "=" * 60)
    print("Example 1: Crystal Lattice")
    print("=" * 60)
    
    # Simple cubic lattice (10 Bohr = 5.29 Angstrom)
    a = 10.0  # Bohr
    lattice = Lattice.cubic(a)
    
    print(f"\nSimple cubic lattice with a = {a} Bohr ({a * BOHR_TO_ANG:.3f} Angstrom)")
    print(f"Volume: {lattice.volume:.2f} Bohr^3")
    print(f"\nLattice vectors:")
    print(f"  a = {lattice.vectors[0]}")
    print(f"  b = {lattice.vectors[1]}")
    print(f"  c = {lattice.vectors[2]}")
    
    print(f"\nReciprocal lattice vectors:")
    print(f"  b1 = {lattice.reciprocal_vectors[0]}")
    print(f"  b2 = {lattice.reciprocal_vectors[1]}")
    print(f"  b3 = {lattice.reciprocal_vectors[2]}")
    
    # FCC lattice
    print("\n--- FCC Primitive Cell ---")
    fcc = Lattice.fcc(a)
    print(f"FCC with conventional a = {a} Bohr")
    print(f"Primitive cell volume: {fcc.volume:.2f} Bohr^3 (1/4 of cubic)")
    
    return lattice


def example_crystal():
    """Demonstrate crystal structure."""
    print("\n" + "=" * 60)
    print("Example 2: Crystal Structure")
    print("=" * 60)
    
    # Diamond Silicon
    crystal = Crystal.diamond_si()
    crystal.display()
    
    return crystal


def example_gvectors(lattice):
    """Demonstrate G-vector generation."""
    print("\n" + "=" * 60)
    print("Example 3: G-Vector Generation")
    print("=" * 60)
    
    # Energy cutoff
    ecut = 5.0  # Hartree
    
    print(f"\nEnergy cutoff: {ecut} Ha ({ecut * HA_TO_EV:.1f} eV)")
    
    gvec = GVector(lattice, ecut)
    
    print(f"Number of G-vectors: {gvec.npw}")
    print(f"G=0 at index: {gvec.g0_index}")
    print(f"\nFirst 10 G-vectors (Miller indices):")
    for i in range(min(10, gvec.npw)):
        m = gvec.miller[i]
        g = gvec.norms[i]
        print(f"  G[{i}] = ({m[0]:2d}, {m[1]:2d}, {m[2]:2d}), |G| = {g:.4f}")
    
    fft_shape = gvec.get_fft_grid_size()
    print(f"\nFFT grid size: {fft_shape}")
    
    return gvec


def example_kpoints(lattice):
    """Demonstrate k-point sampling."""
    print("\n" + "=" * 60)
    print("Example 4: K-Point Sampling")
    print("=" * 60)
    
    # Gamma point only
    gamma = gamma_only(lattice)
    gamma.display()
    
    # Monkhorst-Pack mesh
    print("\n--- Monkhorst-Pack Mesh ---")
    kpts = monkhorst_pack(lattice, 4, 4, 4, shift=False)
    kpts.display()
    
    return kpts


def example_xc():
    """Demonstrate LDA exchange-correlation."""
    print("\n" + "=" * 60)
    print("Example 5: LDA Exchange-Correlation")
    print("=" * 60)
    
    # Test densities (electrons per Bohr^3)
    rho = np.array([0.001, 0.01, 0.1, 1.0])
    
    vxc, exc = lda_xc(rho)
    
    print("\nLDA-PZ Exchange-Correlation:")
    print(f"{'rho (e/Bohr^3)':>15} {'V_xc (Ha)':>12} {'eps_xc (Ha)':>12} {'V_xc (eV)':>12}")
    print("-" * 55)
    for i in range(len(rho)):
        print(f"{rho[i]:15.4f} {vxc[i]:12.6f} {exc[i]:12.6f} {vxc[i]*HA_TO_EV:12.4f}")


def example_ewald():
    """Demonstrate Ewald summation."""
    print("\n" + "=" * 60)
    print("Example 6: Ewald Summation")
    print("=" * 60)
    
    # NaCl-like structure (simplified)
    a = 10.0
    lattice = Lattice.cubic(a)
    
    # Two ions: one at origin, one at center
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    charges = np.array([1.0, -1.0])  # +1 and -1 charges
    
    print(f"\nTwo-ion system in {a} Bohr cube")
    print(f"Positions (fractional): {positions.tolist()}")
    print(f"Charges: {charges.tolist()}")
    
    ewald = Ewald(lattice, positions, charges)
    
    print(f"\nEwald energy: {ewald.energy:.6f} Ha ({ewald.energy * HA_TO_EV:.4f} eV)")
    print(f"\nForces (Ha/Bohr):")
    for i, f in enumerate(ewald.forces):
        print(f"  Ion {i+1}: [{f[0]:8.5f}, {f[1]:8.5f}, {f[2]:8.5f}]")
    
    print(f"\nStress tensor (Ha/Bohr^3):")
    print(ewald.stress)


def example_smearing():
    """Demonstrate smearing functions."""
    print("\n" + "=" * 60)
    print("Example 7: Electronic Smearing")
    print("=" * 60)
    
    # Test eigenvalues
    eigenvalues = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    
    # Fermi-Dirac
    fd = create_smearing('fd', temperature=1000)  # High T for visible effect
    fermi, occ = find_fermi_level(eigenvalues, 1.0, 4.0, fd)
    
    print(f"\nFermi-Dirac at T=1000 K")
    print(f"Fermi level: {fermi:.4f} Ha")
    print(f"{'Energy (Ha)':>12} {'Occupation':>12}")
    print("-" * 26)
    for e, o in zip(eigenvalues, occ):
        print(f"{e:12.4f} {o:12.4f}")
    
    # Gaussian smearing
    gs = create_smearing('gaussian', sigma=0.05)
    fermi, occ = find_fermi_level(eigenvalues, 1.0, 4.0, gs)
    
    print(f"\nGaussian smearing with sigma=0.05 Ha")
    print(f"Fermi level: {fermi:.4f} Ha")


def example_scf_harmonic():
    """Run SCF calculation for electrons in a harmonic trap."""
    print("\n" + "=" * 60)
    print("Example 8: SCF - Electrons in Harmonic Trap")
    print("=" * 60)
    
    # Create a simple cubic box
    a = 15.0  # Bohr (larger box for harmonic oscillator)
    lattice = Lattice.cubic(a)
    
    # Parameters
    ecut = 3.0  # Hartree (lower cutoff for speed)
    n_bands = 4
    n_electrons = 2  # Two electrons (like helium atom in a trap)
    
    print(f"\nBox size: {a} Bohr")
    print(f"Energy cutoff: {ecut} Ha")
    print(f"Electrons: {n_electrons}")
    print(f"Bands: {n_bands}")
    
    # Create solver (uses harmonic potential by default)
    solver = SCFSolver(
        lattice=lattice,
        ecut=ecut,
        n_bands=n_bands,
        n_electrons=n_electrons,
        mixer='broyden'
    )
    
    print(f"Plane waves: {solver.npw}")
    print(f"FFT grid: {solver.fft_shape}")
    
    # Run SCF
    print("\nRunning SCF...")
    energy = solver.run(max_iter=30, tol=1e-5, verbose=True)
    
    return energy


def example_jellium():
    """Run SCF calculation for jellium (uniform electron gas)."""
    print("\n" + "=" * 60)
    print("Example 9: SCF - Jellium Box")
    print("=" * 60)
    
    # Jellium box parameters
    a = 10.0  # Bohr
    n_electrons = 2
    ecut = 3.0  # Hartree
    
    print(f"\nJellium box: {a} Bohr cube")
    print(f"Electrons: {n_electrons}")
    print(f"Energy cutoff: {ecut} Ha")
    
    # Compute Wigner-Seitz radius
    volume = a**3
    rho = n_electrons / volume
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0/3.0)
    print(f"Density: {rho:.6f} e/Bohr^3")
    print(f"Wigner-Seitz radius rs: {rs:.2f} Bohr")
    
    # Create jellium solver
    solver = create_jellium_box(a, n_electrons, ecut)
    
    print(f"Plane waves: {solver.npw}")
    
    # Run SCF
    print("\nRunning SCF...")
    energy = solver.run(max_iter=30, tol=1e-5, verbose=True)
    
    return energy


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# Educational Plane-Wave DFT Examples")
    print("# Version 0.2.0")
    print("#" * 60)
    
    # Basic examples
    lattice = example_lattice()
    example_crystal()
    gvec = example_gvectors(lattice)
    example_kpoints(lattice)
    example_xc()
    example_ewald()
    example_smearing()
    
    # SCF examples
    example_scf_harmonic()
    example_jellium()
    
    print("\n" + "#" * 60)
    print("# All examples completed!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
