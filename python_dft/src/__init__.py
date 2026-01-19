"""
Educational Plane-Wave DFT Implementation

A simplified Python implementation of plane-wave pseudopotential
Density Functional Theory for educational purposes.

Modules:
    constants: Physical constants and unit conversions
    lattice: Crystal lattice operations
    crystal: Crystal structure with atoms
    gvector: G-vector generation for plane-wave basis
    pwbasis: Plane-wave basis set
    kpoints: K-point sampling (Monkhorst-Pack, Gamma)
    xc: Exchange-correlation functionals (LDA-PZ)
    hartree: Hartree potential computation
    hamiltonian: Kohn-Sham Hamiltonian application
    eigensolver: Preconditioned Conjugate Gradient solver
    mixing: Density mixing (Broyden, linear)
    smearing: Occupation smearing (Fermi-Dirac, Gaussian, MP)
    density: Electron density computation
    ewald: Ewald summation for ion-ion energy
    pseudopotential: Pseudopotential handling
    total_energy: Energy computation
    scf: Self-consistent field driver
"""

# Physical constants
from .constants import *

# Lattice and structure
from .lattice import Lattice
from .crystal import Crystal, Atom, read_xyz, make_supercell

# Plane-wave basis
from .gvector import GVector
from .pwbasis import PWBasis

# K-point sampling
from .kpoints import (
    KPoints,
    GammaPoint,
    gamma_only,
    monkhorst_pack,
    automatic_kpoints,
    high_symmetry_path,
    get_kpg_norms,
)

# Exchange-correlation
from .xc import lda_xc, compute_xc_energy, compute_xc_potential_energy

# Potentials
from .hartree import compute_hartree_potential, compute_hartree_energy
from .pseudopotential import (
    AtomicSpecies,
    SimplePseudopotential,
    LocalPotential,
    NonLocalPotential,
    compute_local_potential_g,
    create_jellium_potential,
    create_harmonic_potential,
)

# Hamiltonian
from .hamiltonian import Hamiltonian, g_to_r, r_to_g

# Eigensolver
from .eigensolver import PCGEigensolver, random_initial_guess

# Density mixing
from .mixing import BroydenMixer, LinearMixer

# Smearing
from .smearing import (
    FermiDirac,
    Gaussian,
    MethfesselPaxton,
    FixedOccupation,
    create_smearing,
    find_fermi_level,
    compute_band_energy,
)

# Density
from .density import (
    compute_density,
    compute_density_g,
    initial_density_uniform,
    initial_density_atomic,
    integrate_density,
)

# Energy
from .total_energy import TotalEnergy
from .ewald import Ewald, compute_ewald_energy

# SCF solver
from .scf import SCFSolver, create_jellium_box

__version__ = "0.2.0"

__all__ = [
    # Constants
    "BOHR_TO_ANG", "ANG_TO_BOHR", "HA_TO_EV", "EV_TO_HA", "RY_TO_EV",
    "PI", "TWOPI", "FOURPI", "BOLTZMANN_CONSTANT",
    
    # Lattice/Crystal
    "Lattice", "Crystal", "Atom", "read_xyz", "make_supercell",
    
    # Basis
    "GVector", "PWBasis",
    
    # K-points
    "KPoints", "GammaPoint", "gamma_only", "monkhorst_pack", 
    "automatic_kpoints", "high_symmetry_path", "get_kpg_norms",
    
    # XC
    "lda_xc", "compute_xc_energy", "compute_xc_potential_energy",
    
    # Potentials
    "compute_hartree_potential", "compute_hartree_energy",
    "AtomicSpecies", "SimplePseudopotential",
    "compute_local_potential_g", "create_jellium_potential",
    "create_harmonic_potential",
    
    # Hamiltonian
    "Hamiltonian", "g_to_r", "r_to_g",
    
    # Eigensolver
    "PCGEigensolver", "random_initial_guess",
    
    # Mixing
    "BroydenMixer", "LinearMixer",
    
    # Smearing
    "FermiDirac", "Gaussian", "MethfesselPaxton", "FixedOccupation",
    "create_smearing", "find_fermi_level", "compute_band_energy",
    
    # Density
    "compute_density", "compute_density_g", 
    "initial_density_uniform", "initial_density_atomic",
    "integrate_density",
    
    # Energy
    "TotalEnergy", "Ewald", "compute_ewald_energy",
    
    # SCF
    "SCFSolver", "create_jellium_box",
]
