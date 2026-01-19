"""
Educational Plane-Wave DFT Implementation

A simplified Python implementation of plane-wave pseudopotential
Density Functional Theory for educational purposes.
"""

from .constants import *
from .lattice import Lattice
from .gvector import GVector
from .pwbasis import PWBasis
from .xc import lda_xc
from .hartree import compute_hartree_potential, compute_hartree_energy
from .hamiltonian import Hamiltonian
from .eigensolver import PCGEigensolver
from .mixing import BroydenMixer, LinearMixer
from .scf import SCFSolver

__version__ = "0.1.0"
__all__ = [
    "Lattice",
    "GVector", 
    "PWBasis",
    "lda_xc",
    "compute_hartree_potential",
    "compute_hartree_energy",
    "Hamiltonian",
    "PCGEigensolver",
    "BroydenMixer",
    "LinearMixer",
    "SCFSolver",
]
