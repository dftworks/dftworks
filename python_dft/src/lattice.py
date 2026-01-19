"""
Crystal lattice operations.

Handles lattice vectors, reciprocal lattice, and volume calculations.
"""

import numpy as np
from .constants import TWOPI


class Lattice:
    """
    Crystal lattice defined by three lattice vectors.
    
    The lattice vectors are stored as rows of a 3x3 matrix:
        a = lattice_vectors[0]
        b = lattice_vectors[1]  
        c = lattice_vectors[2]
    
    Attributes:
        vectors: 3x3 array of lattice vectors (rows)
        volume: Unit cell volume
        reciprocal: 3x3 array of reciprocal lattice vectors
    """
    
    def __init__(self, vectors):
        """
        Initialize lattice from lattice vectors.
        
        Args:
            vectors: 3x3 array-like, lattice vectors as rows
                     [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
        """
        self.vectors = np.array(vectors, dtype=float)
        assert self.vectors.shape == (3, 3), "Lattice vectors must be 3x3"
        
        self._volume = None
        self._reciprocal = None
    
    @classmethod
    def cubic(cls, a):
        """
        Create a simple cubic lattice.
        
        Args:
            a: Lattice constant (cube side length)
        
        Returns:
            Lattice object
        """
        vectors = a * np.eye(3)
        return cls(vectors)
    
    @classmethod
    def from_parameters(cls, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
        """
        Create lattice from lattice parameters.
        
        Args:
            a, b, c: Lattice constants
            alpha, beta, gamma: Angles in degrees (default: 90 for orthorhombic)
        
        Returns:
            Lattice object
        """
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Build lattice vectors
        # a along x-axis
        va = np.array([a, 0.0, 0.0])
        
        # b in xy-plane
        vb = np.array([
            b * np.cos(gamma_rad),
            b * np.sin(gamma_rad),
            0.0
        ])
        
        # c in general direction
        cx = c * np.cos(beta_rad)
        cy = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        vc = np.array([cx, cy, cz])
        
        return cls(np.array([va, vb, vc]))
    
    @property
    def volume(self):
        """Unit cell volume (scalar triple product)."""
        if self._volume is None:
            self._volume = abs(np.linalg.det(self.vectors))
        return self._volume
    
    @property
    def reciprocal_vectors(self):
        """
        Reciprocal lattice vectors (2*pi factor included).
        
        b_i = 2*pi * (a_j x a_k) / (a_i . (a_j x a_k))
        """
        if self._reciprocal is None:
            # Transpose of inverse, scaled by 2*pi
            self._reciprocal = TWOPI * np.linalg.inv(self.vectors).T
        return self._reciprocal
    
    def get_vector_a(self):
        """Return first lattice vector."""
        return self.vectors[0]
    
    def get_vector_b(self):
        """Return second lattice vector."""
        return self.vectors[1]
    
    def get_vector_c(self):
        """Return third lattice vector."""
        return self.vectors[2]
    
    def cart_to_frac(self, cart_coords):
        """
        Convert Cartesian coordinates to fractional coordinates.
        
        Args:
            cart_coords: Cartesian coordinates (3,) or (N, 3)
        
        Returns:
            Fractional coordinates
        """
        return np.linalg.solve(self.vectors.T, cart_coords.T).T
    
    def frac_to_cart(self, frac_coords):
        """
        Convert fractional coordinates to Cartesian coordinates.
        
        Args:
            frac_coords: Fractional coordinates (3,) or (N, 3)
        
        Returns:
            Cartesian coordinates
        """
        frac_coords = np.asarray(frac_coords)
        return frac_coords @ self.vectors
    
    def __repr__(self):
        return (f"Lattice(\n"
                f"  a = {self.vectors[0]},\n"
                f"  b = {self.vectors[1]},\n"
                f"  c = {self.vectors[2]},\n"
                f"  volume = {self.volume:.6f})")
