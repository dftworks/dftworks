"""
Plane wave basis set for a specific k-point.

Extends GVector with k-point specific information.
"""

import numpy as np
from .gvector import GVector


class PWBasis:
    """
    Plane wave basis for a specific k-point.
    
    This is essentially a wrapper around GVector that adds k-point
    specific quantities like |k+G|^2 kinetic energies.
    
    Attributes:
        gvec: GVector object (shared among k-points)
        k_cart: k-point in Cartesian coordinates
        kg: |k+G|^2 / 2 kinetic energies
        npw: Number of plane waves
    """
    
    def __init__(self, gvec, k_cart=None):
        """
        Create plane wave basis for a k-point.
        
        Args:
            gvec: GVector object
            k_cart: k-point in Cartesian coordinates (default: Gamma)
        """
        self.gvec = gvec
        self.k_cart = np.zeros(3) if k_cart is None else np.array(k_cart)
        
        # Compute |k+G|^2 / 2
        self.kg = self._compute_kinetic()
        self.npw = gvec.npw
    
    def _compute_kinetic(self):
        """Compute kinetic energies |k+G|^2 / 2."""
        k_plus_g = self.gvec.cart + self.k_cart
        return 0.5 * np.sum(k_plus_g**2, axis=1)
    
    def get_kg(self):
        """Return kinetic energies."""
        return self.kg
    
    def get_npw(self):
        """Return number of plane waves."""
        return self.npw
    
    def get_gindex(self):
        """Return G-vector indices (identity for now)."""
        return np.arange(self.npw)
    
    def __repr__(self):
        return f"PWBasis(npw={self.npw}, k={self.k_cart})"


def create_fft_workspace(gvec, factor=2.0):
    """
    Create FFT workspace arrays.
    
    Args:
        gvec: GVector object
        factor: FFT grid size factor (2.0 for products)
    
    Returns:
        fft_shape: (n1, n2, n3) grid dimensions
        work_r: Real-space workspace array
        work_g: G-space workspace array
    """
    fft_shape = gvec.get_fft_grid_size(factor)
    work_r = np.zeros(fft_shape, dtype=complex)
    work_g = np.zeros(fft_shape, dtype=complex)
    return fft_shape, work_r, work_g
