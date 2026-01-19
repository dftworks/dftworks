"""
G-vector generation for plane wave calculations.

G-vectors are reciprocal lattice vectors used as the plane wave basis.
"""

import numpy as np
from .lattice import Lattice


class GVector:
    """
    G-vector grid for plane wave basis.
    
    Generates all G-vectors within a kinetic energy cutoff and provides
    mapping to/from FFT grids.
    
    Attributes:
        lattice: Lattice object
        ecut: Energy cutoff in Hartree
        miller: (N, 3) array of Miller indices
        cart: (N, 3) array of Cartesian G-vectors
        norms: (N,) array of |G| values
        npw: Number of plane waves
    """
    
    def __init__(self, lattice, ecut):
        """
        Generate G-vectors within energy cutoff.
        
        Args:
            lattice: Lattice object
            ecut: Energy cutoff in Hartree (kinetic energy: 0.5*|G|^2 < ecut)
        """
        self.lattice = lattice
        self.ecut = ecut
        
        # Generate G-vectors
        self._generate_gvectors()
    
    def _generate_gvectors(self):
        """Generate all G-vectors with 0.5*|G|^2 < ecut."""
        b = self.lattice.reciprocal_vectors
        
        # Maximum |G| from energy cutoff: 0.5*|G|^2 = ecut => |G| = sqrt(2*ecut)
        gmax = np.sqrt(2.0 * self.ecut)
        
        # Estimate max Miller indices
        # |G| = |n1*b1 + n2*b2 + n3*b3| <= |n1|*|b1| + |n2|*|b2| + |n3|*|b3|
        b_norms = np.linalg.norm(b, axis=1)
        n_max = (gmax / b_norms).astype(int) + 1
        
        # Generate all candidate Miller indices
        miller_list = []
        cart_list = []
        norm_list = []
        
        for n1 in range(-n_max[0], n_max[0] + 1):
            for n2 in range(-n_max[1], n_max[1] + 1):
                for n3 in range(-n_max[2], n_max[2] + 1):
                    # G = n1*b1 + n2*b2 + n3*b3
                    g_cart = n1 * b[0] + n2 * b[1] + n3 * b[2]
                    g_norm = np.linalg.norm(g_cart)
                    
                    # Check energy cutoff: 0.5*|G|^2 < ecut
                    if 0.5 * g_norm**2 < self.ecut:
                        miller_list.append([n1, n2, n3])
                        cart_list.append(g_cart)
                        norm_list.append(g_norm)
        
        # Convert to arrays
        self.miller = np.array(miller_list, dtype=int)
        self.cart = np.array(cart_list)
        self.norms = np.array(norm_list)
        self.npw = len(self.norms)
        
        # Sort by |G| (G=0 first)
        order = np.argsort(self.norms)
        self.miller = self.miller[order]
        self.cart = self.cart[order]
        self.norms = self.norms[order]
        
        # Find index of G=0
        self.g0_index = np.where(self.norms < 1e-10)[0][0]
    
    def get_fft_grid_size(self, factor=2.0):
        """
        Determine FFT grid size to avoid aliasing.
        
        For proper representation of products (like V*psi), the FFT grid
        must be at least 2x the maximum Miller index in each direction.
        
        Args:
            factor: Safety factor (default 2.0 for products)
        
        Returns:
            (n1, n2, n3): FFT grid dimensions
        """
        max_miller = np.max(np.abs(self.miller), axis=0)
        # FFT size should be at least 2*max + 1, rounded up to next suitable FFT size
        n_fft = (factor * max_miller + 1).astype(int)
        
        # Round up to next power of 2 or product of small primes for efficiency
        n_fft = np.array([self._next_fft_size(n) for n in n_fft])
        
        return tuple(n_fft)
    
    @staticmethod
    def _next_fft_size(n):
        """Find next efficient FFT size >= n."""
        # For simplicity, round up to next power of 2
        return int(2 ** np.ceil(np.log2(max(n, 4))))
    
    def map_to_fft_grid(self, data_g, fft_shape):
        """
        Map 1D G-space data to 3D FFT grid.
        
        Args:
            data_g: (npw,) array of data in G-space
            fft_shape: (n1, n2, n3) FFT grid dimensions
        
        Returns:
            (n1, n2, n3) complex array on FFT grid
        """
        n1, n2, n3 = fft_shape
        grid = np.zeros(fft_shape, dtype=complex)
        
        for ig, (m1, m2, m3) in enumerate(self.miller):
            # Handle negative indices (wrap around)
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            grid[i1, i2, i3] = data_g[ig]
        
        return grid
    
    def map_from_fft_grid(self, grid):
        """
        Extract 1D G-space data from 3D FFT grid.
        
        Args:
            grid: (n1, n2, n3) complex array on FFT grid
        
        Returns:
            (npw,) array of data in G-space
        """
        n1, n2, n3 = grid.shape
        data_g = np.zeros(self.npw, dtype=complex)
        
        for ig, (m1, m2, m3) in enumerate(self.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            data_g[ig] = grid[i1, i2, i3]
        
        return data_g
    
    def get_kinetic_energies(self, k_cart=None):
        """
        Get kinetic energies |k+G|^2/2 for each G-vector.
        
        Args:
            k_cart: k-point in Cartesian coordinates (default: Gamma point)
        
        Returns:
            (npw,) array of kinetic energies
        """
        if k_cart is None:
            k_cart = np.zeros(3)
        
        kg = self.cart + k_cart
        return 0.5 * np.sum(kg**2, axis=1)
    
    def __repr__(self):
        return f"GVector(npw={self.npw}, ecut={self.ecut:.2f} Ha)"
