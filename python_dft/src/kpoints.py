"""
K-Point Sampling for Brillouin Zone Integration.

Implements Monkhorst-Pack mesh generation and related utilities
for sampling the Brillouin zone in periodic DFT calculations.

Reference:
    Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone 
    integrations" Phys. Rev. B 13, 5188 (1976)
"""

import numpy as np
from .constants import TWOPI


class KPoints:
    """
    K-point mesh for Brillouin zone sampling.
    
    Generates a Monkhorst-Pack mesh of k-points in the first Brillouin zone.
    
    Attributes:
        k_frac: K-points in fractional (crystal) coordinates
        k_cart: K-points in Cartesian coordinates
        weights: Integration weights (sum to 1)
        nk: Number of k-points
    """
    
    def __init__(self, lattice, mesh, shift=None):
        """
        Generate Monkhorst-Pack k-point mesh.
        
        Args:
            lattice: Lattice object
            mesh: Tuple (n1, n2, n3) specifying mesh density
            shift: Optional tuple (s1, s2, s3) for mesh shift (0 or 0.5)
        """
        self.lattice = lattice
        self.mesh = mesh
        self.shift = shift if shift is not None else (0, 0, 0)
        
        # Generate k-points
        self.k_frac, self.weights = self._generate_mp_mesh()
        self.nk = len(self.k_frac)
        
        # Convert to Cartesian
        self.k_cart = self._frac_to_cart_all()
    
    def _generate_mp_mesh(self):
        """
        Generate Monkhorst-Pack mesh.
        
        k_i = (2*n - N - 1) / (2*N) + s_i
        
        where n = 1, ..., N and s_i is the shift.
        """
        n1, n2, n3 = self.mesh
        s1, s2, s3 = self.shift
        
        k_points = []
        
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    # Monkhorst-Pack formula
                    k1 = (2*i - n1 + 1) / (2*n1) + s1/n1
                    k2 = (2*j - n2 + 1) / (2*n2) + s2/n2
                    k3 = (2*k - n3 + 1) / (2*n3) + s3/n3
                    
                    k_points.append([k1, k2, k3])
        
        k_points = np.array(k_points)
        
        # All points have equal weight for now
        nk = len(k_points)
        weights = np.ones(nk) / nk
        
        return k_points, weights
    
    def _frac_to_cart_all(self):
        """Convert all k-points from fractional to Cartesian."""
        b = self.lattice.reciprocal_vectors
        
        k_cart = np.zeros_like(self.k_frac)
        for i, k in enumerate(self.k_frac):
            k_cart[i] = k[0] * b[0] + k[1] * b[1] + k[2] * b[2]
        
        return k_cart
    
    def get_k_frac(self, index):
        """Get k-point in fractional coordinates."""
        return self.k_frac[index].copy()
    
    def get_k_cart(self, index):
        """Get k-point in Cartesian coordinates."""
        return self.k_cart[index].copy()
    
    def get_weight(self, index):
        """Get integration weight for k-point."""
        return self.weights[index]
    
    def get_nk(self):
        """Return number of k-points."""
        return self.nk
    
    def display(self):
        """Print k-point information."""
        print(f"\nK-point mesh: {self.mesh[0]} x {self.mesh[1]} x {self.mesh[2]}")
        print(f"Shift: {self.shift}")
        print(f"Total k-points: {self.nk}")
        print()
        print(f"{'Index':>6} {'k1':>10} {'k2':>10} {'k3':>10} {'Weight':>10}")
        print("-" * 50)
        
        for i in range(min(self.nk, 20)):
            k = self.k_frac[i]
            w = self.weights[i]
            print(f"{i+1:6d} {k[0]:10.6f} {k[1]:10.6f} {k[2]:10.6f} {w:10.6f}")
        
        if self.nk > 20:
            print(f"... ({self.nk - 20} more k-points)")


class GammaPoint:
    """
    Single Gamma point (k = 0) for simple calculations.
    
    This is a simplified version for educational purposes where
    we only consider the Gamma point.
    """
    
    def __init__(self, lattice):
        """Initialize with just the Gamma point."""
        self.lattice = lattice
        self.k_frac = np.array([[0.0, 0.0, 0.0]])
        self.k_cart = np.array([[0.0, 0.0, 0.0]])
        self.weights = np.array([1.0])
        self.nk = 1
        self.mesh = (1, 1, 1)
    
    def get_k_frac(self, index=0):
        return np.array([0.0, 0.0, 0.0])
    
    def get_k_cart(self, index=0):
        return np.array([0.0, 0.0, 0.0])
    
    def get_weight(self, index=0):
        return 1.0
    
    def get_nk(self):
        return 1
    
    def display(self):
        print("\nK-point sampling: Gamma point only")
        print("k = (0, 0, 0), weight = 1.0")


def gamma_only(lattice):
    """Create Gamma-point-only k-point sampling."""
    return GammaPoint(lattice)


def monkhorst_pack(lattice, n1, n2, n3, shift=False):
    """
    Create Monkhorst-Pack k-point mesh.
    
    Args:
        lattice: Lattice object
        n1, n2, n3: Mesh density in each direction
        shift: If True, use (0.5, 0.5, 0.5) shift
    
    Returns:
        KPoints object
    """
    s = (0.5, 0.5, 0.5) if shift else (0, 0, 0)
    return KPoints(lattice, (n1, n2, n3), s)


def automatic_kpoints(lattice, kspacing=0.1):
    """
    Generate k-point mesh based on target k-spacing.
    
    Args:
        lattice: Lattice object
        kspacing: Maximum distance between k-points (Bohr^-1)
    
    Returns:
        KPoints object
    """
    b = lattice.reciprocal_vectors
    
    # Compute mesh density based on reciprocal lattice vector lengths
    n1 = max(1, int(np.linalg.norm(b[0]) / kspacing))
    n2 = max(1, int(np.linalg.norm(b[1]) / kspacing))
    n3 = max(1, int(np.linalg.norm(b[2]) / kspacing))
    
    return KPoints(lattice, (n1, n2, n3))


def high_symmetry_path(lattice, path_string, npoints_per_segment=20):
    """
    Generate k-points along high-symmetry path for band structure.
    
    This is a simplified version supporting common paths.
    
    Args:
        lattice: Lattice object
        path_string: String like "GXMGR" for high-symmetry points
        npoints_per_segment: Number of points between each pair
    
    Returns:
        k_frac: K-points in fractional coordinates
        k_dist: Cumulative distance along path (for plotting)
        labels: Labels for high-symmetry points
    
    Note: Only supports cubic lattices in this simplified version.
    """
    # High-symmetry points for cubic lattices
    special_points = {
        'G': np.array([0.0, 0.0, 0.0]),      # Gamma
        'X': np.array([0.5, 0.0, 0.0]),
        'M': np.array([0.5, 0.5, 0.0]),
        'R': np.array([0.5, 0.5, 0.5]),
        'L': np.array([0.5, 0.5, 0.5]),      # Same as R for simple cubic
    }
    
    # Parse path
    points = [special_points[c] for c in path_string if c in special_points]
    
    if len(points) < 2:
        raise ValueError(f"Path must have at least 2 points, got: {path_string}")
    
    # Generate k-points along path
    k_frac = []
    k_dist = []
    labels = []
    label_positions = []
    
    cumulative_dist = 0.0
    b = lattice.reciprocal_vectors
    
    for i in range(len(points) - 1):
        k1, k2 = points[i], points[i+1]
        
        # Record label position
        if i == 0:
            labels.append(path_string[0])
            label_positions.append(0.0)
        
        # Linear interpolation
        for j in range(npoints_per_segment):
            t = j / npoints_per_segment
            k = (1 - t) * k1 + t * k2
            k_frac.append(k)
            
            # Distance in Cartesian reciprocal space
            if len(k_frac) > 1:
                dk = k_frac[-1] - k_frac[-2]
                dk_cart = dk[0] * b[0] + dk[1] * b[1] + dk[2] * b[2]
                cumulative_dist += np.linalg.norm(dk_cart)
            
            k_dist.append(cumulative_dist)
        
        # Add label for end of segment
        labels.append(path_string[i+1])
        label_positions.append(cumulative_dist + np.linalg.norm(
            (k2 - k1)[0] * b[0] + (k2 - k1)[1] * b[1] + (k2 - k1)[2] * b[2]
        ) / npoints_per_segment * (npoints_per_segment - 1))
    
    # Add final point
    k_frac.append(points[-1])
    dk = k_frac[-1] - k_frac[-2]
    dk_cart = dk[0] * b[0] + dk[1] * b[1] + dk[2] * b[2]
    cumulative_dist += np.linalg.norm(dk_cart)
    k_dist.append(cumulative_dist)
    label_positions[-1] = cumulative_dist
    
    return np.array(k_frac), np.array(k_dist), labels, label_positions


def get_kpg_norms(k_cart, gvector):
    """
    Compute |k + G| for all G-vectors at a given k-point.
    
    Args:
        k_cart: K-point in Cartesian coordinates (1D array of length 3)
        gvector: GVector object
    
    Returns:
        kpg: Array of |k + G|^2 values, shape (npw,)
    """
    kpg_squared = np.zeros(gvector.npw)
    
    for i in range(gvector.npw):
        kpg = k_cart + gvector.gvecs[i]
        kpg_squared[i] = np.sum(kpg**2)
    
    return kpg_squared
