"""
Hamiltonian application: H|psi>.

The Kohn-Sham Hamiltonian consists of:
- Kinetic energy: T = -1/2 * nabla^2
- Local potential: V_loc = V_H + V_xc + V_ps_loc
- Non-local potential: V_nl (not implemented in this simplified version)
"""

import numpy as np


class Hamiltonian:
    """
    Kohn-Sham Hamiltonian in plane wave basis.
    
    Applies H|psi> = (T + V_loc)|psi> using FFT for the local potential.
    
    Attributes:
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume
    """
    
    def __init__(self, gvec, volume):
        """
        Initialize Hamiltonian.
        
        Args:
            gvec: GVector object
            volume: Cell volume
        """
        self.gvec = gvec
        self.volume = volume
        self.fft_shape = gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        # Kinetic energies (for Gamma point)
        self.kg = 0.5 * np.sum(gvec.cart**2, axis=1)
        
        # Local potential in real space (set later)
        self.vloc_r = None
        
        # Workspace arrays
        self._work_g = np.zeros(self.fft_shape, dtype=complex)
        self._work_r = np.zeros(self.fft_shape, dtype=complex)
    
    def set_local_potential(self, vloc_r):
        """
        Set the local potential in real space.
        
        Args:
            vloc_r: Local potential V_loc(r) on FFT grid
        """
        self.vloc_r = vloc_r
    
    def apply(self, psi_g):
        """
        Apply Hamiltonian to wavefunction: H|psi>.
        
        H = T + V_loc
        
        Args:
            psi_g: Wavefunction in G-space (npw,) complex array
        
        Returns:
            h_psi_g: H|psi> in G-space
        """
        h_psi_g = np.zeros_like(psi_g, dtype=complex)
        
        # Kinetic: T|psi> = (|G|^2 / 2) * psi(G)
        h_psi_g += self.kg * psi_g
        
        # Local potential: V_loc|psi> via FFT
        if self.vloc_r is not None:
            h_psi_g += self._apply_vloc(psi_g)
        
        return h_psi_g
    
    def _apply_vloc(self, psi_g):
        """
        Apply local potential via FFT.
        
        1. Map psi(G) to FFT grid
        2. IFFT to get psi(r)
        3. Multiply: V_loc(r) * psi(r)
        4. FFT to get (V_loc * psi)(G)
        5. Map back to G-vector list
        """
        # Map to FFT grid
        self._work_g.fill(0)
        self._map_to_fft(psi_g, self._work_g)
        
        # IFFT: G -> r (with proper normalization)
        self._work_r = np.fft.ifftn(self._work_g) * self.n_fft
        
        # Multiply in real space: V_loc(r) * psi(r)
        # Normalize by sqrt(volume) for proper normalization
        self._work_r *= self.vloc_r / np.sqrt(self.volume)
        
        # FFT: r -> G
        self._work_g = np.fft.fftn(self._work_r) / self.n_fft
        
        # Map back to G-vector list
        vloc_psi_g = self._map_from_fft(self._work_g)
        
        return vloc_psi_g * np.sqrt(self.volume)
    
    def _map_to_fft(self, data_g, grid):
        """Map 1D G-space data to 3D FFT grid."""
        n1, n2, n3 = self.fft_shape
        for ig, (m1, m2, m3) in enumerate(self.gvec.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            grid[i1, i2, i3] = data_g[ig]
    
    def _map_from_fft(self, grid):
        """Extract 1D G-space data from 3D FFT grid."""
        n1, n2, n3 = self.fft_shape
        data_g = np.zeros(self.gvec.npw, dtype=complex)
        for ig, (m1, m2, m3) in enumerate(self.gvec.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            data_g[ig] = grid[i1, i2, i3]
        return data_g
    
    def get_diagonal(self):
        """
        Get diagonal of Hamiltonian (kinetic energy).
        
        Used for preconditioning in iterative eigensolvers.
        
        Returns:
            (npw,) array of diagonal elements
        """
        return self.kg.copy()
    
    def compute_expectation(self, psi_g):
        """
        Compute <psi|H|psi>.
        
        Args:
            psi_g: Normalized wavefunction in G-space
        
        Returns:
            Expectation value (energy)
        """
        h_psi = self.apply(psi_g)
        return np.real(np.vdot(psi_g, h_psi))


def g_to_r(psi_g, gvec, fft_shape, volume):
    """
    Transform wavefunction from G-space to real space.
    
    psi(r) = (1/sqrt(V)) * sum_G psi(G) * exp(i*G*r)
    
    Args:
        psi_g: Wavefunction in G-space
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume
    
    Returns:
        psi_r: Wavefunction on real-space grid
    """
    n1, n2, n3 = fft_shape
    n_fft = n1 * n2 * n3
    
    # Map to FFT grid
    work = np.zeros(fft_shape, dtype=complex)
    for ig, (m1, m2, m3) in enumerate(gvec.miller):
        i1 = m1 % n1
        i2 = m2 % n2
        i3 = m3 % n3
        work[i1, i2, i3] = psi_g[ig]
    
    # IFFT with normalization
    psi_r = np.fft.ifftn(work) * n_fft / np.sqrt(volume)
    
    return psi_r


def r_to_g(psi_r, gvec, fft_shape, volume):
    """
    Transform wavefunction from real space to G-space.
    
    psi(G) = (1/sqrt(V)) * integral psi(r) * exp(-i*G*r) dr
    
    Args:
        psi_r: Wavefunction on real-space grid
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume
    
    Returns:
        psi_g: Wavefunction in G-space
    """
    n1, n2, n3 = fft_shape
    n_fft = n1 * n2 * n3
    
    # FFT
    work = np.fft.fftn(psi_r * np.sqrt(volume)) / n_fft
    
    # Map to G-vector list
    psi_g = np.zeros(gvec.npw, dtype=complex)
    for ig, (m1, m2, m3) in enumerate(gvec.miller):
        i1 = m1 % n1
        i2 = m2 % n2
        i3 = m3 % n3
        psi_g[ig] = work[i1, i2, i3]
    
    return psi_g
