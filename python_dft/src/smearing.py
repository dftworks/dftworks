"""
Electronic Smearing Functions for Occupation Numbers.

Implements various smearing schemes for computing fractional occupation
numbers in metallic systems, which is essential for smooth convergence
of the SCF loop.

Reference:
    - Fermi-Dirac: Standard statistical mechanics
    - Gaussian: Fu, C.-L. & Ho, K.-M. Phys. Rev. B 28, 5480 (1983)
    - Methfessel-Paxton: Methfessel, M. & Paxton, A. T. Phys. Rev. B 40, 3616 (1989)
"""

import numpy as np
from scipy.special import erfc
from .constants import BOLTZMANN_CONSTANT, HA_TO_EV


class FermiDirac:
    """
    Fermi-Dirac smearing.
    
    f(E) = 1 / (exp((E - mu) / kT) + 1)
    
    This is the physically correct distribution for electrons at
    finite temperature, but has slow convergence with respect to
    the number of k-points.
    """
    
    def __init__(self, temperature=300.0):
        """
        Initialize Fermi-Dirac smearing.
        
        Args:
            temperature: Electronic temperature in Kelvin
        """
        self.temperature = temperature
        self.kbt = max(BOLTZMANN_CONSTANT * temperature, 1e-10)
    
    def occupation(self, energy, fermi_level):
        """
        Compute occupation number.
        
        Args:
            energy: Eigenvalue (Hartree)
            fermi_level: Fermi level (Hartree)
        
        Returns:
            Occupation number between 0 and 1
        """
        x = (energy - fermi_level) / self.kbt
        
        # Avoid overflow
        if x > 100:
            return 0.0
        elif x < -100:
            return 1.0
        else:
            return 1.0 / (np.exp(x) + 1.0)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.kbt
        x = np.clip(x, -100, 100)
        return 1.0 / (np.exp(x) + 1.0)
    
    def entropy(self, occupations):
        """
        Compute electronic entropy.
        
        S = -kT * sum_i [f*ln(f) + (1-f)*ln(1-f)]
        
        Returns:
            Entropy contribution to free energy (Hartree)
        """
        # Avoid log(0)
        f = np.clip(occupations, 1e-20, 1 - 1e-20)
        s = -self.kbt * np.sum(f * np.log(f) + (1 - f) * np.log(1 - f))
        return s
    
    @property
    def name(self):
        return "Fermi-Dirac"


class Gaussian:
    """
    Gaussian smearing (cold smearing).
    
    f(E) = 0.5 * erfc((E - mu) / sigma)
    
    This smearing has better convergence properties than Fermi-Dirac
    but the entropy term is not physical.
    """
    
    def __init__(self, sigma=0.01):
        """
        Initialize Gaussian smearing.
        
        Args:
            sigma: Smearing width in Hartree (typical: 0.01-0.1 Ha)
        """
        self.sigma = sigma
    
    def occupation(self, energy, fermi_level):
        """Compute occupation number."""
        x = (energy - fermi_level) / self.sigma
        return 0.5 * erfc(x)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.sigma
        return 0.5 * erfc(x)
    
    def entropy(self, occupations):
        """
        Compute entropy correction for Gaussian smearing.
        
        Note: This is not a physical entropy, but a correction term.
        """
        # Simplified approximation
        return 0.0
    
    @property
    def name(self):
        return "Gaussian"


class MethfesselPaxton:
    """
    Methfessel-Paxton smearing (order 1).
    
    Provides better integration accuracy than Gaussian smearing
    while still having fast k-point convergence.
    
    The first-order MP smearing function is:
    f(x) = 0.5*erfc(x) - exp(-x^2) * x / sqrt(pi)
    """
    
    def __init__(self, sigma=0.01, order=1):
        """
        Initialize Methfessel-Paxton smearing.
        
        Args:
            sigma: Smearing width in Hartree
            order: Order of the method (1 or 2)
        """
        self.sigma = sigma
        self.order = order
    
    def occupation(self, energy, fermi_level):
        """Compute occupation number."""
        x = (energy - fermi_level) / self.sigma
        return self._mp_function(x)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.sigma
        return self._mp_function(x)
    
    def _mp_function(self, x):
        """Methfessel-Paxton occupation function."""
        # 0th order = Gaussian
        f = 0.5 * erfc(x)
        
        if self.order >= 1:
            # 1st order correction
            A1 = -1.0 / np.sqrt(np.pi)
            f += A1 * x * np.exp(-x**2)
        
        if self.order >= 2:
            # 2nd order correction
            A2 = -0.5 / np.sqrt(np.pi)
            H2 = 4.0 * x**2 - 2.0  # Hermite polynomial
            f += A2 * H2 * np.exp(-x**2)
        
        return np.clip(f, 0.0, 1.0)
    
    def entropy(self, occupations):
        """Entropy correction term."""
        return 0.0
    
    @property
    def name(self):
        return f"Methfessel-Paxton (order {self.order})"


class FixedOccupation:
    """
    Fixed occupation numbers (no smearing).
    
    For insulators and semiconductors with a gap, fixed occupation
    is often sufficient and provides exact results.
    """
    
    def __init__(self):
        pass
    
    def occupation(self, energy, fermi_level):
        """Occupation is 1 below Fermi level, 0 above."""
        return 1.0 if energy < fermi_level else 0.0
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        return np.where(energies < fermi_level, 1.0, 0.0)
    
    def entropy(self, occupations):
        """No entropy for fixed occupations."""
        return 0.0
    
    @property
    def name(self):
        return "Fixed"


def create_smearing(scheme, **kwargs):
    """
    Factory function to create smearing object.
    
    Args:
        scheme: 'fd', 'gaussian', 'mp', 'mp1', 'mp2', or 'fixed'
        **kwargs: Additional parameters for smearing (temperature, sigma, etc.)
    
    Returns:
        Smearing object
    """
    scheme = scheme.lower()
    
    if scheme == 'fd' or scheme == 'fermi-dirac':
        temperature = kwargs.get('temperature', 300.0)
        return FermiDirac(temperature=temperature)
    
    elif scheme in ('gs', 'gaussian'):
        sigma = kwargs.get('sigma', 0.01)
        return Gaussian(sigma=sigma)
    
    elif scheme in ('mp', 'mp1'):
        sigma = kwargs.get('sigma', 0.01)
        return MethfesselPaxton(sigma=sigma, order=1)
    
    elif scheme == 'mp2':
        sigma = kwargs.get('sigma', 0.01)
        return MethfesselPaxton(sigma=sigma, order=2)
    
    elif scheme == 'fixed':
        return FixedOccupation()
    
    else:
        print(f"Warning: Unknown smearing scheme '{scheme}', using Fermi-Dirac")
        return FermiDirac()


def find_fermi_level(eigenvalues, weights, n_electrons, smearing, 
                     spin_factor=2.0, tol=1e-10, max_iter=100):
    """
    Find Fermi level for given eigenvalues and number of electrons.
    
    Uses bisection method to find the Fermi level that gives the
    correct number of electrons.
    
    Args:
        eigenvalues: Array of shape (nk, n_bands) or (n_bands,)
        weights: K-point weights, shape (nk,) or scalar 1.0
        n_electrons: Target number of electrons
        smearing: Smearing object
        spin_factor: 2.0 for spin-paired (default), 1.0 for spin-polarized
        tol: Tolerance for electron count
        max_iter: Maximum iterations
    
    Returns:
        fermi_level: Fermi level in Hartree
        occupations: Occupation numbers for each eigenvalue
    """
    # Flatten eigenvalues if needed
    if eigenvalues.ndim == 1:
        eigs = eigenvalues
        wts = np.ones(len(eigenvalues))
    else:
        nk, nbands = eigenvalues.shape
        eigs = eigenvalues.flatten()
        if np.isscalar(weights):
            wts = np.ones(nk * nbands) / nk
        else:
            wts = np.repeat(weights, nbands)
    
    # Initial bounds
    e_min = np.min(eigs) - 1.0
    e_max = np.max(eigs) + 1.0
    
    def count_electrons(mu):
        occ = smearing.occupation_array(eigs, mu)
        return spin_factor * np.sum(wts * occ)
    
    # Bisection search
    for _ in range(max_iter):
        mu = 0.5 * (e_min + e_max)
        n_e = count_electrons(mu)
        
        if abs(n_e - n_electrons) < tol:
            break
        
        if n_e < n_electrons:
            e_min = mu
        else:
            e_max = mu
    
    fermi_level = mu
    occupations = smearing.occupation_array(eigs, fermi_level)
    
    # Reshape occupations if input was 2D
    if eigenvalues.ndim > 1:
        occupations = occupations.reshape(eigenvalues.shape)
    
    return fermi_level, occupations * spin_factor


def compute_band_energy(eigenvalues, occupations, weights=1.0):
    """
    Compute band structure energy.
    
    E_band = sum_{n,k} w_k * f_{n,k} * eps_{n,k}
    
    Args:
        eigenvalues: Eigenvalues (nk, nbands) or (nbands,)
        occupations: Occupation numbers (same shape)
        weights: K-point weights
    
    Returns:
        Band energy in Hartree
    """
    if eigenvalues.ndim == 1:
        return np.sum(occupations * eigenvalues)
    else:
        if np.isscalar(weights):
            weights = np.ones(eigenvalues.shape[0]) * weights
        return np.sum(weights[:, np.newaxis] * occupations * eigenvalues)
