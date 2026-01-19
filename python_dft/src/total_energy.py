"""
Total Energy Computation for Plane-Wave DFT.

Computes all contributions to the total energy:
- Band (Kohn-Sham eigenvalue) energy
- Hartree energy
- Exchange-correlation energy
- Ion-ion (Ewald) energy
- Kinetic energy
- Local potential energy
- Non-local potential energy

The total energy can be computed two ways:

1. Harris-Foulkes functional:
   E_tot = E_band - E_H + E_xc - E_vxc + E_ion-ion

2. Direct sum:
   E_tot = E_kin + E_loc + E_nl + E_H + E_xc + E_ion-ion
"""

import numpy as np
from .constants import HA_TO_EV


class TotalEnergy:
    """
    Stores and computes total energy and its components.
    
    Attributes:
        e_band: Band structure energy (sum of eigenvalues)
        e_hartree: Hartree electron-electron repulsion
        e_xc: Exchange-correlation energy
        e_vxc: XC potential energy (double-counting correction)
        e_kinetic: Kinetic energy
        e_local: Local potential energy
        e_nonlocal: Non-local potential energy
        e_ewald: Ion-ion Coulomb energy
        e_entropy: Entropy contribution (for smearing)
        e_total: Total energy
    """
    
    def __init__(self):
        """Initialize energy storage."""
        self.e_band = 0.0
        self.e_hartree = 0.0
        self.e_xc = 0.0
        self.e_vxc = 0.0
        self.e_kinetic = 0.0
        self.e_local = 0.0
        self.e_nonlocal = 0.0
        self.e_ewald = 0.0
        self.e_entropy = 0.0
        self.e_total = 0.0
    
    def compute_total(self, method='harris'):
        """
        Compute total energy from components.
        
        Args:
            method: 'harris' or 'direct'
        
        Returns:
            Total energy in Hartree
        """
        if method == 'harris':
            # Harris-Foulkes: E = E_band - E_H + E_xc - E_vxc + E_ewald
            self.e_total = (self.e_band - self.e_hartree + 
                           self.e_xc - self.e_vxc + self.e_ewald)
        else:
            # Direct sum
            self.e_total = (self.e_kinetic + self.e_local + self.e_nonlocal +
                           self.e_hartree + self.e_xc + self.e_ewald)
        
        # Add entropy contribution (for smearing)
        self.e_total += self.e_entropy
        
        return self.e_total
    
    def display(self, verbose=True):
        """Print energy breakdown."""
        if verbose:
            print("\n" + "=" * 50)
            print("Energy Components (Hartree)")
            print("=" * 50)
            print(f"  Band energy:      {self.e_band:16.8f}")
            print(f"  Hartree energy:   {self.e_hartree:16.8f}")
            print(f"  XC energy:        {self.e_xc:16.8f}")
            print(f"  XC potential:     {self.e_vxc:16.8f}")
            print(f"  Ewald energy:     {self.e_ewald:16.8f}")
            if abs(self.e_entropy) > 1e-10:
                print(f"  Entropy:          {self.e_entropy:16.8f}")
            print("-" * 50)
            print(f"  TOTAL ENERGY:     {self.e_total:16.8f} Ha")
            print(f"                    {self.e_total * HA_TO_EV:16.8f} eV")
            print("=" * 50)
    
    def __str__(self):
        return f"TotalEnergy(E_tot={self.e_total:.8f} Ha)"


def compute_band_energy(eigenvalues, occupations, k_weights=1.0):
    """
    Compute band structure energy.
    
    E_band = sum_{n,k} w_k * f_{n,k} * eps_{n,k}
    
    Args:
        eigenvalues: Eigenvalues, shape (nbands,) or (nk, nbands)
        occupations: Occupation numbers, same shape
        k_weights: K-point weights
    
    Returns:
        Band energy in Hartree
    """
    if eigenvalues.ndim == 1:
        return np.sum(occupations * eigenvalues)
    else:
        if np.isscalar(k_weights):
            k_weights = np.ones(eigenvalues.shape[0]) * k_weights
        return np.sum(k_weights[:, np.newaxis] * occupations * eigenvalues)


def compute_kinetic_energy(evecs, occupations, kg, k_weights=1.0):
    """
    Compute kinetic energy.
    
    E_kin = (1/2) * sum_{n,k} w_k * f_{n,k} * sum_G |c_{n,k}(G)|^2 * |k+G|^2
    
    Args:
        evecs: Wavefunctions in G-space
        occupations: Occupation numbers
        kg: |k+G|^2 values for each plane wave
        k_weights: K-point weights
    
    Returns:
        Kinetic energy in Hartree
    """
    e_kin = 0.0
    
    if evecs.ndim == 2:
        # Single k-point
        nbands = evecs.shape[1]
        for iband in range(nbands):
            if occupations[iband] < 1e-10:
                continue
            e_kin += 0.5 * occupations[iband] * np.sum(
                np.abs(evecs[:, iband])**2 * kg
            )
    else:
        # Multiple k-points
        nk = evecs.shape[0]
        nbands = evecs.shape[2]
        
        if np.isscalar(k_weights):
            k_weights = np.ones(nk) * k_weights
        
        for ik in range(nk):
            for iband in range(nbands):
                occ = occupations[ik, iband] if occupations.ndim > 1 else occupations[iband]
                if occ < 1e-10:
                    continue
                e_kin += 0.5 * k_weights[ik] * occ * np.sum(
                    np.abs(evecs[ik, :, iband])**2 * kg[ik]
                )
    
    return e_kin


def compute_hartree_energy(rho_g, g_norms, volume):
    """
    Compute Hartree energy.
    
    E_H = (V/2) * sum_{G!=0} 4*pi * |rho(G)|^2 / G^2
    
    Args:
        rho_g: Density in G-space
        g_norms: |G| values
        volume: Cell volume
    
    Returns:
        Hartree energy in Hartree
    """
    # Skip G=0
    mask = g_norms > 1e-10
    g2 = g_norms[mask]**2
    rho2 = np.abs(rho_g[mask])**2
    
    e_hartree = 0.5 * volume * 4.0 * np.pi * np.sum(rho2 / g2)
    
    return e_hartree


def compute_xc_energy(rho_r, exc_r, volume, n_fft):
    """
    Compute exchange-correlation energy.
    
    E_xc = int rho(r) * eps_xc(rho(r)) dr
    
    Args:
        rho_r: Density in real space
        exc_r: XC energy density in real space
        volume: Cell volume
        n_fft: Number of FFT grid points
    
    Returns:
        XC energy in Hartree
    """
    dr = volume / n_fft
    return np.sum(rho_r * exc_r) * dr


def compute_vxc_energy(rho_r, vxc_r, volume, n_fft):
    """
    Compute XC potential energy (for double-counting correction).
    
    E_vxc = int rho(r) * V_xc(r) dr
    
    Args:
        rho_r: Density in real space
        vxc_r: XC potential in real space
        volume: Cell volume
        n_fft: Number of FFT grid points
    
    Returns:
        XC potential energy in Hartree
    """
    dr = volume / n_fft
    return np.sum(rho_r * vxc_r) * dr


def compute_local_potential_energy(rho_g, vloc_g, volume):
    """
    Compute local potential energy.
    
    E_loc = V * sum_G rho*(G) * V_loc(G)
    
    Args:
        rho_g: Density in G-space
        vloc_g: Local potential in G-space
        volume: Cell volume
    
    Returns:
        Local potential energy in Hartree
    """
    return volume * np.real(np.sum(np.conj(rho_g) * vloc_g))


def energy_convergence_check(energy_new, energy_old, tol=1e-6):
    """
    Check if energy has converged.
    
    Args:
        energy_new: New total energy
        energy_old: Previous total energy
        tol: Convergence tolerance
    
    Returns:
        converged: Boolean
        de: Energy change
    """
    de = abs(energy_new - energy_old)
    converged = de < tol
    return converged, de


def print_scf_header():
    """Print SCF iteration header."""
    print("\n" + "-" * 70)
    print(f"{'Iter':>5} {'E_total (Ha)':>18} {'dE (Ha)':>14} {'dE (eV)':>14}")
    print("-" * 70)


def print_scf_iteration(iteration, energy, de):
    """Print single SCF iteration line."""
    print(f"{iteration:5d} {energy:18.10f} {de:14.2e} {de * HA_TO_EV:14.2e}")


def print_scf_converged(iterations, energy):
    """Print convergence message."""
    print("-" * 70)
    print(f"SCF converged in {iterations} iterations")
    print(f"Final energy: {energy:.10f} Ha ({energy * HA_TO_EV:.6f} eV)")
    print("-" * 70)
