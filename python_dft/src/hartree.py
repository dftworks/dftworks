"""
Hartree potential computation.

The Hartree potential describes electron-electron Coulomb repulsion.
In reciprocal space: V_H(G) = 4*pi*rho(G) / |G|^2
"""

import numpy as np
from .constants import FOURPI


def compute_hartree_potential(rhog, g_norms):
    """
    Compute Hartree potential in G-space.
    
    V_H(G) = 4*pi * rho(G) / |G|^2  for G != 0
    V_H(0) = 0  (charge neutrality / jellium background)
    
    Args:
        rhog: Density in G-space (complex array, G=0 should be first)
        g_norms: |G| for each G-vector (G=0 should be first)
    
    Returns:
        vhg: Hartree potential in G-space
    """
    vhg = np.zeros_like(rhog, dtype=complex)
    
    # Skip G=0 (set to zero for charge neutrality)
    # This corresponds to adding a neutralizing background charge
    vhg[0] = 0.0
    
    # For G != 0: V_H(G) = 4*pi * rho(G) / G^2
    g2 = g_norms[1:]**2
    vhg[1:] = FOURPI * rhog[1:] / g2
    
    return vhg


def compute_hartree_energy(rhog, g_norms, volume):
    """
    Compute Hartree energy.
    
    E_H = (1/2) * integral(V_H * rho) dr
        = (Omega/2) * sum_{G!=0} 4*pi * |rho(G)|^2 / G^2
    
    Args:
        rhog: Density in G-space
        g_norms: |G| for each G-vector
        volume: Cell volume
    
    Returns:
        Hartree energy
    """
    # Skip G=0
    g2 = g_norms[1:]**2
    rhog_no_g0 = rhog[1:]
    
    # E_H = (Omega / 2) * sum 4*pi * |rho(G)|^2 / G^2
    e_hartree = 0.5 * volume * FOURPI * np.sum(np.abs(rhog_no_g0)**2 / g2)
    
    return e_hartree


def compute_hartree_stress(rhog, g_cart, g_norms, volume):
    """
    Compute Hartree contribution to stress tensor.
    
    sigma_ij = -(1/Omega) * sum_{G!=0} 4*pi * |rho(G)|^2 / G^4 * 
               (delta_ij * G^2 - 2 * G_i * G_j)
    
    Args:
        rhog: Density in G-space
        g_cart: Cartesian G-vectors
        g_norms: |G| for each G-vector
        volume: Cell volume
    
    Returns:
        3x3 stress tensor
    """
    stress = np.zeros((3, 3))
    
    # Skip G=0
    for ig in range(1, len(g_norms)):
        g = g_cart[ig]
        g2 = g_norms[ig]**2
        rho_g2 = np.abs(rhog[ig])**2
        
        prefactor = FOURPI * rho_g2 / (g2 * g2)
        
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                stress[i, j] += prefactor * (delta_ij * g2 - 2.0 * g[i] * g[j])
    
    stress *= -1.0 / volume
    
    return stress
