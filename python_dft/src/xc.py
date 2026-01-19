"""
Exchange-correlation functionals.

Implements LDA (Local Density Approximation) with:
- Slater exchange
- Perdew-Zunger correlation
"""

import numpy as np
from .constants import PI


def lda_xc(rho):
    """
    Compute LDA exchange-correlation potential and energy density.
    
    Uses Slater exchange and Perdew-Zunger (1981) correlation.
    
    Args:
        rho: Electron density array (real, positive)
    
    Returns:
        vxc: Exchange-correlation potential (same shape as rho)
        exc: Exchange-correlation energy density (same shape as rho)
    
    Reference:
        Perdew & Zunger, Phys. Rev. B 23, 5048 (1981)
    """
    rho = np.asarray(rho, dtype=float)
    
    # Avoid division by zero for very small densities
    rho_safe = np.maximum(np.abs(rho), 1e-30)
    
    # Exchange
    vx, ex = slater_exchange(rho_safe)
    
    # Correlation
    vc, ec = pz_correlation(rho_safe)
    
    return vx + vc, ex + ec


def slater_exchange(rho):
    """
    Slater (Dirac) exchange functional.
    
    epsilon_x = -3/4 * (3*rho/pi)^(1/3)
    V_x = -4/3 * epsilon_x / rho = -(3*rho/pi)^(1/3)
    
    Args:
        rho: Electron density (positive)
    
    Returns:
        vx: Exchange potential
        ex: Exchange energy density
    """
    # Exchange coefficient
    cx = -(3.0 / PI) ** (1.0 / 3.0)
    
    # Exchange potential: vx = cx * rho^(1/3)
    rho_third = rho ** (1.0 / 3.0)
    vx = cx * rho_third
    
    # Exchange energy density: ex = 3/4 * vx
    ex = 0.75 * vx
    
    return vx, ex


def pz_correlation(rho):
    """
    Perdew-Zunger parameterization of the correlation energy.
    
    Different formulas for high density (rs < 1) and low density (rs >= 1),
    where rs = (3/(4*pi*rho))^(1/3) is the Wigner-Seitz radius.
    
    Args:
        rho: Electron density (positive)
    
    Returns:
        vc: Correlation potential
        ec: Correlation energy density
    """
    # Wigner-Seitz radius
    rs = (3.0 / (4.0 * PI * rho)) ** (1.0 / 3.0)
    
    # Initialize outputs
    vc = np.zeros_like(rho)
    ec = np.zeros_like(rho)
    
    # Low density regime: rs >= 1
    mask_low = rs >= 1.0
    if np.any(mask_low):
        vc[mask_low], ec[mask_low] = _pz_low_density(rs[mask_low])
    
    # High density regime: rs < 1
    mask_high = ~mask_low
    if np.any(mask_high):
        vc[mask_high], ec[mask_high] = _pz_high_density(rs[mask_high])
    
    return vc, ec


def _pz_low_density(rs):
    """
    PZ correlation for rs >= 1 (low density).
    
    ec = gamma / (1 + beta1*sqrt(rs) + beta2*rs)
    """
    # Parameters for unpolarized electron gas
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    
    rs_sqrt = np.sqrt(rs)
    denom = 1.0 + beta1 * rs_sqrt + beta2 * rs
    
    # Correlation energy density
    ec = gamma / denom
    
    # Correlation potential: vc = d(rho*ec)/d(rho)
    # Using chain rule with rs
    numer = 1.0 + (7.0 / 6.0) * beta1 * rs_sqrt + (4.0 / 3.0) * beta2 * rs
    vc = ec * numer / denom
    
    return vc, ec


def _pz_high_density(rs):
    """
    PZ correlation for rs < 1 (high density).
    
    ec = A*ln(rs) + B + C*rs*ln(rs) + D*rs
    """
    # Parameters
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    
    rs_ln = np.log(rs)
    
    # Correlation energy density
    ec = A * rs_ln + B + C * rs * rs_ln + D * rs
    
    # Correlation potential
    vc = (A * rs_ln + (B - A / 3.0) + 
          (2.0 / 3.0) * C * rs * rs_ln + 
          (1.0 / 3.0) * (2.0 * D - C) * rs)
    
    return vc, ec


def compute_xc_energy(rho, exc, volume, n_grid):
    """
    Compute total XC energy from energy density.
    
    E_xc = integral(rho * epsilon_xc) dr
         = (volume / n_grid) * sum(rho * epsilon_xc)
    
    Args:
        rho: Electron density in real space
        exc: XC energy density
        volume: Cell volume
        n_grid: Total number of grid points
    
    Returns:
        Total XC energy
    """
    return (volume / n_grid) * np.sum(np.real(rho * exc))


def compute_xc_potential_energy(rho, vxc, volume, n_grid):
    """
    Compute XC potential energy (for double-counting correction).
    
    E_vxc = integral(rho * V_xc) dr
    
    Args:
        rho: Electron density in real space
        vxc: XC potential
        volume: Cell volume
        n_grid: Total number of grid points
    
    Returns:
        XC potential energy
    """
    return (volume / n_grid) * np.sum(np.real(rho * vxc))
