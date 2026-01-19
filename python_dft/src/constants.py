"""
Physical and numerical constants for DFT calculations.

All values are in Hartree atomic units unless otherwise specified:
- Length: Bohr
- Energy: Hartree
- Mass: electron mass
- Charge: electron charge
- hbar = m_e = e = 4*pi*epsilon_0 = 1
"""

import numpy as np

# =============================================================================
# Unit Conversions
# =============================================================================

# Length
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# Volume
BOHR3_TO_ANG3 = BOHR_TO_ANG ** 3
ANG3_TO_BOHR3 = ANG_TO_BOHR ** 3

# Energy
RY_TO_EV = 13.605698066
HA_TO_EV = 2.0 * RY_TO_EV  # 27.211396132
HA_TO_RY = 2.0
EV_TO_HA = 1.0 / HA_TO_EV
RY_TO_HA = 0.5

# Force
FORCE_HA_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG  # eV/Angstrom per Hartree/Bohr

# Stress
STRESS_HA_TO_GPA = 29421.02648438959
STRESS_HA_TO_KBAR = 294210.2648438959

# =============================================================================
# Physical Constants
# =============================================================================

# Boltzmann constant in Hartree/Kelvin
BOLTZMANN_CONSTANT = 8.617333262145e-5 * EV_TO_HA

# =============================================================================
# Mathematical Constants
# =============================================================================

PI = np.pi
TWOPI = 2.0 * np.pi
FOURPI = 4.0 * np.pi

# Complex unit
I_COMPLEX = 1.0j

# =============================================================================
# Numerical Tolerances
# =============================================================================

EPS3 = 1e-3
EPS5 = 1e-5
EPS6 = 1e-6
EPS8 = 1e-8
EPS10 = 1e-10
EPS12 = 1e-12
EPS20 = 1e-20

# Default convergence thresholds
DEFAULT_ENERGY_TOL = 1e-6  # Hartree
DEFAULT_DENSITY_TOL = 1e-6


def hartree_to_ev(energy_ha):
    """Convert energy from Hartree to electron volts."""
    return energy_ha * HA_TO_EV


def ev_to_hartree(energy_ev):
    """Convert energy from electron volts to Hartree."""
    return energy_ev * EV_TO_HA


def bohr_to_angstrom(length_bohr):
    """Convert length from Bohr to Angstrom."""
    return length_bohr * BOHR_TO_ANG


def angstrom_to_bohr(length_ang):
    """Convert length from Angstrom to Bohr."""
    return length_ang * ANG_TO_BOHR
