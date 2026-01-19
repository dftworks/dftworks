"""
Ewald Summation for Ion-Ion Coulomb Energy.

Computes the electrostatic energy of point charges in a periodic system
by splitting the slowly converging 1/r sum into real-space and 
reciprocal-space parts.

Reference:
    Ewald, P. P. "Die Berechnung optischer und elektrostatischer 
    Gitterpotentiale" Ann. Phys. 369, 253 (1921)
"""

import numpy as np
from scipy.special import erfc
from .constants import PI, TWOPI, FOURPI


class Ewald:
    """
    Ewald summation for periodic systems.
    
    Computes:
    - Ion-ion Coulomb energy
    - Forces on ions
    - Stress tensor contribution
    
    The Ewald method splits the Coulomb sum into three parts:
    
    E_ewald = E_real + E_recip + E_self + E_G0
    
    Attributes:
        energy: Total Ewald energy (Hartree)
        forces: Forces on each ion (Hartree/Bohr)
        stress: Stress tensor (3x3, Hartree/Bohr^3)
    """
    
    def __init__(self, lattice, atom_positions, charges, gvector=None, eta=None):
        """
        Compute Ewald summation.
        
        Args:
            lattice: Lattice object with vectors and volume
            atom_positions: Fractional coordinates of atoms, shape (natoms, 3)
            charges: Ion charges (positive), shape (natoms,)
            gvector: Optional GVector object for reciprocal space part
            eta: Ewald parameter (computed automatically if None)
        """
        self.lattice = lattice
        self.positions = np.array(atom_positions)
        self.charges = np.array(charges)
        self.natoms = len(charges)
        self.volume = lattice.volume
        
        # Lattice vectors
        self.a = lattice.vectors[0]
        self.b = lattice.vectors[1]
        self.c = lattice.vectors[2]
        
        # Compute optimal eta if not provided
        if eta is None:
            self.eta = self._compute_optimal_eta()
        else:
            self.eta = eta
        
        # Get real-space cutoff
        self.rmax = self._compute_rmax()
        
        # Generate neighbor cells for real-space sum
        self.nn_cells = self._make_near_cells()
        
        # Compute energy
        e_real = self._compute_energy_real_space()
        e_recip = self._compute_energy_reciprocal_space(gvector)
        e_self, e_g0 = self._compute_energy_self_and_g0()
        
        self.energy = e_real + e_recip + e_self + e_g0
        
        # Compute forces
        f_real = self._compute_force_real_space()
        f_recip = self._compute_force_reciprocal_space(gvector)
        
        self.forces = f_real + f_recip
        
        # Compute stress
        self.stress = self._compute_stress(gvector)
    
    def _compute_optimal_eta(self):
        """
        Compute optimal Ewald parameter.
        
        Balances real and reciprocal space convergence.
        eta ~ (natoms / volume)^(2/3)
        """
        rho = self.natoms / self.volume
        eta = (rho ** (2.0/3.0)) * PI
        # Ensure reasonable value
        return max(eta, 0.1)
    
    def _compute_rmax(self, eps=1e-12):
        """Compute real-space cutoff where erfc(sqrt(eta)*r) < eps."""
        rmax = 1.0
        sqrt_eta = np.sqrt(self.eta)
        while erfc(sqrt_eta * rmax) > eps * rmax:
            rmax += 0.5
            if rmax > 100.0:  # Safety limit
                break
        return rmax
    
    def _make_near_cells(self):
        """Generate list of nearby cells for real-space sum."""
        na = int(self.rmax / np.linalg.norm(self.a)) + 2
        nb = int(self.rmax / np.linalg.norm(self.b)) + 2
        nc = int(self.rmax / np.linalg.norm(self.c)) + 2
        
        cells = []
        for ia in range(-na, na + 1):
            for ib in range(-nb, nb + 1):
                for ic in range(-nc, nc + 1):
                    R = ia * self.a + ib * self.b + ic * self.c
                    if np.linalg.norm(R) < self.rmax:
                        cells.append(np.array([ia, ib, ic]))
        
        # Sort by distance
        cells.sort(key=lambda c: np.linalg.norm(
            c[0]*self.a + c[1]*self.b + c[2]*self.c))
        
        return cells
    
    def _frac_to_cart(self, frac):
        """Convert fractional to Cartesian coordinates."""
        return frac[0] * self.a + frac[1] * self.b + frac[2] * self.c
    
    def _compute_energy_real_space(self):
        """
        Compute real-space part of Ewald energy.
        
        E_real = (1/2) * sum_{R} sum_{i!=j} Z_i * Z_j * erfc(sqrt(eta)*r) / r
        """
        sqrt_eta = np.sqrt(self.eta)
        energy = 0.0
        
        for cell in self.nn_cells:
            for i in range(self.natoms):
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    
                    # r_i - r_j - R in fractional
                    dfrac = self.positions[i] - self.positions[j] - cell
                    
                    # Convert to Cartesian
                    dr = self._frac_to_cart(dfrac)
                    r = np.linalg.norm(dr)
                    
                    if r > 1e-10:
                        energy += 0.5 * self.charges[i] * self.charges[j] * \
                                  erfc(sqrt_eta * r) / r
        
        return energy
    
    def _compute_energy_reciprocal_space(self, gvector):
        """
        Compute reciprocal-space part of Ewald energy.
        
        E_recip = (2*pi/V) * sum_{G!=0} exp(-G^2/4eta) / G^2 * |S(G)|^2
        
        where S(G) = sum_i Z_i * exp(i*G*r_i) is the structure factor.
        """
        if gvector is None:
            # Use simple G-vector generation
            return self._compute_energy_recip_simple()
        
        energy = 0.0
        
        # Get G-vectors (skip G=0)
        for i in range(1, gvector.npw):
            m = gvector.miller[i]
            G = gvector.gvecs[i]
            g2 = gvector.norms[i]**2
            
            # Structure factor
            S = 0.0 + 0.0j
            for j, pos in enumerate(self.positions):
                phase = TWOPI * (m[0]*pos[0] + m[1]*pos[1] + m[2]*pos[2])
                S += self.charges[j] * np.exp(1j * phase)
            
            # Add contribution
            energy += np.abs(S)**2 * np.exp(-g2 / (4*self.eta)) / g2
        
        energy *= FOURPI / (2.0 * self.volume)
        
        return energy
    
    def _compute_energy_recip_simple(self):
        """Simple reciprocal space calculation without GVector object."""
        # Get reciprocal lattice vectors
        b1, b2, b3 = self.lattice.reciprocal_vectors
        
        # G-vector cutoff
        gmax = np.sqrt(2 * self.eta) * 5  # 5 times decay length
        
        n1_max = int(gmax / np.linalg.norm(b1)) + 1
        n2_max = int(gmax / np.linalg.norm(b2)) + 1
        n3_max = int(gmax / np.linalg.norm(b3)) + 1
        
        energy = 0.0
        
        for n1 in range(-n1_max, n1_max + 1):
            for n2 in range(-n2_max, n2_max + 1):
                for n3 in range(-n3_max, n3_max + 1):
                    if n1 == 0 and n2 == 0 and n3 == 0:
                        continue
                    
                    G = n1 * b1 + n2 * b2 + n3 * b3
                    g2 = np.sum(G**2)
                    
                    # Structure factor
                    S = 0.0 + 0.0j
                    for j, pos in enumerate(self.positions):
                        phase = TWOPI * (n1*pos[0] + n2*pos[1] + n3*pos[2])
                        S += self.charges[j] * np.exp(1j * phase)
                    
                    energy += np.abs(S)**2 * np.exp(-g2 / (4*self.eta)) / g2
        
        energy *= FOURPI / (2.0 * self.volume)
        
        return energy
    
    def _compute_energy_self_and_g0(self):
        """
        Compute self-energy and G=0 corrections.
        
        E_self = -sqrt(eta/pi) * sum_i Z_i^2
        E_G0 = -pi / (2*V*eta) * (sum_i Z_i)^2
        """
        z_sum = np.sum(self.charges)
        z2_sum = np.sum(self.charges**2)
        
        e_self = -np.sqrt(self.eta / PI) * z2_sum
        e_g0 = -PI / (2.0 * self.volume * self.eta) * z_sum**2
        
        return e_self, e_g0
    
    def _compute_force_real_space(self):
        """
        Compute real-space force contribution.
        
        F_i = -dE/dr_i
        """
        sqrt_eta = np.sqrt(self.eta)
        forces = np.zeros((self.natoms, 3))
        
        for i in range(self.natoms):
            f = np.zeros(3)
            
            for cell in self.nn_cells:
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    
                    dfrac = self.positions[i] - self.positions[j] - cell
                    dr = self._frac_to_cart(dfrac)
                    r = np.linalg.norm(dr)
                    
                    if r > 1e-10:
                        # Derivative of erfc(sqrt(eta)*r)/r
                        coeff = self.charges[i] * self.charges[j] / r**2
                        coeff *= (2.0 * sqrt_eta / np.sqrt(PI) * 
                                 np.exp(-self.eta * r**2) + 
                                 erfc(sqrt_eta * r) / r)
                        
                        f += coeff * dr / r
            
            forces[i] = f
        
        return forces
    
    def _compute_force_reciprocal_space(self, gvector):
        """Compute reciprocal-space force contribution."""
        forces = np.zeros((self.natoms, 3))
        
        if gvector is None:
            return self._compute_force_recip_simple()
        
        for iat in range(self.natoms):
            f = np.zeros(3, dtype=complex)
            
            for i in range(1, gvector.npw):
                m = gvector.miller[i]
                G = gvector.gvecs[i]
                g2 = gvector.norms[i]**2
                
                # Structure factor
                S = 0.0 + 0.0j
                for j, pos in enumerate(self.positions):
                    phase = TWOPI * (m[0]*pos[0] + m[1]*pos[1] + m[2]*pos[2])
                    S += self.charges[j] * np.exp(1j * phase)
                
                # Phase for this atom (negative)
                pos = self.positions[iat]
                neg_phase = -TWOPI * (m[0]*pos[0] + m[1]*pos[1] + m[2]*pos[2])
                
                coeff = S * self.charges[iat] * np.exp(1j * neg_phase)
                coeff *= np.exp(-g2 / (4*self.eta)) / g2
                
                f += 1j * G * coeff
            
            forces[iat] = 2.0 * np.real(f) * FOURPI / (2.0 * self.volume)
        
        return forces
    
    def _compute_force_recip_simple(self):
        """Simple reciprocal space force calculation."""
        b1, b2, b3 = self.lattice.reciprocal_vectors
        
        gmax = np.sqrt(2 * self.eta) * 5
        n1_max = int(gmax / np.linalg.norm(b1)) + 1
        n2_max = int(gmax / np.linalg.norm(b2)) + 1
        n3_max = int(gmax / np.linalg.norm(b3)) + 1
        
        forces = np.zeros((self.natoms, 3))
        
        for iat in range(self.natoms):
            f = np.zeros(3, dtype=complex)
            
            for n1 in range(-n1_max, n1_max + 1):
                for n2 in range(-n2_max, n2_max + 1):
                    for n3 in range(-n3_max, n3_max + 1):
                        if n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        
                        G = n1 * b1 + n2 * b2 + n3 * b3
                        g2 = np.sum(G**2)
                        
                        # Structure factor
                        S = 0.0 + 0.0j
                        for j, pos in enumerate(self.positions):
                            phase = TWOPI * (n1*pos[0] + n2*pos[1] + n3*pos[2])
                            S += self.charges[j] * np.exp(1j * phase)
                        
                        pos = self.positions[iat]
                        neg_phase = -TWOPI * (n1*pos[0] + n2*pos[1] + n3*pos[2])
                        
                        coeff = S * self.charges[iat] * np.exp(1j * neg_phase)
                        coeff *= np.exp(-g2 / (4*self.eta)) / g2
                        
                        f += 1j * G * coeff
            
            forces[iat] = 2.0 * np.real(f) * FOURPI / (2.0 * self.volume)
        
        return forces
    
    def _compute_stress(self, gvector):
        """
        Compute stress tensor contribution.
        
        stress_ij = -dE/d(strain_ij) / V
        """
        stress = np.zeros((3, 3))
        
        # Real-space contribution
        sqrt_eta = np.sqrt(self.eta)
        
        for cell in self.nn_cells:
            for i in range(self.natoms):
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    
                    dfrac = self.positions[i] - self.positions[j] - cell
                    dr = self._frac_to_cart(dfrac)
                    r = np.linalg.norm(dr)
                    
                    if r > 1e-10:
                        coeff = 0.5 * sqrt_eta * self.charges[i] * self.charges[j]
                        coeff *= (-2.0 / np.sqrt(PI) * np.exp(-self.eta * r**2) -
                                  erfc(sqrt_eta * r) / r) / r**2
                        
                        for ii in range(3):
                            for jj in range(3):
                                stress[ii, jj] += coeff * dr[ii] * dr[jj]
        
        # Reciprocal-space contribution (simplified)
        # Add self-energy contribution to diagonal
        z_sum = np.sum(self.charges)
        for ii in range(3):
            stress[ii, ii] += z_sum**2 * PI / (2.0 * self.volume * self.eta)
        
        # Normalize by volume
        stress /= -self.volume
        
        return stress
    
    def get_energy(self):
        """Return Ewald energy in Hartree."""
        return self.energy
    
    def get_forces(self):
        """Return forces on ions in Hartree/Bohr."""
        return self.forces.copy()
    
    def get_stress(self):
        """Return stress tensor in Hartree/Bohr^3."""
        return self.stress.copy()


def compute_ewald_energy(lattice, atom_positions, charges, gvector=None):
    """
    Convenience function to compute just the Ewald energy.
    
    Args:
        lattice: Lattice object
        atom_positions: Fractional coordinates, shape (natoms, 3)
        charges: Ion charges, shape (natoms,)
        gvector: Optional GVector object
    
    Returns:
        Ewald energy in Hartree
    """
    ewald = Ewald(lattice, atom_positions, charges, gvector)
    return ewald.get_energy()
