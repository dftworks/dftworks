"""
Self-Consistent Field (SCF) driver.

Orchestrates the iterative solution of the Kohn-Sham equations.
"""

import numpy as np
from .constants import HA_TO_EV, FOURPI
from .lattice import Lattice
from .gvector import GVector
from .pwbasis import PWBasis
from .xc import lda_xc, compute_xc_energy, compute_xc_potential_energy
from .hartree import compute_hartree_potential, compute_hartree_energy
from .hamiltonian import Hamiltonian, g_to_r, r_to_g
from .eigensolver import PCGEigensolver, random_initial_guess
from .mixing import BroydenMixer, LinearMixer


class SCFSolver:
    """
    Self-Consistent Field solver for plane-wave DFT.
    
    Implements the basic SCF loop:
    1. Build potential from density
    2. Solve eigenvalue problem
    3. Build new density from wavefunctions
    4. Mix densities
    5. Check convergence
    
    Attributes:
        lattice: Crystal lattice
        gvec: G-vector grid
        n_bands: Number of bands
        n_electrons: Number of electrons
    """
    
    def __init__(self, lattice, ecut, n_bands, n_electrons,
                 external_potential=None, mixer='broyden'):
        """
        Initialize SCF solver.
        
        Args:
            lattice: Lattice object
            ecut: Energy cutoff in Hartree
            n_bands: Number of bands to compute
            n_electrons: Number of electrons
            external_potential: Optional external potential V_ext(r) on FFT grid
            mixer: 'broyden' or 'linear'
        """
        self.lattice = lattice
        self.ecut = ecut
        self.n_bands = n_bands
        self.n_electrons = n_electrons
        self.volume = lattice.volume
        
        # Generate G-vectors
        self.gvec = GVector(lattice, ecut)
        self.npw = self.gvec.npw
        
        # FFT grid
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        # Plane wave basis (Gamma point only)
        self.pwbasis = PWBasis(self.gvec)
        
        # Hamiltonian
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)
        
        # External potential (ion potential placeholder)
        if external_potential is not None:
            self.v_ext = external_potential
        else:
            # Default: harmonic potential for testing
            self.v_ext = self._create_harmonic_potential()
        
        # Eigensolver
        self.eigensolver = PCGEigensolver(self.npw, self.n_bands)
        
        # Mixer
        if mixer == 'broyden':
            self.mixer = BroydenMixer(alpha=0.7, n_history=8)
        else:
            self.mixer = LinearMixer(alpha=0.3)
        
        # Occupations (simple filling)
        self.occupations = self._compute_occupations()
        
        # Storage
        self.evecs = None
        self.evals = None
        self.rho_r = None
        self.rho_g = None
    
    def _create_harmonic_potential(self, omega=0.1):
        """
        Create a harmonic confining potential for testing.
        
        V(r) = 0.5 * omega^2 * |r - r_center|^2
        """
        n1, n2, n3 = self.fft_shape
        
        # Real-space grid
        x = np.linspace(0, 1, n1, endpoint=False)
        y = np.linspace(0, 1, n2, endpoint=False)
        z = np.linspace(0, 1, n3, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center at (0.5, 0.5, 0.5) in fractional coordinates
        dx = X - 0.5
        dy = Y - 0.5
        dz = Z - 0.5
        
        # Convert to Cartesian distances
        # For simplicity, use lattice vectors
        a, b, c = self.lattice.vectors
        
        r2 = np.zeros_like(X)
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    r_cart = dx[i,j,k] * a + dy[i,j,k] * b + dz[i,j,k] * c
                    r2[i,j,k] = np.sum(r_cart**2)
        
        return 0.5 * omega**2 * r2
    
    def _compute_occupations(self):
        """Compute occupation numbers (simple filling)."""
        occ = np.zeros(self.n_bands)
        n_filled = self.n_electrons // 2  # Assuming spin-paired
        remainder = self.n_electrons % 2
        
        for i in range(min(n_filled, self.n_bands)):
            occ[i] = 2.0  # Spin-paired
        
        if remainder > 0 and n_filled < self.n_bands:
            occ[n_filled] = float(remainder)
        
        return occ
    
    def run(self, max_iter=50, tol=1e-6, verbose=True):
        """
        Run SCF calculation.
        
        Args:
            max_iter: Maximum SCF iterations
            tol: Energy convergence tolerance (Hartree)
            verbose: Print progress
        
        Returns:
            Total energy (Hartree)
        """
        if verbose:
            print("=" * 60)
            print("SCF Calculation")
            print("=" * 60)
            print(f"  Lattice volume: {self.volume:.4f} Bohr^3")
            print(f"  Energy cutoff: {self.ecut:.2f} Ha ({self.ecut * HA_TO_EV:.2f} eV)")
            print(f"  Plane waves: {self.npw}")
            print(f"  FFT grid: {self.fft_shape}")
            print(f"  Bands: {self.n_bands}")
            print(f"  Electrons: {self.n_electrons}")
            print("-" * 60)
        
        # Initialize density (uniform)
        self._initialize_density()
        
        # Initialize eigenvectors
        self.evecs = random_initial_guess(self.npw, self.n_bands)
        self.evals = np.zeros(self.n_bands)
        
        # SCF loop
        energy_old = 0.0
        converged = False
        
        if verbose:
            print(f"{'Iter':>4} {'Energy (Ha)':>16} {'dE (Ha)':>12} {'dE (eV)':>12}")
            print("-" * 60)
        
        for scf_iter in range(1, max_iter + 1):
            # Build potentials
            self._build_potential()
            
            # Solve eigenvalue problem
            self.eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=self.evecs,
                evals=self.evals,
                tol=1e-8,
                max_iter=50
            )
            
            # Compute new density
            rho_new = self._compute_density()
            
            # Compute total energy
            energy = self._compute_total_energy()
            
            # Check convergence
            de = abs(energy - energy_old)
            
            if verbose:
                print(f"{scf_iter:4d} {energy:16.8f} {de:12.2e} {de * HA_TO_EV:12.2e}")
            
            if de < tol:
                converged = True
                if verbose:
                    print("-" * 60)
                    print(f"SCF converged in {scf_iter} iterations")
                break
            
            # Mix densities
            rho_mixed = self.mixer.mix(self.rho_g, rho_new)
            self.rho_g = rho_mixed
            
            # Update real-space density
            self.rho_r = self._g_to_r_density(self.rho_g)
            
            energy_old = energy
        
        if not converged and verbose:
            print("-" * 60)
            print(f"SCF did not converge in {max_iter} iterations")
        
        # Print final results
        if verbose:
            print("-" * 60)
            print("Final eigenvalues (Ha / eV):")
            for i, (e, occ) in enumerate(zip(self.evals, self.occupations)):
                print(f"  Band {i+1}: {e:12.6f} / {e * HA_TO_EV:12.6f}  occ: {occ:.1f}")
            print(f"\nTotal energy: {energy:.8f} Ha ({energy * HA_TO_EV:.6f} eV)")
            print("=" * 60)
        
        return energy
    
    def _initialize_density(self):
        """Initialize electron density (uniform)."""
        # Uniform density
        rho_0 = self.n_electrons / self.volume
        
        # Real-space density
        self.rho_r = np.full(self.fft_shape, rho_0)
        
        # G-space density
        self.rho_g = self._r_to_g_density(self.rho_r)
    
    def _r_to_g_density(self, rho_r):
        """Transform density from real space to G-space."""
        # FFT
        rho_fft = np.fft.fftn(rho_r) / self.n_fft
        
        # Map to G-vector list
        rho_g = self.gvec.map_from_fft_grid(rho_fft)
        
        return rho_g
    
    def _g_to_r_density(self, rho_g):
        """Transform density from G-space to real space."""
        # Map to FFT grid
        rho_fft = self.gvec.map_to_fft_grid(rho_g, self.fft_shape)
        
        # IFFT
        rho_r = np.fft.ifftn(rho_fft) * self.n_fft
        
        return np.real(rho_r)
    
    def _build_potential(self):
        """Build effective potential V_eff = V_H + V_xc + V_ext."""
        # Hartree potential in G-space
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)
        
        # XC potential in real space
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        v_xc_r, self._exc_r = lda_xc(rho_real)
        
        # Transform Hartree to real space
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)
        
        # Total local potential
        v_local_r = v_hartree_r + v_xc_r + self.v_ext
        
        # Store for energy calculation
        self._v_hartree_r = v_hartree_r
        self._v_xc_r = v_xc_r
        
        # Set in Hamiltonian
        self.hamiltonian.set_local_potential(v_local_r)
    
    def _compute_density(self):
        """Compute new density from wavefunctions."""
        # Real-space density
        rho_r = np.zeros(self.fft_shape, dtype=float)
        
        for i in range(self.n_bands):
            if self.occupations[i] < 1e-10:
                continue
            
            # Transform wavefunction to real space
            psi_r = g_to_r(self.evecs[:, i], self.gvec, 
                          self.fft_shape, self.volume)
            
            # Add to density
            rho_r += self.occupations[i] * np.abs(psi_r)**2
        
        # Transform to G-space
        rho_g = self._r_to_g_density(rho_r)
        
        return rho_g
    
    def _compute_total_energy(self):
        """
        Compute total energy.
        
        E_tot = E_band - E_H + E_xc - E_Vxc
        
        (In real DFT, there would also be ion-ion Ewald energy and
        pseudopotential contributions.)
        """
        # Band energy: sum of occupied eigenvalues
        e_band = np.sum(self.occupations * self.evals)
        
        # Hartree energy
        e_hartree = compute_hartree_energy(self.rho_g, self.gvec.norms, self.volume)
        
        # XC energy
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        e_xc = compute_xc_energy(rho_real, self._exc_r, self.volume, self.n_fft)
        
        # XC potential energy (for double-counting correction)
        e_vxc = compute_xc_potential_energy(rho_real, self._v_xc_r, 
                                            self.volume, self.n_fft)
        
        # External potential energy
        e_ext = (self.volume / self.n_fft) * np.sum(rho_real * self.v_ext)
        
        # Total energy
        # E_tot = E_band - E_H + E_xc - E_Vxc
        # But E_band already includes E_H + E_Vxc + E_ext
        # So: E_tot = E_band - E_H - E_Vxc + E_xc
        e_total = e_band - e_hartree + e_xc - e_vxc
        
        return e_total
    
    def get_eigenvalues(self):
        """Return eigenvalues in Hartree."""
        return self.evals.copy() if self.evals is not None else None
    
    def get_density(self):
        """Return electron density in real space."""
        return self.rho_r.copy() if self.rho_r is not None else None


def create_jellium_box(a, n_electrons, ecut, n_bands=None):
    """
    Create a jellium box calculation.
    
    Jellium: uniform positive background with electrons.
    
    Args:
        a: Box side length (Bohr)
        n_electrons: Number of electrons
        ecut: Energy cutoff (Hartree)
        n_bands: Number of bands (default: n_electrons // 2 + 4)
    
    Returns:
        SCFSolver configured for jellium box
    """
    lattice = Lattice.cubic(a)
    
    if n_bands is None:
        n_bands = n_electrons // 2 + 4
    
    # No external potential (just uniform background)
    solver = SCFSolver(
        lattice=lattice,
        ecut=ecut,
        n_bands=n_bands,
        n_electrons=n_electrons,
        external_potential=None  # Will create harmonic potential by default
    )
    
    # Actually set V_ext to zero for true jellium
    solver.v_ext = np.zeros(solver.fft_shape)
    
    return solver
