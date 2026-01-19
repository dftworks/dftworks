"""
Preconditioned Conjugate Gradient (PCG) eigensolver.

Finds the lowest eigenvalues and eigenvectors of the Kohn-Sham Hamiltonian
using a band-by-band optimization approach.
"""

import numpy as np


class PCGEigensolver:
    """
    Preconditioned Conjugate Gradient eigensolver for Kohn-Sham equations.
    
    Optimizes one band at a time, orthogonalizing to previously converged bands.
    Uses the Teter-Payne-Allan preconditioner.
    
    Reference:
        Teter, Payne, Allan. Phys. Rev. B 40, 12255 (1989)
    """
    
    def __init__(self, npw, n_bands):
        """
        Initialize PCG solver.
        
        Args:
            npw: Number of plane waves
            n_bands: Number of bands to compute
        """
        self.npw = npw
        self.n_bands = n_bands
        
        # Workspace vectors
        self._x0 = np.zeros(npw, dtype=complex)
        self._h_x0 = np.zeros(npw, dtype=complex)
        self._d0 = np.zeros(npw, dtype=complex)
        self._h_d0 = np.zeros(npw, dtype=complex)
        self._g0 = np.zeros(npw, dtype=complex)
        self._g1 = np.zeros(npw, dtype=complex)
        self._pg0 = np.zeros(npw, dtype=complex)
        self._pg1 = np.zeros(npw, dtype=complex)
        self._precond = np.zeros(npw, dtype=complex)
    
    def solve(self, ham_apply, ham_diag, evecs, evals, 
              tol=1e-6, max_iter=100):
        """
        Solve for eigenvalues and eigenvectors.
        
        Args:
            ham_apply: Function to apply Hamiltonian: h_psi = ham_apply(psi)
            ham_diag: Diagonal of Hamiltonian (kinetic energies)
            evecs: (npw, n_bands) initial guess, overwritten with eigenvectors
            evals: (n_bands,) array, overwritten with eigenvalues
            tol: Convergence tolerance for eigenvalues
            max_iter: Maximum CG iterations per band
        
        Returns:
            n_converged: Number of bands converged
            n_hpsi: Total number of H|psi> applications
        """
        n_hpsi = 0
        n_converged = 0
        
        for iband in range(self.n_bands):
            # Orthogonalize to lower bands and normalize
            if iband > 0:
                self._gram_schmidt(evecs, iband)
            
            # Copy current band to workspace
            self._x0[:] = evecs[:, iband]
            self._normalize(self._x0)
            
            # Apply Hamiltonian
            self._h_x0[:] = ham_apply(self._x0)
            n_hpsi += 1
            
            # Initial eigenvalue estimate
            omega0 = np.real(np.vdot(self._x0, self._h_x0))
            
            # CG iterations for this band
            converged = False
            self._d0.fill(0)
            
            for cg_iter in range(max_iter):
                # Compute preconditioner
                self._compute_preconditioner(self._x0, ham_diag)
                
                # Gradient: g = H|x> - omega*|x>
                self._g1[:] = self._h_x0 - omega0 * self._x0
                
                # Orthogonalize gradient to lower bands
                self._orthogonalize_to_bands(evecs, iband, self._g1)
                
                # Apply preconditioner
                self._pg1[:] = self._g1 * self._precond
                
                # Orthogonalize preconditioned gradient
                self._orthogonalize_to_bands(evecs, iband, self._pg1)
                self._normalize(self._pg1)
                
                # Compute beta (Polak-Ribiere)
                if cg_iter == 0:
                    beta = 0.0
                else:
                    num = np.vdot(self._g1, self._pg1 - self._pg0)
                    denom = np.vdot(self._pg0, self._g0)
                    if abs(denom) > 1e-20:
                        beta = max(0.0, np.real(num / denom))
                    else:
                        beta = 0.0
                
                # Update search direction
                self._d0[:] = -self._pg1 + beta * self._d0
                
                # Orthogonalize d to x
                proj = np.vdot(self._x0, self._d0)
                self._d0 -= proj * self._x0
                self._normalize(self._d0)
                
                # Apply Hamiltonian to search direction
                self._h_d0[:] = ham_apply(self._d0)
                n_hpsi += 1
                
                # Solve 2x2 generalized eigenvalue problem
                alpha = self._get_optimal_step()
                
                # Update wavefunction
                t = np.sqrt(1.0 + np.abs(alpha)**2)
                cs = 1.0 / t
                sn = alpha / t
                
                self._x0[:] = cs * self._x0 + sn * self._d0
                self._h_x0[:] = cs * self._h_x0 + sn * self._h_d0
                
                # New eigenvalue
                omega = np.real(np.vdot(self._x0, self._h_x0))
                
                # Check convergence
                if abs(omega - omega0) < tol:
                    converged = True
                    n_converged += 1
                    break
                
                # Save for next iteration
                self._pg0[:] = self._pg1
                self._g0[:] = self._g1
                omega0 = omega
            
            # Store results
            evals[iband] = omega0 if not converged else omega
            evecs[:, iband] = self._x0
        
        return n_converged, n_hpsi
    
    def _gram_schmidt(self, evecs, iband):
        """Orthogonalize band iband to all lower bands."""
        v = evecs[:, iband].copy()
        
        for j in range(iband):
            proj = np.vdot(evecs[:, j], v)
            v -= proj * evecs[:, j]
        
        v /= np.linalg.norm(v)
        evecs[:, iband] = v
    
    def _orthogonalize_to_bands(self, evecs, n_bands, v):
        """Orthogonalize vector v to first n_bands bands."""
        for j in range(n_bands):
            proj = np.vdot(evecs[:, j], v)
            v -= proj * evecs[:, j]
    
    def _normalize(self, v):
        """Normalize vector in place."""
        norm = np.linalg.norm(v)
        if norm > 1e-20:
            v /= norm
    
    def _compute_preconditioner(self, psi, kin):
        """
        Compute Teter-Payne-Allan preconditioner.
        
        P(G) = (27 + 18x + 12x^2 + 8x^3) / (27 + 18x + 12x^2 + 8x^3 + 16x^4)
        where x = E_kin(G) / (1.5 * E_avg)
        """
        # Average kinetic energy
        ek = np.sum(np.abs(psi)**2 * kin)
        ek = max(ek, 1e-10)
        
        x = kin / (1.5 * ek)
        x2 = x * x
        x3 = x * x2
        x4 = x * x3
        
        y = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x3
        self._precond[:] = y / (y + 16.0 * x4)
        self._precond *= 2.0 / (1.5 * ek)
    
    def _get_optimal_step(self):
        """
        Find optimal step by solving 2x2 generalized eigenvalue problem.
        
        Returns the optimal mixing coefficient alpha.
        """
        # Build 2x2 Hamiltonian and overlap matrices
        h00 = np.vdot(self._x0, self._h_x0)
        h01 = np.vdot(self._x0, self._h_d0)
        h10 = np.vdot(self._d0, self._h_x0)
        h11 = np.vdot(self._d0, self._h_d0)
        
        s00 = np.vdot(self._x0, self._x0)
        s01 = np.vdot(self._x0, self._d0)
        s10 = np.vdot(self._d0, self._x0)
        s11 = np.vdot(self._d0, self._d0)
        
        # Form matrices
        H = np.array([[h00, h01], [h10, h11]])
        S = np.array([[s00, s01], [s10, s11]])
        
        # Solve generalized eigenvalue problem
        try:
            # S^(-1) * H
            S_inv = np.linalg.inv(S)
            M = S_inv @ H
            
            # Get eigenvector for lowest eigenvalue
            evals, evecs = np.linalg.eig(M)
            idx = np.argmin(np.real(evals))
            ev = evecs[:, idx]
            
            # alpha = ev[1] / ev[0]
            if abs(ev[0]) > 1e-20:
                alpha = ev[1] / ev[0]
            else:
                alpha = 0.0
        except:
            alpha = 0.0
        
        return alpha


def random_initial_guess(npw, n_bands):
    """
    Generate random initial guess for eigenvectors.
    
    Args:
        npw: Number of plane waves
        n_bands: Number of bands
    
    Returns:
        (npw, n_bands) array of orthonormalized random vectors
    """
    evecs = np.random.randn(npw, n_bands) + 1j * np.random.randn(npw, n_bands)
    
    # Orthonormalize using Gram-Schmidt
    for i in range(n_bands):
        for j in range(i):
            proj = np.vdot(evecs[:, j], evecs[:, i])
            evecs[:, i] -= proj * evecs[:, j]
        evecs[:, i] /= np.linalg.norm(evecs[:, i])
    
    return evecs
