"""
Density mixing schemes for SCF convergence acceleration.

Implements:
- Linear mixing (simple)
- Broyden mixing (modified Broyden II method)
"""

import numpy as np


class LinearMixer:
    """
    Simple linear density mixing.
    
    rho_next = rho_in + alpha * (rho_out - rho_in)
    
    Simple but requires small alpha (0.1-0.3) for stability.
    """
    
    def __init__(self, alpha=0.3):
        """
        Initialize linear mixer.
        
        Args:
            alpha: Mixing parameter (0 < alpha <= 1)
        """
        self.alpha = alpha
    
    def mix(self, rho_in, rho_out):
        """
        Perform linear mixing.
        
        Args:
            rho_in: Input density (from previous iteration)
            rho_out: Output density (computed from wavefunctions)
        
        Returns:
            rho_next: Mixed density for next iteration
        """
        return rho_in + self.alpha * (rho_out - rho_in)
    
    def reset(self):
        """Reset mixer state (no state for linear mixing)."""
        pass


class BroydenMixer:
    """
    Modified Broyden mixing (Broyden II method).
    
    Uses history of input/output densities to estimate the inverse
    Jacobian and accelerate convergence.
    
    Reference:
        Johnson, D.D. Phys. Rev. B 38, 12807 (1988)
    """
    
    def __init__(self, alpha=0.7, n_history=8, omega0=0.01):
        """
        Initialize Broyden mixer.
        
        Args:
            alpha: Mixing parameter for initial linear mixing step
            n_history: Maximum number of iterations to keep in history
            omega0: Regularization parameter for matrix inversion
        """
        self.alpha = alpha
        self.n_history = n_history
        self.omega0 = omega0
        
        # History storage
        self.history_in = []
        self.history_out = []
        self.n_iter = 0
    
    def mix(self, rho_in, rho_out):
        """
        Perform Broyden mixing.
        
        Args:
            rho_in: Input density (flattened complex array)
            rho_out: Output density (flattened complex array)
        
        Returns:
            rho_next: Mixed density for next iteration
        """
        # Ensure arrays are 1D complex
        rho_in = np.asarray(rho_in).flatten()
        rho_out = np.asarray(rho_out).flatten()
        
        # Store in history
        self.history_in.append(rho_in.copy())
        self.history_out.append(rho_out.copy())
        
        # Start with linear mixing step
        rho_next = rho_in + self.alpha * (rho_out - rho_in)
        
        # Apply Broyden correction if we have history
        if len(self.history_in) > 1:
            m = len(self.history_in) - 1
            m = min(m, self.n_history)
            
            # Weights (can be made adaptive)
            omega = np.ones(len(self.history_in))
            
            # Compute overlap matrix A
            A = self._compute_a_matrix(m, omega)
            
            # Compute coefficient matrix C
            C = self._compute_c_matrix(m, omega)
            
            # Compute beta = (A + omega0^2 * I)^(-1)
            beta = self._compute_beta(A)
            
            # Compute gamma = C @ beta
            gamma = C @ beta
            
            # Apply correction
            ng = len(rho_in)
            for n in range(m):
                # Residual and density differences
                dres = self.history_out[n + 1] - self.history_out[n]
                drho = self.history_in[n + 1] - self.history_in[n]
                
                # Normalize
                norm = np.sqrt(np.sum(np.abs(dres)**2))
                if norm > 1e-20:
                    dres = dres / norm
                    drho = drho / norm
                
                # Apply correction
                rho_next -= omega[n] * gamma[m, n] * (self.alpha * dres + drho)
        
        # Trim history to maximum size
        if len(self.history_in) > self.n_history:
            self.history_in.pop(0)
            self.history_out.pop(0)
        
        self.n_iter += 1
        return rho_next
    
    def _compute_a_matrix(self, m, omega):
        """
        Compute overlap matrix A.
        
        A_ij = <dR_i | dR_j> * omega_i * omega_j
        where dR_i = R_{i+1} - R_i and R_i = rho_out_i - rho_in_i
        """
        A = np.zeros((m, m), dtype=complex)
        
        for i in range(m):
            dres_i = self.history_out[i + 1] - self.history_out[i]
            norm_i = np.sqrt(np.sum(np.abs(dres_i)**2))
            if norm_i > 1e-20:
                dres_i = dres_i / norm_i
            
            for j in range(m):
                dres_j = self.history_out[j + 1] - self.history_out[j]
                norm_j = np.sqrt(np.sum(np.abs(dres_j)**2))
                if norm_j > 1e-20:
                    dres_j = dres_j / norm_j
                
                A[i, j] = np.vdot(dres_j, dres_i) * omega[i] * omega[j]
        
        return A
    
    def _compute_c_matrix(self, m, omega):
        """
        Compute coefficient matrix C.
        
        C_ik = <R_i | dR_k> * omega_k
        Dimension: (m+1) x m
        """
        C = np.zeros((m + 1, m), dtype=complex)
        
        for i in range(m + 1):
            res_i = self.history_out[i] - self.history_in[i]
            
            for k in range(m):
                dres_k = self.history_out[k + 1] - self.history_out[k]
                norm_k = np.sqrt(np.sum(np.abs(dres_k)**2))
                if norm_k > 1e-20:
                    dres_k = dres_k / norm_k
                
                C[i, k] = np.vdot(dres_k, res_i) * omega[k]
        
        return C
    
    def _compute_beta(self, A):
        """
        Compute beta = (A + omega0^2 * I)^(-1).
        
        The regularization omega0^2 * I prevents ill-conditioning.
        """
        m = A.shape[0]
        A_reg = A + self.omega0**2 * np.eye(m)
        
        try:
            beta = np.linalg.inv(A_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            beta = np.linalg.pinv(A_reg)
        
        return beta
    
    def reset(self):
        """Reset mixer state (clear history)."""
        self.history_in = []
        self.history_out = []
        self.n_iter = 0


class PulayMixer:
    """
    Pulay mixing (DIIS - Direct Inversion in the Iterative Subspace).
    
    Minimizes the residual in the space spanned by previous iterations.
    """
    
    def __init__(self, alpha=0.5, n_history=8):
        """
        Initialize Pulay mixer.
        
        Args:
            alpha: Mixing parameter for linear mixing step
            n_history: Maximum history size
        """
        self.alpha = alpha
        self.n_history = n_history
        
        self.history_in = []
        self.history_res = []  # Residuals: rho_out - rho_in
    
    def mix(self, rho_in, rho_out):
        """
        Perform Pulay mixing.
        
        Args:
            rho_in: Input density
            rho_out: Output density
        
        Returns:
            rho_next: Mixed density
        """
        rho_in = np.asarray(rho_in).flatten()
        rho_out = np.asarray(rho_out).flatten()
        
        # Residual
        residual = rho_out - rho_in
        
        # Store in history (after linear mixing)
        rho_mixed = rho_in + self.alpha * residual
        self.history_in.append(rho_mixed.copy())
        self.history_res.append(residual.copy())
        
        # If not enough history, return linear mixing result
        if len(self.history_in) < 2:
            return rho_mixed
        
        m = len(self.history_in)
        
        # Build overlap matrix of residuals
        B = np.zeros((m + 1, m + 1), dtype=complex)
        
        for i in range(m):
            for j in range(m):
                B[i, j] = np.vdot(self.history_res[i], self.history_res[j])
        
        # Lagrange constraint row/column
        B[m, :m] = 1.0
        B[:m, m] = 1.0
        B[m, m] = 0.0
        
        # Right-hand side
        rhs = np.zeros(m + 1, dtype=complex)
        rhs[m] = 1.0
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(B, rhs, rcond=None)[0]
        
        # Build optimal density
        rho_next = np.zeros_like(rho_in)
        for i in range(m):
            rho_next += coeffs[i] * self.history_in[i]
        
        # Trim history
        if len(self.history_in) > self.n_history:
            self.history_in.pop(0)
            self.history_res.pop(0)
        
        return rho_next
    
    def reset(self):
        """Reset mixer state."""
        self.history_in = []
        self.history_res = []
