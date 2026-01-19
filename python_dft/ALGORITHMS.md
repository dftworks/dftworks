# Plane-Wave Pseudopotential DFT: Algorithm Documentation

This document describes the core algorithms implemented in a plane-wave pseudopotential density functional theory (DFT) code, based on the dftworks Rust implementation.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Plane Wave Basis Set](#plane-wave-basis-set)
3. [Self-Consistent Field (SCF) Loop](#self-consistent-field-scf-loop)
4. [Hartree Potential](#hartree-potential)
5. [Exchange-Correlation Functional](#exchange-correlation-functional)
6. [Hamiltonian Application](#hamiltonian-application)
7. [Eigensolver (PCG)](#eigensolver-pcg)
8. [Density Mixing](#density-mixing)
9. [Total Energy](#total-energy)
10. [Ewald Summation](#ewald-summation)

---

## Theoretical Background

### The Kohn-Sham Equations

Density Functional Theory reduces the many-body Schrodinger equation to a set of single-particle equations:

$$\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{eff}(\mathbf{r})\right]\psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})$$

where the effective potential is:

$$V_{eff}(\mathbf{r}) = V_{ext}(\mathbf{r}) + V_H(\mathbf{r}) + V_{xc}(\mathbf{r})$$

The electron density is constructed from the occupied orbitals:

$$\rho(\mathbf{r}) = \sum_{i}^{occ} f_i |\psi_i(\mathbf{r})|^2$$

where $f_i$ is the occupation number.

### Atomic Units

Throughout this code, we use Hartree atomic units:
- Energy: 1 Hartree = 27.211 eV
- Length: 1 Bohr = 0.529177 Angstrom
- $\hbar = m_e = e = 4\pi\epsilon_0 = 1$

Key constants:
```python
BOHR_TO_ANG = 0.529177249
HA_TO_EV = 27.211396132
RY_TO_EV = 13.605698066
```

---

## Plane Wave Basis Set

### Bloch's Theorem

For periodic systems, wavefunctions satisfy Bloch's theorem:

$$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})$$

where $u_{n\mathbf{k}}(\mathbf{r})$ has the periodicity of the lattice.

### Plane Wave Expansion

The periodic part is expanded in plane waves:

$$u_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) e^{i\mathbf{G}\cdot\mathbf{r}}$$

where $\mathbf{G}$ are reciprocal lattice vectors:

$$\mathbf{G} = m_1\mathbf{b}_1 + m_2\mathbf{b}_2 + m_3\mathbf{b}_3$$

### Energy Cutoff

Only plane waves with kinetic energy below the cutoff are included:

$$\frac{1}{2}|\mathbf{k}+\mathbf{G}|^2 < E_{cut}$$

This determines the number of plane waves (basis size).

### G-Vector Generation Algorithm

```python
def generate_gvectors(b1, b2, b3, ecut):
    """Generate G-vectors within energy cutoff."""
    gmax = sqrt(2 * ecut)
    gvectors = []
    
    # Determine max Miller indices
    n1_max = int(gmax / norm(b1)) + 1
    n2_max = int(gmax / norm(b2)) + 1
    n3_max = int(gmax / norm(b3)) + 1
    
    for n1 in range(-n1_max, n1_max + 1):
        for n2 in range(-n2_max, n2_max + 1):
            for n3 in range(-n3_max, n3_max + 1):
                G = n1*b1 + n2*b2 + n3*b3
                if 0.5 * norm(G)**2 < ecut:
                    gvectors.append((n1, n2, n3, G))
    
    return sorted(gvectors, key=lambda x: norm(x[3]))
```

---

## Self-Consistent Field (SCF) Loop

The SCF procedure iteratively solves the Kohn-Sham equations:

```
Algorithm: SCF Loop
-------------------
1. Initialize density rho(r) (e.g., from atomic superposition)
2. Transform to reciprocal space: rho(G) = FFT[rho(r)]
3. LOOP until convergence:
   a. Compute Hartree potential: V_H(G) = 4*pi*rho(G)/G^2
   b. Compute XC potential: V_xc(r) = d(E_xc)/d(rho) in real space
   c. Transform V_xc to G-space: V_xc(G) = FFT[V_xc(r)]
   d. Build total local potential: V_loc(G) = V_ps_loc(G) + V_H(G) + V_xc(G)
   e. Solve eigenvalue problem: H|psi_i> = eps_i|psi_i>
   f. Compute new density: rho_new(r) = sum_i f_i |psi_i(r)|^2
   g. Mix densities: rho_next = mix(rho_old, rho_new)
   h. Check convergence: |E_new - E_old| < epsilon
4. Compute forces and stresses (if needed)
```

### Convergence Criteria

Typically check:
- Total energy change: $|E^{(n+1)} - E^{(n)}| < \epsilon_{energy}$
- Density change: $\int |\rho^{(n+1)} - \rho^{(n)}|^2 dr < \epsilon_{density}$

---

## Hartree Potential

The Hartree potential describes electron-electron Coulomb repulsion.

### Poisson Equation

$$\nabla^2 V_H(\mathbf{r}) = -4\pi\rho(\mathbf{r})$$

### Solution in Reciprocal Space

In G-space, this becomes algebraic:

$$V_H(\mathbf{G}) = \frac{4\pi\rho(\mathbf{G})}{|\mathbf{G}|^2} \quad (\mathbf{G} \neq 0)$$

For $\mathbf{G} = 0$, we set $V_H(0) = 0$ (neutralizing background).

### Implementation

```python
def compute_hartree_potential(rhog, g_norms):
    """
    Compute Hartree potential in G-space.
    
    Args:
        rhog: Density in G-space (complex array)
        g_norms: |G| for each G-vector (G=0 should be first)
    
    Returns:
        vhg: Hartree potential in G-space
    """
    vhg = np.zeros_like(rhog)
    # Skip G=0 (set to zero for charge neutrality)
    vhg[0] = 0.0
    # For G != 0
    vhg[1:] = 4.0 * np.pi * rhog[1:] / (g_norms[1:]**2)
    return vhg
```

### Hartree Energy

$$E_H = \frac{1}{2}\int V_H(\mathbf{r})\rho(\mathbf{r})d\mathbf{r} = \frac{\Omega}{2}\sum_{\mathbf{G}\neq 0}\frac{4\pi|\rho(\mathbf{G})|^2}{|\mathbf{G}|^2}$$

---

## Exchange-Correlation Functional

### Local Density Approximation (LDA)

The LDA assumes the XC energy density at each point depends only on the local density:

$$E_{xc}[\rho] = \int \rho(\mathbf{r})\epsilon_{xc}(\rho(\mathbf{r}))d\mathbf{r}$$

### Slater Exchange

The exchange energy density:

$$\epsilon_x(\rho) = -\frac{3}{4}\left(\frac{3\rho}{\pi}\right)^{1/3}$$

The exchange potential:

$$V_x(\rho) = \frac{d(\rho\epsilon_x)}{d\rho} = -\left(\frac{3\rho}{\pi}\right)^{1/3}$$

### Perdew-Zunger Correlation

The correlation uses the Wigner-Seitz radius:

$$r_s = \left(\frac{3}{4\pi\rho}\right)^{1/3}$$

**For $r_s \geq 1$ (low density):**

$$\epsilon_c = \frac{\gamma}{1 + \beta_1\sqrt{r_s} + \beta_2 r_s}$$

Parameters: $\gamma = -0.1423$, $\beta_1 = 1.0529$, $\beta_2 = 0.3334$

The potential:

$$V_c = \epsilon_c \cdot \frac{1 + \frac{7}{6}\beta_1\sqrt{r_s} + \frac{4}{3}\beta_2 r_s}{1 + \beta_1\sqrt{r_s} + \beta_2 r_s}$$

**For $r_s < 1$ (high density):**

$$\epsilon_c = A\ln(r_s) + B + C\cdot r_s\ln(r_s) + D\cdot r_s$$

Parameters: $A = 0.0311$, $B = -0.048$, $C = 0.0020$, $D = -0.0116$

### Implementation

```python
def lda_xc(rho):
    """
    Compute LDA exchange-correlation potential and energy density.
    
    Args:
        rho: Electron density (real array)
    
    Returns:
        vxc: XC potential
        exc: XC energy density
    """
    # Avoid division by zero
    rho = np.maximum(rho, 1e-20)
    
    # Slater exchange
    cx = -(3.0 / np.pi) ** (1.0/3.0)
    vx = cx * rho ** (1.0/3.0)
    ex = 0.75 * vx
    
    # Perdew-Zunger correlation
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0/3.0)
    
    vc = np.zeros_like(rho)
    ec = np.zeros_like(rho)
    
    # Low density (rs >= 1)
    mask_low = rs >= 1.0
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    rs_low = rs[mask_low]
    rs_sqrt = np.sqrt(rs_low)
    denom = 1.0 + beta1 * rs_sqrt + beta2 * rs_low
    ec[mask_low] = gamma / denom
    numer = 1.0 + 7.0/6.0 * beta1 * rs_sqrt + 4.0/3.0 * beta2 * rs_low
    vc[mask_low] = ec[mask_low] * numer / denom
    
    # High density (rs < 1)
    mask_high = ~mask_low
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    rs_high = rs[mask_high]
    rs_ln = np.log(rs_high)
    ec[mask_high] = A * rs_ln + B + C * rs_high * rs_ln + D * rs_high
    vc[mask_high] = (A * rs_ln + (B - A/3.0) + 
                     2.0/3.0 * C * rs_high * rs_ln + 
                     1.0/3.0 * (2*D - C) * rs_high)
    
    return vx + vc, ex + ec
```

---

## Hamiltonian Application

The Kohn-Sham Hamiltonian in the plane wave basis:

$$H = T + V_{loc} + V_{nl}$$

### Kinetic Energy Operator

In G-space, the kinetic energy is diagonal:

$$\langle\mathbf{k}+\mathbf{G}|T|\mathbf{k}+\mathbf{G}'\rangle = \frac{1}{2}|\mathbf{k}+\mathbf{G}|^2\delta_{\mathbf{G},\mathbf{G}'}$$

Implementation:
```python
def kinetic_on_psi(kg, psi_g):
    """Apply kinetic operator: T|psi> = 0.5*|k+G|^2 * psi(G)"""
    return 0.5 * kg * psi_g
```

### Local Potential

The local potential is applied via FFT (convolution theorem):

1. Transform $\psi(\mathbf{G}) \rightarrow \psi(\mathbf{r})$ via inverse FFT
2. Multiply: $\tilde{\psi}(\mathbf{r}) = V_{loc}(\mathbf{r}) \cdot \psi(\mathbf{r})$
3. Transform back: $\tilde{\psi}(\mathbf{G}) = FFT[\tilde{\psi}(\mathbf{r})]$

```python
def vloc_on_psi(psi_g, vloc_r, fft_grid):
    """Apply local potential via FFT."""
    # G -> r
    psi_r = ifft(map_to_fft_grid(psi_g, fft_grid))
    # Multiply in real space
    vpsi_r = vloc_r * psi_r
    # r -> G
    vpsi_g = fft(vpsi_r)
    return map_from_fft_grid(vpsi_g, fft_grid)
```

### Non-Local Potential (Kleinman-Bylander)

The non-local pseudopotential uses separable projectors:

$$V_{nl} = \sum_{\ell m} |\beta_{\ell m}\rangle D_\ell \langle\beta_{\ell m}|$$

Application:
```python
def vnl_on_psi(psi_g, beta, D):
    """Apply non-local potential."""
    # Project onto beta functions
    proj = np.dot(beta.conj(), psi_g)
    # Apply D matrix and sum back
    return D * proj * beta
```

---

## Eigensolver (PCG)

The Preconditioned Conjugate Gradient method finds eigenpairs iteratively.

### Algorithm: Band-by-Band PCG

```
For each band i = 1, ..., N_bands:
    1. Orthogonalize psi_i to all lower bands (Gram-Schmidt)
    2. Normalize psi_i
    3. Compute H|psi_i> and initial eigenvalue omega_0 = <psi|H|psi>
    
    4. CG iterations until converged:
       a. Compute gradient: g = H|psi> - omega*|psi>
       b. Orthogonalize g to lower bands
       c. Apply preconditioner: pg = P*g
       d. Orthogonalize pg to lower bands
       e. Compute search direction: d = -pg + beta*d_prev
          where beta = <g|pg - pg_prev> / <g_prev|pg_prev>
       f. Orthogonalize d to |psi>
       g. Compute H|d>
       h. Solve 2x2 generalized eigenvalue problem to find optimal step
       i. Update: psi = cos(theta)*psi + sin(theta)*d
       j. Update: H|psi> = cos(theta)*H|psi> + sin(theta)*H|d>
       k. omega = <psi|H|psi>
       l. Check convergence: |omega - omega_prev| < tol
    
    5. Store eigenvalue and eigenvector
```

### Preconditioner

The Teter-Payne-Allan preconditioner:

$$P(\mathbf{G}) = \frac{27 + 18x + 12x^2 + 8x^3}{27 + 18x + 12x^2 + 8x^3 + 16x^4}$$

where $x = E_{kin}(\mathbf{G}) / (1.5 \cdot E_{kin,avg})$

```python
def compute_preconditioner(psi_g, kg):
    """Teter-Payne-Allan preconditioner."""
    # Average kinetic energy
    ek = np.sum(np.abs(psi_g)**2 * kg)
    x = kg / (1.5 * ek)
    x2, x3, x4 = x**2, x**3, x**4
    y = 27.0 + 18.0*x + 12.0*x2 + 8.0*x3
    precond = y / (y + 16.0*x4)
    return precond * 2.0 / (1.5 * ek)
```

### Gram-Schmidt Orthogonalization

```python
def gram_schmidt(evecs, iband):
    """Orthogonalize band iband to all lower bands."""
    v = evecs[:, iband].copy()
    for j in range(iband):
        proj = np.vdot(evecs[:, j], v)
        v -= proj * evecs[:, j]
    v /= np.linalg.norm(v)
    evecs[:, iband] = v
```

---

## Density Mixing

### The Problem

Direct iteration (using output density as next input) often diverges. Mixing schemes accelerate and stabilize convergence.

### Simple Linear Mixing

$$\rho^{(n+1)}_{in} = \rho^{(n)}_{in} + \alpha \cdot (\rho^{(n)}_{out} - \rho^{(n)}_{in})$$

where $\alpha \approx 0.1-0.3$ is the mixing parameter.

### Broyden Mixing

The modified Broyden method (Phys. Rev. B 38, 12807, 1988) uses history to estimate the inverse Jacobian.

```
Algorithm: Broyden Mixing
-------------------------
Store history of input/output densities: {rho_in^(n)}, {rho_out^(n)}

1. Simple mixing step:
   rho_next = rho_in + alpha * (rho_out - rho_in)

2. If history available (n > 1):
   a. Compute residual differences: dR^(i) = R^(i+1) - R^(i)
      where R^(i) = rho_out^(i) - rho_in^(i)
   b. Compute input differences: drho^(i) = rho_in^(i+1) - rho_in^(i)
   c. Build overlap matrix: A_ij = <dR^(i) | dR^(j)>
   d. Add regularization: A_ii += omega_0^2
   e. Compute beta = A^(-1)
   f. Compute coefficients: gamma = C * beta
      where C_ij = <R^(n) | dR^(j)>
   g. Apply correction:
      rho_next -= sum_i omega_i * gamma_ni * (alpha * dR^(i) + drho^(i))
```

### Implementation

```python
class BroydenMixer:
    def __init__(self, alpha=0.7, n_history=8):
        self.alpha = alpha
        self.n_history = n_history
        self.history_in = []
        self.history_out = []
    
    def mix(self, rho_in, rho_out):
        self.history_in.append(rho_in.copy())
        self.history_out.append(rho_out.copy())
        
        # Simple mixing step
        rho_next = rho_in + self.alpha * (rho_out - rho_in)
        
        if len(self.history_in) > 1:
            # Apply Broyden correction
            m = len(self.history_in) - 1
            m = min(m, self.n_history)
            
            # Build matrices and apply correction...
            # (see full implementation)
        
        # Trim history
        if len(self.history_in) > self.n_history:
            self.history_in.pop(0)
            self.history_out.pop(0)
        
        return rho_next
```

---

## Total Energy

The total energy consists of several contributions:

$$E_{tot} = E_{band} - E_{H} + E_{xc} - E_{V_{xc}} + E_{Ewald}$$

or equivalently:

$$E_{tot} = E_{kin} + E_{loc} + E_{nl} + E_{H} + E_{xc} + E_{Ewald}$$

### Band Structure Energy

$$E_{band} = \sum_i f_i \varepsilon_i$$

### Kinetic Energy

$$E_{kin} = \frac{1}{2}\sum_{n\mathbf{k}} f_{n\mathbf{k}} \sum_{\mathbf{G}} |c_{n\mathbf{k}}(\mathbf{G})|^2 |\mathbf{k}+\mathbf{G}|^2$$

### Hartree Energy

$$E_H = \frac{\Omega}{2}\sum_{\mathbf{G}\neq 0} \frac{4\pi|\rho(\mathbf{G})|^2}{|\mathbf{G}|^2}$$

### Exchange-Correlation Energy

$$E_{xc} = \int \rho(\mathbf{r})\epsilon_{xc}(\rho(\mathbf{r}))d\mathbf{r}$$

### Double Counting Correction

The band energy includes the Hartree and XC contributions, so we must subtract them:

$$E_{tot} = E_{band} - E_H + E_{xc} - \int V_{xc}(\mathbf{r})\rho(\mathbf{r})d\mathbf{r} + E_{Ewald}$$

---

## Ewald Summation

The Ewald method computes the ion-ion Coulomb energy for periodic systems by splitting it into real-space and reciprocal-space parts.

### The Problem

Direct sum of $1/r$ Coulomb interactions converges slowly (conditionally).

### Ewald Decomposition

$$E_{Ewald} = E_{real} + E_{recip} + E_{self}$$

### Real-Space Part

$$E_{real} = \frac{1}{2}\sum_{\mathbf{R}}\sum_{i \neq j} Z_i Z_j \frac{\text{erfc}(\eta^{1/2}|\mathbf{r}_{ij} + \mathbf{R}|)}{|\mathbf{r}_{ij} + \mathbf{R}|}$$

### Reciprocal-Space Part

$$E_{recip} = \frac{2\pi}{\Omega}\sum_{\mathbf{G}\neq 0} \frac{e^{-|\mathbf{G}|^2/4\eta}}{|\mathbf{G}|^2} |S(\mathbf{G})|^2$$

where the structure factor is:

$$S(\mathbf{G}) = \sum_i Z_i e^{i\mathbf{G}\cdot\mathbf{r}_i}$$

### Self-Energy Correction

$$E_{self} = -\left(\frac{\eta}{\pi}\right)^{1/2}\sum_i Z_i^2$$

### G=0 Term

$$E_{G=0} = -\frac{\pi}{2\Omega\eta}\left(\sum_i Z_i\right)^2$$

### Choice of eta

The parameter $\eta$ controls the splitting. Optimal choice balances real and reciprocal space work:

$$\eta \approx \left(\frac{N_{atoms}}{V}\right)^{2/3}$$

---

## Summary: Data Flow

```
Input: Crystal structure, Pseudopotentials, Parameters
       |
       v
Generate G-vectors and plane wave basis
       |
       v
Initialize density (atomic superposition)
       |
       v
+-------------------------------+
|     SCF Loop                  |
|  +-------------------------+  |
|  | Build V_eff = V_H + V_xc|  |
|  +-------------------------+  |
|             |                 |
|             v                 |
|  +-------------------------+  |
|  | Solve H|psi> = E|psi>   |  |
|  | (PCG eigensolver)       |  |
|  +-------------------------+  |
|             |                 |
|             v                 |
|  +-------------------------+  |
|  | Build new density       |  |
|  | rho = sum |psi|^2       |  |
|  +-------------------------+  |
|             |                 |
|             v                 |
|  +-------------------------+  |
|  | Mix densities           |  |
|  | (Broyden)               |  |
|  +-------------------------+  |
|             |                 |
|      Check convergence        |
+-------------------------------+
       |
       v
Output: Total energy, Eigenvalues, Forces, Stress
```

---

## References

1. Martin, R. M. "Electronic Structure: Basic Theory and Practical Methods" (Cambridge, 2004)
2. Perdew, J. P., & Zunger, A. "Self-interaction correction to density-functional approximations" Phys. Rev. B 23, 5048 (1981)
3. Johnson, D. D. "Modified Broyden's method for accelerating convergence in self-consistent calculations" Phys. Rev. B 38, 12807 (1988)
4. Teter, M. P., Payne, M. C., & Allan, D. C. "Solution of Schrodinger's equation for large systems" Phys. Rev. B 40, 12255 (1989)
5. Ewald, P. P. "Die Berechnung optischer und elektrostatischer Gitterpotentiale" Ann. Phys. 369, 253 (1921)
