# SCF Pipeline Architecture

This document describes the Self-Consistent Field (SCF) iteration architecture in DFTWorks, including the unified iteration engine, spin/nonspin adapters, and execution flow.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [SCF Iteration Flow](#scf-iteration-flow)
4. [Spin vs Nonspin Paths](#spin-vs-nonspin-paths)
5. [Integration with KSCF](#integration-with-kscf)
6. [Convergence and Mixing](#convergence-and-mixing)
7. [Force and Stress Calculation](#force-and-stress-calculation)
8. [Adding New Features](#adding-new-features)

---

## Overview

### Purpose

The SCF module solves the Kohn-Sham equations iteratively:

1. **Build potential** from current density
2. **Solve eigenvalue problem** for each k-point
3. **Compute new density** from eigenstates
4. **Mix densities** to improve convergence
5. **Check convergence** (energy, density change)
6. **Repeat** until converged

### Key Design Goals

- **Unified iteration engine** - Shared loop control for nonspin and spin
- **Channel adapters** - Spin/nonspin-specific behavior via adapters
- **Allocation-free** - Workspaces eliminate hot-loop allocations
- **Modular** - Clear separation: iteration control, physics kernels, I/O

---

## Architecture Components

### Module Structure

```
scf/
├── src/
│   ├── lib.rs              # Public API (SCF trait)
│   ├── engine.rs           # Unified SCF iteration engine
│   ├── nonspin.rs          # Nonspin implementation + adapter
│   ├── spin.rs             # Spin implementation + adapter
│   ├── hubbard.rs          # DFT+U implementation
│   └── utils/
│       ├── mod.rs          # Utils façade
│       ├── potential.rs    # Potential assembly
│       ├── energy.rs       # Energy calculation
│       ├── mixing.rs       # Density mixing
│       ├── diagnostics.rs  # Eigenvalue output, tuning
│       └── symmetry_projection.rs  # Force/stress symmetrization
```

### Core Types

```rust
// Main SCF trait
pub trait SCF {
    fn run(
        &mut self,
        context: /* ... */,
        state: /* ... */,
    ) -> Result<ScfOutput, String>;
}

// Unified iteration engine adapter
pub trait ScfIterationAdapter {
    fn prepare_potential(&mut self, /* ... */) -> IterationPotential;
    fn solve_eigenstates(&mut self, /* ... */) -> IterationEigenstates;
    fn compute_occupations(&mut self, /* ... */) -> IterationOccupations;
    fn compute_harris_energy(&mut self, /* ... */) -> HarrisEnergy;
    fn build_output_density(&mut self, /* ... */) -> OutputDensity;
    fn refresh_input_density(&mut self, /* ... */);
    // ...
}
```

---

## SCF Iteration Flow

### High-Level Flow

```
┌─────────────────────────────────────────┐
│  SCFNonspin::run() or SCFSpin::run()    │
│  ┌───────────────────────────────────┐  │
│  │ Create Workspace (once)           │  │
│  │ Create Adapter (nonspin or spin)  │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              v                           │
│  ┌───────────────────────────────────┐  │
│  │ run_scf_iteration_engine()        │◄─┼─ Unified engine
│  │  ├─ for iter in 1..=max_iter     │  │
│  │  │   ├─ prepare_potential        │  │
│  │  │   ├─ solve_eigenstates        │  │
│  │  │   ├─ compute_occupations      │  │
│  │  │   ├─ compute_harris_energy    │  │
│  │  │   ├─ build_output_density     │  │
│  │  │   ├─ check_convergence        │  │
│  │  │   └─ mix_density              │  │
│  │  └─ return IterationOutput        │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              v                           │
│  ┌───────────────────────────────────┐  │
│  │ Post-SCF Force/Stress (if needed) │  │
│  │  ├─ compute_force_contributions  │  │
│  │  ├─ symmetry_project_forces      │  │
│  │  ├─ compute_stress_contributions │  │
│  │  └─ symmetry_project_stress      │  │
│  └───────────────────────────────────┘  │
│              │                           │
│              v                           │
│  ┌───────────────────────────────────┐  │
│  │ Return ScfOutput                  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Detailed Iteration Steps

Each SCF iteration executes these stages (via adapter):

#### 1. Prepare Potential

```rust
let pot = adapter.prepare_potential(
    iteration,
    rhog_in,      // Input density (G-space)
    rho_3d_in,    // Input density (real-space)
    workspace,
    context,
);
```

**Actions:**
- Compute Hartree potential: `V_H[ρ] = ∫ ρ(r')/|r-r'| dr'`
- Compute XC potential: `V_XC[ρ] = δE_XC/δρ`
- Add pseudopotential local part: `V_ps,loc`
- Add external field (if enabled): `V_ext`
- Add DFT+U potential (if enabled): `V_U`
- Total: `V_KS = V_H + V_XC + V_ps,loc + V_ext + V_U`

**Outputs:**
```rust
struct IterationPotential {
    vloc: Vec<c64>,          // Local potential (G-space)
    vloc_3d: Array3<c64>,    // Local potential (real-space)
    vxc_3d: VXCR,            // XC potential (for energy)
    exc_3d: Array3<c64>,     // XC energy density
}
```

#### 2. Solve Eigenstates

```rust
let eigenstates = adapter.solve_eigenstates(
    iteration,
    &pot.vloc,
    vk_scf,       // Per-k-point SCF state
    context,
);
```

**Actions:**
- **For each k-point** (MPI-distributed):
  - Build Hamiltonian: `H = T + V_loc + V_nl`
  - Solve eigenvalue problem: `H ψ_nk = ε_nk ψ_nk`
  - Store eigenvalues and eigenvectors

**Delegation:**
- Calls `kscf::solve_kpoint()` for each local k-point
- K-point loops are MPI-parallelized (each rank owns subset of k-points)
- Band loops inside k-point solver (sequential or future thread-parallel)

**Outputs:**
```rust
struct IterationEigenstates {
    // Eigenvalues/eigenvectors updated in vk_scf
    // Timing info for diagnostics
}
```

#### 3. Compute Occupations

```rust
let occ = adapter.compute_occupations(
    iteration,
    vk_eigval,
    vk_eigvec,
    smearing,
    context,
);
```

**Actions:**
- Find Fermi level: `∑_nk f(ε_nk, E_F) = N_elec`
- Compute occupations: `f_nk = f(ε_nk, E_F, T_smear)`
- Smearing methods: Gaussian, Fermi-Dirac, Methfessel-Paxton, cold

**Outputs:**
```rust
struct IterationOccupations {
    fermilevel: f64,
    vk_occ: VKOccupation,
}
```

#### 4. Compute Harris Energy

```rust
let harris = adapter.compute_harris_energy(
    iteration,
    rhog_in,
    &pot,
    &occ,
    context,
);
```

**Actions:**
- Compute Harris-Foulkes functional (uses input density):
  - `E_Harris = E_band + E_H[ρ_in] + E_XC[ρ_in] + E_ewald + E_correction`
- Provides energy estimate before mixing

**Outputs:**
```rust
struct HarrisEnergy {
    harris_energy: f64,
    band_energy: f64,
}
```

#### 5. Build Output Density

```rust
let out_density = adapter.build_output_density(
    iteration,
    vk_eigvec,
    &occ.vk_occ,
    workspace,
    context,
);
```

**Actions:**
- Sum over bands and k-points:
  - `ρ_out(r) = ∑_nk f_nk |ψ_nk(r)|²`
- Transform to G-space for mixing
- MPI reduction to collect contributions from all ranks

**Outputs:**
```rust
struct OutputDensity {
    rhog_out: RHOR,        // Output density (G-space)
    rho_3d_out: RHOR,      // Output density (real-space)
}
```

#### 6. Check Convergence

```rust
let convergence = check_convergence(
    iteration,
    rhog_in,
    &out_density.rhog_out,
    harris_energy,
    prev_harris_energy,
    energy_tol,
    density_tol,
);
```

**Criteria:**
- Energy change: `|E_i - E_{i-1}| < ε_E`
- Density change: `∫ |ρ_out - ρ_in| dr < ε_ρ`

**Outputs:**
```rust
struct ConvergenceStatus {
    converged: bool,
    energy_delta: f64,
    density_delta: f64,
}
```

#### 7. Mix Density

```rust
let mixed = mix_density(
    iteration,
    rhog_in,
    &out_density.rhog_out,
    mixer,
    workspace,
);
```

**Actions:**
- Pulay (DIIS) mixing: linear combination of previous densities
- Simple mixing: `ρ_new = α ρ_out + (1-α) ρ_in`
- Kerker preconditioning (for metals)

**Outputs:**
```rust
struct MixedDensity {
    rhog_mixed: RHOR,      // Mixed density for next iteration
}
```

#### 8. Refresh Input Density

```rust
adapter.refresh_input_density(rhog_mixed, workspace);
```

**Actions:**
- Update `rhog_in` and `rho_3d_in` for next iteration
- Inverse FFT: G-space → real-space

---

## Spin vs Nonspin Paths

### Unified Engine, Different Adapters

Both nonspin and spin use the same iteration engine (`run_scf_iteration_engine`), but provide different adapters:

```rust
// Nonspin adapter
struct NonSpinIterationAdapter { /* ... */ }

// Spin adapter
struct SpinIterationAdapter { /* ... */ }

// Both implement ScfIterationAdapter trait
impl ScfIterationAdapter for NonSpinIterationAdapter { /* ... */ }
impl ScfIterationAdapter for SpinIterationAdapter { /* ... */ }
```

### Key Differences

| Aspect | Nonspin | Spin |
|--------|---------|------|
| **Density channels** | 1 (total ρ) | 2 (ρ↑, ρ↓) |
| **Potential channels** | 1 (V_KS) | 2 (V↑, V↓) |
| **XC functional** | LDA/GGA | LSDA/GGA spin-polarized |
| **Eigenvectors** | Real spin (spinor) | Spin-up or spin-down component |
| **Density construction** | `∑_nk f_nk \|ψ_nk\|²` | `ρ↑ = ∑_nk↑ f_nk \|ψ_nk↑\|²` <br> `ρ↓ = ∑_nk↓ f_nk \|ψ_nk↓\|²` |

### Spin Implementation Details

**Channel iteration:**
```rust
// Spin adapter handles two channels
for channel in [SpinChannel::Up, SpinChannel::Down] {
    // Build potential for this channel
    let vloc_ch = build_channel_potential(channel, rhog_up, rhog_dn, ...);

    // Solve k-points for this channel
    for ik in local_k_indices {
        solve_kpoint(vloc_ch, vk_scf_ch[ik], ...);
    }

    // Build output density for this channel
    let rhog_out_ch = build_channel_density(vk_eigvec_ch, vk_occ_ch, ...);
}
```

**Shared vs per-channel:**
- **Shared:** Fermi level (single E_F for both channels)
- **Shared:** K-point cache (spherical harmonics, kinetic energy, FFT indices)
- **Per-channel:** Density, potential, eigenvalues, eigenvectors, occupations

---

## Integration with KSCF

### Delegation to K-Point Solver

SCF iteration calls `kscf::solve_kpoint()` for each k-point:

```rust
// SCF provides:
// - Local potential (vloc)
// - K-point basis (pwbasis)
// - Initial eigenvectors (for warmstart)

kscf::solve_kpoint(
    workspace,        // KSCF workspace (FFT, Hamiltonian scratch)
    shared_cache,     // Immutable per-k data (spherical harmonics, etc.)
    vloc,             // Local KS potential
    pwbasis,          // Plane-wave basis for this k-point
    eigval,           // Output: eigenvalues
    eigvec,           // Input/Output: eigenvectors
    // ...
);
```

### KSCF Responsibilities

- Apply Hamiltonian: `H ψ = (T + V_loc + V_nl) ψ`
- Iterative eigensolve (Davidson, PCG, Lanczos)
- Subspace rotation (Rayleigh-Ritz)
- Orthogonalization (Gram-Schmidt, Cholesky)

### Return to SCF

After all k-points solved:
- Eigenvalues/eigenvectors updated in `vk_scf`
- SCF proceeds to occupation calculation

---

## Convergence and Mixing

### Mixing Algorithms

**Pulay (DIIS) Mixing** (default):
- Stores history of densities and residuals
- Minimizes residual norm via linear combination
- Typically converges in 10-20 iterations

**Simple Mixing** (fallback):
- `ρ_new = α ρ_out + (1-α) ρ_in`
- Slower but more robust for difficult systems

### Convergence Criteria

**Primary:**
- Energy tolerance: `|ΔE| < 1e-6 Ry` (default)
- Density tolerance: `||Δρ||_1 < 1e-5` (default)

**Secondary:**
- Maximum iterations: 50 (default)
- Force convergence (for geometry optimization): `max|F| < 1e-3 Ry/bohr`

### Handling Non-Convergence

```rust
if !converged {
    eprintln!("SCF did not converge after {} iterations", max_iter);
    eprintln!("Final energy delta: {:.3e} Ry", energy_delta);
    eprintln!("Final density delta: {:.3e}", density_delta);
    return Err("SCF convergence failure".to_string());
}
```

**Common causes:**
- Mixing parameter too large (reduce α)
- Poor initial guess (use atomic densities)
- Metallic system (use smearing, Kerker preconditioning)
- Magnetic instability (use spin-polarized calculation)

---

## Force and Stress Calculation

### Post-SCF Evaluation

After SCF convergence, forces and stresses computed via Hellmann-Feynman theorem:

```rust
// Force = -dE/dR_atom
let force_contributions = compute_force_contributions(
    vk_eigval,
    vk_eigvec,
    vk_occ,
    converged_density,
    context,
);

// Symmetry projection (if symmetry enabled)
let forces = symmetry_project_forces(force_contributions, symops);

// Similarly for stress
let stress = compute_and_project_stress(...);
```

### Force Components

1. **Ewald force:** Ionic repulsion (long-range)
2. **Local pseudopotential force:** `∫ ρ(r) ∇V_ps,loc dr`
3. **Non-local pseudopotential force:** `∑_nk f_nk <ψ_nk|∇V_nl|ψ_nk>`
4. **Core-correction force:** NLCC gradient term
5. **Hubbard force:** DFT+U gradient (if enabled)
6. **External field force:** Electric field gradient (if enabled)

### Stress Components

Similar decomposition for stress tensor (needed for variable-cell optimization).

---

## Adding New Features

### Adding a New Potential Term

**Example:** Add a custom external potential `V_custom(r)`

**Step 1:** Add to potential preparation
```rust
// In utils/potential.rs or adapter
fn prepare_potential(...) -> IterationPotential {
    // ... existing potential terms ...

    // Add custom term
    let vcustom = compute_custom_potential(rho_3d, context);
    add_potential_contribution(&mut vloc, &vcustom, gvec);

    // ...
}
```

**Step 2:** Add to energy calculation
```rust
// In utils/energy.rs
let e_custom = compute_custom_energy(rho_3d, vcustom, context);
total_energy += e_custom;
```

**Step 3:** Add to force (if position-dependent)
```rust
// In utils/symmetry_projection.rs or force module
let f_custom = compute_custom_force(eigvec, context);
force_total += f_custom;
```

**Step 4:** Add control knobs
```rust
// In control/src/lib.rs
pub struct Control {
    // ...
    pub custom_potential_enabled: bool,
    pub custom_potential_strength: f64,
}
```

### Adding a New Convergence Criterion

**Example:** Add maximum gradient criterion

**Step 1:** Add to convergence check
```rust
fn check_convergence(...) -> ConvergenceStatus {
    let energy_converged = energy_delta < energy_tol;
    let density_converged = density_delta < density_tol;
    let gradient_converged = max_gradient < gradient_tol;  // NEW

    ConvergenceStatus {
        converged: energy_converged && density_converged && gradient_converged,
        // ...
    }
}
```

**Step 2:** Add to control
```rust
pub struct Control {
    pub convergence_gradient_tol: f64,
}
```

---

## Summary

### Key Architectural Decisions

1. **Unified engine with adapters** - Eliminates spin/nonspin code duplication
2. **Workspace pattern** - Allocation-free hot paths
3. **Modular utilities** - Reusable components (potential, energy, mixing, diagnostics)
4. **Clear delegation** - SCF orchestrates, KSCF solves eigenproblems
5. **Symmetry projection** - Post-SCF force/stress symmetrization

### Module Boundaries

```
┌─────────────────────────────────────────────┐
│ pw (Orchestration)                          │
│  ├─ Input parsing                           │
│  ├─ Geometry loop                           │
│  └─ Calls SCF                               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ scf (SCF Iteration)                         │
│  ├─ Iteration engine                        │
│  ├─ Potential/energy/mixing utilities       │
│  └─ Calls KSCF for each k-point             │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ kscf (K-Point Eigenvalue Solver)            │
│  ├─ Hamiltonian application                 │
│  ├─ Iterative eigensolver (PCG, Davidson)   │
│  └─ Returns eigenvalues/eigenvectors         │
└─────────────────────────────────────────────┘
```

### Further Reading

- `scf/src/engine.rs` - Unified iteration engine implementation
- `scf/src/nonspin.rs` - Nonspin adapter and driver
- `scf/src/spin.rs` - Spin adapter and driver
- `scf/src/utils/` - Reusable SCF utilities
- `kscf/src/lib.rs` - K-point eigenvalue solver
- `WORKSPACE_GUIDE.md` - Workspace pattern documentation
