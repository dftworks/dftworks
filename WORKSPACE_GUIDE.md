# Workspace Architecture Guide

This guide explains the `Context + State + Workspace` pattern used throughout DFTWorks to achieve allocation-free hot paths while maintaining clean separation of concerns.

## Table of Contents

1. [Overview](#overview)
2. [Pattern Components](#pattern-components)
3. [Design Principles](#design-principles)
4. [Implementation Examples](#implementation-examples)
5. [Adding Workspaces to New Modules](#adding-workspaces-to-new-modules)
6. [Performance Validation](#performance-validation)
7. [Anti-Patterns](#anti-patterns)

---

## Overview

### The Problem

Scientific computing codes have hot loops that execute millions of times. Common patterns like this create performance issues:

```rust
// BAD: Allocates on every iteration
for iter in 0..n_iterations {
    let scratch = vec![0.0; n];  // ❌ Allocation in hot loop
    compute_something(&mut scratch);
}
```

In DFT calculations:
- SCF iterations: 10-50 iterations
- Per k-point loops: 1-1000 k-points
- Per band loops: 10-1000 bands
- Per geometry step: 1-100 steps

Nested loops mean **millions of potential allocations**.

### The Solution

The `Context + State + Workspace` pattern:

```rust
// GOOD: Allocate once, reuse everywhere
let workspace = ScfWorkspace::new(npw_rho, fft_shape);  // ✅ One-time allocation
for iter in 0..n_iterations {
    compute_something(&mut workspace.scratch);  // ✅ Reuse
}
```

This pattern achieves:
- ✅ **Zero allocations** in hot loops (after warmup)
- ✅ **Explicit buffer sizing** validated at construction
- ✅ **Clear ownership** of scratch memory
- ✅ **Type safety** for buffer dimensions

---

## Pattern Components

### 1. Context (`Ctx`)

**Purpose**: Immutable runtime configuration and shared state.

**Characteristics:**
- Created once per phase (e.g., per geometry step)
- Read-only during execution
- Contains:
  - Runtime configuration (`Control`, `Crystal`, `GVector`, etc.)
  - Immutable derived data (FFT grids, k-point basis, symmetry ops)
  - Shared caches (when safe to share)

**Example:**
```rust
struct ScfContext<'a> {
    control: &'a Control,
    crystal: &'a Crystal,
    gvec: &'a GVector,
    pwden: &'a PWDensity,
    rgtrans: &'a RGTransform,
    ewald: &'a Ewald,
    // ... other immutable context
}
```

### 2. State (`State`)

**Purpose**: Mutable iteration state that evolves during execution.

**Characteristics:**
- Holds the "current" solution state
- Updated each iteration
- Contains:
  - Density, potentials, energies
  - Wavefunctions, eigenvalues, occupations
  - Convergence history

**Example:**
```rust
struct ScfState {
    rhog: RHOR,           // Current density (G-space)
    rho_3d: RHOR,         // Current density (real-space)
    vloc: Vec<c64>,       // Current local potential
    vk_eigval: VKEigenValue,
    vk_eigvec: VKEigenVector,
    // ... other iteration state
}
```

### 3. Workspace (`Workspace`)

**Purpose**: Reusable scratch buffers for intermediate calculations.

**Characteristics:**
- Allocated **once** at phase/module initialization
- **Sized explicitly** based on problem dimensions
- **Validated** via `debug_assert!` or `validate()` method
- **Reused** in-place across iterations
- Contains:
  - Temporary FFT buffers
  - Intermediate calculation scratch
  - Work arrays for linear algebra

**Example:**
```rust
struct NonSpinScfWorkspace {
    vhg: Vec<c64>,           // Hartree potential (G-space)
    vxcg: VXCG,              // XC potential (G-space)
    vxc_3d: VXCR,            // XC potential (real-space)
    exc_3d: Array3<c64>,     // XC energy density
    rhog_out: Vec<c64>,      // Output density (for mixing)
    rhog_diff: Vec<c64>,     // Density difference
    // ... other scratch buffers
}

impl NonSpinScfWorkspace {
    fn new(npw_rho: usize, fft_shape: [usize; 3]) -> Self {
        // Allocate all buffers once
        Self {
            vhg: vec![c64::zero(); npw_rho],
            vxcg: VXCG::NonSpin(vec![c64::zero(); npw_rho]),
            vxc_3d: VXCR::NonSpin(Array3::<c64>::new(fft_shape)),
            exc_3d: Array3::<c64>::new(fft_shape),
            rhog_out: vec![c64::zero(); npw_rho],
            rhog_diff: vec![c64::zero(); npw_rho],
        }
    }

    fn validate(&self, npw_rho: usize, fft_shape: [usize; 3]) {
        let nfft = fft_shape[0] * fft_shape[1] * fft_shape[2];
        debug_assert_eq!(self.vhg.len(), npw_rho);
        debug_assert_eq!(self.rhog_out.len(), npw_rho);
        debug_assert_eq!(self.exc_3d.as_slice().len(), nfft);
        // ... validate all buffers
    }
}
```

---

## Design Principles

### 1. One-Shot Sizing

**Principle:** All buffer sizes are computed **once** at workspace construction, not dynamically during execution.

```rust
// ✅ GOOD: Explicit sizing at construction
let workspace = KscfWorkspace::new(fft_shape, npw_wfc, npw_rho, nband, hubbard_n_m);

// ❌ BAD: Dynamic sizing during execution
fn compute(&mut self, npw: usize) {
    let scratch = vec![0.0; npw];  // Size depends on runtime parameter
}
```

### 2. Validation Over Panics

**Principle:** Validate buffer sizes in debug builds, but don't check in release builds (zero overhead).

```rust
fn validate(&self, expected_npw: usize) {
    debug_assert_eq!(
        self.buffer.len(),
        expected_npw,
        "Buffer size mismatch: expected {}, got {}",
        expected_npw,
        self.buffer.len()
    );
}
```

### 3. Explicit Ownership

**Principle:** Workspaces own their scratch buffers. No hidden allocations via `Box`, `Vec::push`, etc.

```rust
// ✅ GOOD: Workspace owns buffers
struct Workspace {
    buffer: Vec<c64>,  // Owned, pre-sized
}

// ❌ BAD: Hidden allocations
struct Workspace {
    buffer: Vec<Vec<c64>>,  // Growing vector of vectors
}
```

### 4. Reuse, Don't Reallocate

**Principle:** Buffers are overwritten, not replaced.

```rust
// ✅ GOOD: Reuse existing buffer
workspace.scratch.fill(c64::zero());

// ❌ BAD: Reallocate
workspace.scratch = vec![c64::zero(); n];
```

---

## Implementation Examples

### Example 1: SCF Nonspin Workspace

**Location:** `scf/src/nonspin.rs`

**Problem:** SCF loop runs 10-50 iterations, computing potentials, energies, densities each time.

**Solution:**
```rust
struct NonSpinScfWorkspace {
    vhg: Vec<c64>,           // Hartree potential (G-space)
    vxcg: VXCG,              // XC potential (G-space)
    vxc_3d: VXCR,            // XC potential (real-space)
    exc_3d: Array3<c64>,     // XC energy density
    vextg: Vec<c64>,         // External field potential (G)
    vext_3d: Array3<c64>,    // External field potential (R)
    vpslocg: Vec<c64>,       // Pseudopotential local (G)
    vlocg: Vec<c64>,         // Total local potential (G)
    vloc_3d: Array3<c64>,    // Total local potential (R)
    rhog_out: Vec<c64>,      // Output density for mixing
    rhog_diff: Vec<c64>,     // Density difference
}
```

**Usage:**
```rust
impl SCF for SCFNonspin {
    fn run(&mut self, /* ... */) -> Result<ScfOutput, String> {
        // Create workspace once
        let mut workspace = NonSpinScfWorkspace::new(npw_rho, fft_shape);
        workspace.validate(npw_rho, fft_shape);

        // Run SCF iterations - workspace is reused each iteration
        for iter in 1..=max_iter {
            // Build potential (uses workspace buffers)
            utils::potential::build_local_potential(
                &mut workspace.vhg,
                &mut workspace.vxcg,
                &mut workspace.vxc_3d,
                // ...
            );

            // Solve k-points (workspace passed down)
            for ik in 0..nkpt {
                kscf::solve_kpoint(/* ... */);
            }

            // Mix density (uses workspace buffers)
            mixing::mix(
                &mut workspace.rhog_out,
                &mut workspace.rhog_diff,
                // ...
            );
        }
    }
}
```

**Result:** Zero allocations in SCF loop after workspace creation.

---

### Example 2: KSCF Per-K-Point Workspace

**Location:** `kscf/src/lib.rs`

**Problem:** Per-k-point eigenvalue solve runs for every k-point (1-1000 times), with band-level loops inside.

**Solution:**
```rust
struct KscfWorkspace {
    vunkg_3d: Array3<c64>,              // FFT workspace for u_nk(G)
    unk_3d: Array3<c64>,                // FFT workspace for u_nk(r)
    fft_workspace: Array3<c64>,         // FFT scratch
    hubbard_beta: Vec<c64>,             // Hubbard projector overlap
    hubbard_coeff: Vec<c64>,            // Hubbard matrix elements
    hybrid_workspace: HybridWorkspace,  // HSE06 exchange workspace
    hybrid_prepare_workspace: HybridPrepareWorkspace,
    rayleigh_evecs: Matrix<c64>,        // Subspace rotation matrix
}
```

**Usage:**
```rust
pub fn solve_kpoint(
    workspace: &mut KscfWorkspace,
    shared_cache: &KscfSharedCache,
    // ...
) {
    // Eigensolve iterations
    for band in 0..nband {
        // Use workspace FFT buffers (no allocation)
        apply_hamiltonian(
            &mut workspace.vunkg_3d,
            &mut workspace.unk_3d,
            &mut workspace.fft_workspace,
            // ...
        );
    }

    // Subspace rotation (reuses Rayleigh matrix)
    subspace_rotate(&mut workspace.rayleigh_evecs, /* ... */);
}
```

---

### Example 3: Shared Cache (Immutable Context)

**Location:** `kscf/src/lib.rs`

**Problem:** Some data is expensive to compute but immutable across iterations (e.g., spherical harmonics, kinetic energy diagonal, FFT index maps).

**Solution:** Use a `SharedCache` as part of the context.

```rust
pub struct KscfSharedCache<'a> {
    ik: usize,
    kgylm: KGYLM,                 // Spherical harmonics (expensive, immutable)
    kin: Vec<f64>,                // Kinetic energy diagonal
    fft_linear_index: Vec<usize>, // FFT G-vector indexing
    vnl_terms: Vec<NonLocalTerm<'a>>,  // Non-local pseudopotential terms
}
```

**For spin calculations:** Share cache between spin-up and spin-down channels to save memory.

```rust
// Spin-up and spin-down share immutable cache
let shared_cache = build_shared_cache(ik, /* ... */);

solve_kpoint_spin_up(&mut workspace_up, &shared_cache, /* ... */);
solve_kpoint_spin_dn(&mut workspace_dn, &shared_cache, /* ... */);
```

---

## Adding Workspaces to New Modules

### Step 1: Identify Hot Paths

Profile your code or use `pw/src/bin/workspace_alloc_trace.rs` to find allocations in hot loops.

```bash
cargo run -p pw --bin workspace_alloc_trace
```

Look for:
- Allocations inside loops (iteration, k-point, band)
- Repeated allocations of same-sized buffers
- Large temporary arrays

### Step 2: Define Workspace Struct

```rust
struct MyModuleWorkspace {
    scratch_a: Vec<c64>,
    scratch_b: Array3<f64>,
    intermediate: Matrix<c64>,
}
```

**Naming convention:**
- Use descriptive names for buffers (`vhg` = Hartree potential in G-space)
- Suffix with dimensionality if helpful (`_3d`, `_g`, `_r`)
- Group related buffers together

### Step 3: Implement Constructor

```rust
impl MyModuleWorkspace {
    pub fn new(npw: usize, fft_shape: [usize; 3], nband: usize) -> Self {
        Self {
            scratch_a: vec![c64::zero(); npw],
            scratch_b: Array3::new(fft_shape),
            intermediate: Matrix::new(nband, nband),
        }
    }

    #[cfg(debug_assertions)]
    pub fn validate(&self, npw: usize, fft_shape: [usize; 3], nband: usize) {
        debug_assert_eq!(self.scratch_a.len(), npw);
        debug_assert_eq!(
            self.scratch_b.as_slice().len(),
            fft_shape[0] * fft_shape[1] * fft_shape[2]
        );
        debug_assert_eq!(self.intermediate.rows(), nband);
    }
}
```

### Step 4: Thread Through Call Stack

Update function signatures to accept `&mut Workspace`:

```rust
// Before
pub fn compute_something(/* ... */) {
    let scratch = vec![0.0; n];  // ❌ Allocation
    // ...
}

// After
pub fn compute_something(workspace: &mut MyModuleWorkspace, /* ... */) {
    workspace.scratch_a.fill(c64::zero());  // ✅ Reuse
    // ...
}
```

### Step 5: Validate Performance

```bash
# Run allocation trace before and after
cargo run -p pw --bin workspace_alloc_trace

# Look for:
# - "0 alloc calls, 0 realloc calls" in hot kernels
# - Steady-state after warmup
```

---

## Performance Validation

### Allocation Tracing

DFTWorks includes an allocation tracer to verify workspace effectiveness:

```bash
cargo run -p pw --bin workspace_alloc_trace
```

**Expected output:**
```
=== Allocation Trace Summary ===
gradient_r3d: 0 alloc calls, 0 realloc calls
gradient_norm_r3d: 0 alloc calls, 0 realloc calls
divergence_r3d: 0 alloc calls, 0 realloc calls
scf_iteration: 0 alloc calls, 0 realloc calls (after warmup)
```

### Validation Checklist

After adding a workspace:

- [ ] Run `cargo check` - compiles without errors
- [ ] Run unit tests - behavior unchanged
- [ ] Run `workspace_alloc_trace` - zero allocations in hot path
- [ ] Run phase12 regression - physics unchanged
- [ ] Profile with `perf` or `valgrind` - memory usage constant

---

## Anti-Patterns

### ❌ Anti-Pattern 1: Growing Vectors in Hot Loops

```rust
// BAD: Vector grows dynamically
for item in items {
    result.push(compute(item));  // Reallocations
}

// GOOD: Pre-allocate
result.clear();
result.reserve(items.len());
for item in items {
    result.push(compute(item));
}

// BETTER: Reuse workspace
workspace.result.clear();
for item in items {
    workspace.result.push(compute(item));
}
```

### ❌ Anti-Pattern 2: Hidden Allocations in Constructors

```rust
// BAD: Allocates inside hot loop
for iter in 0..n_iter {
    let helper = Helper::new(n);  // Hidden Vec allocation
    helper.compute();
}

// GOOD: Workspace owns helper
struct Workspace {
    helper: Helper,
}
```

### ❌ Anti-Pattern 3: Conditional Sizing

```rust
// BAD: Size depends on runtime branch
let scratch = if condition {
    vec![0.0; n]
} else {
    vec![0.0; m]
};

// GOOD: Size to maximum
struct Workspace {
    scratch: Vec<f64>,  // Sized to max(n, m)
}
```

### ❌ Anti-Pattern 4: Forgetting `debug_assert!`

```rust
// BAD: No validation
struct Workspace {
    buffer: Vec<c64>,
}

// GOOD: Validate in debug builds
impl Workspace {
    fn validate(&self, expected_size: usize) {
        debug_assert_eq!(self.buffer.len(), expected_size);
    }
}
```

---

## Summary

### Key Takeaways

1. **Pattern:** `Context` (immutable) + `State` (mutable) + `Workspace` (reusable scratch)
2. **Goal:** Zero allocations in hot loops after warmup
3. **Method:** Pre-allocate all buffers at module/phase initialization
4. **Validation:** Use `debug_assert!` and allocation tracing

### When to Use Workspaces

Use workspaces when:
- ✅ Function is called in a loop (iteration, k-point, band, atom)
- ✅ Temporary buffers are allocated each call
- ✅ Buffer sizes are known at phase initialization
- ✅ Module is performance-critical

Don't use workspaces when:
- ❌ Function is called once per run (initialization, I/O)
- ❌ Buffer size varies unpredictably
- ❌ Code clarity would suffer significantly

### Further Reading

- `scf/src/nonspin.rs` - SCF loop workspace example
- `scf/src/spin.rs` - Spin SCF with channel workspaces
- `kscf/src/lib.rs` - Per-k-point workspace + shared cache
- `density/src/nonspin.rs` - Density kernel workspace
- `eigensolver/src/pcg.rs` - PCG eigensolver workspace
- `pw/src/bin/workspace_alloc_trace.rs` - Allocation tracer

### Questions?

If you're adding a workspace and unsure about the design:
1. Look at similar hot paths in existing code
2. Run the allocation tracer before and after
3. Validate with regression tests
4. Document your workspace in module-level comments
