# DFTWorks Optimization and Feature Roadmap

This document consolidates all optimization, architecture, and feature expansion work into a single prioritized backlog.

---

## Completed Work Summary

### Infrastructure
- [x] Phase 1/2 NCL roadmap (PBE kernel on collinear nonspin/spin, enum-driven spin_scheme)
- [x] Regression artifacts for phase 1/2 (`test_example/si-oncv/regression/`)

### Physics Features (Levels 1-4)
- [x] Stable property output layout (`runs/<stage>/properties/`) with machine-readable files
- [x] Property postprocessing module (`property` crate, 1398 lines)
- [x] Timing and memory logging for SCF/NSCF stages
- [x] Total DOS from NSCF eigenvalues with configurable smearing
- [x] Automatic band gap analysis (direct/indirect, VBM/CBM k-point indices)
- [x] Fermi-level consistency checks between SCF and postprocessing
- [x] `symops` crate with shared symmetry-operation representation
- [x] Crystal space-group detection and operation extraction
- [x] k-point little-group / star construction utilities
- [x] Machine-readable symmetry metadata export
- [x] PDOS (atom/orbital projected DOS) using projection weights
- [x] Fat-band output (band structure with projection weights)
- [x] Cached projection matrices across repeated analyses
- [x] HSE06 screened hybrid functional (gamma-only MVP)
- [x] DFT+U Dudarev MVP with projector-based Hamiltonian term
- [x] Wannier90 interface (`w90-win`, `w90-amn`, `w90-proj`)

---

## P1 - Build Portability and Correctness

### 1. Remove hard-coded linker paths from build scripts
**Status**: Open
**Files**: `matrix/build.rs`, `symmetry/build.rs`

- Problem: Hard-coded local library paths break cross-machine builds
- Plan:
  - Use environment-driven discovery (`LAPACK_DIR`) or `pkg-config`
  - Add explicit build-time diagnostics if libraries not found
  - Keep optional per-platform fallback logic gated behind env flags
- Impact: Portable builds across machines and CI runners

---

## P2 - Performance and Hot-Path Optimization

### 2. Introduce reusable workspace objects for hot SCF loops
**Status**: Open
**Files**: `scf/`, `kscf/`, `density/`

- Problem: Repeated per-iteration allocations in SCF/KSCF density and Hamiltonian paths
- Plan:
  - Add `KscfWorkspace` and `DensityWorkspace` structs holding reusable buffers
  - Allocate once per geometry/SCF stage, pass mutable references through loops
  - Keep workspace sizing explicit and validated
- Impact: Lower allocator pressure and better runtime stability for large systems

### 3. Remove artificial delay from eigenvalue output path
**Status**: Open
**Files**: `pw/src/main.rs`

- Problem: Rank-serialized output includes fixed sleeps and barriers
- Plan:
  - Gate verbose eigenvalue printing by verbosity/debug option
  - Remove fixed sleep in production path
  - Keep optional ordered-rank debug print mode for troubleshooting
- Impact: Shorter wall time and cleaner MPI behavior

### 4. Reuse eigensolver scratch memory
**Status**: Open
**Files**: `eigensolver/src/pcg.rs`

- Problem: Repeatedly allocates projection/temp vectors in inner loops
- Plan:
  - Move temporary vectors into solver state, clear/reuse per band iteration
  - Avoid short-lived `Vec` creation in Gram-Schmidt and lower-band orthogonalization
- Impact: Reduced overhead in heavy diagonalization steps

### 5. Expand workspace reuse to all hot compute paths
**Status**: Open
**Files**: `density/`, `eigensolver/`, `force/`, `stress/`, `dwfft3d/`

- Problem: Allocations remain in density/eigensolver and FFT-heavy paths
- Plan:
  - Extend workspace pattern to density, eigensolver, force, stress, FFT helpers
  - Add clear ownership/lifetime strategy for work buffers
  - Track allocation counts in profiling to verify gains
- Impact: Lower allocation churn and improved scaling for large systems

---

## P2/P3 - API and Error Handling

### 6. Remove `process::exit` from library crates
**Status**: Open (9 files affected)
**Files**: `control/src/lib.rs`, `kpts/src/line.rs`, `special/src/lib.rs`, plus wannier90 binaries

- Problem: Library-level code terminates the process directly
- Plan:
  - Return typed errors (`Result<_, Error>`) from library functions
  - Centralize exit behavior in binary entry points only
  - Add context-rich error propagation (`thiserror`/custom error enums)
- Impact: Better composability, testability, and embedding behavior

### 7. Complete framework-wide Result/error model
**Status**: Open

- Problem: Error policy inconsistent between libraries and binaries
- Plan:
  - Enforce `Result`-based error returns for framework/library crates
  - Keep process termination policy only in executable entry points
  - Standardize error taxonomy and propagation format
- Impact: Better composability and cleaner integration surface

---

## P3 - Maintainability and Code Quality

### 8. Refactor oversized orchestration interfaces
**Status**: Open
**Files**: `pw/src/main.rs`, `scf/src/lib.rs`

- Problem: Large parameter lists (e.g., SCF trait `run`) and long monolithic `main`
- Plan:
  - Introduce grouped context structs (`ScfContext`, `RuntimeContext`, `PostprocessContext`)
  - Split `pw/src/main.rs` into phase-oriented functions/modules:
    - input/bootstrap
    - basis/construction
    - SCF execution
    - outputs/postprocessing
  - Keep interfaces explicit and immutable where possible
- Impact: Easier feature evolution and lower regression risk

### 9. Reduce blanket warning suppression
**Status**: Open (36 files affected)
**Files**: Widespread `#![allow(warnings)]` usage

- Problem: Blanket warning suppression hides potential issues
- Plan:
  - Remove blanket suppression incrementally
  - Replace with narrow `#[allow(...)]` only where justified
  - Fix underlying warnings where appropriate
- Impact: Higher code quality signal and safer iteration

### 10. Expand integration test coverage
**Status**: Open
**Files**: `scf/`, `kscf/`, `pw/`

- Problem: Limited tests in SCF/KSCF/PW paths
- Plan:
  - Add integration tests for:
    - SCF convergence on small benchmark systems
    - Energy component consistency checks
    - Deterministic behavior under fixed seeds/settings
  - Add CI gates for `cargo check` + selected tests
- Impact: Higher confidence during refactors

---

## P4 - Future Physics Features

### Level 5: Equation-of-State and Thermodynamics

- [ ] Add automated volume-scan workflow (`-6%` to `+6%`)
- [ ] Fit Birch-Murnaghan EOS and report `V0`, `B0`, `B0'`, `E0`
- [ ] Add static lattice enthalpy vs pressure output
- [ ] Parallelize independent volume points with restart support
- [ ] Add regression case comparing fitted constants to reference range

**Deliverable**: `eos_fit.json` + plot-ready table for energy-volume and pressure-volume curves

### Level 6: Elastic Properties

- [ ] Implement finite-strain generation and stress collection
- [ ] Fit elastic tensor `Cij` (symmetry-aware where available)
- [ ] Compute derived moduli (Voigt/Reuss/Hill bulk and shear, Young's modulus, Poisson ratio)
- [ ] Use symmetry to reduce number of strain calculations
- [ ] Add quality checks (tensor symmetry, mechanical stability criteria)

**Deliverable**: `elastic_tensor.json` and `elastic_summary.md`

### Level 7: Vibrational Properties (Finite Displacement)

- [ ] Implement supercell builder and displacement patterns
- [ ] Compute force constants and dynamical matrices
- [ ] Compute phonon dispersion and phonon DOS
- [ ] Add acoustic sum-rule enforcement and imaginary-mode detection
- [ ] Add convergence workflow for supercell size and displacement amplitude

**Deliverable**: `phonon_bands.dat`, `phonon_dos.dat`, and stability summary

### Level 8: Electric/Optical Response

- [ ] Implement Berry-phase polarization (non-metal cases first)
- [ ] Implement dielectric tensor (finite-field or finite-difference strategy)
- [ ] Implement Born effective charges
- [ ] Add IR-active mode intensities from phonons + Born charges
- [ ] Add symmetry checks for tensor forms and coordinate conventions

**Deliverable**: `polarization.json`, `dielectric_tensor.json`, `born_charges.json`

### Level 9: Advanced Transport and Topological Responses

- [ ] Add band interpolation path (Wannier-based) for dense k-space properties
- [ ] Implement Boltzmann transport (conductivity/Seebeck vs temperature and chemical potential)
- [ ] Implement anomalous Hall/Berry-curvature integration workflows
- [ ] Add scalable parallel execution for dense k/q sampling with deterministic reductions
- [ ] Add benchmark suite for performance and reproducibility on large systems

**Deliverable**: `transport_*.json` and optional `berry_curvature_*.dat` datasets

### NCL Extension (Phase 3+)

- [ ] Implement spinor wavefunction and magnetization-density data model for true NCL physics
- [ ] Extend SCF pipeline for NCL Hamiltonian assembly, mixing, and occupation handling
- [ ] Add NCL-specific XC integration and validation benchmarks
- [ ] Define parity and reference targets for NCL vs trusted external solvers

### Full Symmetry Support

- [ ] Build centralized symmetry context/service from internal symmetry output
- [ ] Use symmetry to reduce k-point workloads (irreducible k-mesh + proper weights)
- [ ] Apply symmetry operations consistently to charge density, potential, forces, stress
- [ ] Add validation tests:
  - Symmetric structure invariants
  - Force cancellation in high-symmetry systems
  - Agreement between full-mesh and irreducible-mesh energies

### HSE06 Extension

- [ ] Extend HSE06 beyond gamma-only to general k-point meshes
- [ ] Add hybrid exchange contribution to force/stress calculations
- [ ] Add benchmark comparison with reference implementations

---

## P5 - Rust Scalability Patterns

### 11. Replace string-based runtime modes with typed enums
**Status**: Open
**Files**: `control/`, `scf/`, `smearing/`, `density/`, `kpts/`

- Problem: String matching allows invalid states and runtime fallback behavior
- Plan:
  - Introduce typed configuration enums parsed once at input load time
  - Use exhaustive `match` on enums across constructors and drivers
  - Remove repeated string comparisons in runtime paths
- Impact: Better compile-time safety and cleaner extension path

### 12. Prefer static dispatch in hot kernels
**Status**: Open
**Files**: `scf/`, `eigensolver/`, `kscf/`

- Problem: Dynamic dispatch costly/opaque in performance-critical inner loops
- Plan:
  - Keep trait objects at high-level orchestration boundaries
  - Use enums/generics for hot compute kernels where implementations known at compile time
  - Measure before/after in SCF and eigensolver-heavy kernels
- Impact: Better inlining and reduced dispatch overhead

### 13. Standardize Context + State + Workspace pattern
**Status**: Open

- Problem: Pattern partially used but not standardized globally
- Plan:
  - Expand into shared convention for SCF, density, force, stress, and geometry drivers
  - Document API template:
    - context = immutable problem data
    - state = iteration-evolving data
    - workspace = reusable scratch memory
  - Refactor modules incrementally to same shape
- Impact: Easier feature scaling and lower coupling

### 14. Parallelize by k-point with deterministic reductions
**Status**: Open

- Problem: Mostly serial in key sections, some output/reduction paths rank-serialized
- Plan:
  - Parallelize independent k-point work using thread-level parallel iterators
  - Preserve deterministic reduction order for reproducible results
  - Keep MPI + thread interplay explicit and benchmarked
- Impact: Stronger scaling on multicore nodes with numerical reproducibility

---

## P6 - Framework Modernization

### 15. Build typed configuration framework layer
**Status**: Open
**Files**: `control/src/lib.rs`, `scf/src/lib.rs`, `smearing/src/lib.rs`, `density/src/lib.rs`, `kpts/src/lib.rs`

- Problem: Runtime behavior driven by string selectors in multiple modules
- Plan:
  - Add single typed config layer parsed once in `control`, passed as typed enums/structs
  - Remove string-based branching from runtime drivers
- Impact: Cleaner framework contracts and safer feature extension

### 16. Redesign driver contracts using Context + State + Workspace
**Status**: Open

- Problem: Orchestration hard to scale with large argument lists and monolithic `main`
- Plan:
  - Introduce framework-level types:
    - `ScfContext`, `ScfState`, `ScfWorkspace`
    - Similar patterns for density/force/stress
  - Split orchestration in `pw/src/main.rs` into phase modules
- Impact: Lower coupling, easier testing, easier onboarding

### 17. Define scalable execution framework
**Status**: Open

- Problem: Current execution not organized as explicit scalable engine
- Plan:
  - Add k-point execution layer supporting thread parallelism and deterministic global reductions
  - Ensure compatibility with MPI decomposition and reproducible summation rules
  - Integrate with existing SCF utilities and KSCF kernels
- Impact: Better multicore scaling and predictable numerical behavior

### 18. Adopt benchmark-driven optimization workflow
**Status**: Open

- Problem: Performance work not consistently guarded by benchmark baselines
- Plan:
  - Add benchmark harnesses (Criterion-based microbench + end-to-end SCF timing cases)
  - Track scaling dimensions: atoms, k-points, thread count
  - Add performance regression checks in CI for representative workloads
- Impact: Data-driven optimization decisions and long-term performance stability

### 19. Formalize framework-level validation
**Status**: Open

- Problem: No unified framework gate for runtime scalability and physics regressions
- Plan:
  - Add framework benchmark and regression suite:
    - End-to-end SCF timing matrix (atoms, k-points, threads)
    - Convergence and energy-consistency checks
    - Reproducibility checks under fixed seed/settings
  - Hook into CI as required status checks
- Impact: Confident framework evolution with measurable performance targets

---

## Cross-Cutting Concerns (Apply at Every Level)

- [ ] Keep hot loops allocation-free via context/state/workspace patterns
- [ ] Replace string-dispatch runtime modes with typed enums in property pipelines
- [ ] Add deterministic reduction utilities for threaded/MPI aggregation
- [ ] Add property-level profiling harnesses and CI regression thresholds
- [ ] Centralize error handling with `Result`-based library APIs (no `process::exit` in libs)

---

## Recommended Execution Order

### Phase A: Correctness and Build Portability
1. Item 1 (hard-coded linker paths)

### Phase B: Hot-Path Memory Reuse
2. Items 2-5 (workspace objects, eigensolver scratch, FFT buffers)

### Phase C: API and Error Refactor
3. Items 6-7 (remove `process::exit`, Result-based errors)
4. Item 8 (refactor orchestration interfaces)

### Phase D: Code Quality and Testing
5. Items 9-10 (warning suppression, integration tests)

### Phase E: Rust Scalability Patterns
6. Items 11-14 (typed enums, static dispatch, Context+State+Workspace, k-point parallelism)

### Phase F: Framework Modernization
7. Items 15-19 (typed config, driver contracts, execution framework, benchmarks)

### Phase G: Physics Features
8. Levels 5-9 (EOS, elastic, phonons, electric/optical, transport)
9. NCL extension, full symmetry, HSE06 extension

---

## Delivery Strategy

Each phase should include:
- Implementation tasks
- Benchmark/validation checks
- Rollback-safe commits

---

## Execution Rule

Finish one level/item only when:
- Unit/integration tests pass
- One reference example is documented
- Runtime and memory metrics are recorded
- Output schema is versioned and backward-compatible
