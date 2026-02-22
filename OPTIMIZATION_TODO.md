# DFTWorks Optimization and Feature Roadmap (Normalized)

This document is the deduplicated execution backlog for optimization, architecture, and feature expansion.

## Completed Work Summary

### Infrastructure
- [x] Phase 1/2 NCL roadmap (PBE kernel on collinear nonspin/spin, enum-driven spin_scheme)
- [x] Regression artifacts for phase 1/2 (`test_example/si-oncv/regression/`)

### Physics Features (Levels 1-4)
- [x] Stable property output layout (`runs/<stage>/properties/`) with machine-readable files
- [x] Property postprocessing module (`property` crate)
- [x] Timing and memory logging for SCF/NSCF stages
- [x] Total DOS from NSCF eigenvalues with configurable smearing
- [x] Automatic band gap analysis (direct/indirect, VBM/CBM k-point indices)
- [x] Fermi-level consistency checks between SCF and postprocessing
- [x] `symops` crate with shared symmetry-operation representation
- [x] Crystal space-group detection and operation extraction
- [x] k-point little-group and star construction utilities
- [x] Machine-readable symmetry metadata export
- [x] PDOS (atom/orbital projected DOS) using projection weights
- [x] Fat-band output (band structure with projection weights)
- [x] Cached projection matrices across repeated analyses
- [x] HSE06 screened hybrid functional (gamma-only MVP)
- [x] DFT+U Dudarev MVP with projector-based Hamiltonian term
- [x] Wannier90 interface (`w90-win`, `w90-amn`, `w90-proj`)

## Normalization Notes (2026-02-22)

The previous list contained intentional overlap to capture related themes. The items below are merged into canonical units:

- `2 + 4 + 5 + 13 + 16` -> `E2` (Workspace architecture and reusable buffers)
- `6 + 7` -> `E4` (Result-based error model and process boundary)
- `11 + 15` -> `E5` (Typed configuration and mode safety)
- `14 + 17` -> `E6` (Scalable k-point execution and deterministic reductions)
- `18 + 19` -> `E11` (Benchmark and validation framework)
- `8` remains standalone as `E7` (orchestration modularization)
- `3` remains standalone as `E3` (quick runtime cleanup)
- `9`, `10`, `12` remain standalone as `E9`, `E10`, `E8`

## Canonical Engineering Backlog (Core)

### E1 - Build Portability and Correctness
**Priority**: P1  
**Status**: Open  
**Files**: `matrix/build.rs`, `symmetry/build.rs`

- Replace hard-coded local linker paths with environment-driven discovery (`LAPACK_DIR`) or `pkg-config`
- Add explicit build-time diagnostics when required libraries are missing
- Keep optional platform-specific fallback only behind explicit env flags

**Acceptance Criteria**
- `cargo check` works on at least two different machines without local path edits
- Build scripts print actionable failure messages when dependencies are missing

### E2 - Workspace Architecture for Hot Paths
**Priority**: P2  
**Status**: Open  
**Files**: `scf/`, `kscf/`, `density/`, `eigensolver/`, `force/`, `stress/`, `dwfft3d/`, `pw/src/main.rs`

- Standardize `Context + State + Workspace` contracts across SCF and related kernels
- Introduce reusable workspace structs (`ScfWorkspace`, `KscfWorkspace`, `DensityWorkspace`, solver scratch)
- Remove per-iteration and per-band short-lived allocations in hot loops
- Keep buffer sizing explicit and validated at stage construction time

**Acceptance Criteria**
- Allocation-heavy hot loops are allocation-free in profiling traces
- Workspace APIs are documented and adopted in SCF and eigensolver paths
- No behavior regressions in reference SCF cases

### E3 - Remove Serialized Eigenvalue Output Delay
**Priority**: P2  
**Status**: Open  
**Files**: `pw/src/main.rs`

- Remove fixed sleeps from production output paths
- Gate ordered-rank debug printing behind explicit verbose/debug flags
- Keep rank ordering support only for debugging workflows

**Acceptance Criteria**
- No fixed sleep calls in production paths
- Wall-time reduction is measured on at least one multi-rank case

### E4 - Result-Based Error Model and Process Boundary
**Priority**: P2/P3  
**Status**: Open  
**Files**: `control/src/lib.rs`, `kpts/src/line.rs`, `special/src/lib.rs`, Wannier90 binaries and related callers

- Remove `process::exit` usage from library crates
- Return typed `Result<_, Error>` from library-level APIs
- Centralize process termination policy in binary entry points only

**Acceptance Criteria**
- No `process::exit` in library crates
- Library APIs propagate typed errors with context
- CLI binaries keep user-friendly exit behavior

### E5 - Typed Configuration and Runtime Mode Safety
**Priority**: P3/P5  
**Status**: Open  
**Files**: `control/`, `scf/`, `smearing/`, `density/`, `kpts/`

- Parse runtime modes and options once into typed enums/structs
- Remove repeated string-based branching in runtime drivers
- Use exhaustive `match` paths to prevent invalid runtime mode states

**Acceptance Criteria**
- String mode dispatch removed from runtime hot paths
- Invalid mode configurations fail at parse/validation time

### E6 - Scalable K-Point Execution with Deterministic Reductions
**Priority**: P5/P6  
**Status**: Open  
**Files**: `scf/`, `kscf/`, orchestration and reduction utilities

- Add explicit execution layer for thread-parallel k-point evaluation
- Preserve deterministic reduction order for reproducibility
- Define MPI/thread interaction policy and reproducibility guarantees

**Acceptance Criteria**
- Thread-level k-point parallel execution is available behind config
- Repeated runs with fixed settings are numerically reproducible
- Scaling measured versus serial baseline

### E7 - Orchestration Modularization
**Priority**: P3/P6  
**Status**: Open  
**Files**: `pw/src/main.rs`, `scf/src/lib.rs`

- Split monolithic orchestration into phase modules:
  - input/bootstrap
  - basis/construction
  - SCF execution
  - outputs/postprocessing
- Replace oversized argument lists with grouped context structs

**Acceptance Criteria**
- `pw/src/main.rs` reduced to high-level orchestration flow
- Phase modules own their scoped logic and interfaces

### E8 - Prefer Static Dispatch in Hot Kernels
**Priority**: P5  
**Status**: Open  
**Files**: `scf/`, `eigensolver/`, `kscf/`

- Keep trait objects at orchestration boundaries only
- Use enums/generics in kernels where implementations are known at compile time
- Measure before/after runtime for SCF and eigensolver-heavy cases

**Acceptance Criteria**
- Kernel dispatch hotspots converted to static dispatch where valid
- Measured performance is neutral or better

### E9 - Warning Policy Cleanup
**Priority**: P3  
**Status**: Open  
**Files**: Widespread `#![allow(warnings)]` usage

- Remove blanket warning suppression incrementally
- Keep only narrow, justified `#[allow(...)]` annotations
- Fix underlying warnings where practical

**Acceptance Criteria**
- Blanket `#![allow(warnings)]` removed from active core modules
- CI warning signal is meaningful

### E10 - Integration Tests and CI Gates
**Priority**: P3  
**Status**: Open  
**Files**: `scf/`, `kscf/`, `pw/`, CI configuration

- Add integration coverage for convergence, energy consistency, and determinism
- Add CI gates for `cargo check` and selected integration tests
- Include at least one reference benchmark system per major workflow

**Acceptance Criteria**
- CI runs selected SCF/KSCF/PW integration tests on each PR
- Deterministic test cases pass under fixed settings

### E11 - Benchmark and Validation Framework
**Priority**: P6  
**Status**: Open  
**Files**: benchmark harnesses, CI perf jobs, regression suite

- Add microbenchmarks (Criterion) and end-to-end SCF timing harnesses
- Track scaling versus atoms, k-points, and thread count
- Add performance regression checks and physics-consistency validation jobs

**Acceptance Criteria**
- Baseline performance dashboard exists for representative workloads
- CI detects meaningful performance regressions and physics regressions

## Legacy to Canonical Mapping

| Legacy Item | Canonical Item |
| --- | --- |
| 1 | E1 |
| 2 | E2 |
| 3 | E3 |
| 4 | E2 |
| 5 | E2 |
| 6 | E4 |
| 7 | E4 |
| 8 | E7 |
| 9 | E9 |
| 10 | E10 |
| 11 | E5 |
| 12 | E8 |
| 13 | E2 |
| 14 | E6 |
| 15 | E5 |
| 16 | E2 |
| 17 | E6 |
| 18 | E11 |
| 19 | E11 |

## Sprint Plan (Sprint-Ready)

**Assumptions**
- Sprint duration: 1 week
- Team focus: complete in-order unless a blocker requires parallel execution
- Done criteria for each sprint: tests pass, one reference example documented, runtime and memory metrics recorded

### Sprint 1 - Build and Quick Runtime Wins
**Scope**: `E1`, `E3`

- Remove hard-coded linker paths and add robust diagnostics
- Remove fixed eigenvalue output sleep path and gate debug rank-printing

**Exit Gates**
- Cross-machine build check complete
- Measured wall-time improvement on one MPI case

### Sprint 2 - Error Model Foundation
**Scope**: `E4`

- Remove library-level `process::exit`
- Implement typed error propagation and binary-only exit policy

**Exit Gates**
- Libraries return `Result` with context
- CLI behavior remains user-friendly and stable

### Sprint 3 - Typed Config Foundation
**Scope**: `E5`

- Introduce typed config parse layer in `control`
- Migrate runtime string selectors in core modules

**Exit Gates**
- Invalid modes rejected at parse time
- Runtime no longer depends on repeated string dispatch

### Sprint 4 - Workspace Phase 1
**Scope**: `E2` (SCF, KSCF, density core)

- Introduce workspace types and API template
- Eliminate key per-iteration allocations in SCF loop

**Exit Gates**
- Allocation profile improves in SCF hot loops
- SCF reference outputs match baseline tolerances

### Sprint 5 - Workspace Phase 2 and Orchestration Split
**Scope**: `E2` (eigensolver/force/stress/FFT), `E7`

- Extend workspace pattern to remaining hot modules
- Split `pw` orchestration into phase modules and context structs

**Exit Gates**
- `pw` flow is phase-modular
- Eigensolver and FFT paths avoid repeated scratch allocation

### Sprint 6 - Parallel Execution and Dispatch Tuning
**Scope**: `E6`, `E8`

- Implement deterministic threaded k-point execution layer
- Shift hot kernel dispatch to static dispatch where valid

**Exit Gates**
- Reproducibility checks pass under fixed settings
- Scaling and kernel timing data captured

### Sprint 7 - Quality and Test Hardening
**Scope**: `E9`, `E10`

- Remove blanket warning suppressions in active core modules
- Add integration tests and CI gates for SCF/KSCF/PW flows

**Exit Gates**
- Warnings are actionable (no blanket suppression in core)
- Integration suite passes in CI

### Sprint 8 - Benchmark and Validation Platform
**Scope**: `E11`

- Add benchmark matrix and regression policy in CI
- Finalize physics consistency and reproducibility checks

**Exit Gates**
- Performance baseline stored and compared in CI
- Regression jobs protect runtime and physics quality

## Feature Track (Post-Core Stabilization)

Run these after Sprints 1-8 establish a stable core execution framework.

### F1 - Equation of State and Thermodynamics
- [ ] Add automated volume-scan workflow (`-6%` to `+6%`)
- [ ] Fit Birch-Murnaghan EOS and report `V0`, `B0`, `B0'`, `E0`
- [ ] Add static lattice enthalpy versus pressure output
- [ ] Parallelize independent volume points with restart support
- [ ] Add regression case against reference range

**Deliverables**
- `eos_fit.json`
- Plot-ready energy-volume and pressure-volume tables

### F2 - Elastic Properties
- [ ] Implement finite-strain generation and stress collection
- [ ] Fit elastic tensor `Cij` (symmetry-aware where available)
- [ ] Compute Voigt/Reuss/Hill derived moduli
- [ ] Use symmetry to reduce required strain calculations
- [ ] Add tensor-symmetry and mechanical-stability checks

**Deliverables**
- `elastic_tensor.json`
- `elastic_summary.md`

### F3 - Vibrational Properties (Finite Displacement)
- [ ] Implement supercell builder and displacement patterns
- [ ] Compute force constants and dynamical matrices
- [ ] Compute phonon dispersion and phonon DOS
- [ ] Add acoustic sum-rule enforcement and imaginary-mode detection
- [ ] Add convergence workflow for supercell size and displacement amplitude

**Deliverables**
- `phonon_bands.dat`
- `phonon_dos.dat`
- Stability summary

### F4 - Electric and Optical Response
- [ ] Implement Berry-phase polarization (non-metal first)
- [ ] Implement dielectric tensor (finite-field or finite-difference)
- [ ] Implement Born effective charges
- [ ] Add IR-active mode intensities from phonons and Born charges
- [ ] Add symmetry checks for tensor forms and coordinate conventions

**Deliverables**
- `polarization.json`
- `dielectric_tensor.json`
- `born_charges.json`

### F5 - Advanced Transport and Topological Responses
- [ ] Add Wannier-based band interpolation for dense k-space properties
- [ ] Implement Boltzmann transport (`conductivity`, `Seebeck`) versus temperature and chemical potential
- [ ] Implement anomalous Hall and Berry-curvature integration workflows
- [ ] Add scalable parallel execution for dense k/q sampling with deterministic reductions
- [ ] Add benchmark suite for performance and reproducibility on large systems

**Deliverables**
- `transport_*.json`
- Optional `berry_curvature_*.dat`

### F6 - NCL Extension (Phase 3+)
- [ ] Implement spinor wavefunction and magnetization-density model for true NCL
- [ ] Extend SCF pipeline for NCL Hamiltonian assembly, mixing, and occupation handling
- [ ] Add NCL-specific XC integration and validation benchmarks
- [ ] Define parity and reference targets against trusted external solvers

### F7 - Full Symmetry Support
- [ ] Build centralized symmetry context/service from internal symmetry output
- [ ] Use symmetry to reduce k-point workloads (irreducible mesh and weights)
- [ ] Apply symmetry operations consistently to density, potential, force, and stress
- [ ] Add validation tests for invariants and full-mesh versus irreducible-mesh agreement

### F8 - HSE06 Extension
- [ ] Extend HSE06 beyond gamma-only to general k-point meshes
- [ ] Add hybrid exchange contribution to force and stress
- [ ] Add benchmark comparison against reference implementations

## Cross-Cutting Rules

Apply to every sprint and feature milestone:

- Keep hot loops allocation-free via workspace APIs
- Keep runtime mode selection typed and parse-time validated
- Keep reductions deterministic under thread and MPI execution
- Keep library crates `Result`-based (no direct process termination)
- Keep schema changes versioned and backward-compatible

## Execution Rule

Finish an item only when:

- Unit and integration tests pass
- One reference example is documented
- Runtime and memory metrics are recorded
- Output schema compatibility is preserved
