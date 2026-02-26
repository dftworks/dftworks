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

### E12 - Restart Semantics and Checkpoint Completeness
**Priority**: P1  
**Status**: Completed (2026-02-25)  
**Files**: `pw/src/main.rs`, `control/src/lib.rs`, `dfttypes/src/lib.rs`, `workflow/src/main.rs`

- Honor `restart` as an explicit runtime policy instead of implicit file-existence checks
- Support warm restart for both nonspin and spin paths (density, eigenvectors, occupations, and SCF bookkeeping)
- Validate checkpoint compatibility (`spin_scheme`, lattice, k-mesh, cutoffs, `nband`, schema version) before load
- Replace checkpoint `unwrap()` failure paths with actionable error messages

**Acceptance Criteria**
- `restart=false` never attempts to read checkpoint artifacts
- `restart=true` can resume nonspin and spin runs without manual file edits
- Incompatible checkpoint inputs fail with clear diagnostics and no panic

**Completion Update (2026-02-25)**
- [x] `pw` now treats `restart` as an explicit policy gate for density checkpoint loading
- [x] Spin and nonspin density restart paths are both wired in `pw`
- [x] Wavefunction warm-start loading added with lattice and local basis compatibility checks
- [x] Restart now emits explicit diagnostics when checkpoint files are missing or incompatible
- [x] Replace panic-based checkpoint readers with `Result`-based loaders in `dfttypes`/`matrix`/`ndarray`
- [x] Persist and validate checkpoint metadata (spin mode, cutoffs, k-mesh, `nband`, schema version)
- [x] Validate wavefunction checkpoint metadata across local k-point files before loading vectors

### E13 - Spin SCF MPI and Symmetry Parity
**Priority**: P1  
**Status**: Completed (2026-02-25)  
**Files**: `scf/src/spin.rs`, `scf/src/utils.rs`, `pw/src/main.rs`

- Align spin SCF reduction semantics with nonspin (`band_energy`, force, stress, and derived totals)
- Apply symmetry projection to spin force/stress paths when symmetry is enabled
- Remove rank-noisy default logs in spin path and enforce root-only summaries
- Add spin multi-rank parity tests versus single-rank references

**Acceptance Criteria**
- Spin total energies/forces/stresses are rank-count invariant within tolerance
- Symmetry-enabled spin runs satisfy force/stress invariance checks
- Default spin output volume is comparable to nonspin output volume

**Completion Update (2026-02-25)**
- [x] Spin SCF now reduces and broadcasts distributed contributions consistently (`band_energy`, hybrid exchange, nonlocal force, kinetic/nonlocal stress)
- [x] Symmetry projection is now applied to spin force and stress decomposition terms when symmetry is enabled
- [x] Spin SCF default logging is now root-only with compact per-iteration summaries
- [x] Added spin MPI parity regression harness (`scripts/run_spin_mpi_parity.sh`) comparing 1-rank vs 2-rank energy/force/stress outputs

### E14 - FFT Planning and Spectral-Operator Workspace Tuning
**Priority**: P2  
**Status**: Open  
**Files**: `dwfft3d/src/lib.rs`, `rgtransform/src/lib.rs`, SCF/XC callers

- Remove hard-coded FFT thread count and add controlled runtime selection
- Add FFT plan reuse policy (including optional wisdom persistence where supported)
- Eliminate transient allocations in gradient/divergence kernels with reusable workspace buffers
- Benchmark FFT planning overhead and GGA-heavy SCF loops before/after changes

**Acceptance Criteria**
- FFT thread policy is configurable and documented
- Repeated transforms avoid redundant planning overhead
- Gradient/divergence paths are allocation-free in steady-state profiling

### E15 - Cost-Aware K-Point Scheduling and Spin Cache Deduplication
**Priority**: P2/P3  
**Status**: Open  
**Files**: `kpts_distribution/src/lib.rs`, `pw/src/main.rs`, `kscf/src/lib.rs`

- Replace pure contiguous k-point partitioning with cost-aware scheduling (e.g., `npw * nband` proxy)
- Add optional dynamic scheduling mode for heterogeneous k-point costs
- Share immutable per-k caches between spin-up/down workers to avoid duplicated precompute/memory
- Track per-rank timing imbalance and memory deltas in scaling reports

**Acceptance Criteria**
- MPI rank wall-time imbalance is reduced on asymmetric k-point workloads
- Spin memory footprint drops measurably on representative systems
- Numerical results remain unchanged versus current partitioning

### E16 - Deterministic Initialization and Run Provenance
**Priority**: P2/P4  
**Status**: Open  
**Files**: `utility/src/lib.rs`, `control/src/lib.rs`, `pw/src/main.rs`, `workflow/src/main.rs`

- Add explicit RNG seed control for wavefunction random initialization
- Record full run manifest (seed, git commit, crate features, MPI/rayon settings, input hashes)
- Surface provenance in properties/workflow outputs to support reproducible reruns
- Add replay checks that reject stale/incompatible manifests when requested

**Acceptance Criteria**
- Fixed seed runs are bitwise or numerically stable under fixed runtime settings
- Every run directory contains a machine-readable provenance manifest
- Reproducibility checks can be automated in CI for at least one reference case

### E17 - Scalable Checkpoint I/O and Artifact Schema Governance
**Priority**: P3/P4  
**Status**: Open  
**Files**: `dfttypes/src/lib.rs`, `pw/src/main.rs`, `workflow/src/main.rs`

- Add scalable checkpoint layout options beyond per-k-point small-file patterns
- Support chunking/compression and batched write/read strategies for large runs
- Introduce explicit schema/version metadata for `rho`/`wfc` artifacts
- Provide migration/compatibility checks across schema revisions

**Acceptance Criteria**
- Large-k workloads produce fewer metadata-heavy I/O bottlenecks
- Checkpoint readers reject incompatible schema versions with actionable guidance
- Restart throughput improves on representative multi-rank filesystems

### E18 - Verbosity Policy and Structured Runtime Logging
**Priority**: P3  
**Status**: Open  
**Files**: `control/src/lib.rs`, `pw/src/main.rs`, `scf/`, `kscf/`

- Implement typed verbosity levels (`quiet`, `normal`, `verbose`, `debug`) and enforce them consistently
- Gate high-volume per-band/per-rank diagnostics behind explicit debug modes
- Emit structured iteration timing/metric logs (`jsonl` or CSV) alongside human-readable output
- Ensure logging overhead is measured and bounded in production defaults

**Acceptance Criteria**
- `verbosity` setting materially changes output behavior across modules
- Default mode avoids high-frequency diagnostic flood in large runs
- Structured logs can drive regression tooling without parsing free-form stdout

## Refactoring Review Update (2026-02-25)

Current code review highlights these structural expansion bottlenecks:

- `pw/src/main.rs` is still a monolithic runtime orchestrator (bootstrap, construction, restart, SCF execution, output, and geom loop in one file)
- `scf/src/nonspin.rs` and `scf/src/spin.rs` duplicate most SCF iteration flow and post-SCF force/stress assembly
- `control/src/lib.rs` keeps a large string-key parser with many `unwrap()` and `process::exit` paths, making extension and diagnostics harder
- `dfttypes/src/lib.rs` repeats spin/nonspin checkpoint I/O logic and still uses panic-style failure paths in several save/load branches

New canonical refactoring items are added below to unblock future feature expansion (NCL, richer checkpoint backends, cleaner workflow composition).

### E19 - Unified SCF Iteration Engine (Spin/Nonspin)
**Priority**: P1/P2  
**Status**: Open  
**Files**: `scf/src/nonspin.rs`, `scf/src/spin.rs`, `scf/src/lib.rs`, `scf/src/utils.rs`

- Extract a shared SCF iteration template/state machine (`potential -> eigensolve -> occupations -> density -> energy -> convergence -> mixing`)
- Move spin-specific versus nonspin-specific behavior behind explicit channel adapters rather than duplicated loops
- Centralize convergence bookkeeping and iteration reporting so both paths use the same metrics schema
- Reuse one shared post-SCF reduction/projection pipeline for force/stress parts with channel contributions as inputs

**Acceptance Criteria**
- SCF loop control flow lives in one shared engine with spin/nonspin adapters
- Spin/nonspin convergence output fields are identical by schema
- MPI parity tests remain green for nonspin and spin after consolidation

### E20 - Typed Run Context and Phase Builders
**Priority**: P1/P2  
**Status**: Open  
**Files**: `pw/src/main.rs`, `scf/src/lib.rs`, `kpts_distribution/src/lib.rs`

- Replace repeated construction blocks with typed phase builders (`RuntimeContext`, `GeometryStepContext`, `ElectronicStepContext`)
- Centralize k-point local/global indexing, local basis, and local VNL construction into a reusable domain object
- Remove repeated spin-branch allocation/initialization code for `VKSCF`, `VKEigenValue`, and `VKEigenVector`
- Keep `main` as orchestration only; move restart/checkpoint and SCF setup into dedicated modules

**Acceptance Criteria**
- `pw/src/main.rs` delegates to phase modules and no longer owns low-level construction details
- No ad-hoc `ik - ik_first` indexing arithmetic remains outside k-point domain helpers
- New phases are unit-testable without invoking full end-to-end `pw` flow

### E21 - Declarative Input Schema and Validation Pipeline
**Priority**: P1  
**Status**: Open  
**Files**: `control/src/lib.rs`

- Replace large manual `match` parser with declarative key schema (type parser + unit conversion + validation rule)
- Return `Result<Control, ControlError>` with line-aware diagnostics instead of `unwrap()` / `process::exit`
- Split validation into stages: syntax, semantic ranges, feature-compatibility matrix
- Add parser tests for invalid keys, invalid values, deprecated aliases, and cross-field constraints

**Acceptance Criteria**
- `Control` loading has no `process::exit` and no parsing `unwrap()` in normal flow
- Input error messages include key name and line number
- Parser/validator test suite covers compatibility gates (HSE, Hubbard U, restart, spin modes)

### E22 - Checkpoint Repository and Codec Abstraction
**Priority**: P2/P3  
**Status**: Open  
**Files**: `dfttypes/src/lib.rs`, `pw/src/main.rs`, `workflow/src/main.rs`

- Split checkpoint domain model from storage codec to allow multiple backends (current HDF5 file-per-k, future packed/chunked formats)
- Remove duplicated spin/nonspin load/save loops via shared channel iteration helpers
- Replace remaining `unwrap()`/`expect()` in checkpoint I/O paths with typed errors and context
- Add backward-compatible schema migration hooks beyond a single version check

**Acceptance Criteria**
- Restart logic depends on a repository trait instead of direct filename logic in `pw`
- Save/load branches share common code for spin channel handling
- Checkpoint compatibility and migration behavior is covered by integration tests

### E23 - SCF Utilities Module Decomposition
**Priority**: P2  
**Status**: Open  
**Files**: `scf/src/utils.rs`, `scf/src/nonspin.rs`, `scf/src/spin.rs`

- Split `scf::utils` into focused modules (`potential`, `energy`, `mixing`, `symmetry_projection`, `diagnostics`)
- Move printing/output routines out of numerical kernels to a diagnostics layer with verbosity controls
- Consolidate duplicated helper logic (`get_eigvalue_epsilon`, eigen display helpers, plane-wave max helpers)
- Keep math helpers (`3x3` transforms/projections) in a dedicated linear-algebra utility module

**Acceptance Criteria**
- `scf/src/utils.rs` is replaced by smaller purpose-specific modules
- Numerical kernels do not directly own user-facing I/O formatting
- Shared helper logic has one source of truth across spin/nonspin paths

### E24 - Capability Matrix and Unsupported-Mode Policy
**Priority**: P2/P4  
**Status**: Open  
**Files**: `control/src/lib.rs`, `pw/src/main.rs`, `scf/src/lib.rs`

- Define explicit capability matrix for `{spin_scheme, xc_scheme, task, restart}` combinations
- Validate unsupported combinations before runtime setup; return actionable errors instead of runtime `panic!`
- Keep feature flags/capability tags on SCF drivers to enable incremental NCL rollout without hard panics
- Ensure checkpoint metadata validation also enforces capability compatibility

**Acceptance Criteria**
- Unsupported runtime combinations fail at input validation/preflight phase
- `panic!` for "not implemented" mode combinations is removed from runtime setup paths
- Adding a new mode requires updating one central capability table and tests

### E25 - K-Point Domain Model and Index Safety
**Priority**: P2  
**Status**: Open  
**Files**: `kpts_distribution/src/lib.rs`, `pw/src/main.rs`, `scf/src/spin.rs`, `scf/src/nonspin.rs`

- Introduce explicit `KPointDomain` (`global_index`, `local_slot`, `weight`, `basis_ref`) to avoid index drift bugs
- Eliminate duplicated k-point loops for up/down channel setup and checkpoint file naming
- Provide stable helpers for local/global index transforms used by logging, restart, and parity scripts
- Add assertions and tests that cover uneven rank partitioning and empty-local-k cases

**Acceptance Criteria**
- K-point loops use typed domain iterators instead of manual index math
- Spin/nonspin setup code reuses the same domain traversal utilities
- Parity tests include uneven rank-to-k distributions and zero-local-k ranks

## Legacy to Canonical Mapping

Legacy mapping covers the original normalized set (`E1`-`E11`).  
Newly added items (`E12`-`E25`) are code-review additions without legacy IDs.

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

### Sprint 9 - Restart and Spin-Path Correctness
**Scope**: `E12`, `E13`

- Implement explicit restart-policy behavior and checkpoint compatibility checks
- Bring spin SCF reductions/symmetry handling to parity with nonspin execution

**Exit Gates**
- Spin and nonspin multi-rank parity tests pass
- Restart behavior is deterministic and policy-controlled

### Sprint 10 - FFT and Scheduling Throughput
**Scope**: `E14`, `E15`

- Add configurable FFT planning/threading and spectral workspace reuse
- Introduce cost-aware k-point scheduling and spin cache deduplication

**Exit Gates**
- GGA-heavy and large-k runs show measured throughput gains
- Rank imbalance and memory duplication are reduced in scaling reports

### Sprint 11 - Reproducibility, I/O, and Logging Hardening
**Scope**: `E16`, `E17`, `E18`

- Implement deterministic seed control and provenance manifests
- Upgrade checkpoint schema/versioning and scalable artifact I/O
- Enforce typed verbosity plus structured runtime logs

**Exit Gates**
- End-to-end reproducibility checks pass on reference workflows
- Restart artifacts are schema-validated and backward-compatible
- Production-default logging overhead is bounded and documented

### Sprint 12 - SCF Structural Consolidation
**Scope**: `E19`, `E23`, `E25`

- Introduce shared SCF iteration engine and channel adapters
- Decompose `scf::utils` and unify duplicated helper logic
- Roll out typed k-point domain/index model in SCF setup and execution

**Exit Gates**
- Spin/nonspin loops run through shared orchestration without behavior regression
- K-point indexing is domain-driven and validated under uneven MPI partitions
- MPI parity scripts pass for both spin and nonspin references

### Sprint 13 - Runtime Contracts and Expansion Interfaces
**Scope**: `E20`, `E21`, `E22`, `E24`

- Modularize `pw` into typed phase contexts/builders
- Replace control parser with declarative schema and typed errors
- Introduce checkpoint repository abstraction and codec split
- Add centralized capability matrix to replace runtime "not implemented" panics

**Exit Gates**
- `pw` orchestration is phase-modular and testable at unit scope
- Input/feature compatibility failures return actionable diagnostics without process abort in libraries
- Restart/checkpoint APIs are backend-ready and schema-governed for future formats

## Feature Track (Post-Core Stabilization)

Run these after Sprints 1-11 establish a stable core execution framework.

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

### F9 - Automated Convergence Campaigns
- [ ] Add workflow mode for automatic `ecut`, `kmesh`, `nband`, and smearing convergence sweeps
- [ ] Support stopping criteria based on user tolerances (energy, force, stress, gap, DOS stability)
- [ ] Reuse checkpoint/restart to skip already-converged sample points
- [ ] Export machine-readable recommendation report and selected final inputs

**Deliverables**
- `convergence_report.json`
- `recommended_in.ctrl` and `recommended_in.kmesh`

## Cross-Cutting Rules

Apply to every sprint and feature milestone:

- Keep hot loops allocation-free via workspace APIs
- Keep runtime mode selection typed and parse-time validated
- Keep reductions deterministic under thread and MPI execution
- Keep library crates `Result`-based (no direct process termination)
- Keep schema changes versioned and backward-compatible
- Keep restart/checkpoint compatibility explicitly validated before resume
- Keep run provenance complete enough for reproducibility audits

## Execution Rule

Finish an item only when:

- Unit and integration tests pass
- One reference example is documented
- Runtime and memory metrics are recorded
- Output schema compatibility is preserved
