# DFTWorks Optimization and Feature Roadmap (Normalized)

This document is the deduplicated execution backlog for optimization, architecture, and feature expansion.

## Validation Policy (Effective 2026-02-26)

- For every major code change, correctness must be validated in Docker before commit/push.
- Baseline gate: run `scripts/run_phase12_regression.sh` inside `rust-dev` Docker (`FORCE_BUILD=1`).
- If spin/MPI behavior is touched, also run `scripts/run_spin_mpi_parity.sh` in Docker.

## Engineering Simplicity Rule (Effective 2026-02-28)

- Rule: "don't overengineer." Prefer direct phase-oriented code paths over wrapper-on-wrapper abstractions.
- Add a new wrapper/context layer only when it removes clear duplication or encodes reusable behavior with tests.
- If a helper only forwards arguments without adding logic, inline/remove it in the next refactor pass.

## Priority Levels (Revised 2026-02-28)

- **P0**: Blockers - prevent team expansion or cause runtime failures
- **P1**: Critical - enable core functionality or prevent correctness issues
- **P2**: High Value - significant performance, maintainability, or feature gaps
- **P3**: Medium Value - code quality, developer experience, or nice-to-have features
- **P4**: Low Value - polish, optimization, or advanced features
- **P5**: Future - long-term improvements or research directions

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

Completed/done engineering tasks are kept here. Uncompleted tasks are moved to the end of this file.

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

## Refactoring Review Update (2026-02-25)

Current code review highlights these structural expansion bottlenecks:

- `pw/src/main.rs` is still a monolithic runtime orchestrator (bootstrap, construction, restart, SCF execution, output, and geom loop in one file)
- `scf/src/nonspin.rs` and `scf/src/spin.rs` duplicate most SCF iteration flow and post-SCF force/stress assembly
- `control/src/lib.rs` keeps a large string-key parser with many `unwrap()` and `process::exit` paths, making extension and diagnostics harder
- `dfttypes/src/lib.rs` repeats spin/nonspin checkpoint I/O logic and still uses panic-style failure paths in several save/load branches

New canonical refactoring items are added below to unblock future feature expansion (NCL, richer checkpoint backends, cleaner workflow composition).

### E19 - Unified SCF Iteration Engine (Spin/Nonspin)
**Priority**: P1/P2  
**Status**: Done (Spin/Nonspin)  
**Files**: `scf/src/nonspin.rs`, `scf/src/spin.rs`, `scf/src/lib.rs`, `scf/src/utils.rs`

- Extract a shared SCF iteration template/state machine (`potential -> eigensolve -> occupations -> density -> energy -> convergence -> mixing`)
- Move spin-specific versus nonspin-specific behavior behind explicit channel adapters rather than duplicated loops
- Centralize convergence bookkeeping and iteration reporting so both paths use the same metrics schema
- Reuse one shared post-SCF reduction/projection pipeline for force/stress parts with channel contributions as inputs

**Implementation Update (2026-02-26)**
- [x] Added shared SCF iteration engine module (`scf/src/engine.rs`) with unified stage ordering and convergence/report output schema
- [x] Replaced duplicated nonspin/spin loop control flow with explicit channel adapters (`NonSpinIterationAdapter`, `SpinIterationAdapter`)
- [x] Consolidated post-SCF force/stress finalize+projection path into shared helpers in `scf/src/utils.rs`
- [x] Validated via Docker correctness gates (`run_phase12_regression.sh`, `run_spin_mpi_parity.sh`)

**Acceptance Criteria**
- SCF loop control flow lives in one shared engine with spin/nonspin adapters
- Spin/nonspin convergence output fields are identical by schema
- MPI parity tests remain green for nonspin and spin after consolidation

### E20 - Typed Run Context and Phase Builders
**Priority**: P1/P2  
**Status**: Done  
**Files**: `pw/src/main.rs`, `scf/src/lib.rs`, `kpts_distribution/src/lib.rs`

- Replace repeated construction blocks with typed phase builders (`RuntimeContext`, `GeometryStepContext`, `ElectronicStepContext`)
- Centralize k-point local/global indexing, local basis, and local VNL construction into a reusable domain object
- Remove repeated spin-branch allocation/initialization code for `VKSCF`, `VKEigenValue`, and `VKEigenVector`
- Keep `main` as orchestration only; move restart/checkpoint and SCF setup into dedicated modules

**Implementation Update (2026-02-26)**
- [x] Added typed runtime/electronic phase setup in `pw` via `RuntimeContext` + `ElectronicStepContext`
- [x] Added typed local k-point domain model (`KPointDomain`, `KPointSlot`) in `kpts_distribution`
- [x] Removed ad-hoc `ik - ik_first` indexing from SCF setup by using domain `global/local` slot mapping
- [x] Consolidated spin/nonspin SCF state allocation (`VKSCF`, `VKEigenValue`, `VKEigenVector`) behind shared builder helpers
- [x] Validated with Docker correctness gates (`run_phase12_regression.sh`, `run_spin_mpi_parity.sh`)

**Acceptance Criteria**
- `pw/src/main.rs` delegates to phase modules and no longer owns low-level construction details
- No ad-hoc `ik - ik_first` indexing arithmetic remains outside k-point domain helpers
- New phases are unit-testable without invoking full end-to-end `pw` flow

### E21 - Declarative Input Schema and Validation Pipeline
**Priority**: P1  
**Status**: Done (2026-02-26)  
**Files**: `control/src/lib.rs`

- Replace large manual `match` parser with declarative key schema (type parser + unit conversion + validation rule)
- Return `Result<Control, ControlError>` with line-aware diagnostics instead of `unwrap()` / `process::exit`
- Split validation into stages: syntax, semantic ranges, feature-compatibility matrix
- Add parser tests for invalid keys, invalid values, deprecated aliases, and cross-field constraints

**Implementation Update (2026-02-26)**
- [x] Replaced manual key dispatch with declarative key schema (`KeySpec`) and typed setter functions (including unit-converting fields)
- [x] Added structured `ControlError` diagnostics with line/key context and fallible APIs (`Control::from_file`, `try_read_file`)
- [x] Split validation pipeline into staged checks (syntax parse -> semantic ranges -> feature compatibility)
- [x] Added compatibility validation for restart/spin unsupported pair (`restart=true` with `spin_scheme=ncl`)
- [x] Added parser/validator tests for unknown keys, typed value failures, deprecated aliases, and cross-field constraints (HSE/Hubbard/restart)
- [x] Validated via Docker correctness gates (`scripts/run_phase12_regression.sh`, `scripts/run_spin_mpi_parity.sh` in `rust-dev`)

**Acceptance Criteria**
- `Control` loading has no `process::exit` and no parsing `unwrap()` in normal flow
- Input error messages include key name and line number
- Parser/validator test suite covers compatibility gates (HSE, Hubbard U, restart, spin modes)

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

## Sprint Plan (Revised 2026-02-28)

**Current Status Summary**
- **Major refactoring completed**: E2, E3, E4, E5, E7, E12-E25 (workspace, typed config, error model, orchestration, SCF unification, checkpoint abstraction, verbosity)
- **Immediate priorities**: E1 (build portability), E10 (CI/testing), E24 (capability matrix), E29 (reproducibility tests)
- **Follow-up work**: E26-E29 (documentation, benchmarks, cleanup, CI)

**Assumptions**
- Sprint duration: 1-2 weeks depending on scope
- Done criteria: tests pass, one reference example documented (if applicable), metrics recorded

### Sprint 1 - Foundation Stability (CURRENT PRIORITY)
**Scope**: `E1`, `E10`, `E24`, `E29`

**E1 - Build Portability** (P0 BLOCKER)
- Remove hard-coded linker paths and add environment-driven discovery
- Add build documentation for macOS, Linux, HPC clusters
- Test on at least 3 different machines/environments

**E10 - Integration Tests and CI** (P1 CRITICAL)
- Add integration test suite with reference systems
- Set up CI gates for `cargo check`, `cargo test`, integration tests
- Automate Docker validation scripts in CI

**E24 - Capability Matrix** (P1 CRITICAL)
- Define explicit capability matrix for mode combinations
- Remove runtime panics for unsupported modes
- Add validation tests for all unsupported combinations

**E29 - CI Reproducibility Tests** (P2 CRITICAL)
- Add fixed-seed reproducibility test in CI
- Add determinism test (1-rank vs 2-rank)
- Add restart parity test

**Exit Gates**
- CI runs successfully on multiple platforms
- Build works on 3+ different machines without path edits
- All unsupported mode combinations fail gracefully with helpful errors
- Reproducibility tests pass in CI

### Sprint 2 - Code Quality and Polish (COMPLETED - Follow-up in E26-E28)
**Status**: Mostly Complete
**Scope**: `E2`, `E3`, `E4`, `E5`, `E7`, `E14`, `E15`, `E16`, `E18`, `E25`

**Completed Work** (2026-02-26 to 2026-02-28)
- ✅ E2: Workspace architecture for hot paths (allocation-free steady state)
- ✅ E3: Removed serialized eigenvalue output delay
- ✅ E4: Result-based error model (no `process::exit` in libraries)
- ✅ E5: Typed configuration (enum-based mode dispatch)
- ✅ E7: Orchestration modularization (phase-based structure)
- ✅ E12: Restart semantics and checkpoint completeness
- ✅ E13: Spin SCF MPI and symmetry parity
- ✅ E14: FFT planning and spectral-operator workspace tuning
- ✅ E15: Cost-aware k-point scheduling and spin cache deduplication
- ✅ E16: Deterministic initialization and run provenance
- ✅ E18: Verbosity policy and structured runtime logging
- ✅ E19: Unified SCF iteration engine
- ✅ E20: Typed run context and phase builders
- ✅ E21: Declarative input schema and validation pipeline
- ✅ E22: Checkpoint repository and codec abstraction
- ✅ E23: SCF utilities module decomposition
- ✅ E25: K-point domain model and index safety

**Remaining Follow-up**
- E26: Workspace documentation
- E27: Performance benchmarking
- E28: Cleanup compatibility wrappers

### Sprint 3 - Performance and Scalability (FUTURE)
**Scope**: `E6`, `E8`, `E11`, `E17`, `E26`, `E27`

**E6 - Thread-Level K-Point Parallelism** (P3)
- Implement thread-parallel k-point execution with deterministic reductions
- Benchmark scaling on systems with many k-points

**E8 - Static Dispatch Optimization** (P4)
- Profile trait dispatch overhead, convert hot paths if beneficial

**E11 - Benchmark Framework** (P3)
- Add Criterion microbenchmarks and performance regression tracking

**E17 - Scalable Checkpoint I/O** (P3)
- Implement packed HDF5 and parallel I/O for large-scale runs

**E26 - Workspace Documentation** (P3)
- Document workspace pattern and provide developer guide

**E27 - Performance Benchmarking** (P3)
- Capture baseline metrics for recent optimizations

**Exit Gates**
- Thread parallelism provides measurable speedup on large systems
- Benchmark framework tracks performance regressions
- Workspace pattern is well-documented for contributors

### Sprint 4 - Code Quality Hardening (FUTURE)
**Scope**: `E9`, `E28`

**E9 - Warning Cleanup** (P2)
- Remove blanket `#![allow(warnings)]` from core modules
- Fix underlying warnings incrementally

**E28 - Compatibility Wrapper Cleanup** (P4)
- Migrate remaining legacy APIs to typed variants
- Add deprecation warnings

**Exit Gates**
- CI runs `cargo clippy` without warnings on new code
- Legacy APIs are marked deprecated with migration guide

## Feature Track (Post-Core Stabilization)

Run these after core engineering tasks (E1, E10, E24, E29) establish a stable foundation.

**Priority Levels for Features:**
- **FP1**: High-impact features with broad user demand
- **FP2**: Valuable features for specific use cases
- **FP3**: Advanced features for specialized research
- **FP4**: Long-term research directions

### F1 - Equation of State and Thermodynamics
**Priority**: FP1
**Status**: Open

- [ ] Add automated volume-scan workflow (`-6%` to `+6%` or custom range)
- [ ] Fit Birch-Murnaghan EOS (2nd, 3rd, 4th order) and report `V0`, `B0`, `B0'`, `E0`
- [ ] Add static lattice enthalpy versus pressure output
- [ ] Parallelize independent volume points with restart support
- [ ] Add regression case against reference range
- [ ] Support thermal expansion via quasi-harmonic approximation (integrate with F3)

**Deliverables**
- `eos_fit.json` with fitting parameters and statistics
- Plot-ready energy-volume and pressure-volume tables
- Automated plotting scripts

### F2 - Elastic Properties
**Priority**: FP1
**Status**: Open

- [ ] Implement finite-strain generation (Voigt notation, symmetry-adapted)
- [ ] Automated stress collection workflow with multiple strain amplitudes
- [ ] Fit elastic tensor `Cij` (symmetry-aware where available)
- [ ] Compute Voigt/Reuss/Hill derived moduli (bulk, shear, Young's, Poisson ratio)
- [ ] Use crystal symmetry to reduce required strain calculations
- [ ] Add tensor-symmetry and mechanical-stability checks (Born criteria)
- [ ] Support both stress-based and energy-based fitting

**Deliverables**
- `elastic_tensor.json` with Cij matrix and derived moduli
- `elastic_summary.md` with mechanical stability analysis
- Convergence plots for strain amplitude

### F3 - Vibrational Properties (Finite Displacement)
**Priority**: FP1
**Status**: Open

- [ ] Implement supercell builder and atomic displacement patterns (symmetry-reduced)
- [ ] Compute force constants (harmonic and optionally anharmonic 3rd order)
- [ ] Build and diagonalize dynamical matrices on q-grid
- [ ] Compute phonon dispersion along high-symmetry paths and phonon DOS
- [ ] Add acoustic sum-rule enforcement and imaginary-mode detection
- [ ] Add convergence workflow for supercell size and displacement amplitude
- [ ] Support thermodynamic property calculation (free energy, entropy, Cv)
- [ ] Interface with Phonopy for advanced analysis

**Deliverables**
- `phonon_bands.dat` and `phonon_dos.dat`
- `phonon_thermodynamics.json` (if requested)
- Stability summary with imaginary mode warnings
- Phonopy-compatible force constants output

### F4 - Electric and Optical Response
**Priority**: FP2
**Status**: Partial (F10 provides basic external field support)

- [ ] Implement Berry-phase polarization (modern theory of polarization, non-metal first)
- [ ] Implement static dielectric tensor (finite-field or DFPT)
- [ ] Implement Born effective charges (finite-difference or DFPT)
- [ ] Add IR-active mode intensities from phonons and Born charges
- [ ] Add optical absorption spectrum (independent-particle or RPA with scissors)
- [ ] Add symmetry checks for tensor forms and coordinate conventions
- [ ] Support metallic systems with appropriate polarization quantum

**Deliverables**
- `polarization.json` with Berry phase and quantum of polarization
- `dielectric_tensor.json` (static and optionally optical)
- `born_charges.json` with Z* tensors per atom
- `ir_intensities.dat` for IR-active modes
- `optical_spectrum.dat` (if requested)

### F5 - Advanced Transport and Topological Responses
**Priority**: FP3
**Status**: Open (requires Wannier interpolation)

- [ ] Add Wannier-based band interpolation for dense k-space properties (requires MLWF, see F11)
- [ ] Implement Boltzmann transport (`conductivity`, `Seebeck`, `kappa_e`) versus T and μ
- [ ] Implement anomalous Hall conductivity and Berry-curvature integration workflows
- [ ] Add Chern number and Z2 topological invariant calculation
- [ ] Add scalable parallel execution for dense k/q sampling with deterministic reductions
- [ ] Support constant relaxation time and energy-dependent scattering
- [ ] Add benchmark suite for performance and reproducibility on large systems

**Deliverables**
- `transport_*.json` with σ(T,μ), S(T,μ), κ_e(T,μ)
- `berry_curvature_*.dat` and `anomalous_hall.json`
- `topological_invariants.json` (Chern, Z2)

### F6 - NCL Extension (Phase 3+)
**Priority**: FP2
**Status**: Open (requires E24 capability matrix)

- [ ] Implement spinor wavefunction and magnetization-density model for true NCL
- [ ] Extend SCF pipeline for NCL Hamiltonian assembly (SOC, non-collinear XC)
- [ ] Add NCL-specific density mixing and occupation handling
- [ ] Add NCL-specific XC integration and validation benchmarks
- [ ] Define parity and reference targets against trusted external solvers (QE, VASP)
- [ ] Support spin-orbit coupling with ultrasoft or PAW pseudopotentials

**Deliverables**
- NCL-enabled SCF for systems with strong SOC (heavy elements, topological materials)
- Validation suite against reference codes

### F7 - Full Symmetry Support
**Priority**: FP2
**Status**: Open (symops crate provides foundation)

- [ ] Build centralized symmetry context/service from internal symmetry output
- [ ] Use symmetry to reduce k-point workloads (irreducible mesh and weights)
- [ ] Apply symmetry operations consistently to density, potential, force, and stress
- [ ] Add automatic irreducible k-point mesh generation from full mesh
- [ ] Add validation tests for invariants and full-mesh versus irreducible-mesh agreement
- [ ] Support symmetry-adapted force constants (for F3)

**Deliverables**
- Automatic irreducible k-mesh generation with symmetry weights
- Symmetry-consistent force/stress/property calculations
- Reduced computational cost for high-symmetry systems

### F8 - HSE06 Extension
**Priority**: FP2
**Status**: Partial (gamma-only MVP complete)

- [ ] Extend HSE06 beyond gamma-only to general k-point meshes with ACE or truncated Coulomb
- [ ] Add hybrid exchange contribution to force and stress (analytical derivatives)
- [ ] Add benchmark comparison against reference implementations (QE, VASP)
- [ ] Optimize hybrid exchange performance (better parallelization, FFT box optimization)
- [ ] Support other hybrid functionals (PBE0, B3LYP)

**Deliverables**
- General k-point HSE06 with forces and stresses
- Benchmark suite against reference codes
- Performance optimization documentation

### F9 - Automated Convergence Campaigns
**Priority**: FP1
**Status**: Open

- [ ] Add workflow mode for automatic `ecut`, `kmesh`, `nband`, and smearing convergence sweeps
- [ ] Support stopping criteria based on user tolerances (energy, force, stress, gap, DOS stability)
- [ ] Reuse checkpoint/restart to skip already-converged sample points
- [ ] Parallelize independent convergence points (multi-job workflow)
- [ ] Export machine-readable recommendation report and selected final inputs
- [ ] Provide visualization tools for convergence trends

**Deliverables**
- `convergence_report.json` with recommended parameters and tolerance curves
- `recommended_in.ctrl` and `recommended_in.kmesh`
- Convergence plots (energy vs ecut, energy vs k-mesh, etc.)

### F10 - Surface Dipole Correction and 2D Electric-Field Effects
**Priority**: FP2
**Status**: Partial (basic implementation complete, validation pending)

**Implementation Update (2026-02-28)**
- [x] Added typed control knobs for slab-field workflows (`electric_field_2d`, `electric_field_axis`, `surface_dipole_correction`)
- [x] Integrated external sawtooth field + dipole correction potential into SCF
- [x] Added external-field electronic energy contribution to total energy
- [x] Validated no-regression via Docker gates

**Remaining Work**
- [ ] Add dedicated 2D-material reference examples (graphene, MoS2, hBN)
- [ ] Include explicit external-field ionic force/stress terms for geometry optimization under field
- [ ] Validate field-response trends (polarization vs field strength)
- [ ] Add Hellmann-Feynman force correction for field-gradient effects

**Deliverables**
- `field_response_summary.json`
- `surface_dipole_correction_report.md`
- Validation examples for 2D materials


### F11 - Maximally Localized Wannier Functions (MLWF)
**Priority**: FP2
**Status**: Open (Wannier90 interface exists but MLWF calculation not integrated)

- [ ] Implement maximal localization minimization (spread functional, Wannier centers)
- [ ] Add disentanglement for entangled bands (outer/inner windows)
- [ ] Integrate with existing Wannier90 interface for seamless workflow
- [ ] Support band interpolation using MLWF for dense k-space properties
- [ ] Add MLWF-based analysis: Wannier centers, spreads, real-space Hamiltonians
- [ ] Support transport calculations using interpolated MLWF bands

**Deliverables**
- `wannier_centers.json` with WF centers and spreads
- `wannier_hr.dat` for tight-binding Hamiltonian
- Interpolated band structure with fine k-mesh
- Integration with Wannier90 and WannierTools

### F12 - Magnetic Properties and Magnetization Analysis
**Priority**: FP2
**Status**: Open

- [ ] Export 3D magnetization density (spin-up minus spin-down) to visualization formats
- [ ] Compute atomic magnetic moments (sphere integration, Lowdin, Mulliken)
- [ ] Add spin texture calculation (spin expectation values for NCL)
- [ ] Compute magnetic anisotropy energy (MAE) for different spin orientations
- [ ] Add Hubbard U correction for orbital-resolved magnetic moments
- [ ] Support non-collinear magnetic structure optimization

**Deliverables**
- `magnetization_density.cube` or `.xsf` for visualization
- `magnetic_moments.json` with atomic-resolved moments
- `spin_texture.dat` for NCL systems
- `mae_analysis.json` with anisotropy energies

### F13 - Geometry Optimization Improvements
**Priority**: FP1
**Status**: Open (basic BFGS exists in workflow, needs enhancement)

- [ ] Implement advanced optimizers (FIRE, L-BFGS with line search, conjugate gradient)
- [ ] Add constraints (fixed atoms, fixed cell, fixed angles, distance constraints)
- [ ] Add lattice optimization (variable cell shape at constant/variable volume)
- [ ] Improve convergence criteria (max force, RMS force, max displacement)
- [ ] Add optimizer state checkpoint/restart for long optimizations
- [ ] Support transition state optimization (dimer method, NEB preliminary)

**Deliverables**
- Multiple optimizer options with documented performance characteristics
- Flexible constraint system for various optimization scenarios
- Robust variable-cell optimization for crystals

### F14 - Nudged Elastic Band (NEB) for Reaction Paths
**Priority**: FP2
**Status**: Open (requires F13 improvements)

- [ ] Implement climbing-image NEB (CI-NEB)
- [ ] Support variable number of images with adaptive refinement
- [ ] Add transition state search from NEB maximum
- [ ] Support string method variant for faster convergence
- [ ] Parallelize independent image calculations with MPI
- [ ] Add automatic initial path generation (linear interpolation, IDPP)

**Deliverables**
- `neb_path.json` with energies and forces along path
- `transition_state.xyz` and barrier heights
- Visualization output for reaction coordinate

### F15 - Molecular Dynamics (Born-Oppenheimer MD)
**Priority**: FP2
**Status**: Open

- [ ] Implement velocity-Verlet integrator for MD trajectories
- [ ] Add thermostats (Nosé-Hoover, Langevin, velocity rescaling)
- [ ] Add barostats for NPT ensemble (Parrinello-Rahman, Berendsen)
- [ ] Support constrained MD (SHAKE/RATTLE for bonds)
- [ ] Add trajectory analysis tools (RDF, MSD, VACF)
- [ ] Support ab initio MD with on-the-fly ML potential training

**Deliverables**
- `trajectory.xyz` with atomic positions/velocities
- `md_log.json` with energy, temperature, pressure vs time
- Analysis outputs (RDF, MSD, diffusion coefficients)

### F16 - Van der Waals Corrections
**Priority**: FP1
**Status**: Open

- [ ] Implement DFT-D3 dispersion correction (Grimme)
- [ ] Add DFT-D3(BJ) variant with Becke-Johnson damping
- [ ] Implement Tkatchenko-Scheffler (TS) vdW correction
- [ ] Add Many-Body Dispersion (MBD) for improved accuracy
- [ ] Support vdW-DF family (vdW-DF, vdW-DF2, rev-vdW-DF2) - non-local functionals
- [ ] Add vdW correction to forces and stresses for geometry optimization

**Deliverables**
- Multiple vdW correction methods with documented accuracy
- Benchmark suite comparing vdW methods on layered materials
- Full support for vdW-corrected geometry optimization

### F17 - Meta-GGA Functionals
**Priority**: FP2
**Status**: Open (GGA functionals exist, needs meta-GGA extension)

- [ ] Implement SCAN meta-GGA functional
- [ ] Add r2SCAN (revised SCAN) for improved accuracy
- [ ] Implement TPSS and revTPSS meta-GGA functionals
- [ ] Add kinetic energy density calculation for meta-GGA
- [ ] Implement meta-GGA forces and stresses
- [ ] Validate against reference implementations and experimental data

**Deliverables**
- SCAN, r2SCAN, TPSS functional implementations
- Benchmark suite on diverse materials
- Performance comparison vs GGA

### F18 - Advanced Smearing and Occupation Methods
**Priority**: FP3
**Status**: Open (basic smearing exists)

- [ ] Implement cold smearing (Marzari-Vanderbilt) for faster convergence
- [ ] Add Methfessel-Paxton smearing (1st, 2nd order)
- [ ] Implement tetrahedron method for Brillouin zone integration
- [ ] Add optimized smearing width selection (entropy-based)
- [ ] Support fixed occupations for core-hole calculations

**Deliverables**
- Multiple smearing methods with automatic width selection
- Tetrahedron method for accurate DOS and properties
- Validation of energy vs smearing width convergence

### F19 - Constrained DFT (cDFT)
**Priority**: FP3
**Status**: Open

- [ ] Implement charge constraints (constrain charge in region/atom)
- [ ] Implement spin constraints (constrain magnetization)
- [ ] Add Lagrange multiplier optimization for constraint enforcement
- [ ] Support diabatic state calculations for electron transfer
- [ ] Add cDFT-based reorganization energy calculation

**Deliverables**
- Charge/spin constrained SCF calculations
- Diabatic state energy differences
- Reorganization energies for charge transfer

### F20 - Time-Dependent DFT (TDDFT)
**Priority**: FP3
**Status**: Open (requires significant infrastructure)

- [ ] Implement Casida equation solver for excitation energies
- [ ] Add real-time TDDFT for optical absorption (Ehrenfest dynamics)
- [ ] Support spin-flip TDDFT for excited states
- [ ] Add oscillator strengths and transition dipole moments
- [ ] Implement Tamm-Dancoff approximation for faster calculations

**Deliverables**
- Excitation energies and oscillator strengths
- Optical absorption spectra
- Transition density analysis

### F21 - GW Approximation (Quasiparticle Energies)
**Priority**: FP3
**Status**: Open (requires significant infrastructure)

- [ ] Implement one-shot G0W0 for quasiparticle band structure
- [ ] Add plasmon-pole approximation for efficiency
- [ ] Support full-frequency integration for accuracy
- [ ] Implement self-consistent GW (scGW)
- [ ] Add GW-based band gap correction workflow

**Deliverables**
- Quasiparticle band structures with corrected gaps
- GW spectral functions
- Validation against experimental photoemission data

### F22 - GPU Acceleration
**Priority**: FP2
**Status**: Open

- [ ] Port FFT operations to GPU (cuFFT, rocFFT)
- [ ] Accelerate linear algebra operations (cuBLAS, rocBLAS)
- [ ] Implement GPU-accelerated eigensolver (Davidson, LOBPCG)
- [ ] Accelerate XC functional evaluation on GPU
- [ ] Add multi-GPU support for large systems
- [ ] Benchmark GPU vs CPU performance and provide usage guidelines

**Deliverables**
- GPU-accelerated hot paths (FFT, BLAS, eigensolver)
- Performance benchmarks showing speedup vs CPU
- Multi-GPU scaling analysis

### F23 - Advanced Parallelization (Band + K-point)
**Priority**: FP2
**Status**: Open (MPI k-point parallelism exists)

- [ ] Implement band parallelization within k-points for large systems
- [ ] Add hybrid MPI+MPI parallelism (k-point ranks × band ranks)
- [ ] Optimize communication patterns for band parallelism
- [ ] Add load balancing for heterogeneous band costs
- [ ] Benchmark scaling efficiency on HPC systems (1000+ cores)

**Deliverables**
- Hybrid k-point + band parallelization
- Scaling benchmarks on large systems
- Best-practice guide for HPC deployment

### F24 - Implicit Solvation Models
**Priority**: FP3
**Status**: Open

- [ ] Implement VASPsol-style implicit solvation
- [ ] Add PCM (Polarizable Continuum Model) variant
- [ ] Support COSMO (Conductor-like Screening Model)
- [ ] Add solvation free energy calculation
- [ ] Support variable dielectric regions (inhomogeneous environments)

**Deliverables**
- Implicit solvent support for aqueous and organic solvents
- Solvation free energies
- Validation against experimental solvation data

### F25 - Machine Learning Potential Integration
**Priority**: FP2
**Status**: Open

- [ ] Add interface to pretrained ML potentials (M3GNet, CHGNet, MACE)
- [ ] Support ML-accelerated geometry optimization
- [ ] Implement active learning workflow (DFT → ML training → ML MD)
- [ ] Add uncertainty quantification for ML predictions
- [ ] Support hybrid DFT/ML workflows (ML screening + DFT refinement)

**Deliverables**
- ML potential interface for fast structure relaxation
- Active learning framework for dataset generation
- Hybrid workflow examples

### F26 - Advanced Convergence Techniques
**Priority**: FP2
**Status**: Open (Pulay mixing exists, needs enhancement)

- [ ] Implement Broyden mixing variants (Broyden1, Broyden2)
- [ ] Add Kerker preconditioning for metallic systems
- [ ] Implement Anderson acceleration for faster convergence
- [ ] Add adaptive mixing parameter selection
- [ ] Support real-space mixing for large systems
- [ ] Add convergence diagnostics and troubleshooting tools

**Deliverables**
- Multiple mixing schemes with auto-selection
- Convergence acceleration for difficult systems (metals, magnetic, strongly correlated)
- Diagnostic tools for convergence failures

### F27 - Core-Level Spectroscopy
**Priority**: FP3
**Status**: Open

- [ ] Implement core-hole calculations for XPS/EELS
- [ ] Add initial/final state approximations
- [ ] Support delta-SCF for core excitations
- [ ] Add EELS spectrum calculation (core-loss edges)
- [ ] Implement Bethe-Salpeter equation (BSE) for improved EELS

**Deliverables**
- XPS binding energies and chemical shifts
- EELS spectra for core edges
- Validation against experimental spectroscopy

### F28 - Raman Intensities and IR Spectroscopy
**Priority**: FP3
**Status**: Open (requires F3 phonons and F4 Born charges)

- [ ] Implement Raman tensor calculation (polarizability derivatives)
- [ ] Add Raman intensity calculation for all phonon modes
- [ ] Combine with F4 Born charges for complete IR intensities
- [ ] Add resonance Raman enhancement (if TDDFT available)
- [ ] Support oriented single-crystal geometries

**Deliverables**
- Raman spectra with intensities
- Combined IR and Raman mode assignment
- Comparison with experimental vibrational spectroscopy

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
- One reference example is documented (if user-facing feature)
- Runtime and memory metrics are recorded (if performance-related)
- Output schema compatibility is preserved
- Docker correctness gates pass (phase12 regression, spin MPI parity)

## Current Status Summary (2026-02-28)

### Recent Accomplishments (2026-02-26 to 2026-02-28)
- ✅ **Major refactoring wave completed**: 17 engineering tasks (E2-E5, E7, E12-E25)
- ✅ **Workspace architecture**: Allocation-free hot paths with reusable buffers
- ✅ **Typed configuration**: Enum-based runtime mode dispatch
- ✅ **Error model**: Result-based APIs, no library-level panics
- ✅ **SCF unification**: Shared iteration engine for spin/nonspin
- ✅ **Orchestration**: Phase-modular structure in `pw`
- ✅ **Advanced features**: K-point scheduling, spin cache dedup, FFT tuning, provenance tracking
- ✅ **Code quality**: Structured logging, verbosity control, checkpoint abstraction

### Immediate Priorities (Sprint 1)
1. **E1 (P0)**: Build portability - enable team expansion
2. **E10 (P1)**: CI/testing infrastructure - prevent regressions
3. **E24 (P1)**: Capability matrix - eliminate runtime panics
4. **E29 (P2)**: Reproducibility tests - ensure determinism

### High-Value Follow-up (Sprint 2-4)
- E6: Thread-level k-point parallelism
- E9: Warning cleanup for CI
- E11: Benchmark framework
- E26-E28: Documentation, benchmarks, API cleanup

### Feature Roadmap (Post-Stabilization)
- **FP1 (High Priority)**: F1 (EOS), F2 (Elastic), F3 (Phonons), F9 (Convergence), F13 (Geometry Opt), F16 (vdW)
- **FP2 (Medium Priority)**: F4 (Optical), F6 (NCL), F7 (Symmetry), F8 (HSE), F11 (MLWF), F12 (Magnetic), F14 (NEB), F15 (MD), F17 (Meta-GGA), F22 (GPU), F23 (Band||), F25 (ML), F26 (Mixing)
- **FP3 (Advanced)**: F5 (Transport), F18 (Smearing), F19 (cDFT), F20 (TDDFT), F21 (GW), F24 (Solvation), F27 (Core-level), F28 (Raman)

### Priority Actions for Next Session
1. Start E1: Fix build portability on multiple platforms
2. Set up E10: Create CI pipeline with basic integration tests
3. Implement E24: Define capability matrix and remove panics
4. Add E29: Reproducibility tests in CI

## Uncompleted Engineering Tasks (Moved To End)

### E1 - Build Portability and Correctness
**Priority**: P0 (BLOCKER - prevents team expansion and cross-platform development)
**Status**: Open
**Files**: `matrix/build.rs`, `symmetry/build.rs`, `dwfft3d/build.rs`

- Replace hard-coded local linker paths with environment-driven discovery (`LAPACK_DIR`, `FFTW_DIR`) or `pkg-config`
- Add explicit build-time diagnostics when required libraries are missing
- Support common package managers (Homebrew, apt, conda) for dependency discovery
- Add build documentation with platform-specific instructions (macOS, Linux, HPC clusters)
- Keep optional platform-specific fallback only behind explicit env flags

**Acceptance Criteria**
- `cargo check` works on at least three different machines (macOS, Linux, HPC) without local path edits
- Build scripts print actionable failure messages when dependencies are missing
- CI builds successfully on multiple platforms
- Build documentation covers common setups


### E2 - Workspace Architecture for Hot Paths
**Priority**: P2
**Status**: Complete (2026-02-26) - Minor documentation follow-up in E26
**Files**: `scf/`, `kscf/`, `density/`, `eigensolver/`, `force/`, `stress/`, `dwfft3d/`, `pw/src/main.rs`

**Implementation Summary (2026-02-26)**
- [x] Added explicit reusable SCF workspaces for nonspin and spin paths with one-shot size construction
- [x] Refactored SCF hot-loop scratch storage into workspace-owned buffers
- [x] Extended workspace pattern to `kscf`, `density`, `eigensolver`, `force`, `stress` modules
- [x] Refactored FFT-gradient operators to reuse thread-local spectral scratch buffers
- [x] Added allocation-trace benchmark binary (`pw/src/bin/workspace_alloc_trace.rs`)
- [x] Confirmed steady-state allocation profile: zero alloc/realloc calls across traced kernels after warmup

**Acceptance Criteria Met**
- ✓ Allocation-heavy hot loops are allocation-free in profiling traces
- ✓ Workspace APIs adopted in SCF and eigensolver paths
- ✓ No behavior regressions in reference SCF cases

**Follow-up**: See E26 for workspace documentation and API guidelines


### E3 - Remove Serialized Eigenvalue Output Delay
**Priority**: P2
**Status**: Complete (2026-02-27) - Benchmark follow-up in E27
**Files**: `pw/src/main.rs`, `scf/src/utils.rs`, `scf/src/nonspin.rs`

**Implementation Summary (2026-02-27)**
- [x] Removed fixed tail sleep in `pw` main before MPI finalize
- [x] Gated ordered rank-by-rank eigenvalue output behind explicit verbosity (`verbose`/`debug`)
- [x] Switched default eigenvalue output to root-only print path without rank-serialized barriers
- [x] Validated via Docker correctness gates

**Acceptance Criteria Met**
- ✓ No fixed sleep calls in production paths
- ✓ Rank serialization removed from default output

**Follow-up**: See E27 for multi-rank wall-time benchmarking


### E4 - Result-Based Error Model and Process Boundary
**Priority**: P1
**Status**: Complete (2026-02-27) - Cleanup follow-up in E28
**Files**: `control/src/lib.rs`, `kpts/src/line.rs`, `special/src/lib.rs`, Wannier90 binaries

**Implementation Summary (2026-02-27)**
- [x] Added typed error types (`KptsError`, `SpecialError`) with fallible constructors
- [x] Moved process termination policy to binary entry points only
- [x] Audited codebase: remaining `process::exit` usage confined to binaries
- [x] Docker correctness gates passed

**Acceptance Criteria Met**
- ✓ No `process::exit` in library crates
- ✓ Library APIs propagate typed errors with context
- ✓ CLI binaries keep user-friendly exit behavior

**Follow-up**: See E28 for compatibility wrapper migration


### E5 - Typed Configuration and Runtime Mode Safety
**Priority**: P1
**Status**: Complete (2026-02-27) - Cleanup follow-up in E28
**Files**: `control/`, `kscf/`, `scf/`, `smearing/`, `xc/`, `eigensolver/`, `pspot/`, `kpts/`, `pw/`, `wannier90/`

**Implementation Summary (2026-02-27)**
- [x] Added typed runtime-mode enums in `control` with parser-level conversion
- [x] Migrated core runtime factories to typed dispatch
- [x] Replaced hot-path string comparisons with enum-based `match` checks
- [x] Added validation guard to reject unimplemented modes early
- [x] Validated via Docker correctness gates

**Acceptance Criteria Met**
- ✓ String mode dispatch removed from runtime hot paths
- ✓ Invalid mode configurations fail at parse/validation time

**Follow-up**: See E28 for compatibility string getter removal


### E6 - Scalable K-Point Execution with Deterministic Reductions
**Priority**: P3 (High value for performance but non-blocking)
**Status**: Open
**Files**: `scf/`, `kscf/`, orchestration and reduction utilities

- Add explicit execution layer for thread-parallel k-point evaluation (Rayon or custom thread pool)
- Preserve deterministic reduction order for reproducibility (ordered sum, Kahan summation)
- Define MPI/thread interaction policy and reproducibility guarantees
- Add runtime config for thread-parallel k-point mode with thread count control
- Benchmark scaling efficiency versus serial baseline

**Acceptance Criteria**
- Thread-level k-point parallel execution is available behind config (`kpoint_parallelism=thread`)
- Repeated runs with fixed settings are numerically reproducible (bitwise or within tolerance)
- Scaling measured versus serial baseline on systems with 50+ k-points
- No interference with MPI k-point distribution


### E7 - Orchestration Modularization
**Priority**: P3/P6  
**Status**: Done (2026-02-28)  
**Files**: `pw/src/main.rs`, `pw/src/orchestration/{bootstrap.rs,construction.rs,electronic.rs,outputs.rs,mod.rs}`, `scf/src/lib.rs`

- Split monolithic orchestration into phase modules:
  - input/bootstrap
  - basis/construction
  - SCF execution
  - outputs/postprocessing
- Replace oversized argument lists with grouped context structs

**Implementation Update (2026-02-28)**
- [x] Added explicit orchestration phase modules under `pw/src/orchestration/`:
  `bootstrap` (input/runtime bootstrap), `construction` (geometry-step basis/density/symmetry build), `electronic` (k-point scheduling + SCF/eigenstate allocation helpers), `outputs` (checkpoint/output persistence and convergence exit policy)
- [x] Refactored `pw::main` to a high-level orchestration loop that delegates phase work to those modules
- [x] Introduced grouped phase contexts where they added phase clarity (`BootstrapData`, `GeometryPhaseInput/Artifacts`)
- [x] Simplified thin wrapper layers by removing pure pass-through SCF execution/output context wrappers; `main` now calls SCF execution and output helpers directly
- [x] Preserved existing runtime behavior for restart/checkpoint, symmetry diagnostics, SCF execution, and postprocessing while reducing monolithic control flow in `main.rs`
- [x] Validated via Docker correctness gates (`scripts/run_phase12_regression.sh`, `scripts/run_spin_mpi_parity.sh`)

**Acceptance Criteria**
- `pw/src/main.rs` reduced to high-level orchestration flow
- Phase modules own their scoped logic and interfaces


### E8 - Prefer Static Dispatch in Hot Kernels
**Priority**: P4 (Optimization - may have limited impact)
**Status**: Open
**Files**: `scf/`, `eigensolver/`, `kscf/`, `xc/`, `smearing/`

- Profile trait-object dispatch overhead in hot paths (XC functional calls, eigensolver iterations)
- Keep trait objects at orchestration boundaries only
- Use enums/generics in kernels where implementations are known at compile time
- Measure before/after runtime for SCF and eigensolver-heavy cases

**Acceptance Criteria**
- Profiling confirms trait dispatch is a measurable bottleneck (>2% overhead)
- Kernel dispatch hotspots converted to static dispatch where valid
- Measured performance improvement documented (wall-time or instructions)
- Code complexity increase is justified by performance gain


### E9 - Warning Policy Cleanup
**Priority**: P2 (Code quality - affects maintainability)
**Status**: Open
**Files**: Widespread `#![allow(warnings)]` usage, especially `matrix/`, `ndarray/`, `lattice/`, `pwbasis/`

- Audit all `#![allow(warnings)]` and `#[allow(warnings)]` usage
- Remove blanket warning suppression incrementally (start with leaf crates)
- Keep only narrow, justified `#[allow(...)]` annotations with comments
- Fix underlying warnings where practical (unused variables, dead code, deprecated patterns)
- Add clippy configuration for project-specific lints

**Acceptance Criteria**
- Blanket `#![allow(warnings)]` removed from all active core modules
- CI runs `cargo clippy` and enforces warning-free builds on new code
- Remaining `#[allow(...)]` annotations have justification comments
- `cargo check` and `cargo clippy` provide meaningful feedback


### E10 - Integration Tests and CI Gates
**Priority**: P1 (Critical - prevents regressions and enables confident refactoring)
**Status**: Open
**Files**: `scf/`, `kscf/`, `pw/`, `tests/`, `.github/workflows/`

- Add integration test suite with reference systems (Si, Al, Fe spin, molecular)
- Test coverage: convergence, energy consistency, force/stress accuracy, restart parity
- Add CI gates for `cargo check`, `cargo test`, and selected integration tests
- Include at least one reference benchmark system per major workflow (nonspin, spin, HSE, Hubbard)
- Add determinism tests with fixed seeds
- Move existing Docker validation scripts to CI automation

**Acceptance Criteria**
- CI runs on every PR with `cargo check`, `cargo test`, and integration tests
- At least 4 reference systems tested: nonspin SCF, spin SCF, HSE gamma-only, DFT+U
- Deterministic test cases pass under fixed settings (bitwise or <1e-10 Ry tolerance)
- Phase12 regression and spin MPI parity gates run in CI
- Test runtime is under 10 minutes for full suite


### E11 - Benchmark and Validation Framework
**Priority**: P3 (Medium - enables performance tracking)
**Status**: Open
**Files**: `benches/`, benchmark harnesses, CI perf jobs, regression suite

- Add Criterion microbenchmarks for hot kernels (FFT, eigensolver, XC, density)
- Add end-to-end SCF timing harnesses for representative systems (8-128 atoms, 1-64 k-points)
- Track scaling versus atoms, k-points, MPI ranks, and thread count
- Add performance regression detection (wall-time and memory)
- Add physics-consistency validation jobs (energy, force, stress tolerances)
- Store baseline performance data in repository or artifact storage

**Acceptance Criteria**
- Criterion benchmarks cover at least 5 hot kernels with <5% run-to-run variance
- Baseline performance dashboard exists for representative workloads (small/medium/large systems)
- CI detects >10% performance regressions on reference benchmarks
- Physics regression checks validate energy/force/stress within documented tolerances


### E14 - FFT Planning and Spectral-Operator Workspace Tuning
**Priority**: P2
**Status**: Complete (2026-02-27)
**Files**: `dwfft3d/src/lib.rs`, `rgtransform/src/lib.rs`, SCF/XC callers

**Implementation Summary (2026-02-27)**
- [x] Added typed FFT runtime knobs in `control` (`fft_threads`, `fft_planner`, `fft_wisdom_file`)
- [x] Added runtime FFT backend policy in `dwfft3d` with configurable thread count and planning mode
- [x] Added optional FFTW wisdom import/export hook
- [x] Wired `pw` runtime setup to pass typed FFT policy into `dwfft3d`
- [x] Benchmarked planning/execution policy deltas (`estimate` vs `measure` modes)
- [x] Confirmed steady-state allocation profile for spectral operators is allocation-free

**Acceptance Criteria Met**
- ✓ FFT thread policy is configurable and documented
- ✓ Repeated transforms avoid redundant planning overhead
- ✓ Gradient/divergence paths are allocation-free in steady-state profiling


### E15 - Cost-Aware K-Point Scheduling and Spin Cache Deduplication
**Priority**: P2
**Status**: Complete (2026-02-28) - Benchmark follow-up in E27
**Files**: `control/src/lib.rs`, `kpts_distribution/src/lib.rs`, `pw/src/main.rs`, `kscf/src/lib.rs`, `dfttypes/src/lib.rs`, `wannier90/src/{lib.rs,eig.rs}`

**Implementation Summary (2026-02-28)**
- [x] Added typed `kpoint_schedule` control mode (`contiguous`, `cost_aware`, `dynamic`)
- [x] Added deterministic `KPointSchedulePlan` in `kpts_distribution` with cost-aware LPT assignment
- [x] Wired `pw` electronic setup to estimate per-k costs and print rank load imbalance summaries
- [x] Added spin-channel immutable cache dedup in `kscf` with runtime saved-memory reporting
- [x] Extended wavefunction checkpoint/Wannier EIG I/O for non-contiguous local domains
- [x] Validated no-regression with Docker correctness gates

**Acceptance Criteria Met**
- ✓ MPI rank scheduling infrastructure is in place with multiple modes
- ✓ Spin memory footprint reduced via cache sharing
- ✓ Numerical results unchanged versus current partitioning

**Follow-up**: See E27 for wall-time imbalance benchmark on asymmetric workloads


### E16 - Deterministic Initialization and Run Provenance
**Priority**: P2
**Status**: Complete (2026-02-27) - CI replay test in E29
**Files**: `utility/src/lib.rs`, `control/src/lib.rs`, `kscf/src/lib.rs`, `pw/src/main.rs`, `property/src/lib.rs`, `workflow/src/main.rs`

**Implementation Summary (2026-02-27)**
- [x] Added explicit `random_seed` control parsing and provenance controls
- [x] Added deterministic seeded wavefunction random initialization in `kscf`
- [x] Added root-authored machine-readable `run.provenance.json` manifest emission
- [x] Added replay guard that validates schema and fingerprint against existing manifest
- [x] Surfaced provenance in downstream outputs (`workflow`, `property`)
- [x] Validated via Docker correctness gates

**Acceptance Criteria Met**
- ✓ Fixed seed runs are deterministic under fixed runtime settings
- ✓ Every run directory contains a machine-readable provenance manifest
- ✓ Replay checks reject stale/incompatible manifests when requested

**Follow-up**: See E29 for CI replay reproducibility test


### E17 - Scalable Checkpoint I/O and Artifact Schema Governance
**Priority**: P3 (Important for large-scale runs but E22 provides foundation)
**Status**: Open
**Files**: `dfttypes/src/lib.rs`, `dfttypes/src/checkpoint_repo.rs`, `pw/src/main.rs`, `workflow/src/main.rs`

- Add scalable checkpoint layout options beyond per-k-point small-file patterns (packed HDF5, chunked storage)
- Support chunking/compression and batched write/read strategies for large runs (>100 k-points)
- Introduce explicit schema/version metadata for `rho`/`wfc` artifacts with forward compatibility
- Provide migration/compatibility checks across schema revisions
- Add parallel I/O support (MPI-IO or collective HDF5 writes) for large-scale systems
- Benchmark I/O throughput on representative filesystems (local, NFS, Lustre)

**Acceptance Criteria**
- Large-k workloads (100+ k-points) produce fewer metadata-heavy I/O bottlenecks (measured IOPS reduction)
- Checkpoint readers reject incompatible schema versions with actionable guidance
- Restart throughput improves >2x on representative multi-rank filesystems for large runs
- Parallel I/O mode is available and tested


### E18 - Verbosity Policy and Structured Runtime Logging
**Priority**: P2
**Status**: Complete (2026-02-28) - Overhead benchmark in E27
**Files**: `control/src/lib.rs`, `pw/src/main.rs`, `scf/src/{engine.rs,utils.rs,nonspin.rs,spin.rs}`

**Implementation Summary (2026-02-28)**
- [x] Added typed verbosity model in `control` (`quiet`, `normal`, `verbose`, `debug`)
- [x] Added structured SCF iteration logging controls (`scf_log_format`, `scf_log_file`)
- [x] Wired SCF engine structured logs with per-iteration metrics and phase timings
- [x] Applied verbosity gating in SCF eigenvalue diagnostics and PW orchestration
- [x] Applied verbosity gating to high-volume startup/symmetry diagnostics

**Acceptance Criteria Met**
- ✓ `verbosity` setting materially changes output behavior across modules
- ✓ Default mode avoids high-frequency diagnostic flood in large runs
- ✓ Structured logs can drive regression tooling without parsing free-form stdout

**Follow-up**: See E27 for logging overhead benchmark


### E22 - Checkpoint Repository and Codec Abstraction
**Priority**: P2/P3  
**Status**: Done (2026-02-28)  
**Files**: `dfttypes/src/{lib.rs,checkpoint_repo.rs}`, `pw/src/{restart.rs,orchestration/outputs.rs,main.rs}`

- Split checkpoint domain model from storage codec to allow multiple backends (current HDF5 file-per-k, future packed/chunked formats)
- Remove duplicated spin/nonspin load/save loops via shared channel iteration helpers
- Replace remaining `unwrap()`/`expect()` in checkpoint I/O paths with typed errors and context
- Add backward-compatible schema migration hooks beyond a single version check

**Acceptance Criteria**
- Restart logic depends on a repository trait instead of direct filename logic in `pw`
- Save/load branches share common code for spin channel handling
- Checkpoint compatibility and migration behavior is covered by integration tests

**Implementation Update (2026-02-28)**
- [x] Added checkpoint repository/codec abstraction in `dfttypes` (`CheckpointRepository`, `CheckpointCodec`, `Hdf5FilePerKCheckpointRepository`) and re-exported it via `dfttypes::lib`
- [x] Centralized spin/nonspin channel handling with shared helpers for file naming, channel iteration, and metadata merge behavior
- [x] Refactored `VKEigenVector` and `RHOR` checkpoint save/load paths to delegate through repository methods, with fallible `try_*` APIs and contextual string errors
- [x] Added checkpoint metadata migration hook (`v0` -> current schema) and validation path updates so restart accepts legacy-compatible metadata payloads
- [x] Rewired `pw` restart and output persistence paths to depend on repository APIs for required filenames/load/save instead of hardcoded filename loops
- [x] Added repository-focused unit tests (channel filename mapping + spin metadata merge contract) and migration tests in `dfttypes`
- [x] Validated with Docker correctness gates (`scripts/run_phase12_regression.sh`, `scripts/run_spin_mpi_parity.sh`)


### E23 - SCF Utilities Module Decomposition
**Priority**: P2  
**Status**: Done (2026-02-28)  
**Files**: `scf/src/utils/{mod.rs,potential.rs,mixing.rs,diagnostics.rs,symmetry_projection.rs,energy.rs}`, `scf/src/nonspin.rs`, `scf/src/spin.rs`

- Split `scf::utils` into focused modules (`potential`, `energy`, `mixing`, `symmetry_projection`, `diagnostics`)
- Move printing/output routines out of numerical kernels to a diagnostics layer with verbosity controls
- Consolidate duplicated helper logic (`get_eigvalue_epsilon`, eigen display helpers, plane-wave max helpers)
- Keep math helpers (`3x3` transforms/projections) in a dedicated linear-algebra utility module

**Implementation Update (2026-02-28)**
- [x] Replaced monolithic `scf/src/utils.rs` with module tree and compatibility façade (`scf/src/utils/mod.rs`)
- [x] Moved potential/XC/external-field assembly helpers into `utils/potential.rs`
- [x] Moved density Fourier transform and mixing helpers into `utils/mixing.rs`
- [x] Moved eigen diagnostics and eigensolver tuning helpers into `utils/diagnostics.rs`, including shared spin/nonspin eigenvalue display entrypoints
- [x] Moved symmetry projection math and force/stress finalize+assembly routines into `utils/symmetry_projection.rs`
- [x] Moved nonspin total-energy assembly helpers into `utils/energy.rs`
- [x] Removed duplicated spin-side helpers (`get_eigvalue_epsilon`, `get_n_plane_waves_max`, eigen print block) by wiring `scf/src/spin.rs` to shared diagnostics helpers
- [x] Validated via Docker correctness gates (`scripts/run_phase12_regression.sh`, `scripts/run_spin_mpi_parity.sh`)

**Acceptance Criteria**
- `scf/src/utils.rs` is replaced by smaller purpose-specific modules
- Numerical kernels do not directly own user-facing I/O formatting
- Shared helper logic has one source of truth across spin/nonspin paths


### E24 - Capability Matrix and Unsupported-Mode Policy
**Priority**: P1 (Critical - prevents runtime panics and improves user experience)
**Status**: Open
**Files**: `control/src/lib.rs`, `pw/src/main.rs`, `scf/src/lib.rs`

- Define explicit capability matrix for `{spin_scheme, xc_scheme, task, restart, eigensolver}` combinations
- Validate unsupported combinations before runtime setup; return actionable errors instead of runtime `panic!`
- Keep feature flags/capability tags on SCF drivers to enable incremental NCL rollout without hard panics
- Ensure checkpoint metadata validation also enforces capability compatibility
- Document supported feature combinations in user guide
- Add validation tests for all unsupported combinations

**Acceptance Criteria**
- Unsupported runtime combinations fail at input validation/preflight phase with helpful error messages
- All `panic!` for "not implemented" mode combinations are removed from runtime setup paths
- Adding a new mode requires updating one central capability table and tests
- Capability matrix is documented and tested
- Users receive clear guidance on supported vs unsupported combinations


### E25 - K-Point Domain Model and Index Safety
**Priority**: P2
**Status**: Complete (2026-02-26) - Assumes Docker gates passed in prior sessions
**Files**: `kpts_distribution/src/lib.rs`, `pw/src/main.rs`, `scf/src/spin.rs`, `scf/src/utils.rs`

**Implementation Summary (2026-02-26)**
- [x] Extended `KPointDomain` with explicit slot helpers and clearer local/global semantics
- [x] Added k-domain invariants tests covering uneven partitioning and empty-local domains
- [x] Refactored `pw` wavefunction restart/checkpoint paths to use typed `KPointDomain` iteration
- [x] Consolidated spin/nonspin checkpoint filename traversal via shared helpers
- [x] Reworked SCF eigenvalue display loops to avoid manual index drift

**Acceptance Criteria Met**
- ✓ K-point loops use typed domain iterators instead of manual index math
- ✓ Spin/nonspin setup code reuses the same domain traversal utilities
- ✓ Tests cover uneven rank-to-k distributions and zero-local-k ranks

**Note**: Docker correctness gates validated in earlier commits; implementation is stable


### E26 - Workspace Architecture Documentation and Guidelines
**Priority**: P3 (Code quality - supports maintainability)
**Status**: Open
**Files**: `WORKSPACE_GUIDE.md`, workspace module documentation

- Document workspace architecture pattern (`Context + State + Workspace` separation)
- Provide examples of proper workspace implementation for new modules
- Document workspace buffer sizing and validation strategies
- Add inline API documentation for workspace constructors and lifecycle
- Create developer guide for adding workspaces to new hot paths

**Acceptance Criteria**
- `WORKSPACE_GUIDE.md` exists with architecture overview and examples
- All workspace types have documented constructors and sizing logic
- New contributors can implement workspace pattern without reverse-engineering existing code


### E27 - Performance Benchmarking and Regression Baseline
**Priority**: P3 (Enables tracking improvements)
**Status**: Open
**Files**: `benches/`, benchmark scripts, performance tracking

- Capture baseline wall-time benchmarks for completed optimizations:
  - E3: Multi-rank wall-time improvement from eigenvalue output removal
  - E15: Rank imbalance improvement for asymmetric k-meshes (`contiguous` vs `cost_aware`/`dynamic`)
  - E18: Structured logging overhead (`scf_log_format=none` vs `jsonl/csv`)
- Document benchmark methodology and reference systems
- Store baseline data for regression tracking

**Acceptance Criteria**
- Wall-time benchmarks documented for E3, E15, E18 optimizations
- Benchmark scripts are repeatable and documented
- Baseline performance data stored in repository


### E28 - Cleanup Compatibility Wrappers and Legacy APIs
**Priority**: P4 (Code quality - reduces tech debt)
**Status**: Open
**Files**: `control/src/lib.rs`, `kpts/src/lib.rs`, `special/src/lib.rs`

- Migrate remaining compatibility wrappers to fully result-based callsites:
  - E4 follow-up: `control.read_file`, `kpts::new`, `special::spherical_bessel_jn`
  - E5 follow-up: remove compatibility string getters after typed enum adoption
- Add deprecation warnings to compatibility APIs
- Provide migration guide for downstream code

**Acceptance Criteria**
- All new code uses `try_*` APIs and typed enums
- Compatibility wrappers are marked deprecated with migration guidance
- Migration guide documents API transitions


### E29 - CI Reproducibility and Determinism Tests
**Priority**: P2 (Critical for long-term stability)
**Status**: Open
**Files**: `tests/`, `.github/workflows/`, CI configuration

- Add CI replay test for E16 provenance:
  - Run fixed-seed reference SCF twice
  - Enforce numeric stability (energy/force/stress within tolerance)
  - Validate manifest stability (identical provenance fingerprints)
- Add determinism tests with different MPI ranks but same results
- Test restart reproducibility (run-to-checkpoint vs checkpoint-restart)

**Acceptance Criteria**
- CI runs fixed-seed reproducibility test on every PR
- Determinism test passes for 1-rank vs 2-rank runs (within tolerance)
- Restart parity test validates checkpoint reproducibility


### E30 - Migrate `vector3` to `nalgebra` and Retire Custom Vector Crate
**Priority**: P1 (Correctness + maintainability)
**Status**: In Progress (final cleanup in progress, 2026-03-02)
**Files**: `types/`, `gvector/`, `lattice/`, `crystal/`, `force/`, `ewald/`, `geom/`, `kpts/`, `pwbasis/`, `utility/`, `special/`, `vdw/`, workspace `Cargo.toml`

- Goal: replace project-local `vector3` math types with `nalgebra` `Vector3` equivalents, then remove `vector3` crate.
- Rationale:
  - remove layout-dependent `unsafe` slice casting in `vector3`
  - standardize on one math backend (already used in `matrix`)
  - reduce duplicated vector API maintenance

**Phase 0 - Design and Compatibility Strategy**
- Define canonical aliases:
  - `type Vec3f = nalgebra::Vector3<f64>`
  - `type Vec3i = nalgebra::Vector3<i32>`
- Define migration policy for field/component access (`x/y/z`), dot/cross/norm semantics, and formatting behavior.
- Identify all callsites using flattening helpers (`as_slice_of_element`, `as_mut_slice_of_element`) and classify:
  - read-only flatten to scalar buffer
  - mutable flatten for optimizer/BLAS-style APIs

**Phase 1 - Introduce Bridging Layer**
- Add a small `math3` compatibility module (in `vector3` crate first, then move/rename later) that wraps nalgebra types and exposes temporary compatibility helpers.
- Keep API-compatible constructors and convenience methods used across crates (`new`, `zeros`, dot/cross/norm aliases).
- Replace layout-dependent flattening with explicit safe representations where possible:
  - prefer `Vec<[f64; 3]>`/`Vec<[i32; 3]>` for bulk contiguous data paths
  - use explicit copy-in/copy-out adapters for mutable flat-slice APIs when zero-copy cannot be guaranteed safely

**Phase 2 - Incremental Crate Migration**
- Migrate leaf and utility crates first (`utility`, `special`, `fhkl`, `vdw`) to reduce risk.
- Migrate core geometry/data crates next (`lattice`, `crystal`, `gvector`, `pwbasis`, `kpts`).
- Migrate force/optimization paths last (`force`, `ewald`, `geom`) because they heavily use flat-slice mutation and are most regression-prone.
- After each crate batch:
  - run `cargo check` for full workspace
  - run targeted crate tests
  - run numerical spot checks for representative systems

**Phase 3 - Remove Legacy APIs**
- Deprecate and remove `vector3::as_slice_of_element` and `vector3::as_mut_slice_of_element`.
- Remove custom impl duplication that nalgebra already provides (vector arithmetic traits, dot/cross/norm wrappers where redundant).
- Switch all downstream crates from `vector3::Vector3f64/Vector3i32` imports to canonical nalgebra aliases.

**Phase 4 - Retire `vector3` Crate**
- Remove `vector3` crate from workspace members and crate dependencies.
- If a compatibility shim is still needed, keep a thin `math3` module in a shared crate with no `unsafe`.
- Update developer docs and migration notes with before/after API mapping examples.

**Validation Gates**
- `cargo check --workspace` and `cargo test --workspace` pass.
- Docker correctness gate passes: `scripts/run_phase12_regression.sh` (`FORCE_BUILD=1`).
- If force/stress/geometry paths changed, run `scripts/run_spin_mpi_parity.sh`.
- Numerical parity:
  - energies/forces/stresses within existing tolerances versus pre-migration baseline
  - no new non-determinism in fixed-seed replay tests.

**Implementation Update (2026-03-02)**
- [x] Replaced `vector3` crate core type with nalgebra alias (`Vector3<T> = nalgebra::Vector3<T>`)
- [x] Removed custom `vector3_f64.rs` and `vector3_i32.rs` operator/method implementations
- [x] Migrated existing callsites from custom methods to nalgebra API (`dot`, `cross`, `norm`, constructor literals)
- [x] Replaced vector `.to_vec()` usages with `.as_slice().to_vec()` where needed
- [x] Workspace compiles with `cargo check --workspace`
- [x] Ran Docker phase12 regression gate (`scripts/run_phase12_regression.sh` with `FORCE_BUILD=1`) and confirmed tolerance pass
- [x] Removed/replaced remaining `as_slice_of_element` unsafe bridge usage with safe flatten/scatter adapters
- [x] Retired `vector3` crate from workspace members and dependencies; vector aliases now live in `types`

**Acceptance Criteria**
- No layout-dependent `unsafe` remains in vector math access paths.
- All current `vector3` callsites compile against nalgebra types (directly or via temporary compatibility alias).
- Phase12 regression and spin/MPI parity gates pass after final cutover.
- `vector3` crate is removed from workspace (or reduced to a zero-unsafe compatibility shim with deprecation plan and sunset date).


### E31 - Replace `matrix` Core with `nalgebra` and Keep Only Domain-Specific Wrappers
**Priority**: P1/P2 (Maintainability + numerical safety)
**Status**: Open
**Files**: `matrix/`, `linalg/`, `kscf/`, `eigensolver/`, `stress/`, `force/`, `dfttypes/`, `workflow/`

- Goal: migrate matrix math/storage responsibilities to nalgebra, then reduce `matrix` crate to project-specific adapters only (I/O, checkpoints, compatibility helpers).
- Rationale:
  - remove duplicated linear algebra implementations
  - standardize numerical behavior and APIs across crates
  - reduce maintenance burden of custom indexing/ops code

**Phase 0 - Inventory and API Freeze**
- Catalog `matrix` APIs currently used by downstream crates:
  - construction (`new`, shape helpers, identity/zeros)
  - indexing and slicing patterns
  - arithmetic ops and linear solves/inverse/pseudo-inverse
  - file/HDF5 serialization hooks
- Mark external-facing APIs into:
  - keep (domain-specific wrappers)
  - migrate (pure algebra API)
  - deprecate (legacy compatibility methods)

**Phase 1 - Introduce nalgebra-Backed Internal Representation**
- Switch `matrix::Matrix<T>` storage internals to nalgebra-owned buffers (`DMatrix<T>` or static matrices where dimensions are fixed).
- Preserve existing public API signatures initially to minimize downstream churn.
- Add strict shape checks and convert panic-prone paths to `Result` where practical.

**Phase 2 - Migrate Call Sites to Native nalgebra APIs**
- Migrate hot and algebra-heavy call sites first (`linalg`, `eigensolver`, `kscf`, `stress`):
  - use nalgebra decomposition/solve paths directly
  - replace custom dot/matmul helper chains with nalgebra operations
- Keep `matrix` wrapper only where project-specific semantics are required.

**Phase 3 - Remove Duplicated Algebra Surface**
- Deprecate/remove duplicated methods in `matrix` that are thin wrappers over nalgebra (manual inverse/pinv adapters, ad-hoc arithmetic helpers, etc.).
- Keep only:
  - project-specific HDF5/checkpoint encoding helpers
  - compatibility constructors/parsers needed by control/workflow layers
  - explicit conversion glue (`from_nalgebra`, `into_nalgebra`) where still needed

**Phase 4 - Final Simplification**
- Decide final structure:
  - either retain `matrix` as a thin domain crate, or
  - fully inline remaining wrappers into `dfttypes`/I/O modules and retire `matrix`.
- Remove dead compatibility code and update developer docs with the new matrix policy.

**Validation Gates**
- `cargo check --workspace` and `cargo test --workspace` pass at each phase checkpoint.
- Numerical parity checks for representative kernels (eigensolver convergence, stress tensor, force assembly) remain within documented tolerances.
- Docker correctness gate passes: `scripts/run_phase12_regression.sh` (`FORCE_BUILD=1`).
- If force/stress/spin paths are touched, run `scripts/run_spin_mpi_parity.sh`.

**Acceptance Criteria**
- Algebra-heavy runtime paths use nalgebra directly or via zero-overhead wrappers.
- No duplicated custom linear algebra implementation remains for operations already covered by nalgebra.
- Project-specific `matrix` code is limited to domain adapters (I/O/checkpoint/compat), with clear boundaries.
- Regression and parity gates remain green after final cutover.
