# Workspace Optimization Plan

This document captures prioritized software engineering improvements identified in the workspace review.

## P1 - Correctness and Reliability

### 3) Remove hard-coded linker paths from build scripts
- Problem:
  - `matrix/build.rs` and `symmetry/build.rs` hardcode local library paths.
- Plan:
  - Use environment-driven discovery (`LAPACK_DIR`, `SPGLIB_DIR`) or `pkg-config`.
  - Add explicit build-time diagnostics if libraries are not found.
  - Keep optional per-platform fallback logic gated behind env flags.
- Expected impact:
  - More portable builds across machines and CI runners.

## P2 - Performance and Throughput

### 4) Introduce reusable workspace objects for hot SCF loops
- Problem:
  - Repeated per-iteration allocations in SCF/KSCF density and Hamiltonian paths.
- Plan:
  - Add `KscfWorkspace` and `DensityWorkspace` structs holding reusable buffers.
  - Allocate once per geometry/SCF stage and pass mutable references through loops.
  - Keep workspace sizing explicit and validated.
- Expected impact:
  - Lower allocator pressure and better runtime stability for large systems.

### 5) Remove artificial delay from eigenvalue output path
- Problem:
  - Rank-serialized output includes fixed sleeps and barriers.
- Plan:
  - Gate verbose eigenvalue printing by verbosity/debug option.
  - Remove fixed sleep in production path.
  - Keep optional ordered-rank debug print mode for troubleshooting.
- Expected impact:
  - Shorter wall time and cleaner MPI behavior in production runs.

### 6) Reuse eigensolver scratch memory
- Problem:
  - `eigensolver/src/pcg.rs` repeatedly allocates projection/temp vectors in inner loops.
- Plan:
  - Move temporary vectors into solver state and clear/reuse per band iteration.
  - Avoid short-lived `Vec` creation in Gram-Schmidt and lower-band orthogonalization.
- Expected impact:
  - Reduced overhead in heavy diagonalization steps.

## P2/P3 - API and Error Handling

### 7) Remove `process::exit` from library crates
- Problem:
  - Library-level code (`density`, `control`) terminates the process directly.
- Plan:
  - Return typed errors (`Result<_, Error>`) from library functions.
  - Centralize exit behavior in `pw/src/main.rs`.
  - Add context-rich error propagation (`thiserror`/custom error enums).
- Expected impact:
  - Better composability, testability, and embedding behavior.

## P3 - Maintainability and Development Velocity

### 8) Refactor oversized orchestration interfaces
- Problem:
  - Large parameter lists (e.g., SCF trait `run`) and long monolithic `main`.
- Plan:
  - Introduce grouped context structs (`ScfContext`, `RuntimeContext`, `PostprocessContext`).
  - Split `pw/src/main.rs` into phase-oriented functions/modules:
    - input/bootstrap
    - basis/construction
    - SCF execution
    - outputs/postprocessing
  - Keep interfaces explicit and immutable where possible.
- Expected impact:
  - Easier evolution of features and lower regression risk during refactors.

### 9) Reduce blanket warning suppression and improve test coverage
- Problem:
  - Widespread `#![allow(warnings)]` usage and limited tests in SCF/KSCF/PW paths.
- Plan:
  - Remove blanket warning suppression incrementally, replacing with narrow `allow(...)` only where justified.
  - Add integration tests for:
    - SCF convergence on small benchmark systems
    - energy component consistency checks
    - deterministic behavior under fixed seeds/settings
  - Add CI gates for `cargo check` + selected tests.
- Expected impact:
  - Higher code quality signal and safer iteration speed.

## P4 - Future Physics Features

### 10) Advance non-collinear roadmap (post-Phase 1/2)
- Completed on 2026-02-13 (removed from active backlog):
  - Phase 1 complete: production PBE kernel on collinear `nonspin` and `spin`, including `drho` plumbing and tests.
  - Phase 2 complete: enum-driven `spin_scheme` (`nonspin`/`spin`/`ncl`) with NCL extension hooks and parser/runtime factory refactor.
  - Regression artifacts added:
    - `test_example/si-oncv/regression/phase12_reference.tsv`
    - `test_example/si-oncv/regression/phase12_parity_report.md`
- Remaining roadmap (Phase 3+):
  - Implement spinor wavefunction and magnetization-density data model for true NCL physics.
  - Extend SCF pipeline for NCL Hamiltonian assembly, mixing, and occupation handling.
  - Add NCL-specific XC integration and validation benchmarks.
  - Define parity and reference targets for NCL vs trusted external solvers.
- Expected impact:
  - Keeps plan focused on remaining NCL deliverables after finishing the PBE and spin-mode groundwork.

### 11) Add full symmetry support (irreducible k-mesh and field symmetrization)
- Problem:
  - Symmetry data is present in parts of the codebase, but end-to-end exploitation is incomplete/inconsistent.
- Plan:
  - Build a centralized symmetry context/service from spglib output (operations, mapping, irreducible sets).
  - Use symmetry to reduce k-point workloads (irreducible k-mesh generation + proper weights).
  - Apply symmetry operations consistently to:
    - charge density / potential
    - forces / stress
    - optional occupation and postprocessing paths where relevant
  - Add validation tests:
    - symmetric structure invariants
    - force cancellation in high-symmetry systems
    - agreement between full-mesh and irreducible-mesh energies within tolerance
- Expected impact:
  - Lower runtime cost and improved numerical robustness/consistency.

## P5 - Rust Scalability Patterns

### 12) Replace string-based runtime modes with typed enums
- Problem:
  - Many factories rely on string matching (`spin_scheme`, `smearing`, `kpts`, etc.), allowing invalid states and runtime fallback behavior.
- Plan:
  - Introduce typed configuration enums parsed once at input load time.
  - Use exhaustive `match` on enums across constructors and drivers.
  - Remove repeated string comparisons in runtime paths.
- Expected impact:
  - Better compile-time safety and cleaner extension path for new modes (PBE/symmetry variants).

### 13) Prefer static dispatch in hot kernels
- Problem:
  - Dynamic dispatch is useful for boundaries, but costly/opaque in performance-critical inner loops.
- Plan:
  - Keep trait objects at high-level orchestration boundaries.
  - Use enums/generics for hot compute kernels where implementations are known at compile time.
  - Measure before/after in SCF and eigensolver-heavy kernels.
- Expected impact:
  - Better inlining and reduced dispatch overhead in performance-critical code.

### 14) Standardize Context + State + Workspace pattern across drivers
- Problem:
  - This pattern is partially used but not standardized globally.
- Plan:
  - Expand item (8) into a shared convention for SCF, density, force, stress, and geometry drivers.
  - Document API template:
    - context = immutable problem data
    - state = iteration-evolving data
    - workspace = reusable scratch memory
  - Refactor modules incrementally to the same shape.
- Expected impact:
  - Easier scaling of features and lower coupling between algorithmic phases.

### 15) Parallelize by k-point with deterministic reductions
- Problem:
  - Current loops are mostly serial in key sections and some output/reduction paths are rank-serialized.
- Plan:
  - Parallelize independent k-point work using thread-level parallel iterators.
  - Preserve deterministic reduction order for reproducible results.
  - Keep MPI + thread interplay explicit and benchmarked.
- Expected impact:
  - Stronger scaling on multicore nodes while keeping numerical reproducibility.

### 16) Complete migration to Result-based library error handling
- Problem:
  - Some library layers still use process termination or panic-style behavior.
- Plan:
  - Extend item (7): all library crates should return typed errors.
  - Reserve process exit policy for binary entry points only.
  - Add structured error context at module boundaries.
- Expected impact:
  - Better composability, testing, and integration for larger workflows.

### 17) Expand reusable workspace coverage to all hot compute paths
- Problem:
  - Workspace reuse has started, but allocations remain in density/eigensolver and related loops.
- Plan:
  - Extend item (4) to include density, eigensolver, force, stress, and FFT-heavy helper paths.
  - Add clear ownership/lifetime strategy for work buffers.
  - Track allocation counts in profiling to verify gains.
- Expected impact:
  - Lower allocation churn and improved scaling for large systems.

### 18) Adopt benchmark-driven optimization workflow
- Problem:
  - Performance work is not consistently guarded by benchmark baselines.
- Plan:
  - Add benchmark harnesses (for example, Criterion-based microbench + end-to-end SCF timing cases).
  - Track scaling dimensions:
    - number of atoms
    - number of k-points
    - thread count
  - Add performance regression checks in CI for representative workloads.
- Expected impact:
  - Data-driven optimization decisions and better long-term performance stability.

## P6 - Framework Modernization

### 19) Build a typed configuration framework layer
- Problem:
  - Runtime behavior is still driven by string selectors in multiple modules.
- Plan:
  - Add a single typed config layer parsed once in `control`, then passed as typed enums/structs.
  - Remove string-based branching from runtime drivers.
  - Start migration in:
    - `control/src/lib.rs`
    - `scf/src/lib.rs`
    - `smearing/src/lib.rs`
    - `density/src/lib.rs`
    - `kpts/src/lib.rs`
- Expected impact:
  - Cleaner framework contracts and safer feature extension.

### 20) Redesign driver contracts using Context + State + Workspace
- Problem:
  - Orchestration is hard to scale with large argument lists and a monolithic `main`.
- Plan:
  - Introduce framework-level types:
    - `ScfContext`, `ScfState`, `ScfWorkspace`
    - similar patterns for density/force/stress
  - Split orchestration in `pw/src/main.rs` into phase modules.
- Expected impact:
  - Lower coupling, easier testing, and easier onboarding for new feature work.

### 21) Define scalable execution framework (threaded k-point engine + deterministic reductions)
- Problem:
  - Current execution is not organized as an explicit scalable engine.
- Plan:
  - Add a k-point execution layer that supports thread parallelism and deterministic global reductions.
  - Ensure compatibility with MPI decomposition and reproducible summation rules.
  - Integrate with existing SCF utilities and KSCF kernels.
- Expected impact:
  - Better multicore scaling and predictable numerical behavior.

### 22) Formalize framework-level validation (performance + correctness)
- Problem:
  - No unified framework gate for runtime scalability and physics regressions.
- Plan:
  - Add framework benchmark and regression suite:
    - end-to-end SCF timing matrix (atoms, k-points, threads)
    - convergence and energy-consistency checks
    - reproducibility checks under fixed seed/settings
  - Hook into CI as required status checks.
- Expected impact:
  - Confident framework evolution with measurable performance targets.

### 23) Complete framework-wide Result/error model
- Problem:
  - Error policy is inconsistent between libraries and binary entry points.
- Plan:
  - Enforce `Result`-based error returns for framework/library crates.
  - Keep process termination policy only in executable entry points.
  - Standardize error taxonomy and propagation format.
- Expected impact:
  - Better composability and cleaner integration surface for future tooling.

## Recommended Execution Order

1. P1 item (3) first to eliminate correctness and portability risk.
2. P2 items (4-6) next for runtime gains.
3. P2/P3 item (7) then API cleanup (8).
4. Finish current hardening with (9).
5. Implement future feature roadmap (10-11) with benchmark validation, in this order: PBE on existing modes -> non-collinear extension -> symmetry expansion.
6. Apply Rust scalability patterns (12-18), prioritizing 12, 15, and 18 first.
7. Execute framework modernization track (19-23), beginning with 19 and 20.

## Delivery Strategy

- Phase A: correctness + build portability
- Phase B: hot-path memory reuse
- Phase C: API/error refactor
- Phase D: tests + lint tightening
- Phase E: feature enablement (phased PBE + non-collinear + full symmetry)
- Phase F: Rust scalability architecture + benchmarking
- Phase G: framework modernization and execution engine

Each phase should include:
- implementation tasks
- benchmark/validation checks
- rollback-safe commits
