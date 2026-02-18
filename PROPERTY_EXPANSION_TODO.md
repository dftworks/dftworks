# Property Expansion and Optimization TODO Roadmap

This roadmap is ordered from simple to complex so you can execute it as a staged development plan.

## Current baseline (already available)

- SCF total energy
- Forces
- Stress
- Bands workflow
- Wannier export pipeline (`w90-win`, `w90-amn`)

## Level 1 - Foundation hardening (Simple)

- [x] Define a stable property output layout (`runs/<stage>/properties/`) with machine-readable files (CSV/JSON).
- [x] Add a small postprocessing crate/module (`property`) for shared parsing, unit conversion, and formatting.
- [x] Add regression tests for existing properties (energy/force/stress) with tolerance checks.
- [x] Add timing and memory logging for SCF/NSCF stages to create optimization baselines.
- [x] Remove avoidable allocations in hot SCF loops by reusing existing workspace buffers.

Deliverable:
- One command to export standardized property files for a completed run.

## Level 2 - Basic electronic properties (Simple -> Medium)

- [x] Implement total DOS from NSCF eigenvalues with configurable smearing/broadening.
- [x] Implement automatic band gap analysis (direct/indirect, VBM/CBM k-point indices).
- [x] Add Fermi-level consistency checks between SCF and postprocessing.
- [x] Add CLI options for DOS grids and output formats.
- [x] Add tests using a known semiconductor case (gap and DOS shape sanity checks).

Deliverable:
- Reproducible `dos.dat` and `band_gap.json` for a standard test case.

## Level 3 - Projected properties (Medium)

- [ ] Implement PDOS (atom/orbital projected DOS) using projection weights.
- [ ] Implement fat-band output (band structure with projection weights per band/k-point).
- [ ] Reuse/cached projection matrices across repeated analyses to avoid recomputation.
- [ ] Add validation that sum(PDOS) approximately matches total DOS within tolerance.
- [ ] Add examples for `sp`, `sp2`, `sp3`, and `d` projector sets.

Deliverable:
- `pdos_*.dat` and `fatband_*.dat` with validation report.

## Level 4 - Equation-of-state and thermodynamic basics (Medium)

- [ ] Add automated volume-scan workflow (e.g., `-6%` to `+6%`).
- [ ] Fit Birch-Murnaghan EOS and report `V0`, `B0`, `B0'`, `E0`.
- [ ] Add static lattice enthalpy vs pressure output.
- [ ] Parallelize independent volume points and add restart support for interrupted scans.
- [ ] Add regression case comparing fitted constants to a reference range.

Deliverable:
- `eos_fit.json` + plot-ready table for energy-volume and pressure-volume curves.

## Level 5 - Elastic properties (Medium -> Hard)

- [ ] Implement finite-strain generation and stress collection.
- [ ] Fit elastic tensor `Cij` (symmetry-aware where available).
- [ ] Compute derived moduli (Voigt/Reuss/Hill bulk and shear, Young's modulus, Poisson ratio).
- [ ] Use symmetry to reduce number of strain calculations.
- [ ] Add quality checks (tensor symmetry, mechanical stability criteria for crystal class).

Deliverable:
- `elastic_tensor.json` and `elastic_summary.md`.

## Level 6 - Vibrational properties via finite displacement (Hard)

- [ ] Implement supercell builder and displacement patterns.
- [ ] Compute force constants and dynamical matrices.
- [ ] Compute phonon dispersion and phonon DOS.
- [ ] Add acoustic sum-rule enforcement and imaginary-mode detection.
- [ ] Add convergence workflow for supercell size and displacement amplitude.

Deliverable:
- `phonon_bands.dat`, `phonon_dos.dat`, and stability summary.

## Level 7 - Electric/optical response (Hard)

- [ ] Implement Berry-phase polarization (non-metal cases first).
- [ ] Implement dielectric tensor (start with finite-field or finite-difference strategy).
- [ ] Implement Born effective charges.
- [ ] Add IR-active mode intensities from phonons + Born charges.
- [ ] Add symmetry checks for tensor forms and coordinate conventions.

Deliverable:
- `polarization.json`, `dielectric_tensor.json`, `born_charges.json`.

## Level 8 - Advanced transport and topological responses (Very Hard)

- [ ] Add band interpolation path (Wannier-based) for dense k-space properties.
- [ ] Implement Boltzmann transport (conductivity/Seebeck vs temperature and chemical potential).
- [ ] Implement anomalous Hall/Berry-curvature integration workflows.
- [ ] Add scalable parallel execution for dense k/q sampling with deterministic reductions.
- [ ] Add benchmark suite for performance and reproducibility on large systems.

Deliverable:
- `transport_*.json` and optional `berry_curvature_*.dat` datasets.

## Cross-cutting optimization backlog (apply at every level)

- [ ] Keep hot loops allocation-free via context/state/workspace patterns.
- [ ] Replace string-dispatch runtime modes with typed enums in property pipelines.
- [ ] Add deterministic reduction utilities for threaded/MPI aggregation.
- [ ] Add property-level profiling harnesses and CI regression thresholds.
- [ ] Centralize error handling with `Result`-based library APIs (no `process::exit` in libs).

## Suggested execution rule

- Finish one level only when:
  - unit/integration tests pass,
  - one reference example is documented,
  - runtime and memory metrics are recorded,
  - output schema is versioned and backward-compatible.
