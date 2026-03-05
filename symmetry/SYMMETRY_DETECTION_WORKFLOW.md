# Symmetry Detection Workflow (`symmetry` crate)

This document explains how symmetry detection is performed in dftworks today, from structure input to usable symmetry operations and irreducible k-point mapping.

The implementation is split between:

- `symmetry` crate: runtime-facing driver, atom mapping, FFT/k-mesh filters, irreducible k-mesh generation.
- `symops` crate: symmetry-operation algebra, operation detection, and lightweight classification.

No external symmetry library is required.

## 1. Primary Entry Point: `SymmetryDriverInternal::new`

The main constructor is `symmetry::SymmetryDriverInternal::new(latt, position, types, symprec)`.

### Step 1: Input normalization and detection options

- If `symprec <= 0`, fallback to `1.0e-6`.
- Build `DetectOptions`:
  - `symprec = symprec`
  - `metric_tol = symprec * 10.0`
  - `validate_group = true`
- Normalize lattice rows via `normalize_lattice_rows`:
  - Uses first 3 rows when present.
  - Falls back to identity lattice if input is undersized.

### Step 2: Convert to `symops::Structure`

- Construct:
  - `lattice: [[f64; 3]; 3]`
  - `positions: Vec<[f64; 3]>`
  - `atom_types: Vec<i32>`

### Step 3: Detect operations (`symops::detect_symmetry`)

- Call `detect_symmetry(&structure, options)`.
- On success, use returned operation list.
- On failure (or empty list), fallback to identity only:
  - `vec![SymOp::identity()]`

### Step 4: Classify operation set (`symops::classify_symmetry`)

- Call `classify_symmetry(&operations)`.
- Use classification when available:
  - `spacegroup_number` (limited heuristic, often `None`)
  - `point_group_hint` (string hint)
- Fallbacks in `symmetry` driver:
  - `spacegroup_number = 0` when unavailable
  - `point_group_hint = "1"` when unavailable
- `hall_number` is currently fixed to `0`.

### Step 5: Internal data conversion

- Extract and store:
  - `rotations: Vec<[[i32; 3]; 3]>`
  - `translations: Vec<[f64; 3]>`

### Step 6: Build atom mapping table (`sym_atom`)

- Build `sym_atom[iat][isym]` with `build_sym_atom_map`.
- For each atom and operation:
  - Apply `mapped = R * x + t` (`apply_operation`).
  - Wrap mapped and target positions with `wrap_centered`.
  - Match atom index with same species and coordinate difference `< symprec`.
- If no exact match is found, fallback to original atom index (`iat`) to remain in-bounds.

This completes the runtime symmetry dataset used by SCF/force/stress/k-point workflows.

## 2. Detection Algorithm in `symops::detect_symmetry`

The core detector follows this sequence:

### Step A: Validate structure inputs

- Reject empty structure.
- Reject mismatch between number of positions and atom types.
- Reject non-positive tolerances.

### Step B: Normalize positions

- Fractional positions are normalized into canonical fractional range.

### Step C: Build lattice metric tensor

- Compute `G = A * A^T` from lattice rows.

### Step D: Generate candidate rotations

- Enumerate all `3^9` integer matrices with entries in `{-1, 0, 1}`.
- Keep only matrices with determinant `+1` or `-1`.
- Keep only metric-preserving candidates:
  - `R^T G R ~= G` within `metric_tol`.

### Step E: Generate translation candidates (anchor method)

- Use anchor atom `0`.
- Rotate anchor position with candidate `R`.
- For same-species target atoms, form candidate:
  - `t = target - R * anchor` (fractional, normalized).

### Step F: Validate each `(R, t)` globally

- Check that every atom maps onto a same-species atom under `(R, t)` within `symprec`.
- Build `SymOp` only for valid operations.
- Deduplicate approximately equal operations (modulo lattice translations).

### Step G: Canonicalization and validation

- Canonical sort for deterministic ordering.
- Validate lattice consistency.
- Optional group validation (`validate_group = true`):
  - identity exists
  - closure
  - inverses

### 2.1 Rotation candidate math (explicit)

Given direct lattice rows `A = [a; b; c]`, metric is:

- `G_ij = a_i · a_j`

For each integer candidate rotation `R`, the detector checks:

- `R^T G R ≈ G`

Implementation detail:

- Element-wise absolute error is used.
- Candidate passes only if every component satisfies `| (R^T G R)_ij - G_ij | <= metric_tol`.
- Determinant gate (`det(R) = ±1`) is applied before metric test.

### 2.2 Anchor-based translation generation

For each accepted `R`, translation candidates are generated from anchor atom `a0`:

- Rotate anchor: `x0' = R * x0`
- For every same-species target atom `xj`, candidate translation:
  - `t_j = wrap_frac(xj - x0')`

Dedup is applied with modulo-lattice equivalence:

- two translations are equal if their centered wrapped difference is within `symprec`.

This converts translation search from an unconstrained 3D problem into a finite set derived from actual atomic sites.

### 2.3 Full operation validation (`is_valid_operation`)

Every `(R, t)` candidate is validated by mapping all atoms:

- `x_i^mapped = wrap_frac(R * x_i + t)`

Validation enforces:

- species preservation (`type(mapped_target) == type(i)`)
- one-to-one assignment (each target atom can be used once)
- positional match modulo lattice within `symprec`

Current implementation uses greedy matching over target atoms with a `used[]` mask.

Important implication:

- This is deterministic for fixed atom ordering.
- It is not a globally optimal assignment solver (for current crystal use-cases this is acceptable and fast).

### 2.4 Canonicalization and deterministic ordering

After collecting valid operations:

- approximate duplicates are removed (rotation exact + translation modulo-lattice tolerance)
- output is lexicographically sorted by:
  - rotation matrix entries
  - then translation components

This guarantees stable operation ordering across runs with the same floating-point behavior.

### 2.5 Validation stages and why both exist

- `validate_lattice_consistency`:
  - checks each operation independently against lattice metric invariance
  - catches numerically inconsistent operations even if atom mapping happened to pass
- `validate_group`:
  - checks set-level algebraic properties (identity, inverse, closure)
  - catches incomplete operation sets

Both are enabled in the default runtime path via `DetectOptions { validate_group: true, ... }`.

### 2.6 Rough complexity profile

Let:

- `N = number of atoms`
- `M = number of candidate rotations after metric filtering`
- `T_r = number of translation candidates for a given rotation`

Then dominant detection cost is approximately:

- `O( M * T_r * N^2 )`

Reason:

- each `(R,t)` validation maps all `N` atoms
- each mapped atom may scan up to `N` targets in greedy matching

In practice, metric filtering and species constraints reduce `M` and effective `T_r` strongly for most structures.

## 3. Classification in `symops::classify_symmetry`

`classify_symmetry` is intentionally lightweight:

- Detects coarse crystal-system class from observed rotation orders.
- Reports:
  - `point_group_hint`
  - `has_inversion`
  - `n_operations`
  - `n_proper_rotations`
  - `max_rotation_order`
- Space-group number inference is very limited:
  - `1` for identity-only (`P1`)
  - `2` when exactly identity + inversion (`P-1`)
  - otherwise `None` (driver maps to `0`)

## 4. Runtime Utility Workflow in `symmetry`

### 4.1 Applying operations to vectors

- `operation_on_vector(isym, v)`:
  - rotate via integer matrix
  - then translate via fractional translation

- `center_vector(v)`:
  - wraps each component to centered cell convention (about `[-0.5, 0.5]`).

### 4.2 FFT/k-mesh commensurate operation filtering

- `get_fft_commensurate_ops(fftmesh, kmesh, symprec)` keeps only operations where translation components are commensurate with both:
  - FFT mesh divisibility checks
  - k-mesh divisibility checks

This is used to avoid operations that break discrete mesh consistency.

## 5. Irreducible Reciprocal Mesh Workflow

Entry point: `get_ir_reciprocal_mesh(mesh, is_shift, lattice, position, types, symprec)`.

### Step 1: Build full grid

- Generate all fractional k-points and integer grid addresses for `mesh`.
- Apply `is_shift` in the same formula used by the rest of the code.

### Step 2: Build symmetry driver

- Instantiate `SymmetryDriverInternal` from the same lattice/positions/types/symprec.

### Step 3: Construct mapping table

- Initialize `ir_mapping_table` with `-1`.
- For each yet-unmapped k-point:
  - mark itself as representative
  - apply all symmetry rotations (`R*k`) to find equivalent points
  - map rotated point back to grid index with `map_k_to_grid_index`
  - assign representative index

Note: reciprocal mapping uses rotation only. Translation does not affect `k` mapping.

### 5.1 Grid index reconstruction algorithm

`map_k_to_grid_index` converts a rotated fractional `k` back to mesh index robustly:

- Normalize component `k_norm` into fractional canonical range.
- Solve index from shifted mesh equation:
  - `raw = 2 * n * k_norm - shift`
  - `i = round(raw / 2)`
- Wrap index into valid range `[0, n-1]`.
- Reconstruct fractional value from index:
  - `k_reconstructed = (2*i + shift) / (2*n)`
- Accept mapping only if centered wrapped difference from `k_norm` is within tolerance.

This reject-on-reconstruction check prevents accidental index aliasing due to floating-point drift.

### Step 4: Finalize representatives and degeneracies

- Any unmapped point maps to itself.
- Build unique representative list `ir_ikpt`.
- Count multiplicities `ir_ikpt_degeneracy`.

Return values:

- `kpts`: full mesh points
- `ir_mapping_table`: full-to-irreducible mapping
- `ir_ikpt`: representative indices
- `ir_ikpt_degeneracy`: multiplicity for each representative

## 6. Tolerances and Numerical Behavior

- Driver-level fallback: `symprec = 1.0e-6` when non-positive.
- Detector metric tolerance is set to `symprec * 10`.
- Equality checks use wrapped comparisons (`wrap_centered` / modulo-lattice logic).
- `map_k_to_grid_index` has a lower bound tolerance of `1.0e-8`.

### 6.1 Practical tolerance coupling

- `symprec` controls atom mapping strictness and operation dedup.
- `metric_tol` controls rotation admissibility against lattice metric.
- Current default coupling (`metric_tol = 10 * symprec`) is intentionally looser for metric filtering than atom-position matching.

If `symprec` is tightened too much:

- valid operations can drop due to finite precision in fractional coordinates.

If `symprec` is loosened too much:

- near-symmetry noise can be interpreted as true symmetry.

## 7. Failure/edge-case behavior (runtime contract)

- If `detect_symmetry` fails or returns empty:
  - runtime driver falls back to identity-only operation set.
- If atom mapping lookup in `build_sym_atom_map` cannot find a match:
  - mapping entry falls back to the source atom index.
- If reciprocal mesh input is invalid (`mesh[d] <= 0`):
  - `get_ir_reciprocal_mesh` returns empty vectors.

## 8. Current Limits (Important)

- Hall number is not inferred (`hall_number = 0`).
- Space-group number support is heuristic and limited (mostly 1/2, else 0 in driver).
- Rotation search is restricted to integer entries in `{-1, 0, 1}`.
- If atom mapping under an operation fails at lookup time, mapping falls back to self index.
- The stack is designed as an in-tree bootstrap detector, not a full table-driven crystallographic engine.

## 9. Practical Debug/Validation Hooks

- `display()` and `display_brief()` print detected dataset summary.
- Unit tests in `symmetry/src/lib.rs` cover:
  - trivial cell operation detection
  - nontrivial silicon symmetry
  - irreducible k-mesh mapping consistency
- Additional logic tests live in `symops` (`detect.rs`, `classify.rs` tests).

## 10. Minimal Call Graph

1. `symmetry::new(...)`
2. `SymmetryDriverInternal::new(...)`
3. `symops::detect_symmetry(...)`
4. `symops::classify_symmetry(...)`
5. `build_sym_atom_map(...)`
6. downstream usage:
   - vector operations
   - FFT/k commensurate filtering
   - `get_ir_reciprocal_mesh(...)`
