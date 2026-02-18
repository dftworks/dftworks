//! Symmetry-operation detection from a crystal structure.
//!
//! Current strategy (first-pass implementation):
//! 1. Build metric tensor from lattice vectors.
//! 2. Enumerate small integer rotation candidates (`-1/0/1` entries) and keep
//!    those that preserve the metric tensor within tolerance.
//! 3. For each rotation, generate candidate translations from one anchor atom.
//! 4. Verify full species-preserving atom mapping for each `(R, t)`.
//! 5. Canonicalize, deduplicate, and optionally validate group properties.
//!
//! This is intentionally narrower than full table-driven crystallographic behavior. It is designed as
//! an internal bootstrap detector for dftworks workflows and can be expanded.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use crate::{
    approx_eq_mod_lattice, determinant, sym_op_approx_eq, validate_group,
    validate_lattice_consistency, Rotation, SymOp, SymOpError, Vector3,
};

#[derive(Clone, Debug)]
pub struct Structure {
    /// Direct lattice vectors in Cartesian coordinates as rows `[a, b, c]`.
    ///
    /// Metric is computed as `G_ij = a_i · a_j` from these rows.
    pub lattice: [[f64; 3]; 3],
    /// Atomic fractional coordinates in the same basis.
    ///
    /// Coordinates can be outside `[0,1)`; they are normalized internally.
    pub positions: Vec<Vector3>,
    /// Species labels aligned with positions.
    ///
    /// Mapping checks only allow atoms to map onto atoms with equal label.
    pub atom_types: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// Position tolerance in fractional coordinates.
    ///
    /// Used in atom mapping and operation deduplication comparisons.
    pub symprec: f64,
    /// Lattice-metric invariance tolerance for candidate rotations.
    ///
    /// Candidate `R` is accepted when `R^T G R ≈ G` under this tolerance.
    pub metric_tol: f64,
    /// Validate the resulting set as a closed group.
    pub validate_group: bool,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            symprec: 1.0e-6,
            metric_tol: 1.0e-6,
            validate_group: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DetectedSymmetry {
    /// Canonicalized operation list.
    pub operations: Vec<SymOp>,
    /// Number of accepted rotation candidates prior to translation search.
    ///
    /// This is useful for profiling/tuning detector scope.
    pub candidate_rotations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DetectionError {
    /// Input structure has no sites.
    EmptyStructure,
    /// Number of coordinates does not match number of species labels.
    MismatchedInputs { positions: usize, atom_types: usize },
    /// One or more tolerances are non-positive.
    NonPositiveTolerance,
    /// No valid `(R,t)` survived filtering.
    NoOperationsDetected,
    /// Construction of `SymOp` failed (typically non-unimodular rotation).
    InvalidOperation(SymOpError),
    /// Operation set failed lattice-consistency checks (`R^T G R ≈ G`).
    LatticeValidationFailed(SymOpError),
    /// Optional group validation failed.
    GroupValidationFailed(SymOpError),
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectionError::EmptyStructure => write!(f, "structure has no atoms"),
            DetectionError::MismatchedInputs {
                positions,
                atom_types,
            } => write!(
                f,
                "mismatched inputs: {} positions but {} atom types",
                positions, atom_types
            ),
            DetectionError::NonPositiveTolerance => {
                write!(f, "tolerances must be positive")
            }
            DetectionError::NoOperationsDetected => write!(
                f,
                "failed to detect any symmetry operation with current candidate set"
            ),
            DetectionError::InvalidOperation(err) => {
                write!(f, "invalid symmetry operation: {}", err)
            }
            DetectionError::LatticeValidationFailed(err) => {
                write!(f, "detected operations failed lattice validation: {}", err)
            }
            DetectionError::GroupValidationFailed(err) => {
                write!(f, "detected operations failed group validation: {}", err)
            }
        }
    }
}

impl Error for DetectionError {}

/// Detects symmetry operations for the provided structure.
///
/// Notes:
/// - Rotation candidate search is currently limited to matrices with entries in
///   `{-1, 0, 1}`.
/// - Translation candidates are seeded from one anchor atom and then validated
///   globally against all atoms.
/// - Returned operations are deduplicated and stable-sorted.
pub fn detect_symmetry(
    structure: &Structure,
    options: DetectOptions,
) -> Result<DetectedSymmetry, DetectionError> {
    if structure.positions.is_empty() {
        return Err(DetectionError::EmptyStructure);
    }
    if structure.positions.len() != structure.atom_types.len() {
        return Err(DetectionError::MismatchedInputs {
            positions: structure.positions.len(),
            atom_types: structure.atom_types.len(),
        });
    }
    if options.symprec <= 0.0 || options.metric_tol <= 0.0 {
        return Err(DetectionError::NonPositiveTolerance);
    }

    let positions: Vec<Vector3> = structure
        .positions
        .iter()
        .copied()
        .map(normalize_fractional)
        .collect();
    let atom_types = &structure.atom_types;

    let metric = metric_tensor(structure.lattice);
    let rotations = generate_rotation_candidates(metric, options.metric_tol);

    // Anchor-atom method:
    // for each candidate rotation, possible translations are generated by
    // mapping this atom onto same-species sites.
    let anchor = 0usize;
    let anchor_type = atom_types[anchor];
    let mut operations = Vec::new();
    for rotation in rotations.iter().copied() {
        let anchor_rot = apply_rotation(rotation, positions[anchor]);
        let translations = candidate_translations(
            anchor_rot,
            anchor_type,
            &positions,
            atom_types,
            options.symprec,
        );

        for translation in translations {
            if !is_valid_operation(
                rotation,
                translation,
                &positions,
                atom_types,
                options.symprec,
            ) {
                continue;
            }

            let op = SymOp::new(rotation, translation).map_err(DetectionError::InvalidOperation)?;
            if !operations
                .iter()
                .any(|existing| sym_op_approx_eq(existing, &op, options.symprec))
            {
                operations.push(op);
            }
        }
    }

    if operations.is_empty() {
        return Err(DetectionError::NoOperationsDetected);
    }

    operations = standardize_operations(&operations, options.symprec);

    validate_lattice_consistency(&operations, structure.lattice, options.metric_tol)
        .map_err(DetectionError::LatticeValidationFailed)?;

    if options.validate_group {
        // Optional expensive check: useful for development and CI.
        validate_group(&operations, options.symprec)
            .map_err(DetectionError::GroupValidationFailed)?;
    }

    Ok(DetectedSymmetry {
        operations,
        candidate_rotations: rotations.len(),
    })
}

/// Produces a canonical operation list:
/// - removes approximate duplicates (`tol` modulo lattice),
/// - sorts by rotation then translation for deterministic output.
pub fn standardize_operations(ops: &[SymOp], tol: f64) -> Vec<SymOp> {
    let mut out = Vec::new();
    for op in ops.iter() {
        if !out
            .iter()
            .any(|existing| sym_op_approx_eq(existing, op, tol))
        {
            out.push(op.clone());
        }
    }
    canonical_sort_ops(&mut out);
    out
}

/// Computes direct-lattice metric tensor `G = A * A^T`.
fn metric_tensor(lattice: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut metric = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            metric[i][j] = lattice[i][0] * lattice[j][0]
                + lattice[i][1] * lattice[j][1]
                + lattice[i][2] * lattice[j][2];
        }
    }
    metric
}

/// Enumerates candidate integer rotations that preserve metric within tolerance.
///
/// Current enumeration space is `3^9` matrices with entries in `{-1,0,1}`,
/// then filtered by `det = ±1` and metric invariance.
fn generate_rotation_candidates(metric: [[f64; 3]; 3], tol: f64) -> Vec<Rotation> {
    let mut out = Vec::new();
    for code in 0..3_usize.pow(9) {
        let mut x = code;
        let mut vals = [0_i32; 9];
        for v in vals.iter_mut() {
            *v = (x % 3) as i32 - 1;
            x /= 3;
        }

        let rotation = [
            [vals[0], vals[1], vals[2]],
            [vals[3], vals[4], vals[5]],
            [vals[6], vals[7], vals[8]],
        ];
        let det = determinant(rotation);
        if det != 1 && det != -1 {
            continue;
        }
        if metric_preserving(rotation, metric, tol) {
            out.push(rotation);
        }
    }

    out.sort_by(cmp_rotation);
    out
}

/// Returns true if `R^T G R` matches `G` elementwise within `tol`.
fn metric_preserving(rotation: Rotation, metric: [[f64; 3]; 3], tol: f64) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            let mut transformed = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    transformed += rotation[a][i] as f64 * metric[a][b] * rotation[b][j] as f64;
                }
            }
            if (transformed - metric[i][j]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Generates translation candidates by matching rotated anchor to same-species atoms.
fn candidate_translations(
    rotated_anchor: Vector3,
    anchor_type: i32,
    positions: &[Vector3],
    atom_types: &[i32],
    tol: f64,
) -> Vec<Vector3> {
    let mut out = Vec::new();
    for (idx, target) in positions.iter().enumerate() {
        if atom_types[idx] != anchor_type {
            continue;
        }
        let candidate = normalize_fractional(sub(*target, rotated_anchor));
        if !out
            .iter()
            .any(|existing| approx_eq_mod_lattice(*existing, candidate, tol))
        {
            out.push(candidate);
        }
    }
    out
}

/// Verifies that `(rotation, translation)` maps every atom onto a unique
/// same-species atom under modulo-lattice comparison.
fn is_valid_operation(
    rotation: Rotation,
    translation: Vector3,
    positions: &[Vector3],
    atom_types: &[i32],
    tol: f64,
) -> bool {
    // Greedy one-to-one matching buffer.
    let mut used = vec![false; positions.len()];

    for (idx, position) in positions.iter().enumerate() {
        let mapped = normalize_fractional(add(apply_rotation(rotation, *position), translation));
        let atom_type = atom_types[idx];

        let mut found = None;
        for target_idx in 0..positions.len() {
            if used[target_idx] || atom_types[target_idx] != atom_type {
                continue;
            }
            if approx_eq_mod_lattice(mapped, positions[target_idx], tol) {
                found = Some(target_idx);
                break;
            }
        }

        match found {
            Some(target_idx) => used[target_idx] = true,
            None => return false,
        }
    }

    true
}

/// Deterministic ordering used for stable output and reproducible tests.
fn canonical_sort_ops(operations: &mut [SymOp]) {
    operations.sort_by(|lhs, rhs| {
        let rot_cmp = cmp_rotation(lhs.rotation(), rhs.rotation());
        if rot_cmp != Ordering::Equal {
            return rot_cmp;
        }
        let ltr = lhs.translation();
        let rtr = rhs.translation();
        for i in 0..3 {
            match ltr[i].partial_cmp(&rtr[i]) {
                Some(Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => return Ordering::Equal,
            }
        }
        Ordering::Equal
    });
}

/// Lexicographic compare for integer rotation matrices.
fn cmp_rotation(lhs: &Rotation, rhs: &Rotation) -> Ordering {
    for i in 0..3 {
        for j in 0..3 {
            let cmp = lhs[i][j].cmp(&rhs[i][j]);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
    }
    Ordering::Equal
}

fn apply_rotation(rotation: Rotation, vector: Vector3) -> Vector3 {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = rotation[i][0] as f64 * vector[0]
            + rotation[i][1] as f64 * vector[1]
            + rotation[i][2] as f64 * vector[2];
    }
    out
}

fn add(lhs: Vector3, rhs: Vector3) -> Vector3 {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn sub(lhs: Vector3, rhs: Vector3) -> Vector3 {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn normalize_fractional(vector: Vector3) -> Vector3 {
    [
        wrap_fractional(vector[0]),
        wrap_fractional(vector[1]),
        wrap_fractional(vector[2]),
    ]
}

fn wrap_fractional(x: f64) -> f64 {
    let mut wrapped = x - x.floor();
    if wrapped >= 1.0 {
        wrapped -= 1.0;
    }
    if wrapped < 0.0 {
        wrapped += 1.0;
    }
    if wrapped.abs() < 1.0e-12 || (1.0 - wrapped).abs() < 1.0e-12 {
        0.0
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_ops_map_atoms(
        operations: &[SymOp],
        positions: &[Vector3],
        atom_types: &[i32],
        tol: f64,
    ) -> bool {
        operations.iter().all(|op| {
            is_valid_operation(*op.rotation(), op.translation(), positions, atom_types, tol)
        })
    }

    #[test]
    fn detect_cubic_single_atom() {
        let structure = Structure {
            lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            positions: vec![[0.0, 0.0, 0.0]],
            atom_types: vec![1],
        };
        let detected = detect_symmetry(&structure, DetectOptions::default()).unwrap();
        assert_eq!(detected.operations.len(), 48);
        assert!(detected.operations.iter().any(|op| sym_op_approx_eq(
            op,
            &SymOp::identity(),
            1.0e-9
        )));
    }

    #[test]
    fn detect_mismatched_input_lengths() {
        let structure = Structure {
            lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            positions: vec![[0.0, 0.0, 0.0]],
            atom_types: vec![],
        };
        assert!(matches!(
            detect_symmetry(&structure, DetectOptions::default()),
            Err(DetectionError::MismatchedInputs {
                positions: 1,
                atom_types: 0
            })
        ));
    }

    #[test]
    fn standardize_operations_deduplicates() {
        let id = SymOp::identity();
        let c2z = SymOp::new([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0.0, 0.0, 0.0]).unwrap();
        let standardized =
            standardize_operations(&[c2z.clone(), id.clone(), c2z.clone(), id.clone()], 1.0e-9);
        assert_eq!(standardized.len(), 2);
        assert!(standardized
            .iter()
            .any(|op| sym_op_approx_eq(op, &id, 1.0e-9)));
        assert!(standardized
            .iter()
            .any(|op| sym_op_approx_eq(op, &c2z, 1.0e-9)));
    }

    #[test]
    fn detect_high_symmetry_structure_has_many_valid_mappings() {
        let structure = Structure {
            lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            positions: vec![[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            atom_types: vec![1, 1],
        };
        let detected = detect_symmetry(&structure, DetectOptions::default()).unwrap();
        assert!(detected.operations.len() > 1);
        assert!(all_ops_map_atoms(
            &detected.operations,
            &structure.positions,
            &structure.atom_types,
            1.0e-6
        ));
    }

    #[test]
    fn detect_low_symmetry_structure_is_identity_only() {
        let structure = Structure {
            lattice: [[1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.3, 0.4, 0.9]],
            positions: vec![[0.113, 0.271, 0.389], [0.457, 0.613, 0.791]],
            atom_types: vec![1, 2],
        };
        let detected = detect_symmetry(&structure, DetectOptions::default()).unwrap();
        assert_eq!(detected.operations.len(), 1);
        assert!(sym_op_approx_eq(
            &detected.operations[0],
            &SymOp::identity(),
            1.0e-9
        ));
        assert!(all_ops_map_atoms(
            &detected.operations,
            &structure.positions,
            &structure.atom_types,
            1.0e-6
        ));
    }
}
