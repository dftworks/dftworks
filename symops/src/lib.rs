//! Core symmetry-operation algebra for crystal and k-space workflows.
//!
//! This module intentionally focuses on:
//! - robust operation math (`R`, `t`, composition, inverse),
//! - tolerance-aware equality modulo lattice translations,
//! - lightweight group checks and k-space helpers.
//!
//! It does **not** attempt full space-group detection/classification parity with
//! external libraries. Detection/classification scaffolding lives in sibling
//! modules (`detect`, `classify`) and can evolve independently.

use std::error::Error;
use std::fmt;

/// Integer 3x3 rotation matrix in fractional-coordinate basis.
///
/// For crystallographic operations this should be unimodular with
/// determinant `+1` (proper) or `-1` (improper).
pub type Rotation = [[i32; 3]; 3];

/// 3-vector used for fractional coordinates and translations.
pub type Vector3 = [f64; 3];

pub mod classify;
pub mod detect;

pub use classify::*;
pub use detect::*;

const WRAP_EPS: f64 = 1.0e-12;

/// A single affine symmetry operation in fractional coordinates:
/// `x' = R * x + t (mod 1)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SymOp {
    rotation: Rotation,
    translation: Vector3,
}

/// Errors returned by symmetry-operation construction and group checks.
#[derive(Clone, Debug, PartialEq)]
pub enum SymOpError {
    /// Rotation has determinant different from `±1`.
    NonUnimodularRotation { det: i32 },
    /// Group validation was requested on an empty set.
    EmptyGroup,
    /// Group validation could not find the identity element.
    MissingIdentity,
    /// Group validation found an element without inverse in the set.
    MissingInverse { index: usize },
    /// Group validation found `ops[left] * ops[right]` outside the set.
    NotClosed { left: usize, right: usize },
    /// Operation rotation violates lattice metric invariance within tolerance.
    LatticeInconsistent { index: usize, deviation: f64 },
}

impl fmt::Display for SymOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymOpError::NonUnimodularRotation { det } => {
                write!(f, "rotation determinant must be ±1, got {}", det)
            }
            SymOpError::EmptyGroup => write!(f, "symmetry group is empty"),
            SymOpError::MissingIdentity => write!(f, "symmetry group is missing identity"),
            SymOpError::MissingInverse { index } => {
                write!(f, "symmetry op {} has no inverse in group", index)
            }
            SymOpError::NotClosed { left, right } => write!(
                f,
                "symmetry group is not closed: op {} composed with op {} is missing",
                left, right
            ),
            SymOpError::LatticeInconsistent { index, deviation } => write!(
                f,
                "symmetry op {} violates lattice consistency (max metric deviation = {:.3e})",
                index, deviation
            ),
        }
    }
}

impl Error for SymOpError {}

/// One k-star representative and the operation indices that generate it.
#[derive(Clone, Debug, PartialEq)]
pub struct KStarPoint {
    /// Star point representative in centered fractional coordinates.
    pub k: Vector3,
    /// Indices in the provided operation list that map to this point.
    pub operation_indices: Vec<usize>,
}

impl SymOp {
    /// Constructs a symmetry operation and normalizes translation to `[0, 1)`.
    ///
    /// The rotation must be unimodular (`det = ±1`), otherwise the operation
    /// is not invertible over the integer lattice.
    pub fn new(rotation: Rotation, translation: Vector3) -> Result<Self, SymOpError> {
        let det = determinant(rotation);
        if det != 1 && det != -1 {
            return Err(SymOpError::NonUnimodularRotation { det });
        }

        Ok(Self {
            rotation,
            translation: normalize_fractional(translation),
        })
    }

    /// Returns identity operation (`R = I`, `t = 0`).
    pub fn identity() -> Self {
        Self {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Returns the integer rotation part.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// Returns normalized translation in `[0, 1)`.
    pub fn translation(&self) -> Vector3 {
        self.translation
    }

    /// Applies full affine operation and wraps result to fractional cell.
    pub fn apply_fractional(&self, vector: Vector3) -> Vector3 {
        normalize_fractional(add(mat_vec_i32(self.rotation, vector), self.translation))
    }

    /// Applies only rotation part (used frequently for k-space mapping).
    pub fn apply_rotation(&self, vector: Vector3) -> Vector3 {
        mat_vec_i32(self.rotation, vector)
    }

    /// Group composition `self ∘ rhs`:
    /// `(R1, t1) * (R2, t2) = (R1*R2, R1*t2 + t1)`.
    pub fn compose(&self, rhs: &SymOp) -> SymOp {
        let rotation = mat_mul_i32(self.rotation, rhs.rotation);
        // Translation is wrapped to keep canonical representative in [0,1).
        let translation = normalize_fractional(add(
            mat_vec_i32(self.rotation, rhs.translation),
            self.translation,
        ));
        SymOp {
            rotation,
            translation,
        }
    }

    /// Computes inverse operation:
    /// `(R, t)^-1 = (R^-1, -R^-1*t)`.
    pub fn inverse(&self) -> Result<SymOp, SymOpError> {
        let det = determinant(self.rotation);
        if det != 1 && det != -1 {
            return Err(SymOpError::NonUnimodularRotation { det });
        }

        let inv_rotation = inverse_unimodular(self.rotation, det);
        // Fractional wrap keeps inverse translation numerically stable.
        let inv_translation =
            normalize_fractional(scale(mat_vec_i32(inv_rotation, self.translation), -1.0));
        Ok(SymOp {
            rotation: inv_rotation,
            translation: inv_translation,
        })
    }
}

/// Determinant of 3x3 integer rotation matrix.
pub fn determinant(rotation: Rotation) -> i32 {
    rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
}

/// Validates that `ops` behaves like a finite symmetry group under composition.
///
/// Checks:
/// - non-empty set,
/// - identity exists,
/// - each element has inverse,
/// - closure under composition.
///
/// Translation comparison uses `tol` modulo lattice vectors.
pub fn validate_group(ops: &[SymOp], tol: f64) -> Result<(), SymOpError> {
    if ops.is_empty() {
        return Err(SymOpError::EmptyGroup);
    }

    let identity = SymOp::identity();
    if !ops.iter().any(|op| sym_op_approx_eq(op, &identity, tol)) {
        return Err(SymOpError::MissingIdentity);
    }

    for (idx, op) in ops.iter().enumerate() {
        let has_inverse = ops.iter().any(|candidate| {
            let left = op.compose(candidate);
            let right = candidate.compose(op);
            sym_op_approx_eq(&left, &identity, tol) && sym_op_approx_eq(&right, &identity, tol)
        });
        if !has_inverse {
            return Err(SymOpError::MissingInverse { index: idx });
        }
    }

    for (i, left) in ops.iter().enumerate() {
        for (j, right) in ops.iter().enumerate() {
            let composed = left.compose(right);
            if !ops
                .iter()
                .any(|candidate| sym_op_approx_eq(&composed, candidate, tol))
            {
                return Err(SymOpError::NotClosed { left: i, right: j });
            }
        }
    }

    Ok(())
}

/// Returns indices of operations that leave `k` invariant modulo reciprocal lattice.
pub fn little_group_indices(k: Vector3, ops: &[SymOp], tol: f64) -> Vec<usize> {
    let mut out = Vec::new();
    for (idx, op) in ops.iter().enumerate() {
        let mapped = op.apply_rotation(k);
        if approx_eq_mod_lattice(mapped, k, tol) {
            out.push(idx);
        }
    }
    out
}

/// Computes k-star (`{Rk}`) with deduplication modulo lattice vectors.
pub fn k_star(k: Vector3, ops: &[SymOp], tol: f64) -> Vec<Vector3> {
    k_star_with_mappings(k, ops, tol)
        .into_iter()
        .map(|entry| entry.k)
        .collect()
}

/// Computes k-star and records which operations map to each representative.
pub fn k_star_with_mappings(k: Vector3, ops: &[SymOp], tol: f64) -> Vec<KStarPoint> {
    let mut out: Vec<KStarPoint> = Vec::new();
    for (idx, op) in ops.iter().enumerate() {
        // Use centered wrapping for more interpretable representative values.
        let mapped = normalize_centered(op.apply_rotation(k));
        if let Some(existing) = out
            .iter_mut()
            .find(|candidate| approx_eq_mod_lattice(candidate.k, mapped, tol))
        {
            existing.operation_indices.push(idx);
        } else {
            out.push(KStarPoint {
                k: mapped,
                operation_indices: vec![idx],
            });
        }
    }
    out
}

/// Computes lattice metric tensor `G = A * A^T` from direct lattice rows.
pub fn lattice_metric(lattice: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

/// Validates operation lattice consistency using `R^T G R ≈ G` within `tol`.
pub fn validate_lattice_consistency(
    ops: &[SymOp],
    lattice: [[f64; 3]; 3],
    tol: f64,
) -> Result<(), SymOpError> {
    let metric = lattice_metric(lattice);
    for (idx, op) in ops.iter().enumerate() {
        let det = determinant(*op.rotation());
        if det != 1 && det != -1 {
            return Err(SymOpError::NonUnimodularRotation { det });
        }
        let deviation = rotation_metric_deviation(*op.rotation(), metric);
        if deviation > tol {
            return Err(SymOpError::LatticeInconsistent {
                index: idx,
                deviation,
            });
        }
    }
    Ok(())
}

/// Returns maximum `|R^T G R - G|` entry over all operations.
pub fn max_lattice_deviation(ops: &[SymOp], lattice: [[f64; 3]; 3]) -> f64 {
    let metric = lattice_metric(lattice);
    let mut max_deviation = 0.0;
    for op in ops.iter() {
        let deviation = rotation_metric_deviation(*op.rotation(), metric);
        if deviation > max_deviation {
            max_deviation = deviation;
        }
    }
    max_deviation
}

/// Approximate equality of two operations with translation compared modulo lattice.
pub fn sym_op_approx_eq(lhs: &SymOp, rhs: &SymOp, tol: f64) -> bool {
    lhs.rotation == rhs.rotation && approx_eq_mod_lattice(lhs.translation, rhs.translation, tol)
}

/// Compares vectors modulo integer lattice shifts using centered wrapping.
pub fn approx_eq_mod_lattice(lhs: Vector3, rhs: Vector3, tol: f64) -> bool {
    for i in 0..3 {
        if wrap_centered(lhs[i] - rhs[i]).abs() > tol {
            return false;
        }
    }
    true
}

fn add(lhs: Vector3, rhs: Vector3) -> Vector3 {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale(vector: Vector3, factor: f64) -> Vector3 {
    [factor * vector[0], factor * vector[1], factor * vector[2]]
}

fn mat_vec_i32(rotation: Rotation, vector: Vector3) -> Vector3 {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = rotation[i][0] as f64 * vector[0]
            + rotation[i][1] as f64 * vector[1]
            + rotation[i][2] as f64 * vector[2];
    }
    out
}

fn mat_mul_i32(lhs: Rotation, rhs: Rotation) -> Rotation {
    let mut out = [[0_i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lhs[i][0] * rhs[0][j] + lhs[i][1] * rhs[1][j] + lhs[i][2] * rhs[2][j];
        }
    }
    out
}

fn rotation_metric_deviation(rotation: Rotation, metric: [[f64; 3]; 3]) -> f64 {
    let mut max_deviation = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let mut transformed = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    transformed += rotation[a][i] as f64 * metric[a][b] * rotation[b][j] as f64;
                }
            }
            let delta = (transformed - metric[i][j]).abs();
            if delta > max_deviation {
                max_deviation = delta;
            }
        }
    }
    max_deviation
}

fn inverse_unimodular(rotation: Rotation, det: i32) -> Rotation {
    let mut inv = [[0_i32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let cof = cofactor(rotation, row, col);
            inv[col][row] = cof / det;
        }
    }
    inv
}

fn cofactor(rotation: Rotation, row: usize, col: usize) -> i32 {
    let minor = minor_det(rotation, row, col);
    if (row + col) % 2 == 0 {
        minor
    } else {
        -minor
    }
}

fn minor_det(rotation: Rotation, row: usize, col: usize) -> i32 {
    let mut vals = [0_i32; 4];
    let mut idx = 0;
    for i in 0..3 {
        if i == row {
            continue;
        }
        for j in 0..3 {
            if j == col {
                continue;
            }
            vals[idx] = rotation[i][j];
            idx += 1;
        }
    }
    vals[0] * vals[3] - vals[1] * vals[2]
}

fn normalize_fractional(vector: Vector3) -> Vector3 {
    [
        wrap_fractional(vector[0]),
        wrap_fractional(vector[1]),
        wrap_fractional(vector[2]),
    ]
}

fn normalize_centered(vector: Vector3) -> Vector3 {
    [
        wrap_centered(vector[0]),
        wrap_centered(vector[1]),
        wrap_centered(vector[2]),
    ]
}

fn wrap_fractional(x: f64) -> f64 {
    // Canonical fractional representative in [0,1).
    let mut wrapped = x - x.floor();
    if wrapped >= 1.0 {
        wrapped -= 1.0;
    }
    if wrapped < 0.0 {
        wrapped += 1.0;
    }
    // Clamp values that are numerically at 0/1 boundary.
    if wrapped.abs() < WRAP_EPS || (1.0 - wrapped).abs() < WRAP_EPS {
        0.0
    } else {
        wrapped
    }
}

fn wrap_centered(x: f64) -> f64 {
    // Representative in [-0.5, 0.5), useful for nearest-image comparisons.
    let mut wrapped = x - x.round();
    if wrapped >= 0.5 {
        wrapped -= 1.0;
    }
    if wrapped < -0.5 {
        wrapped += 1.0;
    }
    if wrapped.abs() < WRAP_EPS {
        0.0
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-9;

    fn c2z() -> SymOp {
        SymOp::new([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0.0, 0.0, 0.0]).unwrap()
    }

    #[test]
    fn compose_with_inverse_is_identity() {
        let op = SymOp::new([[0, -1, 0], [1, -1, 0], [0, 0, 1]], [0.5, 0.25, 0.0]).unwrap();
        let inv = op.inverse().unwrap();
        let id = SymOp::identity();

        let left = op.compose(&inv);
        let right = inv.compose(&op);
        assert!(sym_op_approx_eq(&left, &id, TOL));
        assert!(sym_op_approx_eq(&right, &id, TOL));
    }

    #[test]
    fn validate_small_group() {
        let ops = vec![SymOp::identity(), c2z()];
        assert_eq!(validate_group(&ops, TOL), Ok(()));
    }

    #[test]
    fn validate_missing_inverse() {
        let ops = vec![
            SymOp::identity(),
            SymOp::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.25, 0.0, 0.0]).unwrap(),
        ];
        assert_eq!(
            validate_group(&ops, TOL),
            Err(SymOpError::MissingInverse { index: 1 })
        );
    }

    #[test]
    fn star_and_little_group() {
        let ops = vec![SymOp::identity(), c2z()];
        let k = [0.25, 0.0, 0.0];
        let star = k_star(k, &ops, TOL);
        assert_eq!(star.len(), 2);
        assert!(star
            .iter()
            .any(|v| approx_eq_mod_lattice(*v, [0.25, 0.0, 0.0], TOL)));
        assert!(star
            .iter()
            .any(|v| approx_eq_mod_lattice(*v, [-0.25, 0.0, 0.0], TOL)));

        let little_gamma = little_group_indices([0.0, 0.0, 0.0], &ops, TOL);
        assert_eq!(little_gamma, vec![0, 1]);
        let little_x = little_group_indices(k, &ops, TOL);
        assert_eq!(little_x, vec![0]);
    }

    #[test]
    fn star_with_mappings_tracks_generating_operations() {
        let ops = vec![SymOp::identity(), c2z()];
        let mapped = k_star_with_mappings([0.25, 0.0, 0.0], &ops, TOL);
        assert_eq!(mapped.len(), 2);
        assert!(mapped.iter().any(
            |entry| approx_eq_mod_lattice(entry.k, [0.25, 0.0, 0.0], TOL)
                && entry.operation_indices == vec![0]
        ));
        assert!(mapped.iter().any(
            |entry| approx_eq_mod_lattice(entry.k, [-0.25, 0.0, 0.0], TOL)
                && entry.operation_indices == vec![1]
        ));
    }

    #[test]
    fn lattice_validation_rejects_non_metric_rotation() {
        let shear = SymOp::new([[1, 1, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0]).unwrap();
        let err = validate_lattice_consistency(
            &[SymOp::identity(), shear],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            1.0e-9,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SymOpError::LatticeInconsistent {
                index: 1,
                deviation: _
            }
        ));
    }

    #[test]
    fn lattice_validation_accepts_metric_preserving_group() {
        let ops = vec![SymOp::identity(), c2z()];
        assert_eq!(
            validate_lattice_consistency(
                &ops,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
                1.0e-9
            ),
            Ok(())
        );
    }
}
