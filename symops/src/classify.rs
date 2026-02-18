//! Lightweight symmetry classification heuristics.
//!
//! This module consumes an operation set and extracts practical descriptors:
//! - crystal-system guess,
//! - point-group hint string,
//! - operation-count statistics,
//! - limited first-pass space-group number inference.
//!
//! The logic is intentionally conservative and **not** a full replacement for
//! table-driven Hall/space-group classification.

use std::error::Error;
use std::fmt;

use crate::{determinant, Rotation, SymOp};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrystalSystem {
    /// No rotational symmetry constraints.
    Triclinic,
    /// One principal 2-fold axis (heuristic classification).
    Monoclinic,
    /// Three mutually orthogonal 2-fold axes (heuristic classification).
    Orthorhombic,
    /// Presence of proper 4-fold axis.
    Tetragonal,
    /// Presence of proper 3-fold axis without cubic signature.
    Trigonal,
    /// Presence of proper 6-fold axis.
    Hexagonal,
    /// Rich 3-fold content typical of cubic point groups.
    Cubic,
}

/// Classification summary produced from symmetry operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymmetryClassification {
    /// Broad crystal-system class inferred from detected rotational orders.
    pub crystal_system: CrystalSystem,
    /// Human-readable point-group hint (not guaranteed unique).
    pub point_group_hint: &'static str,
    /// Whether inversion operation appears in the set.
    pub has_inversion: bool,
    /// Total number of operations (proper + improper).
    pub n_operations: usize,
    /// Count of operations with determinant `+1`.
    pub n_proper_rotations: usize,
    /// Maximum detected rotational order among proper rotations.
    pub max_rotation_order: u8,
    /// Optional first-pass mapping when unambiguous from operation count.
    pub space_group_number: Option<u16>,
}

/// Classification failure reasons.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassificationError {
    /// No operations were provided.
    EmptyOperations,
}

impl fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassificationError::EmptyOperations => write!(f, "no symmetry operations provided"),
        }
    }
}

impl Error for ClassificationError {}

/// Classifies an operation set into a coarse crystallographic summary.
///
/// Heuristic rules currently inspect:
/// - inversion presence,
/// - counts of proper rotations by order (2,3,4,6),
/// - total operation count for very limited SG inference (`P1`, `P-1`).
pub fn classify_symmetry(ops: &[SymOp]) -> Result<SymmetryClassification, ClassificationError> {
    if ops.is_empty() {
        return Err(ClassificationError::EmptyOperations);
    }

    let mut has_inversion = false;
    let mut n_proper_rotations = 0usize;
    let mut max_rotation_order = 1u8;
    let mut n_order2 = 0usize;
    let mut n_order3 = 0usize;
    let mut n_order4 = 0usize;
    let mut n_order6 = 0usize;

    let inversion = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];

    for op in ops.iter() {
        let rotation = *op.rotation();
        if rotation == inversion {
            has_inversion = true;
        }

        if determinant(rotation) != 1 {
            continue;
        }
        n_proper_rotations += 1;

        if let Some(order) = rotation_order(rotation) {
            max_rotation_order = max_rotation_order.max(order);
            match order {
                2 => n_order2 += 1,
                3 => n_order3 += 1,
                4 => n_order4 += 1,
                6 => n_order6 += 1,
                _ => {}
            }
        }
    }

    // Simple crystal-system decision tree based on observed rotation-order
    // signatures. This keeps behavior transparent and easy to tune for
    // dftworks-specific workloads.
    let crystal_system = if n_order3 >= 8 {
        CrystalSystem::Cubic
    } else if n_order6 > 0 {
        CrystalSystem::Hexagonal
    } else if n_order4 > 0 {
        CrystalSystem::Tetragonal
    } else if n_order3 > 0 {
        CrystalSystem::Trigonal
    } else if n_order2 >= 3 {
        CrystalSystem::Orthorhombic
    } else if n_order2 > 0 {
        CrystalSystem::Monoclinic
    } else {
        CrystalSystem::Triclinic
    };

    let point_group_hint = match (crystal_system, has_inversion) {
        (CrystalSystem::Triclinic, false) => "1",
        (CrystalSystem::Triclinic, true) => "-1",
        (CrystalSystem::Monoclinic, false) => "2",
        (CrystalSystem::Monoclinic, true) => "2/m",
        (CrystalSystem::Orthorhombic, false) => "222",
        (CrystalSystem::Orthorhombic, true) => "mmm",
        (CrystalSystem::Tetragonal, false) => "4mm",
        (CrystalSystem::Tetragonal, true) => "4/mmm",
        (CrystalSystem::Trigonal, false) => "3m",
        (CrystalSystem::Trigonal, true) => "-3m",
        (CrystalSystem::Hexagonal, false) => "6mm",
        (CrystalSystem::Hexagonal, true) => "6/mmm",
        (CrystalSystem::Cubic, false) => "432",
        (CrystalSystem::Cubic, true) => "m-3m",
    };

    let space_group_number = infer_space_group_number(ops.len(), has_inversion);

    Ok(SymmetryClassification {
        crystal_system,
        point_group_hint,
        has_inversion,
        n_operations: ops.len(),
        n_proper_rotations,
        max_rotation_order,
        space_group_number,
    })
}

/// Very limited SG number mapping for unambiguous trivial cases only.
fn infer_space_group_number(n_operations: usize, has_inversion: bool) -> Option<u16> {
    if n_operations == 1 {
        Some(1)
    } else if n_operations == 2 && has_inversion {
        Some(2)
    } else {
        None
    }
}

/// Computes multiplicative order of integer rotation matrix.
///
/// Returns `None` if identity is not reached by order 12.
fn rotation_order(rotation: Rotation) -> Option<u8> {
    let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let mut power = identity;
    for order in 1..=12 {
        power = mat_mul(power, rotation);
        if power == identity {
            return Some(order as u8);
        }
    }
    None
}

fn mat_mul(lhs: Rotation, rhs: Rotation) -> Rotation {
    let mut out = [[0_i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lhs[i][0] * rhs[0][j] + lhs[i][1] * rhs[1][j] + lhs[i][2] * rhs[2][j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::detect::{detect_symmetry, DetectOptions, Structure};

    use super::*;

    #[test]
    fn classify_p1() {
        let class = classify_symmetry(&[SymOp::identity()]).unwrap();
        assert_eq!(class.crystal_system, CrystalSystem::Triclinic);
        assert_eq!(class.point_group_hint, "1");
        assert_eq!(class.space_group_number, Some(1));
    }

    #[test]
    fn classify_p_minus_1() {
        let inversion = SymOp::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [0.0, 0.0, 0.0]).unwrap();
        let class = classify_symmetry(&[SymOp::identity(), inversion]).unwrap();
        assert_eq!(class.crystal_system, CrystalSystem::Triclinic);
        assert_eq!(class.point_group_hint, "-1");
        assert_eq!(class.space_group_number, Some(2));
    }

    #[test]
    fn classify_cubic_detector_output() {
        let structure = Structure {
            lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            positions: vec![[0.0, 0.0, 0.0]],
            atom_types: vec![1],
        };
        let detected = detect_symmetry(&structure, DetectOptions::default()).unwrap();
        let class = classify_symmetry(&detected.operations).unwrap();
        assert_eq!(class.crystal_system, CrystalSystem::Cubic);
    }
}
