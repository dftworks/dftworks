use std::error::Error;
use std::fmt;

pub type Rotation = [[i32; 3]; 3];
pub type Vector3 = [f64; 3];

const WRAP_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct SymOp {
    rotation: Rotation,
    translation: Vector3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymOpError {
    NonUnimodularRotation { det: i32 },
    EmptyGroup,
    MissingIdentity,
    MissingInverse { index: usize },
    NotClosed { left: usize, right: usize },
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
        }
    }
}

impl Error for SymOpError {}

impl SymOp {
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

    pub fn identity() -> Self {
        Self {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    pub fn translation(&self) -> Vector3 {
        self.translation
    }

    pub fn apply_fractional(&self, vector: Vector3) -> Vector3 {
        normalize_fractional(add(mat_vec_i32(self.rotation, vector), self.translation))
    }

    pub fn apply_rotation(&self, vector: Vector3) -> Vector3 {
        mat_vec_i32(self.rotation, vector)
    }

    pub fn compose(&self, rhs: &SymOp) -> SymOp {
        let rotation = mat_mul_i32(self.rotation, rhs.rotation);
        let translation = normalize_fractional(add(
            mat_vec_i32(self.rotation, rhs.translation),
            self.translation,
        ));
        SymOp {
            rotation,
            translation,
        }
    }

    pub fn inverse(&self) -> Result<SymOp, SymOpError> {
        let det = determinant(self.rotation);
        if det != 1 && det != -1 {
            return Err(SymOpError::NonUnimodularRotation { det });
        }

        let inv_rotation = inverse_unimodular(self.rotation, det);
        let inv_translation =
            normalize_fractional(scale(mat_vec_i32(inv_rotation, self.translation), -1.0));
        Ok(SymOp {
            rotation: inv_rotation,
            translation: inv_translation,
        })
    }
}

pub fn determinant(rotation: Rotation) -> i32 {
    rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
}

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

pub fn k_star(k: Vector3, ops: &[SymOp], tol: f64) -> Vec<Vector3> {
    let mut out = Vec::new();
    for op in ops.iter() {
        let mapped = normalize_centered(op.apply_rotation(k));
        if !out
            .iter()
            .any(|candidate| approx_eq_mod_lattice(*candidate, mapped, tol))
        {
            out.push(mapped);
        }
    }
    out
}

pub fn sym_op_approx_eq(lhs: &SymOp, rhs: &SymOp, tol: f64) -> bool {
    lhs.rotation == rhs.rotation && approx_eq_mod_lattice(lhs.translation, rhs.translation, tol)
}

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
    let mut wrapped = x - x.floor();
    if wrapped >= 1.0 {
        wrapped -= 1.0;
    }
    if wrapped < 0.0 {
        wrapped += 1.0;
    }
    if wrapped.abs() < WRAP_EPS || (1.0 - wrapped).abs() < WRAP_EPS {
        0.0
    } else {
        wrapped
    }
}

fn wrap_centered(x: f64) -> f64 {
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
}
