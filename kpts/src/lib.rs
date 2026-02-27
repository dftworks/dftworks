mod line;
use line::*;

mod mesh;
use mesh::*;

use crystal::Crystal;
use lattice::Lattice;
use std::error::Error;
use std::fmt;
use vector3::Vector3f64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KptsError {
    message: String,
}

impl KptsError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn with_context(context: &str, message: impl Into<String>) -> Self {
        Self::new(format!("{}: {}", context, message.into()))
    }
}

impl fmt::Display for KptsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for KptsError {}

// K-point provider interface.
//
// Implementations expose a unified view of:
// - fractional coordinates
// - integration weights
// - degeneracy bookkeeping (mostly informational in current code paths)
// - optional mesh metadata (for kmesh-based workflows such as Wannier export)
pub trait KPTS {
    fn get_k_frac(&self, k_index: usize) -> Vector3f64;
    fn get_k_degeneracy(&self, k_index: usize) -> usize;
    fn get_k_weight(&self, k_index: usize) -> f64;
    fn get_n_kpts(&self) -> usize;
    fn frac_to_cart(&self, k_frac: &Vector3f64, blatt: &Lattice) -> Vector3f64;
    fn get_k_mesh(&self) -> [i32; 3];
    fn display(&self);
}

// Factory for k-point generation modes.
pub fn new(scheme: &str, crystal: &Crystal, symmetry: bool) -> Box<dyn KPTS> {
    match scheme {
        "kmesh" => Box::new(KptsMesh::new(crystal, symmetry)),
        "kline" => Box::new(KptsLine::new()),
        other => panic!("unsupported k-point scheme '{}'", other),
    }
}

pub fn try_new(scheme: &str, crystal: &Crystal, symmetry: bool) -> Result<Box<dyn KPTS>, KptsError> {
    match scheme {
        "kmesh" => Ok(Box::new(KptsMesh::try_new(crystal, symmetry)?)),
        "kline" => Ok(Box::new(KptsLine::try_new()?)),
        other => Err(KptsError::new(format!(
            "unsupported k-point scheme '{}'",
            other
        ))),
    }
}
