mod line;
use line::*;

mod mesh;
use mesh::*;

use crystal::Crystal;
use lattice::Lattice;
use vector3::Vector3f64;

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
