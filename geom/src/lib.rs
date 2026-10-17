mod bfgs;
use bfgs::*;

mod diis;
use diis::*;

use control::GeomOptScheme;
use crystal::Crystal;
use lattice::Lattice;
use types::*;

pub trait GeomOptimizationDriver {
    fn compute_next_configuration(
        &mut self,
        crystal: &mut Crystal,
        force: &[Vector3f64],
        stress: &Matrix<f64>,
        force_mask: &[Vector3f64],
        stress_mask: &Matrix<f64>,
        latt0: &Lattice,
        bcell_move: bool,
    );
}

pub fn new(scheme: GeomOptScheme, alpha: f64, nstep: usize) -> Box<dyn GeomOptimizationDriver> {
    match scheme {
        GeomOptScheme::Bfgs => Box::new(BFGS::new(alpha, nstep)),
        GeomOptScheme::Diis => Box::new(DIIS::new(alpha, nstep)),
    }
}
