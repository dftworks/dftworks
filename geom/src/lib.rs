mod bfgs;
use bfgs::*;

mod diis;
use diis::*;

use crystal::Crystal;
use lattice::Lattice;
use matrix::*;
use vector3::*;

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

pub fn new(scheme: &str, alpha: f64, nstep: usize) -> Box<dyn GeomOptimizationDriver> {
    let optim: Box<dyn GeomOptimizationDriver>;

    match scheme.to_lowercase().as_str() {
        "bfgs" => {
            optim = Box::new(BFGS::new(alpha, nstep));
        }

        "diis" => {
            optim = Box::new(DIIS::new(alpha, nstep));
        }

        other => {
            panic!("unsupported geom_optim_scheme '{}'", other);
        }
    }

    optim
}
