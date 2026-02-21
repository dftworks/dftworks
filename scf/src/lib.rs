mod hartree;
mod hubbard;
mod utils;

mod nonspin;
use nonspin::*;

mod ncl;
use ncl::*;

mod spin;
use spin::*;

use control::{Control, SpinScheme};
use crystal::Crystal;
use dfttypes::*;
use ewald::Ewald;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use ndarray::Array3;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;

// Unified SCF driver interface.
//
// Each concrete implementation (non-spin, collinear spin, NCL) follows the
// same high-level contract:
// 1) consume the current density/potential state
// 2) iterate Kohn-Sham solve + density update until convergence (or max steps)
// 3) return converged eigenpairs plus derived observables (forces/stress)
//    through the mutable output arguments.
pub trait SCF {
    fn run(
        &self,
        geom_iter: usize,
        control: &Control,
        crystal: &Crystal,
        gvec: &GVector,
        pwden: &PWDensity,
        pots: &PSPot,
        rgtrans: &RGTransform,
        kpts: &dyn KPTS,
        ewald: &Ewald,
        vpwwfc: &[PWBasis],
        vkscf: &mut VKSCF,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
        rhocore_3d: &Array3<c64>,
        vkevals: &mut VKEigenValue,
        vkevecs: &mut VKEigenVector,
        symdrv: &dyn SymmetryDriver,
        stress_total: &mut Matrix<f64>,
        force_total: &mut Vec<Vector3f64>,
    );
}

// Factory that selects the SCF algorithm matching the requested spin scheme.
pub fn new(spin_scheme: SpinScheme) -> Box<dyn SCF> {
    match spin_scheme {
        SpinScheme::NonSpin => Box::new(SCFNonspin::new()),
        SpinScheme::Spin => Box::new(SCFSpin::new()),
        SpinScheme::Ncl => Box::new(SCFNcl::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scf_factory_supports_spin_modes() {
        let _ = new(SpinScheme::NonSpin);
        let _ = new(SpinScheme::Spin);
        let _ = new(SpinScheme::Ncl);
    }
}
