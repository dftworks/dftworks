mod hartree;
mod utils;

mod nonspin;
use nonspin::*;

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

pub fn new(spin_scheme: SpinScheme) -> Box<dyn SCF> {
    match spin_scheme {
        SpinScheme::NonSpin => Box::new(SCFNonspin::new()),
        SpinScheme::Spin => Box::new(SCFSpin::new()),
        SpinScheme::Ncl => panic!("unsupported spin_scheme 'ncl' in scf::new"),
    }
}
