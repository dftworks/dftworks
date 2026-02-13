use crate::SCF;
use control::Control;
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
use types::c64;
use vector3::Vector3f64;

pub struct SCFNcl {}

impl SCFNcl {
    pub fn new() -> SCFNcl {
        SCFNcl {}
    }
}

impl SCF for SCFNcl {
    fn run(
        &self,
        _geom_iter: usize,
        _control: &Control,
        _crystal: &Crystal,
        _gvec: &GVector,
        _pwden: &PWDensity,
        _pots: &PSPot,
        _rgtrans: &RGTransform,
        _kpts: &dyn KPTS,
        _ewald: &Ewald,
        _vpwwfc: &[PWBasis],
        _vkscf: &mut VKSCF,
        _rhog: &mut RHOG,
        _rho_3d: &mut RHOR,
        _rhocore_3d: &Array3<c64>,
        _vkevals: &mut VKEigenValue,
        _vkevecs: &mut VKEigenVector,
        _symdrv: &dyn SymmetryDriver,
        _stress_total: &mut Matrix<f64>,
        _force_total: &mut Vec<Vector3f64>,
    ) {
        panic!("SCFNcl::run is not implemented yet");
    }
}
