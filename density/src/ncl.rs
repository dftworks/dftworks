use crate::Density;
use crystal::Crystal;
use dfttypes::*;
use gvector::GVector;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::c64;

pub struct DensityNcl {}

impl DensityNcl {
    pub fn new() -> DensityNcl {
        DensityNcl {}
    }
}

impl Density for DensityNcl {
    fn from_atomic_super_position(
        &self,
        _pspot: &PSPot,
        _crystal: &Crystal,
        _rgtrans: &RGTransform,
        _gvec: &GVector,
        _pwden: &PWDensity,
        _rhog: &mut RHOG,
        _rho_3d: &mut RHOR,
    ) {
        panic!("DensityNcl::from_atomic_super_position is not implemented yet");
    }

    fn get_change_in_density(&self, _rhog: &[c64], _rhog_new: &[c64]) -> f64 {
        panic!("DensityNcl::get_change_in_density is not implemented yet");
    }

    fn compute_charge_density(
        &self,
        _vkscf: &VKSCF,
        _rgtrans: &RGTransform,
        _vkevecs: &VKEigenVector,
        _volume: f64,
        _rho_3d: &mut RHOR,
    ) {
        panic!("DensityNcl::compute_charge_density is not implemented yet");
    }
}
