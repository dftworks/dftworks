mod nonspin;
mod spin;

use nonspin::*;
use spin::*;

use crystal::Crystal;
use dfttypes::*;
use gvector::GVector;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::*;

pub trait Density {
    fn from_atomic_super_position(
        &self,
        pspot: &PSPot,
        crystal: &Crystal,
        rgtrans: &RGTransform,
        gvec: &GVector,
        pwden: &PWDensity,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
    );

    fn get_change_in_density(&self, rhog: &[c64], rhog_new: &[c64]) -> f64;

    fn compute_charge_density(
        &self,
        vkscf: &VKSCF,
        rgtrans: &RGTransform,
        vkevecs: &VKEigenVector,
        volume: f64,
        rho_3d: &mut RHOR,
    );
}

pub fn new(spin_scheme: &str) -> Box<dyn Density> {
    match spin_scheme {
        "nonspin" => Box::new(DensityNonspin::new()),
        "spin" => Box::new(DensitySpin::new()),
        other => panic!("unsupported spin_scheme '{}'", other),
    }
}
