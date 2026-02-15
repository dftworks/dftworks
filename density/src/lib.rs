mod nonspin;
mod spin;
mod ncl;

use ncl::*;
use nonspin::*;
use spin::*;

use control::SpinScheme;
use crystal::Crystal;
use dfttypes::*;
use gvector::GVector;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::*;

// Common density driver interface used by SCF.
//
// Responsibilities:
// - build physically reasonable initial density from atomic superposition
// - evaluate iterative density-change metric for SCF convergence logic
// - rebuild rho(r) from occupied KS states and eigenvectors
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

// Factory that dispatches to the density workflow matching spin treatment.
pub fn new(spin_scheme: SpinScheme) -> Box<dyn Density> {
    match spin_scheme {
        SpinScheme::NonSpin => Box::new(DensityNonspin::new()),
        SpinScheme::Spin => Box::new(DensitySpin::new()),
        SpinScheme::Ncl => Box::new(DensityNcl::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_factory_supports_spin_modes() {
        let _ = new(SpinScheme::NonSpin);
        let _ = new(SpinScheme::Spin);
        let _ = new(SpinScheme::Ncl);
    }
}
