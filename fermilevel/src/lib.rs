mod nonspin;
use nonspin::*;

mod ncl;
use ncl::*;

mod spin;
use spin::*;

use control::SpinScheme;
use dfttypes::*;

pub trait FermiLevel {
    fn get_fermi_level(&self, vkscf: &mut VKSCF, nelec: f64, vevals: &VKEigenValue) -> f64;
    fn set_occ(
        &self,
        vkscf: &mut VKSCF,
        nelec: f64,
        vevals: &VKEigenValue,
        fermi_level: f64,
        occ_inversion: f64,
    ) -> Option<f64>;
}

pub fn new(spin_scheme: SpinScheme) -> Box<dyn FermiLevel> {
    match spin_scheme {
        SpinScheme::NonSpin => Box::new(FermiLevelNonspin::new()),
        SpinScheme::Spin => Box::new(FermiLevelSpin::new()),
        SpinScheme::Ncl => Box::new(FermiLevelNcl::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermilevel_factory_supports_spin_modes() {
        let _ = new(SpinScheme::NonSpin);
        let _ = new(SpinScheme::Spin);
        let _ = new(SpinScheme::Ncl);
    }
}
