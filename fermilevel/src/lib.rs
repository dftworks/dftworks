mod nonspin;
use nonspin::*;

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
        SpinScheme::Ncl => panic!("unsupported spin_scheme 'ncl' in fermilevel::new"),
    }
}
