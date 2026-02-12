mod nonspin;
use nonspin::*;

mod spin;
use spin::*;

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

pub fn new(spin_scheme: &str) -> Box<dyn FermiLevel> {
    match spin_scheme {
        "nonspin" => Box::new(FermiLevelNonspin::new()),
        "spin" => Box::new(FermiLevelSpin::new()),
        other => panic!("unsupported spin_scheme '{}'", other),
    }
}
