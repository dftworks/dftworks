mod nonspin;
use nonspin::*;

mod spin;
use spin::*;

use dfttypes::*;

pub trait FermiLevel {
    fn get_fermi_level(&self, vkscf: &mut VKSCF, nelec: f64, vevals: &VKEigenValue) -> f64;
}

pub fn new(spin_scheme: &str) -> Box<dyn FermiLevel> {
    let fl: Box<dyn FermiLevel>;

    match spin_scheme {
        "nonspin" => {
            fl = Box::new(FermiLevelNonspin::new());
        }

        "spin" => {
            fl = Box::new(FermiLevelSpin::new());
        }

        &_ => {
            println!("{} not implemented", spin_scheme);
            panic!();
        }
    }

    fl
}
