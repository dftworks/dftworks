mod fd;
use fd::*;
mod gs;
use gs::*;
mod mp1;
use mp1::*;
mod mp2;
use mp2::*;

pub trait Smearing {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64;
}

pub fn new(smearing_scheme: &str) -> Box<dyn Smearing> {
    match smearing_scheme {
        "fd" => Box::new(SmearingFD {}),
        "gs" => Box::new(SmearingGS {}),
        "mp1" => Box::new(SmearingMP1 {}),
        "mp2" => Box::new(SmearingMP2 {}),
        other => panic!("unsupported smearing_scheme '{}'", other),
    }
}
