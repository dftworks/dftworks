mod fd;
use fd::*;
mod gs;
use gs::*;
mod mp1;
use mp1::*;
mod mp2;
use mp2::*;
use control::SmearingScheme;

pub trait Smearing {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64;
}

pub fn new(smearing_scheme: SmearingScheme) -> Box<dyn Smearing> {
    match smearing_scheme {
        SmearingScheme::Fd => Box::new(SmearingFD {}),
        SmearingScheme::Gs => Box::new(SmearingGS {}),
        SmearingScheme::Mp1 => Box::new(SmearingMP1 {}),
        SmearingScheme::Mp2 => Box::new(SmearingMP2 {}),
    }
}
