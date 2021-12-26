use crate::Smearing;
use dwconsts::*;

pub struct SmearingFD {}

impl Smearing for SmearingFD {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64 {
        let kbt = (BOLTZMANN_CONSTANT * temperature).max(EPS30);

        let x = (electron_energy - fermi_level) / kbt;

        1.0 / (x.exp() + 1.0)
    }
}
