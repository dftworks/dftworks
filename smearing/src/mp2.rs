use crate::Smearing;
use dwconsts::*;
use special;

pub struct SmearingMP2 {}

impl Smearing for SmearingMP2 {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64 {
        let kbt = (BOLTZMANN_CONSTANT * temperature).max(EPS30);

        let x = (electron_energy - fermi_level) / kbt;

        0.5 * (1.0
            - special::erf(x)
            - 1.0 / PI.sqrt() * x * (7.0 / 4.0 - 0.5 * x * x) * (-x * x).exp())
    }
}
