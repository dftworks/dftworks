use crate::Smearing;
use dwconsts::*;
use special;

// Marzari-Vanderbilt "cold" smearing.
//
// x = (e - ef) / (k_B T)
// f(x) = 0.5 * erfc(x + 1/sqrt(2)) + exp(-(x + 1/sqrt(2))^2) / sqrt(2*pi)
pub struct SmearingMV {}

impl Smearing for SmearingMV {
    fn get_occupation_number(
        &self,
        fermi_level: f64,
        temperature: f64,
        electron_energy: f64,
    ) -> f64 {
        let kbt = (BOLTZMANN_CONSTANT * temperature).max(EPS30);
        let x = (electron_energy - fermi_level) / kbt;
        let x_shift = x + 1.0 / 2.0_f64.sqrt();

        let occ = 0.5 * special::erfc(x_shift)
            + (-x_shift * x_shift).exp() / (2.0 * PI).sqrt();

        occ.max(0.0).min(1.0)
    }
}
