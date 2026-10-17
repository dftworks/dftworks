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

        // Not clamped to [0, 1]: f rises to about 1.083 at x = -sqrt(2) before it
        // settles at 1, and clamping would change the number of electrons.
        0.5 * special::erfc(x_shift) + (-x_shift * x_shift).exp() / (2.0 * PI).sqrt()
    }
}
