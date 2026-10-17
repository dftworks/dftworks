// Conformance tests: every smearing scheme, built through `smearing::new`, must pass
// the same checks. Energies are in Hartree, temperatures in Kelvin.

use control::SmearingScheme;
use dwconsts::BOLTZMANN_CONSTANT;

/// Every smearing scheme. Adding a scheme to `SmearingScheme` makes the `match`
/// below fail to compile until the new scheme is listed here too.
fn all_schemes() -> Vec<SmearingScheme> {
    use SmearingScheme::*;
    let all = vec![Fd, Gs, Mp1, Mp2, Mv];
    for scheme in &all {
        match scheme {
            Fd | Gs | Mp1 | Mp2 | Mv => {}
        }
    }
    all
}

const FERMI: f64 = 0.25;

#[test]
fn test_full_below_and_empty_above_the_fermi_level() {
    // 1 Ha from the Fermi level is more than 100 kT even at 3000 K.
    for scheme in all_schemes() {
        let smearing = smearing::new(scheme);
        for temperature in [0.0, 300.0, 3000.0] {
            let below = smearing.get_occupation_number(FERMI, temperature, FERMI - 1.0);
            let above = smearing.get_occupation_number(FERMI, temperature, FERMI + 1.0);
            assert!(
                (below - 1.0).abs() < 1e-10,
                "{:?} at {} K: occupation {} well below the Fermi level",
                scheme,
                temperature,
                below
            );
            assert!(
                above.abs() < 1e-10,
                "{:?} at {} K: occupation {} well above the Fermi level",
                scheme,
                temperature,
                above
            );
        }
    }
}

#[test]
fn test_finite_near_the_fermi_level() {
    // Includes the Fermi level itself and temperature 0 (a step function).
    for scheme in all_schemes() {
        let smearing = smearing::new(scheme);
        for temperature in [0.0, 300.0] {
            for i in -50..=50 {
                let energy = FERMI + i as f64 * 1e-4;
                let occ = smearing.get_occupation_number(FERMI, temperature, energy);
                assert!(
                    occ.is_finite(),
                    "{:?} at {} K: occupation {} at e - ef = {}",
                    scheme,
                    temperature,
                    occ,
                    energy - FERMI
                );
            }
        }
    }
}

#[test]
fn test_electron_count_matches_the_step_function() {
    // For a constant density of states, smearing must not change the number of
    // electrons: the integral of f(e) - step(ef - e) over e vanishes.
    let temperature = 1000.0;
    let kt = BOLTZMANN_CONSTANT * temperature;
    let de = kt / 200.0;
    let n = 20 * 200; // integrate over +/- 20 kT

    for scheme in all_schemes() {
        let smearing = smearing::new(scheme);
        let mut excess = 0.0;
        for i in -n..n {
            let energy = FERMI + (i as f64 + 0.5) * de;
            let step = if energy < FERMI { 1.0 } else { 0.0 };
            excess += (smearing.get_occupation_number(FERMI, temperature, energy) - step) * de;
        }
        assert!(
            excess.abs() < 1e-6 * kt,
            "{:?}: smearing changes the electron count by {} kT",
            scheme,
            excess / kt
        );
    }
}
