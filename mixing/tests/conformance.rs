// Conformance tests: every density mixer, built through `mixing::new`, must pass the
// same checks. As in the SCF loop, the mixer gets the input density and the residual
// rho_out - rho_in, and replaces the input density with the next one.

use control::{MixingParams, MixingScheme};
use types::c64;

/// Every mixing scheme. Adding a scheme to `MixingScheme` makes the `match` below
/// fail to compile until the new scheme is listed here too.
fn all_schemes() -> Vec<MixingScheme> {
    use MixingScheme::*;
    let all = vec![Pulay, Broyden];
    for scheme in &all {
        match scheme {
            Pulay | Broyden => {}
        }
    }
    all
}

fn params(scheme: MixingScheme) -> MixingParams {
    MixingParams {
        scheme,
        ..MixingParams::default()
    }
}

/// |G| values for `n` plane waves, as the SCF loop passes them.
fn g_norms(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.1 * i as f64).collect()
}

fn start_density(n: usize) -> Vec<c64> {
    (0..n)
        .map(|i| c64::new(1.0 / (1.0 + i as f64), 0.1 * (i as f64).sin()))
        .collect()
}

#[test]
fn test_converged_density_stays_unchanged() {
    // Once the residual is zero the density is self-consistent, and the next step
    // must leave it where it is. The mixers first build up history from a few
    // ordinary steps, as in an SCF run that has just converged.
    let n = 16;
    for scheme in all_schemes() {
        let mut mixer = mixing::new(&params(scheme));
        let mut rho = start_density(n);
        for step in 1..4 {
            // Linearly independent residuals, as in a real SCF history.
            let residual: Vec<c64> = (0..n)
                .map(|i| c64::new(0.01 * ((i * step) as f64).sin(), 0.0))
                .collect();
            mixer.compute_next_density(&g_norms(n), &mut rho, &residual);
        }

        let converged = rho.clone();
        let zero = vec![c64::new(0.0, 0.0); n];
        mixer.compute_next_density(&g_norms(n), &mut rho, &zero);

        for (a, b) in rho.iter().zip(converged.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "{:?}: density moved by {} with zero residual",
                scheme,
                (a - b).norm()
            );
        }
    }
}

#[test]
fn test_first_step_is_linear_mixing() {
    // With no history yet, every mixer takes rho_in + alpha * (rho_out - rho_in).
    let n = 16;
    for scheme in all_schemes() {
        let p = params(scheme);
        let mut mixer = mixing::new(&p);
        let mut rho = start_density(n);
        let residual: Vec<c64> = (0..n).map(|i| c64::new(0.01 * i as f64, -0.02)).collect();

        let expected: Vec<c64> = rho
            .iter()
            .zip(residual.iter())
            .map(|(r, d)| r + p.alpha * d)
            .collect();
        mixer.compute_next_density(&g_norms(n), &mut rho, &residual);

        for (a, b) in rho.iter().zip(expected.iter()) {
            assert!(
                (a - b).norm() < 1e-14,
                "{:?}: first step is not linear mixing",
                scheme
            );
        }
    }
}

#[test]
fn test_solves_a_linear_scf_problem() {
    // Model SCF map rho_out = M rho_in + b with a diagonal M whose entries lie in
    // [-0.8, 0.8]. Plain linear mixing with alpha = 0.7 converges slowly; every
    // mixer must move much faster towards the fixed point rho* = b / (1 - M).
    // Entry 0 is G = 0, the total charge: as in a real SCF run it starts at its
    // fixed point and its residual stays zero.
    let n = 16;
    let mut m: Vec<f64> = (0..n)
        .map(|i| -0.8 + 1.6 * i as f64 / (n - 1) as f64)
        .collect();
    m[0] = 0.0;
    let b: Vec<c64> = (0..n)
        .map(|i| c64::new(0.5 + 0.1 * i as f64, 0.05))
        .collect();
    let fixed_point: Vec<c64> = (0..n).map(|i| b[i] / (1.0 - m[i])).collect();

    for scheme in all_schemes() {
        let mut mixer = mixing::new(&params(scheme));
        let mut rho = start_density(n);
        rho[0] = fixed_point[0];

        let residual_of =
            |rho: &[c64]| -> Vec<c64> { (0..n).map(|i| m[i] * rho[i] + b[i] - rho[i]).collect() };
        let norm = |v: &[c64]| v.iter().map(|d| d.norm_sqr()).sum::<f64>().sqrt();

        let initial = norm(&residual_of(&rho));
        for _ in 0..40 {
            let residual = residual_of(&rho);
            mixer.compute_next_density(&g_norms(n), &mut rho, &residual);
        }
        let last = norm(&residual_of(&rho));

        // Linear mixing alone would reduce the residual only by about 0.86^40 ~ 2e-3.
        assert!(
            last < 1e-6 * initial,
            "{:?}: residual went from {} to {} in 40 steps",
            scheme,
            initial,
            last
        );
        for (a, b) in rho.iter().zip(fixed_point.iter()) {
            assert!((a - b).norm() < 1e-5, "{:?}: wrong fixed point", scheme);
        }
    }
}
