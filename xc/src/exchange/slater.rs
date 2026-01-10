//! Slater exchange functional
//!
//! The Slater exchange is the simplest local exchange approximation,
//! derived from the uniform electron gas model.

/// One-third constant, used frequently in LDA formulas (e.g., rho^(1/3))
const T13: f64 = 1.0 / 3.0;

/// Computes the Slater exchange potential and energy density (unpolarized)
///
/// # Arguments
/// * `rho` - Electron density at a point
///
/// # Returns
/// * `(vx, ex)` - Exchange potential and energy density
///
/// The Slater exchange functional is:
/// - vx = cx * rho^(1/3), where cx = -(3/π)^(1/3)
/// - ex = (3/4) * vx
///
/// This is the simplest local exchange approximation, derived from
/// the uniform electron gas model.
#[inline]
pub fn slater_unpolarized(rho: f64) -> (f64, f64) {
    // Slater exchange coefficient: cx = -(3/π)^(1/3)
    let cx: f64 = -(3.0 / std::f64::consts::PI).powf(T13);

    // Exchange potential: vx = cx * rho^(1/3)
    let vx = cx * rho.powf(T13);

    // Exchange energy density: ex = (3/4) * vx
    // This relationship comes from the functional derivative of the exchange energy
    let ex = 0.75 * vx;

    (vx, ex)
}

/// Computes the Slater exchange potential and energy density (spin-polarized)
///
/// # Arguments
/// * `rho` - Total electron density at a point
/// * `zeta` - Spin polarization: (rho_up - rho_dn) / rho
///
/// # Returns
/// * `(vx_up, vx_dn, ex)` - Exchange potentials for up/down spins and energy density
///
/// For spin-polarized systems, the exchange is computed separately for each spin:
/// - vx_up = cx * ((1 + zeta) * rho)^(1/3)
/// - vx_dn = cx * ((1 - zeta) * rho)^(1/3)
/// - ex = weighted average of exchange energies
#[inline]
pub fn slater_polarized(rho: f64, zeta: f64) -> (f64, f64, f64) {
    // Slater exchange coefficient: cx = -(3/π)^(1/3)
    let cx: f64 = -(3.0 / std::f64::consts::PI).powf(T13);

    // Exchange potential for up spin: vx_up = cx * ((1 + zeta) * rho)^(1/3)
    let vx_up = cx * ((1.0 + zeta) * rho).powf(T13);
    let ex_up = 0.75 * vx_up;

    // Exchange potential for down spin: vx_dn = cx * ((1 - zeta) * rho)^(1/3)
    let vx_dn = cx * ((1.0 - zeta) * rho).powf(T13);
    let ex_dn = 0.75 * vx_dn;

    // Weighted average of exchange energies
    let ex = 0.5 * ((1.0 + zeta) * ex_up + (1.0 - zeta) * ex_dn);

    (vx_up, vx_dn, ex)
}
