//! Perdew-Zunger correlation functional
//!
//! The Perdew-Zunger correlation functional uses different parameterizations
//! for high-density (rs < 1) and low-density (rs >= 1) regimes, where
//! rs = (3/(4π*rho))^(1/3) is the Wigner-Seitz radius.
//!
//! The functional is based on quantum Monte Carlo data for the uniform electron gas.

/// One-third constant, used frequently in LDA formulas
const T13: f64 = 1.0 / 3.0;

/// Perdew-Zunger correlation parameters
#[derive(Debug, Clone, Copy)]
pub struct PZParams {
    /// Low-density regime parameters (rs >= 1)
    pub gamma: f64,
    pub beta1: f64,
    pub beta2: f64,
    /// High-density regime parameters (rs < 1)
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

/// Unpolarized (paramagnetic) PZ correlation parameters
pub const PZ_UNPOLARIZED: PZParams = PZParams {
    gamma: -0.1423,
    beta1: 1.0529,
    beta2: 0.3334,
    a: 0.0311,
    b: -0.048,
    c: 0.0020,
    d: -0.0116,
};

/// Polarized (ferromagnetic) PZ correlation parameters
pub const PZ_POLARIZED: PZParams = PZParams {
    gamma: -0.0843,
    beta1: 1.3981,
    beta2: 0.2611,
    a: 0.01555,
    b: -0.0269,
    c: 0.0007,
    d: -0.0048,
};

/// Computes the Perdew-Zunger correlation potential and energy density
///
/// # Arguments
/// * `rho` - Electron density at a point
/// * `params` - PZ correlation parameters (unpolarized or polarized)
///
/// # Returns
/// * `(vc, ec)` - Correlation potential and energy density
///
/// The functional uses different parameterizations for high-density (rs < 1)
/// and low-density (rs >= 1) regimes.
#[inline]
pub fn pz_correlation(rho: f64, params: PZParams) -> (f64, f64) {
    const FOURPI: f64 = 4.0 * std::f64::consts::PI;

    // Wigner-Seitz radius: rs = (3/(4π*rho))^(1/3)
    // This characterizes the density: small rs = high density, large rs = low density
    let rs = (3.0 / FOURPI / rho).powf(T13);

    if rs > 1.0 {
        // Low-density regime (rs >= 1): uses a Padé-like form
        let rroot = rs.sqrt();

        // Denominator for correlation energy: dt = 1 + β1*√rs + β2*rs
        let dt = 1.0 + params.beta1 * rroot + params.beta2 * rs;

        // Correlation energy density: ec = γ / dt
        let ec = params.gamma / dt;

        // Numerator for correlation potential: nt = 1 + (7/6)*β1*√rs + (4/3)*β2*rs
        // This comes from the functional derivative: vc = d(ρ*ec)/dρ
        let nt = 1.0 + 7.0 / 6.0 * params.beta1 * rroot + 4.0 / 3.0 * params.beta2 * rs;

        // Correlation potential: vc = ec * nt / dt
        let vc = ec * nt / dt;

        (vc, ec)
    } else {
        // High-density regime (rs < 1): uses a logarithmic expansion
        let rln = rs.ln();

        // Correlation potential: vc = A*ln(rs) + (B - A/3) + (2/3)*C*rs*ln(rs) + (1/3)*(2*D - C)*rs
        // The (B - A/3) term comes from the functional derivative of the energy
        let vc = params.a * rln
            + (params.b - params.a / 3.0)
            + 2.0 / 3.0 * params.c * rs * rln
            + 1.0 / 3.0 * (2.0 * params.d - params.c) * rs;

        // Correlation energy density: ec = A*ln(rs) + B + C*rs*ln(rs) + D*rs
        let ec = params.a * rln + params.b + params.c * rs * rln + params.d * rs;

        (vc, ec)
    }
}

/// Computes the Perdew-Zunger correlation for unpolarized systems
#[inline]
pub fn pz_unpolarized(rho: f64) -> (f64, f64) {
    pz_correlation(rho, PZ_UNPOLARIZED)
}

/// Computes the Perdew-Zunger correlation for polarized systems
///
/// # Arguments
/// * `rho` - Total electron density
/// * `zeta` - Spin polarization: (rho_up - rho_dn) / rho
///
/// # Returns
/// * `(vc_up, vc_dn, ec)` - Correlation potentials for up/down spins and energy density
///
/// Uses interpolation between unpolarized and fully polarized correlation
/// based on the spin polarization parameter zeta.
#[inline]
pub fn pz_polarized(rho: f64, zeta: f64) -> (f64, f64, f64) {
    // Compute correlation for unpolarized (u) and fully polarized (p) cases
    let (vc_u, ec_u) = pz_correlation(rho, PZ_UNPOLARIZED);
    let (vc_p, ec_p) = pz_correlation(rho, PZ_POLARIZED);

    // Interpolation function: f(zeta) = [(1+zeta)^(4/3) + (1-zeta)^(4/3) - 2] / [2^(4/3) - 2]
    let mut f = (1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0;
    f /= 2.0_f64.powf(4.0 / 3.0) - 2.0;

    // Interpolated correlation energy
    let ec = ec_u + f * (ec_p - ec_u);

    // Common part of correlation potential
    let vc_comm = vc_u + f * (vc_p - vc_u);

    // Derivative of interpolation function with respect to zeta
    let mut df = (1.0 + zeta).powf(1.0 / 3.0) - (1.0 - zeta).powf(1.0 / 3.0);
    df *= 4.0 / 3.0;
    df /= 2.0_f64.powf(4.0 / 3.0) - 2.0;

    // Spin-dependent correlation potentials
    let vc_up = vc_comm + df * (ec_p - ec_u) * (1.0 - zeta);
    let vc_dn = vc_comm + df * (ec_p - ec_u) * (-1.0 - zeta);

    (vc_up, vc_dn, ec)
}
