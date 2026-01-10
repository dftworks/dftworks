//! Local Density Approximation (LDA) exchange-correlation functional
//! using Slater exchange and Perdew-Zunger correlation.
//!
//! This module implements the LDA-PZ functional, which combines:
//! - Slater exchange: the simplest local exchange approximation
//! - Perdew-Zunger correlation: a parameterized correlation functional
//!   with separate forms for high and low density regimes

use crate::correlation::pz::pz_unpolarized;
use crate::exchange::slater::slater_unpolarized;
use dfttypes::*;
use ndarray::Array3;
use types::c64;

/// Computes the exchange-correlation potential and energy density for LDA-PZ
///
/// # Arguments
/// * `rho` - Electron density (must be non-spin-polarized)
/// * `_drho` - Density gradient (not used in LDA, only needed for GGA functionals)
/// * `vxc` - Output: exchange-correlation potential (mutated in place)
/// * `exc` - Output: exchange-correlation energy density (mutated in place)
///
/// The LDA functional depends only on the local density, not its gradient.
/// For each grid point, we compute:
/// - Slater exchange potential and energy
/// - Perdew-Zunger correlation potential and energy
/// - Sum them to get the total XC potential and energy
pub fn compute(
    rho: &RHOR,
    _drho: Option<&DRHOR>,
    vxc: &mut VXCR,
    exc: &mut Array3<c64>,
) {
    // Extract non-spin-polarized density (LDA-PZ is for closed-shell systems)
    let rho = rho.as_non_spin().unwrap();
    let vxc = vxc.as_non_spin_mut().unwrap();

    // Get mutable slices for efficient iteration
    let rho = rho.as_slice();
    let vxc = vxc.as_mut_slice();

    let exc = exc.as_mut_slice();

    let n = rho.len();

    // Compute XC potential and energy at each grid point
    for i in 0..n {
        // Use norm to handle cases where rho[i] might be slightly negative
        // (due to numerical errors in density construction)
        let t = rho[i].norm();

        // Compute Slater exchange potential and energy
        let (vx, ex) = slater_unpolarized(t);

        // Compute Perdew-Zunger correlation potential and energy
        let (vc, ec) = pz_unpolarized(t);

        // Total XC potential = exchange + correlation
        vxc[i] = c64 {
            re: vx + vc,
            im: 0.0,
        };

        // Total XC energy density = exchange + correlation
        exc[i] = c64 {
            re: ex + ec,
            im: 0.0,
        };
    }
}
