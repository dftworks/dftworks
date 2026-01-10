//! Local Spin Density Approximation (LSDA) exchange-correlation functional
//! using Slater exchange and Perdew-Zunger correlation.
//!
//! This module implements the LSDA-PZ functional for spin-polarized systems,
//! which combines:
//! - Slater exchange: spin-polarized version
//! - Perdew-Zunger correlation: interpolated between unpolarized and polarized limits

use crate::correlation::pz::pz_polarized;
use crate::exchange::slater::slater_polarized;
use dfttypes::*;
use ndarray::Array3;
use types::c64;

/// Computes the exchange-correlation potential and energy density for LSDA-PZ
///
/// # Arguments
/// * `rho` - Electron density (must be spin-polarized)
/// * `_drho` - Density gradient (not used in LDA, only needed for GGA functionals)
/// * `vxc` - Output: exchange-correlation potential (mutated in place)
/// * `exc` - Output: exchange-correlation energy density (mutated in place)
///
/// The LSDA functional depends only on the local density, not its gradient.
/// For each grid point, we compute:
/// - Spin-polarized Slater exchange potential and energy
/// - Spin-polarized Perdew-Zunger correlation potential and energy
/// - Sum them to get the total XC potential and energy for each spin
pub fn compute(
    rho: &RHOR,
    _drho: Option<&DRHOR>,
    vxc: &mut VXCR,
    exc: &mut Array3<c64>,
) {
    // Extract spin-polarized density
    let (rho_up, rho_dn) = rho.as_spin().unwrap();
    let rho_up = rho_up.as_slice();
    let rho_dn = rho_dn.as_slice();

    let (vxc_up, vxc_dn) = vxc.as_spin_mut().unwrap();
    let vxc_up = vxc_up.as_mut_slice();
    let vxc_dn = vxc_dn.as_mut_slice();

    let exc = exc.as_mut_slice();

    let n = rho_up.len();

    for i in 0..n {
        // Total density
        let rho = rho_up[i].norm() + rho_dn[i].norm();

        // Spin polarization: zeta = (rho_up - rho_dn) / rho
        // Clamp to avoid division by zero
        let zeta = (rho_up[i].norm() - rho_dn[i].norm()) / rho.max(1.0E-20);

        // Compute spin-polarized Slater exchange
        let (vx_up, vx_dn, ex) = slater_polarized(rho, zeta);

        // Compute spin-polarized Perdew-Zunger correlation
        let (vc_up, vc_dn, ec) = pz_polarized(rho, zeta);

        // Total XC potential for up spin = exchange + correlation
        vxc_up[i] = c64 {
            re: vx_up + vc_up,
            im: 0.0,
        };

        // Total XC potential for down spin = exchange + correlation
        vxc_dn[i] = c64 {
            re: vx_dn + vc_dn,
            im: 0.0,
        };

        // Total XC energy density = exchange + correlation
        exc[i] = c64 {
            re: ex + ec,
            im: 0.0,
        };
    }
}
