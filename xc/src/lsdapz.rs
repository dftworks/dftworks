use dfttypes::*;
use ndarray::Array3;
use types::c64;

use crate::XC;

const T13: f64 = 1.0 / 3.0;

pub struct XCLSDAPZ {}

impl XCLSDAPZ {
    pub fn new() -> XCLSDAPZ {
        XCLSDAPZ {}
    }
}

impl XC for XCLSDAPZ {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        _drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        let (rho_up, rho_dn) = rho.as_spin().unwrap();
        let rho_up = rho_up.as_slice();
        let rho_dn = rho_dn.as_slice();

        let (vxc_up, vxc_dn) = vxc.as_spin_mut().unwrap();
        let vxc_up = vxc_up.as_mut_slice();
        let vxc_dn = vxc_dn.as_mut_slice();

        let exc = exc.as_mut_slice();

        let n = rho_up.len();

        for i in 0..n {
            let rho = rho_up[i].norm() + rho_dn[i].norm();

            let zeta = (rho_up[i].norm() - rho_dn[i].norm()) / rho.max(1.0E-20);

            let (vx_up, vx_dn, ex) = exch_slater(rho, zeta);

            let (vc_up, vc_dn, ec) = corr_pz(rho, zeta);

            vxc_up[i] = c64 {
                re: vx_up + vc_up,
                im: 0.0,
            };

            vxc_dn[i] = c64 {
                re: vx_dn + vc_dn,
                im: 0.0,
            };

            exc[i] = c64 {
                re: ex + ec,
                im: 0.0,
            };
        }
    }
}

// (vx_up, vx_dn, ex)

fn exch_slater(rho: f64, zeta: f64) -> (f64, f64, f64) {
    let cx: f64 = -(3.0 / std::f64::consts::PI).powf(T13);

    let vx_up = cx * ((1.0 + zeta) * rho).powf(T13);
    let ex_up = 0.75 * vx_up;

    let vx_dn = cx * ((1.0 - zeta) * rho).powf(T13);
    let ex_dn = 0.75 * vx_dn;

    let ex = 0.5 * ((1.0 + zeta) * ex_up + (1.0 - zeta) * ex_dn);

    (vx_up, vx_dn, ex)
}

// (vc_up, vc_dn, ec)

fn corr_pz(rho: f64, zeta: f64) -> (f64, f64, f64) {
    let (vc_u, ec_u) = evc_pz_u(rho);

    let (vc_p, ec_p) = evc_pz_p(rho);

    let mut f = (1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0;

    f /= 2.0_f64.powf(4.0 / 3.0) - 2.0;

    let ec = ec_u + f * (ec_p - ec_u);

    let vc_comm = vc_u + f * (vc_p - vc_u);

    let mut df = (1.0 + zeta).powf(1.0 / 3.0) - (1.0 - zeta).powf(1.0 / 3.0);

    df *= 4.0 / 3.0;

    df /= 2.0_f64.powf(4.0 / 3.0) - 2.0;

    let vc_up = vc_comm + df * (ec_p - ec_u) * (1.0 - zeta);

    let vc_dn = vc_comm + df * (ec_p - ec_u) * (-1.0 - zeta);

    (vc_up, vc_dn, ec)
}

fn evc_pz_u(rho: f64) -> (f64, f64) {
    const FOURPI: f64 = 4.0 * std::f64::consts::PI;

    let rs = (3.0 / FOURPI / rho).powf(T13);

    if rs > 1.0 {
        const GAMMA: f64 = -0.1423;
        const BETA1: f64 = 1.0529;
        const BETA2: f64 = 0.3334;

        let rroot = rs.sqrt();

        let dt = 1.0 + BETA1 * rroot + BETA2 * rs;

        let ec = GAMMA / dt;

        let nt = 1.0 + 7.0 / 6.0 * BETA1 * rroot + 4.0 / 3.0 * BETA2 * rs;

        let vc = ec * nt / dt;

        (vc, ec)
    } else {
        const A: f64 = 0.0311;
        const B: f64 = -0.048;
        const C: f64 = 0.0020;
        const D: f64 = -0.0116;

        let rln = rs.ln();

        let vc =
            A * rln + (B - A / 3.0) + 2.0 / 3.0 * C * rs * rln + 1.0 / 3.0 * (2.0 * D - C) * rs;

        let ec = A * rln + B + C * rs * rln + D * rs;

        (vc, ec)
    }
}

fn evc_pz_p(rho: f64) -> (f64, f64) {
    const FOURPI: f64 = 4.0 * std::f64::consts::PI;

    let rs = (3.0 / FOURPI / rho).powf(T13);

    if rs > 1.0 {
        const GAMMA: f64 = -0.0843;
        const BETA1: f64 = 1.3981;
        const BETA2: f64 = 0.2611;

        let rroot = rs.sqrt();

        let dt = 1.0 + BETA1 * rroot + BETA2 * rs;

        let ec = GAMMA / dt;

        let nt = 1.0 + 7.0 / 6.0 * BETA1 * rroot + 4.0 / 3.0 * BETA2 * rs;

        let vc = ec * nt / dt;

        (vc, ec)
    } else {
        const A: f64 = 0.01555;
        const B: f64 = -0.0269;
        const C: f64 = 0.0007;
        const D: f64 = -0.0048;

        let rln = rs.ln();

        let vc =
            A * rln + (B - A / 3.0) + 2.0 / 3.0 * C * rs * rln + 1.0 / 3.0 * (2.0 * D - C) * rs;

        let ec = A * rln + B + C * rs * rln + D * rs;

        (vc, ec)
    }
}
