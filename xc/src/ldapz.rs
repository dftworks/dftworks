use dfttypes::*;
use ndarray::Array3;
use types::c64;

use crate::XC;

const T13: f64 = 1.0 / 3.0;

pub struct XCLDAPZ {}

impl XCLDAPZ {
    pub fn new() -> XCLDAPZ {
        XCLDAPZ {}
    }
}

impl XC for XCLDAPZ {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        _drho: &Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        let rho = rho.as_non_spin().unwrap();
        let vxc = vxc.as_non_spin_mut().unwrap();

        let rho = rho.as_slice();
        let vxc = vxc.as_mut_slice();

        let exc = exc.as_mut_slice();

        let n = rho.len();

        for i in 0..n {
            let t = rho[i].norm(); // in case rho[i] is slightly negative

            let (vx, ex) = evx_slater(t);

            let (vc, ec) = evc_pz(t);

            vxc[i] = c64 {
                re: vx + vc,
                im: 0.0,
            };

            exc[i] = c64 {
                re: ex + ec,
                im: 0.0,
            };
        }
    }
}

fn evx_slater(rho: f64) -> (f64, f64) {
    let cx: f64 = -4.0 / 4.0 * (3.0 / std::f64::consts::PI).powf(T13);

    let vx = cx * rho.powf(T13);

    let ex = 0.75 * vx;

    (vx, ex)
}

fn evc_pz(rho: f64) -> (f64, f64) {
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
