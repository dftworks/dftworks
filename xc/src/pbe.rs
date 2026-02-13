#![allow(warnings)]

use dfttypes::*;
use ndarray::Array3;
use types::c64;

use crate::XC;

pub struct XCPBE {}

impl XCPBE {
    pub fn new() -> XCPBE {
        XCPBE {}
    }
}

impl XC for XCPBE {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        match (rho, vxc) {
            (RHOR::NonSpin(rho), VXCR::NonSpin(vxc)) => {
                let rho = rho.as_slice();
                let vxc = vxc.as_mut_slice();
                let exc = exc.as_mut_slice();

                let grad = match drho {
                    Some(DRHOR::NonSpin(g)) => Some(g.as_slice()),
                    _ => None,
                };

                for i in 0..rho.len() {
                    let r = rho[i].norm().max(RHO_FLOOR);
                    let grad_norm = grad.map(|g| g[i].norm()).unwrap_or(0.0);

                    let eps = eps_xc_pbe_nonspin(r, grad_norm);
                    let pot = vxc_from_eps_nonspin(r, grad_norm);

                    vxc[i] = c64 { re: pot, im: 0.0 };
                    exc[i] = c64 { re: eps, im: 0.0 };
                }
            }
            (RHOR::Spin(rho_up, rho_dn), VXCR::Spin(vxc_up, vxc_dn)) => {
                let rho_up = rho_up.as_slice();
                let rho_dn = rho_dn.as_slice();
                let vxc_up = vxc_up.as_mut_slice();
                let vxc_dn = vxc_dn.as_mut_slice();
                let exc = exc.as_mut_slice();

                let (grad_up, grad_dn) = match drho {
                    Some(DRHOR::Spin(g_up, g_dn)) => (Some(g_up.as_slice()), Some(g_dn.as_slice())),
                    _ => (None, None),
                };

                for i in 0..rho_up.len() {
                    let ru = rho_up[i].norm().max(RHO_FLOOR);
                    let rd = rho_dn[i].norm().max(RHO_FLOOR);
                    let gu = grad_up.map(|g| g[i].norm()).unwrap_or(0.0);
                    let gd = grad_dn.map(|g| g[i].norm()).unwrap_or(0.0);

                    let eps_up = eps_xc_pbe_nonspin(ru, gu);
                    let eps_dn = eps_xc_pbe_nonspin(rd, gd);

                    let vu = vxc_from_eps_nonspin(ru, gu);
                    let vd = vxc_from_eps_nonspin(rd, gd);

                    let rho_tot = ru + rd;
                    let eps_tot = if rho_tot > RHO_FLOOR {
                        (ru * eps_up + rd * eps_dn) / rho_tot
                    } else {
                        0.0
                    };

                    vxc_up[i] = c64 { re: vu, im: 0.0 };
                    vxc_dn[i] = c64 { re: vd, im: 0.0 };
                    exc[i] = c64 {
                        re: eps_tot,
                        im: 0.0,
                    };
                }
            }
            _ => panic!("PBE XC called with inconsistent rho/vxc spin variants"),
        }
    }
}

const PI: f64 = std::f64::consts::PI;
const FOURPI: f64 = 4.0 * PI;
const ONE_THIRD: f64 = 1.0 / 3.0;
const TWO_THIRD: f64 = 2.0 / 3.0;
const FOUR_THIRD: f64 = 4.0 / 3.0;

const RHO_FLOOR: f64 = 1.0E-20;

// PBE constants
const KAPPA: f64 = 0.804;
const MU: f64 = 0.219_514_972_764_517_1;
const BETA: f64 = 0.066_724_550_603_149_22;
const GAMMA: f64 = 0.031_090_690_869_654_9; // (1 - ln 2) / pi^2

#[inline]
fn eps_xc_pbe_nonspin(rho: f64, grad_norm: f64) -> f64 {
    let ex = eps_x_pbe_nonspin(rho, grad_norm);
    let ec = eps_c_pbe_nonspin(rho, grad_norm);
    ex + ec
}

#[inline]
fn vxc_from_eps_nonspin(rho: f64, grad_norm: f64) -> f64 {
    // Local derivative at fixed |grad rho|. This is an approximation to the
    // full GGA functional derivative but provides a stable SCF potential path.
    let h = (rho * 1.0E-6).max(1.0E-12);
    let rp = rho + h;
    let rm = (rho - h).max(RHO_FLOOR);

    let fp = rp * eps_xc_pbe_nonspin(rp, grad_norm);
    let fm = rm * eps_xc_pbe_nonspin(rm, grad_norm);

    (fp - fm) / (rp - rm)
}

#[inline]
fn eps_x_pbe_nonspin(rho: f64, grad_norm: f64) -> f64 {
    // LDA exchange: eps_x = Cx * rho^(1/3)
    let cx = -0.75 * (3.0 / PI).powf(ONE_THIRD);
    let eps_x_lda = cx * rho.powf(ONE_THIRD);

    // Reduced gradient s = |grad rho| / (2 k_F rho), k_F = (3 pi^2 rho)^(1/3)
    let kf = (3.0 * PI * PI * rho).powf(ONE_THIRD);
    let s = grad_norm / (2.0 * kf * rho).max(RHO_FLOOR);
    let s2 = s * s;

    let fx = 1.0 + KAPPA - KAPPA / (1.0 + MU * s2 / KAPPA);
    eps_x_lda * fx
}

#[inline]
fn eps_c_pbe_nonspin(rho: f64, grad_norm: f64) -> f64 {
    let (_vc_lda, eps_c_lda) = evc_pz_unpolarized(rho);

    // PBE correlation gradient correction H(rs, t)
    let kf = (3.0 * PI * PI * rho).powf(ONE_THIRD);
    let ks = (FOURPI * kf / PI).sqrt();
    let t = grad_norm / (2.0 * ks * rho).max(RHO_FLOOR);
    let t2 = t * t;

    let exp_arg = (-eps_c_lda / GAMMA).clamp(-50.0, 50.0);
    let a = (BETA / GAMMA) / (exp_arg.exp() - 1.0).max(1.0E-14);
    let at2 = a * t2;
    let h = GAMMA * (1.0 + (BETA / GAMMA) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2)).ln();

    eps_c_lda + h
}

#[inline]
fn evc_pz_unpolarized(rho: f64) -> (f64, f64) {
    let rs = (3.0 / FOURPI / rho).powf(ONE_THIRD);

    if rs > 1.0 {
        const GAMMA_U: f64 = -0.1423;
        const BETA1_U: f64 = 1.0529;
        const BETA2_U: f64 = 0.3334;

        let rroot = rs.sqrt();
        let dt = 1.0 + BETA1_U * rroot + BETA2_U * rs;
        let ec = GAMMA_U / dt;
        let nt = 1.0 + 7.0 / 6.0 * BETA1_U * rroot + 4.0 / 3.0 * BETA2_U * rs;
        let vc = ec * nt / dt;
        (vc, ec)
    } else {
        const A_U: f64 = 0.0311;
        const B_U: f64 = -0.048;
        const C_U: f64 = 0.0020;
        const D_U: f64 = -0.0116;

        let rln = rs.ln();
        let vc = A_U * rln
            + (B_U - A_U * ONE_THIRD)
            + TWO_THIRD * C_U * rs * rln
            + ONE_THIRD * (2.0 * D_U - C_U) * rs;
        let ec = A_U * rln + B_U + C_U * rs * rln + D_U * rs;
        (vc, ec)
    }
}

#[test]
fn test_pbe_nonspin_finite_and_gradient_sensitive() {
    let n = 8;
    let shape = [2, 2, 2];

    let mut rho_data = Vec::with_capacity(n);
    for i in 0..n {
        rho_data.push(c64::new(0.02 + i as f64 * 0.001, 0.0));
    }
    let rho_arr = Array3::from_vec(shape, rho_data);
    let rho = RHOR::NonSpin(rho_arr.clone());

    let mut vxc_no_grad = VXCR::NonSpin(Array3::new(shape));
    let mut exc_no_grad = Array3::<c64>::new(shape);

    let pbe = XCPBE::new();
    pbe.potential_and_energy(&rho, None, &mut vxc_no_grad, &mut exc_no_grad);

    let mut grad_data = Vec::with_capacity(n);
    for i in 0..n {
        grad_data.push(c64::new(0.001 + i as f64 * 1.0E-4, 0.0));
    }
    let drho = DRHOR::NonSpin(Array3::from_vec(shape, grad_data));

    let mut vxc_with_grad = VXCR::NonSpin(Array3::new(shape));
    let mut exc_with_grad = Array3::<c64>::new(shape);
    pbe.potential_and_energy(&rho, Some(&drho), &mut vxc_with_grad, &mut exc_with_grad);

    let v0 = vxc_no_grad.as_non_spin().unwrap();
    let v0s = v0.as_slice();
    let e0 = exc_no_grad.as_slice();
    let v1 = vxc_with_grad.as_non_spin().unwrap();
    let v1s = v1.as_slice();
    let e1 = exc_with_grad.as_slice();

    for i in 0..n {
        assert!(v0s[i].re.is_finite());
        assert!(e0[i].re.is_finite());
        assert!(v1s[i].re.is_finite());
        assert!(e1[i].re.is_finite());
    }

    let gradient_changed = v0s
        .iter()
        .zip(v1s.iter())
        .any(|(a, b)| (a.re - b.re).abs() > 1.0E-12);
    assert!(gradient_changed);
}

#[test]
fn test_pbe_spin_finite_outputs() {
    let n = 8;
    let shape = [2, 2, 2];

    let mut rho_up = Vec::with_capacity(n);
    let mut rho_dn = Vec::with_capacity(n);
    let mut grad_up = Vec::with_capacity(n);
    let mut grad_dn = Vec::with_capacity(n);

    for i in 0..n {
        rho_up.push(c64::new(0.015 + i as f64 * 8.0E-4, 0.0));
        rho_dn.push(c64::new(0.010 + i as f64 * 5.0E-4, 0.0));
        grad_up.push(c64::new(8.0E-4 + i as f64 * 5.0E-5, 0.0));
        grad_dn.push(c64::new(6.0E-4 + i as f64 * 4.0E-5, 0.0));
    }

    let rho = RHOR::Spin(
        Array3::from_vec(shape, rho_up),
        Array3::from_vec(shape, rho_dn),
    );
    let drho = DRHOR::Spin(
        Array3::from_vec(shape, grad_up),
        Array3::from_vec(shape, grad_dn),
    );

    let mut vxc = VXCR::Spin(Array3::new(shape), Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);

    let pbe = XCPBE::new();
    pbe.potential_and_energy(&rho, Some(&drho), &mut vxc, &mut exc);

    let (vup, vdn) = vxc.as_spin().unwrap();
    let e = exc.as_slice();
    for i in 0..n {
        assert!(vup.as_slice()[i].re.is_finite());
        assert!(vdn.as_slice()[i].re.is_finite());
        assert!(e[i].re.is_finite());
    }
}
