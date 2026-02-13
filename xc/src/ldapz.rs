use dfttypes::*;
use ndarray::Array3;
use types::c64;

use crate::XC;

const PI: f64 = std::f64::consts::PI;
const FOURPI: f64 = 4.0 * PI;
const ONE_THIRD: f64 = 1.0 / 3.0;
const FOUR_THIRD: f64 = 4.0 / 3.0;
const RHO_FLOOR: f64 = 1.0E-20;

const PZ_U_HIGH_RS_GAMMA: f64 = -0.1423;
const PZ_U_HIGH_RS_BETA1: f64 = 1.0529;
const PZ_U_HIGH_RS_BETA2: f64 = 0.3334;

const PZ_U_LOW_RS_A: f64 = 0.0311;
const PZ_U_LOW_RS_B: f64 = -0.048;
const PZ_U_LOW_RS_C: f64 = 0.0020;
const PZ_U_LOW_RS_D: f64 = -0.0116;

const PZ_P_HIGH_RS_GAMMA: f64 = -0.0843;
const PZ_P_HIGH_RS_BETA1: f64 = 1.3981;
const PZ_P_HIGH_RS_BETA2: f64 = 0.2611;

const PZ_P_LOW_RS_A: f64 = 0.01555;
const PZ_P_LOW_RS_B: f64 = -0.0269;
const PZ_P_LOW_RS_C: f64 = 0.0007;
const PZ_P_LOW_RS_D: f64 = -0.0048;

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
        _drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        match (rho, vxc) {
            (RHOR::NonSpin(rho), VXCR::NonSpin(vxc)) => {
                let rho = rho.as_slice();
                let vxc = vxc.as_mut_slice();
                let exc = exc.as_mut_slice();

                for i in 0..rho.len() {
                    let r = rho[i].norm().max(RHO_FLOOR);
                    let (pot, eps) = lda_pz_nonspin_point(r);

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

                for i in 0..rho_up.len() {
                    let ru = rho_up[i].norm();
                    let rd = rho_dn[i].norm();

                    let (vu, vd, eps) = lsda_pz_spin_point(ru, rd);

                    vxc_up[i] = c64 { re: vu, im: 0.0 };
                    vxc_dn[i] = c64 { re: vd, im: 0.0 };
                    exc[i] = c64 { re: eps, im: 0.0 };
                }
            }
            _ => panic!("LDA/LSDA-PZ XC called with inconsistent rho/vxc spin variants"),
        }
    }
}

#[inline]
fn lda_pz_nonspin_point(rho: f64) -> (f64, f64) {
    let (vx, ex) = slater_exchange_unpolarized(rho);
    let (vc, ec) = pz_correlation_unpolarized(rho);
    (vx + vc, ex + ec)
}

#[inline]
fn lsda_pz_spin_point(rho_up: f64, rho_dn: f64) -> (f64, f64, f64) {
    let ru = rho_up.max(0.0);
    let rd = rho_dn.max(0.0);
    let rho = (ru + rd).max(RHO_FLOOR);
    let zeta = ((ru - rd) / rho).clamp(-1.0, 1.0);

    let (vx_up, vx_dn, ex) = slater_exchange_spin(rho, zeta);
    let (vc_up, vc_dn, ec) = pz_correlation_spin(rho, zeta);

    (vx_up + vc_up, vx_dn + vc_dn, ex + ec)
}

#[inline]
fn slater_exchange_unpolarized(rho: f64) -> (f64, f64) {
    let cx = -(3.0 / PI).powf(ONE_THIRD);
    let vx = cx * rho.powf(ONE_THIRD);
    let ex = 0.75 * vx;
    (vx, ex)
}

#[inline]
fn slater_exchange_spin(rho: f64, zeta: f64) -> (f64, f64, f64) {
    let cx = -(3.0 / PI).powf(ONE_THIRD);

    let rho_up = ((1.0 + zeta) * rho).max(0.0);
    let rho_dn = ((1.0 - zeta) * rho).max(0.0);

    let vx_up = cx * rho_up.powf(ONE_THIRD);
    let vx_dn = cx * rho_dn.powf(ONE_THIRD);

    let ex_up = 0.75 * vx_up;
    let ex_dn = 0.75 * vx_dn;
    let ex = 0.5 * ((1.0 + zeta) * ex_up + (1.0 - zeta) * ex_dn);

    (vx_up, vx_dn, ex)
}

#[inline]
fn pz_correlation_unpolarized(rho: f64) -> (f64, f64) {
    pz_correlation_parametric(
        rho,
        PZ_U_HIGH_RS_GAMMA,
        PZ_U_HIGH_RS_BETA1,
        PZ_U_HIGH_RS_BETA2,
        PZ_U_LOW_RS_A,
        PZ_U_LOW_RS_B,
        PZ_U_LOW_RS_C,
        PZ_U_LOW_RS_D,
    )
}

#[inline]
fn pz_correlation_polarized_limit(rho: f64) -> (f64, f64) {
    pz_correlation_parametric(
        rho,
        PZ_P_HIGH_RS_GAMMA,
        PZ_P_HIGH_RS_BETA1,
        PZ_P_HIGH_RS_BETA2,
        PZ_P_LOW_RS_A,
        PZ_P_LOW_RS_B,
        PZ_P_LOW_RS_C,
        PZ_P_LOW_RS_D,
    )
}

#[inline]
fn pz_correlation_parametric(
    rho: f64,
    gamma: f64,
    beta1: f64,
    beta2: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> (f64, f64) {
    let rs = (3.0 / (FOURPI * rho.max(RHO_FLOOR))).powf(ONE_THIRD);

    if rs > 1.0 {
        let rroot = rs.sqrt();
        let dt = 1.0 + beta1 * rroot + beta2 * rs;
        let ec = gamma / dt;
        let nt = 1.0 + 7.0 / 6.0 * beta1 * rroot + 4.0 / 3.0 * beta2 * rs;
        let vc = ec * nt / dt;
        (vc, ec)
    } else {
        let rln = rs.ln();
        let vc =
            a * rln + (b - a / 3.0) + 2.0 / 3.0 * c * rs * rln + 1.0 / 3.0 * (2.0 * d - c) * rs;
        let ec = a * rln + b + c * rs * rln + d * rs;
        (vc, ec)
    }
}

#[inline]
fn pz_correlation_spin(rho: f64, zeta: f64) -> (f64, f64, f64) {
    let (vc_u, ec_u) = pz_correlation_unpolarized(rho);
    let (vc_p, ec_p) = pz_correlation_polarized_limit(rho);

    let denom = 2.0_f64.powf(FOUR_THIRD) - 2.0;
    let f = ((1.0 + zeta).powf(FOUR_THIRD) + (1.0 - zeta).powf(FOUR_THIRD) - 2.0) / denom;

    let ec = ec_u + f * (ec_p - ec_u);
    let vc_common = vc_u + f * (vc_p - vc_u);

    let df = (4.0 / 3.0) * ((1.0 + zeta).powf(ONE_THIRD) - (1.0 - zeta).powf(ONE_THIRD)) / denom;
    let dec = ec_p - ec_u;

    let vc_up = vc_common + df * dec * (1.0 - zeta);
    let vc_dn = vc_common + df * dec * (-1.0 - zeta);

    (vc_up, vc_dn, ec)
}

#[test]
fn test_lda_pz_nonspin_finite_outputs() {
    let shape = [2, 2, 2];
    let rho = RHOR::NonSpin(Array3::from_vec(
        shape,
        vec![
            c64::new(0.01, 0.0),
            c64::new(0.02, 0.0),
            c64::new(0.03, 0.0),
            c64::new(0.04, 0.0),
            c64::new(0.05, 0.0),
            c64::new(0.06, 0.0),
            c64::new(0.07, 0.0),
            c64::new(0.08, 0.0),
        ],
    ));
    let mut vxc = VXCR::NonSpin(Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);

    XCLDAPZ::new().potential_and_energy(&rho, None, &mut vxc, &mut exc);

    for (v, e) in vxc
        .as_non_spin()
        .unwrap()
        .as_slice()
        .iter()
        .zip(exc.as_slice().iter())
    {
        assert!(v.re.is_finite());
        assert!(e.re.is_finite());
    }
}

#[test]
fn test_lsda_pz_spin_finite_outputs() {
    let shape = [2, 2, 2];
    let rho = RHOR::Spin(
        Array3::from_vec(
            shape,
            vec![
                c64::new(0.03, 0.0),
                c64::new(0.02, 0.0),
                c64::new(0.04, 0.0),
                c64::new(0.01, 0.0),
                c64::new(0.05, 0.0),
                c64::new(0.02, 0.0),
                c64::new(0.04, 0.0),
                c64::new(0.03, 0.0),
            ],
        ),
        Array3::from_vec(
            shape,
            vec![
                c64::new(0.01, 0.0),
                c64::new(0.03, 0.0),
                c64::new(0.02, 0.0),
                c64::new(0.02, 0.0),
                c64::new(0.01, 0.0),
                c64::new(0.04, 0.0),
                c64::new(0.03, 0.0),
                c64::new(0.02, 0.0),
            ],
        ),
    );
    let mut vxc = VXCR::Spin(Array3::new(shape), Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);

    XCLDAPZ::new().potential_and_energy(&rho, None, &mut vxc, &mut exc);

    let (vup, vdn) = vxc.as_spin().unwrap();
    for ((vu, vd), e) in vup
        .as_slice()
        .iter()
        .zip(vdn.as_slice().iter())
        .zip(exc.as_slice().iter())
    {
        assert!(vu.re.is_finite());
        assert!(vd.re.is_finite());
        assert!(e.re.is_finite());
    }
}
