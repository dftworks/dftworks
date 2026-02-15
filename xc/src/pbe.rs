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
        gvec: &gvector::GVector,
        pwden: &pwdensity::PWDensity,
        rgtrans: &rgtransform::RGTransform,
        rho: &RHOR,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        match (rho, vxc) {
            (RHOR::NonSpin(rho), VXCR::NonSpin(vxc)) => {
                compute_nonspin_potential(gvec, pwden, rgtrans, rho, vxc, exc);
            }
            (RHOR::Spin(rho_up, rho_dn), VXCR::Spin(vxc_up, vxc_dn)) => {
                compute_spin_potential(gvec, pwden, rgtrans, rho_up, rho_dn, vxc_up, vxc_dn, exc);
            }
            _ => panic!("PBE XC called with inconsistent rho/vxc spin variants"),
        }
    }
}

fn compute_nonspin_potential(
    gvec: &gvector::GVector,
    pwden: &pwdensity::PWDensity,
    rgtrans: &rgtransform::RGTransform,
    rho: &Array3<c64>,
    vxc: &mut Array3<c64>,
    exc: &mut Array3<c64>,
) {
    let nfft = rho.as_slice().len();

    let mut grad_x = Array3::<c64>::new(rho.shape());
    let mut grad_y = Array3::<c64>::new(rho.shape());
    let mut grad_z = Array3::<c64>::new(rho.shape());
    rgtrans.gradient_r3d(
        gvec,
        pwden,
        rho.as_slice(),
        grad_x.as_mut_slice(),
        grad_y.as_mut_slice(),
        grad_z.as_mut_slice(),
    );

    let mut local_part = vec![0.0; nfft];
    let mut bx = vec![c64::new(0.0, 0.0); nfft];
    let mut by = vec![c64::new(0.0, 0.0); nfft];
    let mut bz = vec![c64::new(0.0, 0.0); nfft];

    build_channel_terms(
        rho.as_slice(),
        grad_x.as_slice(),
        grad_y.as_slice(),
        grad_z.as_slice(),
        &mut local_part,
        exc.as_mut_slice(),
        &mut bx,
        &mut by,
        &mut bz,
    );

    let mut div_b = vec![c64::new(0.0, 0.0); nfft];
    rgtrans.divergence_r3d(gvec, pwden, &bx, &by, &bz, &mut div_b);

    let vxc_slice = vxc.as_mut_slice();
    for i in 0..nfft {
        vxc_slice[i] = c64::new(local_part[i] - div_b[i].re, 0.0);
    }
}

fn compute_spin_potential(
    gvec: &gvector::GVector,
    pwden: &pwdensity::PWDensity,
    rgtrans: &rgtransform::RGTransform,
    rho_up: &Array3<c64>,
    rho_dn: &Array3<c64>,
    vxc_up: &mut Array3<c64>,
    vxc_dn: &mut Array3<c64>,
    exc: &mut Array3<c64>,
) {
    let nfft = rho_up.as_slice().len();

    let mut grad_up_x = Array3::<c64>::new(rho_up.shape());
    let mut grad_up_y = Array3::<c64>::new(rho_up.shape());
    let mut grad_up_z = Array3::<c64>::new(rho_up.shape());
    rgtrans.gradient_r3d(
        gvec,
        pwden,
        rho_up.as_slice(),
        grad_up_x.as_mut_slice(),
        grad_up_y.as_mut_slice(),
        grad_up_z.as_mut_slice(),
    );

    let mut grad_dn_x = Array3::<c64>::new(rho_dn.shape());
    let mut grad_dn_y = Array3::<c64>::new(rho_dn.shape());
    let mut grad_dn_z = Array3::<c64>::new(rho_dn.shape());
    rgtrans.gradient_r3d(
        gvec,
        pwden,
        rho_dn.as_slice(),
        grad_dn_x.as_mut_slice(),
        grad_dn_y.as_mut_slice(),
        grad_dn_z.as_mut_slice(),
    );

    let mut local_up = vec![0.0; nfft];
    let mut eps_up = vec![c64::new(0.0, 0.0); nfft];
    let mut bup_x = vec![c64::new(0.0, 0.0); nfft];
    let mut bup_y = vec![c64::new(0.0, 0.0); nfft];
    let mut bup_z = vec![c64::new(0.0, 0.0); nfft];

    build_channel_terms(
        rho_up.as_slice(),
        grad_up_x.as_slice(),
        grad_up_y.as_slice(),
        grad_up_z.as_slice(),
        &mut local_up,
        &mut eps_up,
        &mut bup_x,
        &mut bup_y,
        &mut bup_z,
    );

    let mut local_dn = vec![0.0; nfft];
    let mut eps_dn = vec![c64::new(0.0, 0.0); nfft];
    let mut bdn_x = vec![c64::new(0.0, 0.0); nfft];
    let mut bdn_y = vec![c64::new(0.0, 0.0); nfft];
    let mut bdn_z = vec![c64::new(0.0, 0.0); nfft];

    build_channel_terms(
        rho_dn.as_slice(),
        grad_dn_x.as_slice(),
        grad_dn_y.as_slice(),
        grad_dn_z.as_slice(),
        &mut local_dn,
        &mut eps_dn,
        &mut bdn_x,
        &mut bdn_y,
        &mut bdn_z,
    );

    let mut div_up = vec![c64::new(0.0, 0.0); nfft];
    let mut div_dn = vec![c64::new(0.0, 0.0); nfft];
    rgtrans.divergence_r3d(gvec, pwden, &bup_x, &bup_y, &bup_z, &mut div_up);
    rgtrans.divergence_r3d(gvec, pwden, &bdn_x, &bdn_y, &bdn_z, &mut div_dn);

    let rho_up_slice = rho_up.as_slice();
    let rho_dn_slice = rho_dn.as_slice();
    let vxc_up_slice = vxc_up.as_mut_slice();
    let vxc_dn_slice = vxc_dn.as_mut_slice();
    let exc_slice = exc.as_mut_slice();

    for i in 0..nfft {
        let ru = rho_up_slice[i].norm().max(RHO_FLOOR);
        let rd = rho_dn_slice[i].norm().max(RHO_FLOOR);
        let rt = ru + rd;

        vxc_up_slice[i] = c64::new(local_up[i] - div_up[i].re, 0.0);
        vxc_dn_slice[i] = c64::new(local_dn[i] - div_dn[i].re, 0.0);

        let eps = if rt > RHO_FLOOR {
            (ru * eps_up[i].re + rd * eps_dn[i].re) / rt
        } else {
            0.0
        };
        exc_slice[i] = c64::new(eps, 0.0);
    }
}

fn build_channel_terms(
    rho: &[c64],
    grad_x: &[c64],
    grad_y: &[c64],
    grad_z: &[c64],
    local_part: &mut [f64],
    eps_out: &mut [c64],
    bx: &mut [c64],
    by: &mut [c64],
    bz: &mut [c64],
) {
    for i in 0..rho.len() {
        let r = rho[i].norm().max(RHO_FLOOR);

        let gx = grad_x[i].re;
        let gy = grad_y[i].re;
        let gz = grad_z[i].re;
        let grad_norm = (gx * gx + gy * gy + gz * gz).sqrt();

        let (df_drho, df_dgrad, eps) = pbe_local_derivatives(r, grad_norm);
        local_part[i] = df_drho;
        eps_out[i] = c64::new(eps, 0.0);

        if grad_norm > GRAD_FLOOR {
            let coeff = df_dgrad / grad_norm;
            bx[i] = c64::new(coeff * gx, 0.0);
            by[i] = c64::new(coeff * gy, 0.0);
            bz[i] = c64::new(coeff * gz, 0.0);
        } else {
            bx[i] = c64::new(0.0, 0.0);
            by[i] = c64::new(0.0, 0.0);
            bz[i] = c64::new(0.0, 0.0);
        }
    }
}

#[inline]
fn pbe_local_derivatives(rho: f64, grad_norm: f64) -> (f64, f64, f64) {
    let rho0 = rho.max(RHO_FLOOR);
    let g0 = grad_norm.max(0.0);

    let eps = eps_xc_pbe_nonspin(rho0, g0);

    let drho = (rho0 * 1.0E-6).max(1.0E-10);
    let rho_p = rho0 + drho;
    let rho_m = (rho0 - drho).max(RHO_FLOOR);

    let f_p = pbe_energy_density(rho_p, g0);
    let f_m = pbe_energy_density(rho_m, g0);
    let df_drho = (f_p - f_m) / (rho_p - rho_m);

    let dgrad = (g0 * 1.0E-6).max(1.0E-10);
    let g_p = g0 + dgrad;
    let g_m = (g0 - dgrad).max(0.0);

    let df_dgrad = if g_p > g_m {
        let fg_p = pbe_energy_density(rho0, g_p);
        let fg_m = pbe_energy_density(rho0, g_m);
        (fg_p - fg_m) / (g_p - g_m)
    } else {
        0.0
    };

    (df_drho, df_dgrad, eps)
}

#[inline]
fn pbe_energy_density(rho: f64, grad_norm: f64) -> f64 {
    let rho = rho.max(RHO_FLOOR);
    rho * eps_xc_pbe_nonspin(rho, grad_norm.max(0.0))
}

const PI: f64 = std::f64::consts::PI;
const FOURPI: f64 = 4.0 * PI;
const ONE_THIRD: f64 = 1.0 / 3.0;
const TWO_THIRD: f64 = 2.0 / 3.0;

const RHO_FLOOR: f64 = 1.0E-20;
const GRAD_FLOOR: f64 = 1.0E-16;

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

#[cfg(test)]
fn make_xc_test_context(
    shape: [usize; 3],
    ecut: f64,
) -> (gvector::GVector, pwdensity::PWDensity, rgtransform::RGTransform) {
    let latt = lattice::Lattice::new(&[8.0, 0.0, 0.0], &[0.0, 8.0, 0.0], &[0.0, 0.0, 8.0]);
    let gvec = gvector::GVector::new(&latt, shape[0], shape[1], shape[2]);
    let pwden = pwdensity::PWDensity::new(ecut, &gvec);
    let rgtrans = rgtransform::RGTransform::new(shape[0], shape[1], shape[2]);
    (gvec, pwden, rgtrans)
}

#[cfg(test)]
fn eval_nonspin_vxc_exc(
    rho_data: Vec<c64>,
    shape: [usize; 3],
    gvec: &gvector::GVector,
    pwden: &pwdensity::PWDensity,
    rgtrans: &rgtransform::RGTransform,
) -> (Array3<c64>, Array3<c64>) {
    let rho = RHOR::NonSpin(Array3::from_vec(shape, rho_data));
    let mut vxc = VXCR::NonSpin(Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);
    XCPBE::new().potential_and_energy(gvec, pwden, rgtrans, &rho, &mut vxc, &mut exc);
    (vxc.as_non_spin().unwrap().clone(), exc)
}

#[test]
fn test_pbe_nonspin_finite_outputs() {
    let shape = [4, 4, 4];
    let n = shape[0] * shape[1] * shape[2];

    let mut rho_data = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / n as f64;
        rho_data.push(c64::new(0.02 + 0.005 * (2.0 * PI * x).sin(), 0.0));
    }

    let rho = RHOR::NonSpin(Array3::from_vec(shape, rho_data));
    let mut vxc = VXCR::NonSpin(Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);

    let (gvec, pwden, rgtrans) = make_xc_test_context(shape, 30.0);
    XCPBE::new().potential_and_energy(&gvec, &pwden, &rgtrans, &rho, &mut vxc, &mut exc);

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
fn test_pbe_nonspin_variational_consistency() {
    let shape = [6, 6, 6];
    let n = shape[0] * shape[1] * shape[2];
    let alpha = 1.0E-4;

    let mut rho0 = Vec::with_capacity(n);
    let mut drho = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        let p = 2.0 * PI * t;

        // Keep rho well above the floor so the derivative is smooth.
        let rho = 0.03 + 0.004 * p.sin() + 0.002 * (2.0 * p).cos();
        let delta = 0.01 * (p.cos() + 0.5 * (3.0 * p).sin());

        rho0.push(c64::new(rho, 0.0));
        drho.push(c64::new(delta, 0.0));
    }

    let (gvec, pwden, rgtrans) = make_xc_test_context(shape, 30.0);

    let (vxc0, exc0) = eval_nonspin_vxc_exc(rho0.clone(), shape, &gvec, &pwden, &rgtrans);

    let mut rho_p = Vec::with_capacity(n);
    let mut rho_m = Vec::with_capacity(n);
    for i in 0..n {
        rho_p.push(c64::new((rho0[i].re + alpha * drho[i].re).max(RHO_FLOOR), 0.0));
        rho_m.push(c64::new((rho0[i].re - alpha * drho[i].re).max(RHO_FLOOR), 0.0));
    }

    let (_vxc_p, exc_p) = eval_nonspin_vxc_exc(rho_p.clone(), shape, &gvec, &pwden, &rgtrans);
    let (_vxc_m, exc_m) = eval_nonspin_vxc_exc(rho_m.clone(), shape, &gvec, &pwden, &rgtrans);

    let mut e0 = 0.0;
    let mut ep = 0.0;
    let mut em = 0.0;
    let mut linear = 0.0;

    for i in 0..n {
        e0 += rho0[i].re * exc0.as_slice()[i].re;
        ep += rho_p[i].re * exc_p.as_slice()[i].re;
        em += rho_m[i].re * exc_m.as_slice()[i].re;
        linear += vxc0.as_slice()[i].re * drho[i].re;
    }

    let n_inv = 1.0 / n as f64;
    e0 *= n_inv;
    ep *= n_inv;
    em *= n_inv;
    linear *= n_inv;

    let fd = (ep - em) / (2.0 * alpha);
    let abs_err = (fd - linear).abs();
    let rel_err = abs_err / fd.abs().max(1.0E-12);

    // Use a conservative tolerance because both v_xc and FD are computed numerically.
    assert!(
        rel_err < 5.0E-3,
        "PBE variational consistency failed: fd={:.6e}, linear={:.6e}, abs={:.3e}, rel={:.3e}, e0={:.6e}",
        fd,
        linear,
        abs_err,
        rel_err,
        e0
    );
}

#[test]
fn test_pbe_gradient_changes_energy_density() {
    let rho = 0.03;
    let e0 = eps_xc_pbe_nonspin(rho, 0.0);
    let e1 = eps_xc_pbe_nonspin(rho, 1.0E-3);
    assert!((e0 - e1).abs() > 1.0E-12);
}

#[test]
fn test_pbe_spin_finite_outputs() {
    let shape = [4, 4, 4];
    let n = shape[0] * shape[1] * shape[2];

    let mut rho_up_data = Vec::with_capacity(n);
    let mut rho_dn_data = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / n as f64;
        rho_up_data.push(c64::new(0.018 + 0.004 * (2.0 * PI * x).sin(), 0.0));
        rho_dn_data.push(c64::new(0.012 + 0.003 * (2.0 * PI * x).cos(), 0.0));
    }

    let rho = RHOR::Spin(
        Array3::from_vec(shape, rho_up_data),
        Array3::from_vec(shape, rho_dn_data),
    );

    let mut vxc = VXCR::Spin(Array3::new(shape), Array3::new(shape));
    let mut exc = Array3::<c64>::new(shape);

    let (gvec, pwden, rgtrans) = make_xc_test_context(shape, 30.0);
    XCPBE::new().potential_and_energy(&gvec, &pwden, &rgtrans, &rho, &mut vxc, &mut exc);

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
