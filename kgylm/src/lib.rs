//#![allow(warnings)]

use dwconsts::*;
use gvector::GVector;
use pwbasis::PWBasis;
use special;
use std::collections::HashMap;
use vector3::Vector3f64;
//
// Y_lm(k+G)
// dY_lm(k+G) / dx, dY_lm(k+G) / dy, dY_lm(k+G) / dz
//
// For each k point, there is a KGYLM instance
//
pub struct KGYLM {
    data: HashMap<(usize, i32), Vec<f64>>,
    data_diff: HashMap<(usize, i32), Vec<Vector3f64>>,
}

impl KGYLM {
    pub fn new(xk: Vector3f64, lmax: usize, gvec: &GVector, pwwfc: &PWBasis) -> KGYLM {
        let data = compute_data(xk, lmax, gvec, pwwfc);

        let data_diff = compute_data_derivatives(xk, lmax, gvec, pwwfc);

        KGYLM { data, data_diff }
    }

    pub fn get_data(&self, l: usize, m: i32) -> &[f64] {
        self.data.get(&(l, m)).unwrap()
    }

    pub fn get_data_derivatives(&self, l: usize, m: i32) -> &[Vector3f64] {
        self.data_diff.get(&(l, m)).unwrap()
    }
}

fn compute_data(
    xk: Vector3f64,
    lmax: usize,
    gvec: &GVector,
    pwwfc: &PWBasis,
) -> HashMap<(usize, i32), Vec<f64>> {
    let mut ylms = HashMap::new();

    let gindex = pwwfc.get_gindex();

    for l in 0..lmax + 1 {
        for m in utility::get_quant_num_m(l) {
            let t = compute_ylm_batch(gvec, xk, gindex, l, m);

            ylms.insert((l, m), t);
        }
    }

    ylms
}

fn compute_data_derivatives(
    xk: Vector3f64,
    lmax: usize,
    gvec: &GVector,
    pwwfc: &PWBasis,
) -> HashMap<(usize, i32), Vec<Vector3f64>> {
    let mut ylms = HashMap::new();

    let gindex = pwwfc.get_gindex();

    for l in 0..lmax + 1 {
        for m in utility::get_quant_num_m(l) {
            let t = compute_dylm_batch(gvec, xk, gindex, l, m);

            ylms.insert((l, m), t);
        }
    }

    ylms
}

fn compute_ylm_batch(
    gvec: &GVector,
    xk: Vector3f64,
    gindex: &[usize],
    l: usize,
    m: i32,
) -> Vec<f64> {
    let gcart = gvec.get_cart();

    let ng = gindex.len();

    let mut ylm = vec![0.0; ng];

    for (i, j) in gindex.iter().enumerate() {
        let mut xkg = xk + gcart[*j];

        if xkg.norm2() < EPS16 {
            xkg.y = xkg.y.signum() * EPS16;
        }

        ylm[i] = special::real_spherical_harmonics(l, m, xkg);
    }

    ylm
}

fn compute_dylm_batch(
    gvec: &GVector,
    xk: Vector3f64,
    gindex: &[usize],
    l: usize,
    m: i32,
) -> Vec<Vector3f64> {
    let gcart = gvec.get_cart();

    let ng = gindex.len();

    let mut ylm = vec![Vector3f64::zeros(); ng];

    for (i, j) in gindex.iter().enumerate() {
        let mut xkg = xk + gcart[*j];

        if xkg.norm2() < EPS16 {
            xkg.y = EPS16 * xkg.y.signum();
        }

        ylm[i] = special::real_spherical_harmonics_diff(l, m, xkg);
    }

    ylm
}
