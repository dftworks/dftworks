#![allow(warnings)]

use itertools::multizip;
use matrix::*;
use ndarray::*;
use rand::{thread_rng, Rng};
use std::f64::consts;
use types::*;
use vector3::*;

const ZERO_C64: c64 = c64 { re: 0.0, im: 0.0 };

pub fn get_quant_num_m(l: usize) -> Vec<i32> {
    (0..2 * l + 1).map(|im| im as i32 - l as i32).collect()
}

pub fn get_slice_up_dn<T>(v: &[T]) -> (&[T], &[T]) {
    let n = v.len() / 2;

    (&v[0..n], &v[n..])
}

pub fn get_mut_slice_up_dn<T>(v: &mut [T]) -> (&mut [T], &mut [T]) {
    let n = v.len() / 2;

    let ptr = v.as_mut_ptr();

    (unsafe { std::slice::from_raw_parts_mut(ptr, n) }, unsafe {
        std::slice::from_raw_parts_mut(ptr.add(n), n)
    })
}

pub fn add_and_scale(inp: &[c64], out: &mut [c64], factor: f64) {
    assert_eq!(inp.len(), out.len());

    for (x, y) in multizip((inp.iter(), out.iter_mut())) {
        *y += *x * factor;
    }
}

pub fn add_and_zscale(inp: &[c64], out: &mut [c64], factor: c64) {
    assert_eq!(inp.len(), out.len());

    for (x, y) in multizip((inp.iter(), out.iter_mut())) {
        *y += *x * factor;
    }
}

pub fn dot_product_v3i32_v3f64(g: Vector3i32, r: Vector3f64) -> f64 {
    f64::from(g.x) * r.x + f64::from(g.y) * r.y + f64::from(g.z) * r.z
}

pub fn zdot_product(u: &[c64], v: &[c64]) -> c64 {
    assert_eq!(u.len(), v.len());

    // u.iter().zip(v.iter()).map(|(x, y)| x.conj() * (*y)).sum()

    multizip((u.iter(), v.iter()))
        .map(|(x, y)| x.conj() * (*y))
        .sum()
}

pub fn zdot_product_metric(u: &[c64], v: &[c64], metric: &[f64]) -> c64 {
    assert_eq!(u.len(), v.len());

    // u.iter()
    //     .zip(v.iter())
    //     .zip(metric.iter())
    //     .map(|((x, y), z)| x.conj() * (*z) * (*y))
    //     .sum()

    multizip((u.iter(), v.iter(), metric.iter()))
        .map(|(x, y, m)| x.conj() * (*m) * (*y))
        .sum()
}

pub fn dot_product_2(u: &[c64], v: &[c64]) -> c64 {
    assert_eq!(u.len(), v.len());
    let size = u.len() as isize;
    let mut ap = u.as_ptr();
    let mut bp = v.as_ptr();
    let mut tot = [ZERO_C64, ZERO_C64];

    unsafe {
        let end_ptr = ap.offset(size);
        const BLOCK_SIZE: isize = 32;
        let block_end_ptr = ap.offset(size & !(BLOCK_SIZE - 1));

        while ap != block_end_ptr {
            for i in 0..BLOCK_SIZE {
                tot[i as usize % 2] += (*ap.offset(i)).conj() * *bp.offset(i);
            }
            ap = ap.offset(BLOCK_SIZE);
            bp = bp.offset(BLOCK_SIZE);
        }

        tot[0] += tot[1];

        while ap != end_ptr {
            tot[0] += (*ap).conj() * *bp;
            ap = ap.offset(1);
            bp = bp.offset(1);
        }
    }

    tot[0]
}

pub fn ddot_product(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len());

    //return u.iter().zip(v.iter()).map(|(x, y)| (*x) * (*y)).sum();

    multizip((u.iter(), v.iter()))
        .map(|(x, y)| (*x) * (*y))
        .sum()
}

pub fn make_matrix(n: usize) -> Matrix<c64> {
    let mut m = Matrix::<c64>::new(n, n);

    for i in 0..n {
        for j in (i + 1)..n {
            m[[j, i]] = c64 {
                //re: rng.gen_range(0.0, 1.0),
                //im: rng.gen_range(0.0, 0.01),
                re: 0.1 * i as f64,
                im: 0.001 * j as f64,
            };

            m[[i, j]] = m[[j, i]].conj();
        }
        m[[i, i]] = c64 {
            //re: rng.gen_range(1.0, 1000.0),
            re: ((i + 1) as f64) * 1.0 - 0.1,
            im: 0.0,
        };
    }

    m
}

pub fn make_normalized_rand_vector(v: &mut [c64]) {
    let mut rng = thread_rng();

    for y in v.iter_mut() {
        let t = rng.gen_range(-0.5f64, 0.5f64);
        let theta = t * 2.0 * consts::PI;

        let re = t * theta.cos();
        let im = t * theta.sin();

        *y = c64 { re, im };
    }

    normalize_vector_c64(v);
}

pub fn vec_norm(v: &[c64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

pub fn l2_norm(v: &[c64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

pub fn normalize_vector_c64(v: &mut [c64]) {
    let s = vec_norm(&v);

    // for x in v.iter_mut() {
    //     *x /= s;
    // }

    v.iter_mut().for_each(|x| *x /= s);
}

pub fn hadamard_product(a: &[c64], b: &[c64], c: &mut [c64]) {
    for (x, y, z) in multizip((a.iter(), b.iter(), c.iter_mut())) {
        *z = x * y;
    }
}

pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();

    //
    // if r is close to zero, we will assume the r is approaching zero along the y axis.
    // So theta is set to pi/2 * sign_of_y.
    // This is very important when using complex spherical harmonics.
    //
    let mut theta = consts::PI / 2.0 * y.signum();

    let EPS12 = 1.0E-12;

    if r > EPS12 {
        theta = (z / r).acos();
    }

    let phi: f64;

    if x > EPS12 {
        phi = (y / x).atan();
    } else if x < -EPS12 {
        phi = (y / x).atan() + consts::PI;
    } else {
        phi = consts::PI / 2.0 * y.signum();
    }

    (r, theta, phi)
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    let x = r * theta.sin() * phi.cos();

    let y = r * theta.sin() * phi.sin();

    let z = r * theta.cos();

    (x, y, z)
}

pub fn argsort<T: PartialOrd>(v: &[T]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();

    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());

    idx
}

/// N even, 8
///
/// n : 0 1 2 3 4 5 6 7
///
/// i : 0 1 2 3 4 -3 -2 -1
///
/// N Odd, 7
///
/// n : 0 1 2 3 4 5 6
///
/// i : 0 1 2 3 -3 -2 -1
pub fn fft_left_end(n: usize) -> i32 {
    let nn = n as i32;

    if n % 2 == 0 {
        -(nn - 2) / 2
    } else {
        -(nn - 1) / 2
    }
}

pub fn fft_right_end(n: usize) -> i32 {
    let nn = n as i32;

    if n % 2 == 0 {
        nn / 2
    } else {
        (nn - 1) / 2
    }
}

pub fn fft_i2n(i: i32, ntot: usize) -> usize {
    if i < 0 {
        (i + ntot as i32) as usize
    } else {
        i as usize
    }
}

pub fn fft_n2i(n: usize, ntot: usize) -> i32 {
    if n > ntot / 2 {
        n as i32 - ntot as i32
    } else {
        n as i32
    }
}

pub fn compute_fft_linear_index_map(
    miller: &[Vector3i32],
    gindex: &[usize],
    n1: usize,
    n2: usize,
    n3: usize,
) -> Vec<usize> {
    let mut linear_index = Vec::with_capacity(gindex.len());

    for ig in gindex.iter() {
        let mi = miller[*ig];

        let idx0 = fft_i2n(mi.x, n1);
        let idx1 = fft_i2n(mi.y, n2);
        let idx2 = fft_i2n(mi.z, n3);

        debug_assert!(idx2 < n3);
        linear_index.push(idx0 + idx1 * n1 + idx2 * n1 * n2);
    }

    linear_index
}

pub fn map_3d_to_1d_with_linear_index(linear_index: &[usize], v3d: &Array3<c64>, v1d: &mut [c64]) {
    assert_eq!(linear_index.len(), v1d.len());

    let v3d_slice = v3d.as_slice();
    for (i, &idx) in linear_index.iter().enumerate() {
        v1d[i] = v3d_slice[idx];
    }
}

pub fn map_1d_to_3d_with_linear_index(linear_index: &[usize], v1d: &[c64], v3d: &mut Array3<c64>) {
    assert_eq!(linear_index.len(), v1d.len());
    v3d.set_value(ZERO_C64);

    let v3d_slice = v3d.as_mut_slice();
    for (i, &idx) in linear_index.iter().enumerate() {
        v3d_slice[idx] = v1d[i];
    }
}

pub fn map_3d_to_1d(
    miller: &[Vector3i32],
    gindex: &[usize],
    n1: usize,
    n2: usize,
    n3: usize,
    v3d: &Array3<c64>,
    v1d: &mut [c64],
) {
    v1d.iter_mut().map(|z| *z = ZERO_C64);

    for (i, ig) in gindex.iter().enumerate() {
        let mi = miller[*ig];

        let idx0 = fft_i2n(mi.x, n1);
        let idx1 = fft_i2n(mi.y, n2);
        let idx2 = fft_i2n(mi.z, n3);

        v1d[i] = v3d[[idx0, idx1, idx2]];
    }
}

pub fn map_1d_to_3d(
    miller: &[Vector3i32],
    gindex: &[usize],
    n1: usize,
    n2: usize,
    n3: usize,
    v1d: &[c64],
    v3d: &mut Array3<c64>,
) {
    v3d.set_value(ZERO_C64);

    for (i, ig) in gindex.iter().enumerate() {
        let mi = miller[*ig];

        let idx0 = fft_i2n(mi.x, n1);
        let idx1 = fft_i2n(mi.y, n2);
        let idx2 = fft_i2n(mi.z, n3);

        v3d[[idx0, idx1, idx2]] = v1d[i];
    }
}

#[test]
fn test_fft_end() {
    for n in 7..9 {
        let n1 = fft_left_end(n);
        let n2 = fft_right_end(n);
        let vec: Vec<i32> = (n1..n2 + 1).collect();

        println!("FFT range for N = {} : {:?}", n, vec);
    }
}

#[test]
fn test_fft_i2n() {
    let ntot = 7 as usize;

    for i in [0, 1, 2, 3, -3, -2, -1].iter() {
        let n = fft_i2n(*i, ntot);

        println!("{:5} {:5}", *i, n);
    }
}

#[test]
fn test_fft_n2i() {
    let ntot = 8 as usize;

    for n in [0, 1, 2, 3, 4, 5, 6, 7].iter() {
        let i = fft_n2i(*n, ntot);

        println!("{:5} {:5}", i, n);
    }
}

#[test]
fn test_cartesian_to_spherical() {
    let x = 2.01;
    let y = -0.0110911;
    let z = 1.00111;
    let (r, theta, phi) = cartesian_to_spherical(x, y, z);
    println!("r = {}, theta = {}, phi = {}", r, theta, phi);

    let (x, y, z) = spherical_to_cartesian(r, theta, phi);
    println!("x = {}, y = {}, z = {}", x, y, z);
}
