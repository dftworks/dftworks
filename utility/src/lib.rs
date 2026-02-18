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
    let n = u.len();
    let mut i = 0;
    let mut acc0 = ZERO_C64;
    let mut acc1 = ZERO_C64;
    let mut acc2 = ZERO_C64;
    let mut acc3 = ZERO_C64;

    while i + 3 < n {
        acc0 += u[i].conj() * v[i];
        acc1 += u[i + 1].conj() * v[i + 1];
        acc2 += u[i + 2].conj() * v[i + 2];
        acc3 += u[i + 3].conj() * v[i + 3];
        i += 4;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < n {
        sum += u[i].conj() * v[i];
        i += 1;
    }

    sum
}

pub fn zdot_product_metric(u: &[c64], v: &[c64], metric: &[f64]) -> c64 {
    assert_eq!(u.len(), v.len());
    assert_eq!(u.len(), metric.len());

    let n = u.len();
    let mut i = 0;
    let mut acc0 = ZERO_C64;
    let mut acc1 = ZERO_C64;
    let mut acc2 = ZERO_C64;
    let mut acc3 = ZERO_C64;

    while i + 3 < n {
        acc0 += u[i].conj() * metric[i] * v[i];
        acc1 += u[i + 1].conj() * metric[i + 1] * v[i + 1];
        acc2 += u[i + 2].conj() * metric[i + 2] * v[i + 2];
        acc3 += u[i + 3].conj() * metric[i + 3] * v[i + 3];
        i += 4;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < n {
        sum += u[i].conj() * metric[i] * v[i];
        i += 1;
    }

    sum
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
    let n = u.len();
    let mut i = 0;
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;

    while i + 3 < n {
        acc0 += u[i] * v[i];
        acc1 += u[i + 1] * v[i + 1];
        acc2 += u[i + 2] * v[i + 2];
        acc3 += u[i + 3] * v[i + 3];
        i += 4;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < n {
        sum += u[i] * v[i];
        i += 1;
    }

    sum
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
    v1d.iter_mut().for_each(|z| *z = ZERO_C64);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_f64(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "lhs = {lhs}, rhs = {rhs}, |diff| = {}",
            (lhs - rhs).abs()
        );
    }

    fn assert_close_c64(lhs: c64, rhs: c64, tol: f64) {
        assert!(
            (lhs - rhs).norm() <= tol,
            "lhs = {lhs}, rhs = {rhs}, |diff| = {}",
            (lhs - rhs).norm()
        );
    }

    #[test]
    fn test_get_quant_num_m() {
        assert_eq!(get_quant_num_m(0), vec![0]);
        assert_eq!(get_quant_num_m(2), vec![-2, -1, 0, 1, 2]);
    }

    #[test]
    fn test_get_slice_up_dn_and_mutate() {
        let mut data = vec![1, 2, 3, 4, 5, 6];

        let (up, dn) = get_slice_up_dn(&data);
        assert_eq!(up, &[1, 2, 3]);
        assert_eq!(dn, &[4, 5, 6]);

        let (up_mut, dn_mut) = get_mut_slice_up_dn(&mut data);
        up_mut[0] = 10;
        dn_mut[2] = 60;

        assert_eq!(data, vec![10, 2, 3, 4, 5, 60]);
    }

    #[test]
    fn test_add_scale_and_hadamard_ops() {
        let inp = vec![
            c64::new(1.0, -2.0),
            c64::new(0.5, 0.25),
            c64::new(-1.5, 0.75),
        ];
        let mut out = vec![
            c64::new(0.0, 1.0),
            c64::new(1.0, 0.0),
            c64::new(-2.0, 2.0),
        ];

        add_and_scale(&inp, &mut out, 2.0);
        assert_close_c64(out[0], c64::new(2.0, -3.0), 1.0e-12);
        assert_close_c64(out[1], c64::new(2.0, 0.5), 1.0e-12);
        assert_close_c64(out[2], c64::new(-5.0, 3.5), 1.0e-12);

        add_and_zscale(&inp, &mut out, c64::new(0.0, 1.0));
        assert_close_c64(out[0], c64::new(4.0, -2.0), 1.0e-12);
        assert_close_c64(out[1], c64::new(1.75, 1.0), 1.0e-12);
        assert_close_c64(out[2], c64::new(-5.75, 2.0), 1.0e-12);

        let mut hadamard = vec![ZERO_C64; inp.len()];
        hadamard_product(&inp, &out, &mut hadamard);
        assert_close_c64(hadamard[0], c64::new(0.0, -10.0), 1.0e-12);
        assert_close_c64(hadamard[1], c64::new(0.625, 0.9375), 1.0e-12);
        assert_close_c64(hadamard[2], c64::new(7.125, -7.3125), 1.0e-12);
    }

    #[test]
    fn test_dot_products_match_reference() {
        let u = vec![
            c64::new(1.0, -2.0),
            c64::new(0.5, 3.0),
            c64::new(-1.0, 1.5),
            c64::new(2.0, -0.5),
            c64::new(-0.25, 0.75),
            c64::new(1.25, -1.25),
            c64::new(3.5, 0.5),
            c64::new(-2.0, -2.0),
            c64::new(0.125, 0.25),
        ];
        let v = vec![
            c64::new(-1.0, 1.0),
            c64::new(2.5, -0.5),
            c64::new(0.0, 3.0),
            c64::new(-1.0, -1.0),
            c64::new(1.5, 0.25),
            c64::new(-2.0, 2.0),
            c64::new(0.5, -3.0),
            c64::new(1.0, 1.0),
            c64::new(2.0, 0.0),
        ];
        let metric = vec![1.0, 0.5, 2.0, 1.5, 1.0, 0.75, 2.0, 0.25, 1.25];

        let ref_zdot: c64 = u.iter().zip(v.iter()).map(|(a, b)| a.conj() * b).sum();
        let ref_metric: c64 = u
            .iter()
            .zip(v.iter())
            .zip(metric.iter())
            .map(|((a, b), &m)| a.conj() * b * m)
            .sum();

        assert_close_c64(zdot_product(&u, &v), ref_zdot, 1.0e-12);
        assert_close_c64(dot_product_2(&u, &v), ref_zdot, 1.0e-12);
        assert_close_c64(zdot_product_metric(&u, &v, &metric), ref_metric, 1.0e-12);

        let uf = vec![1.0, -2.0, 3.0, 4.5, -1.5, 0.25, -0.5];
        let vf = vec![0.5, 2.0, -1.0, 2.0, 3.0, -4.0, 1.5];
        let ref_ddot: f64 = uf.iter().zip(vf.iter()).map(|(a, b)| a * b).sum();
        assert_close_f64(ddot_product(&uf, &vf), ref_ddot, 1.0e-12);

        let g = Vector3i32::new(1, -2, 3);
        let r = Vector3f64::new(0.5, 2.0, -1.0);
        assert_close_f64(dot_product_v3i32_v3f64(g, r), -6.5, 1.0e-12);
    }

    #[test]
    fn test_cartesian_spherical_roundtrip() {
        let (x, y, z) = (2.01, -0.0110911, 1.00111);
        let (r, theta, phi) = cartesian_to_spherical(x, y, z);
        let (xr, yr, zr) = spherical_to_cartesian(r, theta, phi);

        assert_close_f64(x, xr, 1.0e-10);
        assert_close_f64(y, yr, 1.0e-10);
        assert_close_f64(z, zr, 1.0e-10);
    }

    #[test]
    fn test_argsort_and_fft_index_roundtrip() {
        let values = vec![3.0, -1.0, 2.0, -1.5];
        assert_eq!(argsort(&values), vec![3, 1, 2, 0]);

        let cases = [(7usize, -3i32, 3i32), (8usize, -3i32, 4i32)];
        for (ntot, left, right) in cases {
            assert_eq!(fft_left_end(ntot), left);
            assert_eq!(fft_right_end(ntot), right);

            let mut linear = Vec::new();
            for i in left..=right {
                let n = fft_i2n(i, ntot);
                linear.push(n);
                assert_eq!(fft_n2i(n, ntot), i);
            }
            linear.sort_unstable();
            assert_eq!(linear, (0..ntot).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_make_matrix_is_hermitian() {
        let n = 4;
        let m = make_matrix(n);
        assert_eq!(m.nrow(), n);
        assert_eq!(m.ncol(), n);

        for i in 0..n {
            assert_close_f64(m[[i, i]].im, 0.0, 1.0e-12);
            for j in 0..n {
                assert_close_c64(m[[i, j]], m[[j, i]].conj(), 1.0e-12);
            }
        }
    }

    #[test]
    fn test_fft_linear_index_map_and_3d_1d_roundtrip() {
        let (n1, n2, n3) = (4, 3, 2);
        let mut v3d = Array3::<c64>::new([n1, n2, n3]);
        for (idx, value) in v3d.as_mut_slice().iter_mut().enumerate() {
            let x = idx as f64;
            *value = c64::new(x, -x);
        }

        let miller = vec![
            Vector3i32::new(0, 0, 0),
            Vector3i32::new(-1, 0, 0),
            Vector3i32::new(1, -1, 0),
            Vector3i32::new(0, 0, 1),
        ];
        let gindex = vec![0usize, 1, 2, 3];
        let linear_index = compute_fft_linear_index_map(&miller, &gindex, n1, n2, n3);

        let mut mapped_direct = vec![c64::new(99.0, -99.0); 6];
        map_3d_to_1d(&miller, &gindex, n1, n2, n3, &v3d, &mut mapped_direct);
        assert_eq!(mapped_direct[4], ZERO_C64);
        assert_eq!(mapped_direct[5], ZERO_C64);

        let mut mapped_linear = vec![ZERO_C64; linear_index.len()];
        map_3d_to_1d_with_linear_index(&linear_index, &v3d, &mut mapped_linear);
        assert_eq!(&mapped_direct[..linear_index.len()], mapped_linear.as_slice());

        let mut restored_direct = Array3::<c64>::new([n1, n2, n3]);
        let mut restored_linear = Array3::<c64>::new([n1, n2, n3]);
        map_1d_to_3d(
            &miller,
            &gindex,
            n1,
            n2,
            n3,
            &mapped_linear,
            &mut restored_direct,
        );
        map_1d_to_3d_with_linear_index(&linear_index, &mapped_linear, &mut restored_linear);
        assert_eq!(restored_direct.as_slice(), restored_linear.as_slice());

        let src = v3d.as_slice();
        let dst = restored_linear.as_slice();
        for idx in 0..dst.len() {
            if linear_index.contains(&idx) {
                assert_eq!(dst[idx], src[idx]);
            } else {
                assert_eq!(dst[idx], ZERO_C64);
            }
        }
    }
}
