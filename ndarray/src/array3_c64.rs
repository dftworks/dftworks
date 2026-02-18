use crate::Array3;

use itertools::multizip;
use rayon::prelude::*;

use types::*;

use std::convert::TryInto;

const PARALLEL_MIN_LEN: usize = 8192;

#[inline]
fn use_parallel_for_len(len: usize) -> bool {
    len >= PARALLEL_MIN_LEN && rayon::current_num_threads() > 1
}

impl Array3<c64> {
    pub fn scale(&mut self, f: f64) {
        self.data.iter_mut().for_each(|x| *x *= f);
    }

    pub fn norm2(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
    }

    pub fn scaled_assign(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = *s * factor;
        }
    }

    pub fn diff_norm_sum(&self, rhs: &Array3<c64>) -> f64 {
        let prhs = rhs.as_slice();
        let plhs = self.as_slice();

        assert_eq!(prhs.len(), plhs.len());

        let mut sum = 0.0;

        for (&l, &r) in multizip((plhs.iter(), prhs.iter())) {
            sum += (l - r).norm_sqr();
        }

        sum.sqrt()
    }

    pub fn sqr_add(&mut self, rhs: &Array3<c64>) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        if use_parallel_for_len(pdst.len()) {
            pdst.par_iter_mut().zip(psrc.par_iter()).for_each(|(d, s)| {
                *d += s.norm_sqr();
            });
        } else {
            for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
                *d += s.norm_sqr();
            }
        }
    }

    pub fn scaled_sqr_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        if use_parallel_for_len(pdst.len()) {
            pdst.par_iter_mut().zip(psrc.par_iter()).for_each(|(d, s)| {
                *d += s.norm_sqr() * factor;
            });
        } else {
            for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
                *d += s.norm_sqr() * factor;
            }
        }
    }

    pub fn scaled_add(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        if use_parallel_for_len(pdst.len()) {
            pdst.par_iter_mut().zip(psrc.par_iter()).for_each(|(d, s)| {
                *d += (*s) * factor;
            });
        } else {
            for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
                *d += (*s) * factor;
            }
        }
    }

    pub fn mix(&mut self, rhs: &Array3<c64>, factor: f64) {
        let psrc = rhs.as_slice();
        let pdst = self.as_mut_slice();

        assert_eq!(psrc.len(), pdst.len());

        for (s, d) in multizip((psrc.iter(), pdst.iter_mut())) {
            *d = (*d) * factor + (*s) * (1.0 - factor);
        }
    }

    pub fn add(&mut self, addition: f64) {
        let pdst = self.as_mut_slice();

        for d in pdst.iter_mut() {
            *d = *d + addition;
        }
    }

    /// Save the array to a HDF5 file. The shape, and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        // Write shape
        let _dataset_shape = group
            .new_dataset_builder()
            .with_data(&self.shape)
            .create("shape")
            .unwrap();

        let real_data: Vec<f64> = self.data.iter().map(|&c| c.re).collect();
        let imag_data: Vec<f64> = self.data.iter().map(|&c| c.im).collect();

        // Write real part
        let _dataset_real = group
            .new_dataset_builder()
            .with_data(&real_data)
            .create("real")
            .unwrap();

        // Write imaginary part
        let _dataset_imag = group
            .new_dataset_builder()
            .with_data(&imag_data)
            .create("imag")
            .unwrap();
    }

    /// Load the array from a HDF5 group as saved by the save_hdf5 function.
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        // Read shape
        let shape: Vec<usize> = group.dataset("shape").unwrap().read().unwrap().to_vec();
        let shape: [usize; 3] = shape.try_into().unwrap();

        // Read data
        let real_data: Vec<f64> = group.dataset("real").unwrap().read().unwrap().to_vec();
        let imag_data: Vec<f64> = group.dataset("imag").unwrap().read().unwrap().to_vec();
        let data: Vec<c64> = real_data
            .iter()
            .zip(imag_data)
            .map(|(&r, i)| c64::new(r, i))
            .collect();

        Self::from_vec(shape, data)
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

    fn assert_close_array(lhs: &Array3<c64>, rhs: &Array3<c64>, tol: f64) {
        assert_eq!(lhs.shape(), rhs.shape());
        for (&l, &r) in lhs.as_slice().iter().zip(rhs.as_slice().iter()) {
            assert_close_c64(l, r, tol);
        }
    }

    #[test]
    fn test_c64_scaled_assign_scale_and_norm() {
        let shape = [2, 2, 1];
        let rhs = Array3::from_vec(
            shape,
            vec![
                c64::new(1.0, -1.0),
                c64::new(2.0, 0.5),
                c64::new(-3.0, 2.0),
                c64::new(0.25, -0.75),
            ],
        );

        let mut lhs = Array3::<c64>::new(shape);
        lhs.scaled_assign(&rhs, 0.5);

        let expected_half = Array3::from_vec(
            shape,
            rhs.as_slice().iter().map(|&x| x * 0.5).collect::<Vec<_>>(),
        );
        assert_close_array(&lhs, &expected_half, 1.0e-12);

        lhs.scale(2.0);
        assert_close_array(&lhs, &rhs, 1.0e-12);

        let expected_norm = rhs.as_slice().iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert_close_f64(lhs.norm2(), expected_norm, 1.0e-12);
    }

    #[test]
    fn test_c64_diff_norm_sum() {
        let shape = [2, 2, 1];
        let lhs = Array3::from_vec(
            shape,
            vec![
                c64::new(1.0, 0.0),
                c64::new(2.0, 1.0),
                c64::new(-1.0, 2.0),
                c64::new(0.5, -0.5),
            ],
        );
        let mut rhs = lhs.clone();
        rhs.as_mut_slice()[0] += c64::new(1.0, -2.0);

        assert_close_f64(lhs.diff_norm_sum(&rhs), 5.0f64.sqrt(), 1.0e-12);
    }

    #[test]
    fn test_c64_sqr_add_and_scaled_sqr_add() {
        let shape = [2, 2, 1];
        let rhs = Array3::from_vec(
            shape,
            vec![
                c64::new(1.0, 2.0),
                c64::new(-0.5, 1.5),
                c64::new(3.0, -1.0),
                c64::new(0.0, -2.0),
            ],
        );
        let base = Array3::from_vec(
            shape,
            vec![
                c64::new(0.0, 1.0),
                c64::new(1.0, -1.0),
                c64::new(-2.0, 0.5),
                c64::new(0.25, -0.25),
            ],
        );

        let mut sqr_added = base.clone();
        sqr_added.sqr_add(&rhs);
        let expected_sqr = Array3::from_vec(
            shape,
            base.as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(&b, &r)| b + c64::new(r.norm_sqr(), 0.0))
                .collect::<Vec<_>>(),
        );
        assert_close_array(&sqr_added, &expected_sqr, 1.0e-12);

        let mut scaled_sqr_added = base.clone();
        scaled_sqr_added.scaled_sqr_add(&rhs, 0.25);
        let expected_scaled_sqr = Array3::from_vec(
            shape,
            base.as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(&b, &r)| b + c64::new(r.norm_sqr() * 0.25, 0.0))
                .collect::<Vec<_>>(),
        );
        assert_close_array(&scaled_sqr_added, &expected_scaled_sqr, 1.0e-12);
    }

    #[test]
    fn test_c64_scaled_add_mix_and_add() {
        let shape = [2, 2, 1];
        let rhs = Array3::from_vec(
            shape,
            vec![
                c64::new(1.0, -1.0),
                c64::new(2.0, 0.5),
                c64::new(-3.0, 2.0),
                c64::new(0.25, -0.75),
            ],
        );
        let mut dst = Array3::from_vec(
            shape,
            vec![
                c64::new(-1.0, 1.0),
                c64::new(0.5, -2.0),
                c64::new(4.0, 0.0),
                c64::new(-0.5, 0.25),
            ],
        );
        let original = dst.clone();

        dst.scaled_add(&rhs, -0.5);
        let expected_scaled_add = Array3::from_vec(
            shape,
            original
                .as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(&d, &r)| d + r * -0.5)
                .collect::<Vec<_>>(),
        );
        assert_close_array(&dst, &expected_scaled_add, 1.0e-12);

        let mut mixed = original.clone();
        mixed.mix(&rhs, 0.2);
        let expected_mix = Array3::from_vec(
            shape,
            original
                .as_slice()
                .iter()
                .zip(rhs.as_slice().iter())
                .map(|(&d, &r)| d * 0.2 + r * 0.8)
                .collect::<Vec<_>>(),
        );
        assert_close_array(&mixed, &expected_mix, 1.0e-12);

        mixed.add(0.5);
        let expected_plus_real = Array3::from_vec(
            shape,
            expected_mix
                .as_slice()
                .iter()
                .map(|&z| z + c64::new(0.5, 0.0))
                .collect::<Vec<_>>(),
        );
        assert_close_array(&mixed, &expected_plus_real, 1.0e-12);
    }
}
