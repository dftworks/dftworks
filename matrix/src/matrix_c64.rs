use crate::{Dot, Matrix};
use dwconsts::*;

use itertools::multizip;
use nalgebra::DMatrix;
use num_traits::Zero;
use std::ops::AddAssign;
use std::ops::Mul;
use types::c64;

impl Matrix<c64> {
    pub fn identity(n: usize) -> Matrix<c64> {
        let mut mat = Matrix::<c64>::new(n, n);

        for v in mat.as_mut_slice() {
            *v = c64::zero();
        }

        for i in 0..n {
            mat[[i, i]] = ONE_C64;
        }

        mat
    }

    pub fn adjoint(&self) -> Matrix<c64> {
        let mut data = Vec::with_capacity(self.nrow * self.ncol);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                data.push(self[[i, j]].clone().conj())
            }
        }
        Matrix {
            nrow: self.ncol,
            ncol: self.nrow,
            data,
        }
    }

    pub fn sum(&self) -> c64 {
        return self.data.iter().sum();
    }

    pub fn action(&self, vin: &[c64], vout: &mut [c64]) {
        vout.iter_mut().for_each(|x| *x = c64::zero());

        for i in 0..self.ncol {
            for j in 0..self.nrow {
                vout[j] += self[[j, i]] * vin[i];
            }
        }
    }

    // pub fn dot(&self, v: &[c64]) -> Vec<c64> {
    //     let mut vout = vec![ZERO_C64; self.nrow];

    //     for i in 0..self.ncol {
    //         for j in 0..self.nrow {
    //             vout[j] += self[[j, i]] * v[i];
    //         }
    //     }

    //     vout
    // }

    pub fn inv(&mut self) {
        assert_eq!(self.nrow, self.ncol, "Matrix::inv requires a square matrix");

        let mat = DMatrix::<c64>::from_column_slice(self.nrow, self.ncol, self.as_slice());

        if let Some(inv) = mat.try_inverse() {
            self.data.copy_from_slice(inv.as_slice());
        } else {
            self.pinv();
        }
    }

    pub fn pinv(&mut self) {
        assert_eq!(
            self.nrow, self.ncol,
            "Matrix::pinv requires a square matrix"
        );

        let mat = DMatrix::<c64>::from_column_slice(self.nrow, self.ncol, self.as_slice());
        let pinv = mat
            .svd(true, true)
            .pseudo_inverse(EPS30)
            .expect("nalgebra SVD pseudo-inverse failed");

        self.data.copy_from_slice(pinv.as_slice());
    }

    pub fn mat_mul(&self, rhs: &Matrix<c64>) -> Matrix<c64> {
        self.dot(rhs)
    }

    /// Save the matrix to a HDF5 file. The shape (an array [nrow, ncol]), and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        // Write nrow, ncol
        let _dataset_nrow = group
            .new_dataset_builder()
            .with_data(&[self.nrow, self.ncol])
            .create("shape")
            .unwrap();

        let real_data: Vec<f64> = self.data.iter().map(|&c| c.re.into()).collect();
        let imag_data: Vec<f64> = self.data.iter().map(|&c| c.im.into()).collect();

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

    /// Fallible HDF5 loader used by restart/checkpoint paths.
    pub fn try_load_hdf5(group: &hdf5::Group) -> Result<Self, String> {
        let shape_ds = group
            .dataset("shape")
            .map_err(|e| format!("failed to open dataset 'shape': {}", e))?;
        let shape: Vec<usize> = shape_ds
            .read()
            .map_err(|e| format!("failed to read dataset 'shape': {}", e))?
            .to_vec();
        if shape.len() != 2 {
            return Err(format!(
                "invalid matrix shape length: expected 2, got {}",
                shape.len()
            ));
        }
        let nrow = shape[0];
        let ncol = shape[1];

        let real_data: Vec<f64> = group
            .dataset("real")
            .map_err(|e| format!("failed to open dataset 'real': {}", e))?
            .read()
            .map_err(|e| format!("failed to read dataset 'real': {}", e))?
            .to_vec();
        let imag_data: Vec<f64> = group
            .dataset("imag")
            .map_err(|e| format!("failed to open dataset 'imag': {}", e))?
            .read()
            .map_err(|e| format!("failed to read dataset 'imag': {}", e))?
            .to_vec();

        if real_data.len() != imag_data.len() {
            return Err(format!(
                "invalid matrix payload: real_len={} imag_len={}",
                real_data.len(),
                imag_data.len()
            ));
        }

        let expected_len = nrow * ncol;
        if real_data.len() != expected_len {
            return Err(format!(
                "invalid matrix payload length: expected {}, got {}",
                expected_len,
                real_data.len()
            ));
        }

        let data = real_data
            .iter()
            .zip(imag_data)
            .map(|(&r, i)| c64::new(r, i))
            .collect();

        Ok(Self { nrow, ncol, data })
    }

    /// Load the matrix from HDF5 and panic on malformed input.
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        Self::try_load_hdf5(group).expect("failed to load Matrix<c64> from HDF5")
    }
}

impl Mul<f64> for Matrix<c64> {
    type Output = Matrix<c64>;

    fn mul(self, rhs: f64) -> Matrix<c64> {
        let mut mat = self.clone();

        for v in mat.data.iter_mut() {
            *v *= rhs;
        }

        mat
    }
}

impl AddAssign<Matrix<f64>> for Matrix<c64> {
    fn add_assign(&mut self, rhs: Matrix<f64>) {
        for (s, d) in multizip((rhs.as_slice().iter(), self.as_mut_slice().iter_mut())) {
            *d += *s;
        }
    }
}
