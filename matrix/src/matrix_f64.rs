use crate::Matrix;

use dwconsts::*;
use nalgebra::DMatrix;
use std::ops::Mul;

impl Mul<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Matrix<f64> {
        let mut mat = self.clone();

        for v in mat.data.iter_mut() {
            *v *= rhs;
        }

        mat
    }
}

impl Matrix<f64> {
    pub fn identity(n: usize) -> Matrix<f64> {
        let mut mat = Matrix::<f64>::new(n, n);

        for v in mat.as_mut_slice() {
            *v = 0.0;
        }

        for i in 0..n {
            mat[[i, i]] = 1.0;
        }

        mat
    }

    pub fn unit(n: usize) -> Matrix<f64> {
        let mut mat = Matrix::<f64>::new(n, n);

        for v in mat.as_mut_slice() {
            *v = 0.0;
        }

        for i in 0..n {
            mat[[i, i]] = 1.0;
        }

        mat
    }

    pub fn action(&self, vin: &[f64], vout: &mut [f64]) {
        vout.iter_mut().for_each(|x| *x = 0.0);

        for i in 0..self.ncol {
            for j in 0..self.nrow {
                vout[j] += self[[j, i]] * vin[i];
            }
        }
    }

    pub fn symmetrize(&mut self) {
        for i in 0..self.ncol {
            for j in i..self.nrow {
                let a = self[[j, i]].clone();
                let b = self[[i, j]].clone();
                self[[i, j]] = (a + b) / 2.0;
            }
        }

        for i in 0..self.ncol {
            for j in i..self.nrow {
                self[[j, i]] = self[[i, j]].clone();
            }
        }
    }

    pub fn inv(&mut self) {
        assert_eq!(self.nrow, self.ncol, "Matrix::inv requires a square matrix");

        let mat = DMatrix::<f64>::from_column_slice(self.nrow, self.ncol, self.as_slice());

        if let Some(inv) = mat.try_inverse() {
            self.data.copy_from_slice(inv.as_slice());
        } else {
            self.pinv();
        }
    }

    /// https://software.intel.com/content/www/us/en/develop/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl.html
    pub fn pinv(&mut self) {
        assert_eq!(
            self.nrow, self.ncol,
            "Matrix::pinv requires a square matrix"
        );

        let mat = DMatrix::<f64>::from_column_slice(self.nrow, self.ncol, self.as_slice());
        let pinv = mat
            .svd(true, true)
            .pseudo_inverse(EPS30)
            .expect("nalgebra SVD pseudo-inverse failed");

        self.data.copy_from_slice(pinv.as_slice());
    }

    pub fn sum(&self) -> f64 {
        return self.data.iter().sum();
    }

    /// Save the matrix to a HDF5 file. The shape (an array [nrow, ncol]), and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        // Write nrow, ncol
        let _dataset_nrow = group
            .new_dataset_builder()
            .with_data(&[self.nrow, self.ncol])
            .create("shape")
            .unwrap();

        // Write data
        let _dataset_real = group
            .new_dataset_builder()
            .with_data(&self.data)
            .create("data")
            .unwrap();
    }

    /// Load the array from a HDF5 group as saved by the save_hdf5 function.
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        let mut mat = Self::default();

        // Read nrow and ncol
        let shape: Vec<usize> = group.dataset("shape").unwrap().read().unwrap().to_vec();
        mat.nrow = *shape.get(0).unwrap();
        mat.ncol = *shape.get(1).unwrap();

        // Read data
        mat.data = group.dataset("data").unwrap().read().unwrap().to_vec();

        mat
    }
}
