use crate::Matrix;

use dwconsts::*;
use std::ops::Mul;

impl Mul<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Matrix<f64> {
        let mut mat = self.clone();

        for v in mat.as_mut_slice() {
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

    pub fn symmetrize(&mut self) {
        for i in 0..self.ncol() {
            for j in i..self.nrow() {
                let a = self[[j, i]];
                let b = self[[i, j]];
                self[[i, j]] = (a + b) / 2.0;
            }
        }

        for i in 0..self.ncol() {
            for j in i..self.nrow() {
                self[[j, i]] = self[[i, j]];
            }
        }
    }

    pub fn inv(&mut self) {
        assert_eq!(self.nrow(), self.ncol(), "Matrix::inv requires a square matrix");

        if let Some(inv) = self.data.clone().try_inverse() {
            self.data = inv;
        } else {
            self.pinv();
        }
    }

    /// https://software.intel.com/content/www/us/en/develop/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl.html
    pub fn pinv(&mut self) {
        assert_eq!(
            self.nrow(),
            self.ncol(),
            "Matrix::pinv requires a square matrix"
        );

        let pinv = self
            .data
            .clone()
            .svd(true, true)
            .pseudo_inverse(EPS30)
            .expect("nalgebra SVD pseudo-inverse failed");

        self.data = pinv;
    }

    /// Save the matrix to a HDF5 file. The shape (an array [nrow, ncol]), and the real and the imaginary parts of the data are saved in their respective datasets.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        // Write nrow, ncol
        let _dataset_nrow = group
            .new_dataset_builder()
            .with_data(&[self.nrow(), self.ncol()])
            .create("shape")
            .unwrap();

        // Write data
        let _dataset_real = group
            .new_dataset_builder()
            .with_data(self.as_slice())
            .create("data")
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

        let data: Vec<f64> = group
            .dataset("data")
            .map_err(|e| format!("failed to open dataset 'data': {}", e))?
            .read()
            .map_err(|e| format!("failed to read dataset 'data': {}", e))?
            .to_vec();

        let expected_len = nrow * ncol;
        if data.len() != expected_len {
            return Err(format!(
                "invalid matrix payload length: expected {}, got {}",
                expected_len,
                data.len()
            ));
        }

        Ok(Self::from_column_slice(nrow, ncol, &data))
    }

    /// Load the matrix from HDF5 and panic on malformed input.
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        Self::try_load_hdf5(group).expect("failed to load Matrix<f64> from HDF5")
    }
}
