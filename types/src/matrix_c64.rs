use crate::Matrix;
use std::ops::AddAssign;
use std::ops::Mul;

use crate::c64;

impl Matrix<c64> {
    pub fn identity(n: usize) -> Matrix<c64> {
        let mut mat = Matrix::<c64>::new(n, n);

        for v in mat.as_mut_slice() {
            *v = c64::new(0.0, 0.0);
        }

        for i in 0..n {
            mat[[i, i]] = c64::new(1.0, 0.0);
        }

        mat
    }

    pub fn inv(&mut self) {
        assert_eq!(self.nrow(), self.ncol(), "Matrix::inv requires a square matrix");

        if let Some(inv) = self.as_dmatrix().clone().try_inverse() {
            self.as_dmatrix_mut().copy_from(&inv);
        } else {
            self.pinv();
        }
    }

    pub fn pinv(&mut self) {
        assert_eq!(
            self.nrow(),
            self.ncol(),
            "Matrix::pinv requires a square matrix"
        );

        let pinv = self
            .as_dmatrix()
            .clone()
            .svd(true, true)
            .pseudo_inverse(1.0e-30)
            .expect("nalgebra SVD pseudo-inverse failed");

        self.as_dmatrix_mut().copy_from(&pinv);
    }

    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        let _dataset_nrow = group
            .new_dataset_builder()
            .with_data(&[self.nrow(), self.ncol()])
            .create("shape")
            .unwrap();

        let real_data: Vec<f64> = self.as_slice().iter().map(|&c| c.re).collect();
        let imag_data: Vec<f64> = self.as_slice().iter().map(|&c| c.im).collect();

        let _dataset_real = group
            .new_dataset_builder()
            .with_data(&real_data)
            .create("real")
            .unwrap();

        let _dataset_imag = group
            .new_dataset_builder()
            .with_data(&imag_data)
            .create("imag")
            .unwrap();
    }

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
            .collect::<Vec<_>>();

        Ok(Self::from_column_slice(nrow, ncol, &data))
    }

    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        Self::try_load_hdf5(group).expect("failed to load Matrix<c64> from HDF5")
    }
}

impl Mul<f64> for Matrix<c64> {
    type Output = Matrix<c64>;

    fn mul(self, rhs: f64) -> Matrix<c64> {
        let mut mat = self.clone();

        for v in mat.as_mut_slice() {
            *v *= rhs;
        }

        mat
    }
}

impl AddAssign<Matrix<f64>> for Matrix<c64> {
    fn add_assign(&mut self, rhs: Matrix<f64>) {
        for (s, d) in rhs.as_slice().iter().zip(self.as_mut_slice().iter_mut()) {
            *d += *s;
        }
    }
}
