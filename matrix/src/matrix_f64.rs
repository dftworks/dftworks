use crate::Matrix;

use dwconsts::*;
use lapack_sys::*;
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
        //self.pinv();

        //return;

        let nn = self.nrow;
        let n = nn as i32;

        let mut ipiv = vec![0i32; nn];
        let lwork = n * n;
        let mut work = vec![0.0f64; lwork as usize];
        let mut info = 0i32;

        unsafe {
            dgetrf_(&n, &n, self.as_ptr(), &n, ipiv.as_mut_ptr(), &mut info);
            dgetri_(
                &n,
                self.as_mut_ptr(),
                &n,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
    }

    /// https://software.intel.com/content/www/us/en/develop/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl.html
    pub fn pinv(&mut self) {
        let n = self.nrow as i32;
        let nrhs = n;

        let mut s = vec![0.0; n as usize];
        let rcond: f64 = EPS30;
        let lwork = 3 * n + 2 * n; //3*min(M,N) + max( 2*min(M,N), max(M,N), NRHS )
        let mut work = vec![0.0f64; lwork as usize];
        let mut info = 0i32;

        let mut b = Matrix::<f64>::identity(n as usize);

        unsafe {
            dgelss_(
                &n,
                &n,
                &nrhs,
                self.as_mut_ptr(),
                &n,
                b.as_mut_ptr(),
                &n,
                s.as_mut_ptr(),
                &rcond,
                &n,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }

        self.data.copy_from_slice(b.as_slice());
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
    pub fn load_hdf5(&mut self, group: &mut hdf5::Group) {
        // Read nrow and ncol
        let shape: Vec<usize> = group.dataset("shape").unwrap().read().unwrap().to_vec();
        self.nrow = *shape.get(0).unwrap();
        self.ncol = *shape.get(1).unwrap();

        // Read data
        self.data = group.dataset("data").unwrap().read().unwrap().to_vec();
    }
}
