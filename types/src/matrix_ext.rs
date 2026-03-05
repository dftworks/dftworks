use crate::{c64, Matrix};
use nalgebra::DMatrix;
use num_traits::identities::Zero;

pub trait MatrixExt<T>
where
    T: nalgebra::Scalar + Zero + Default + Copy + Clone,
{
    fn new(nrow: usize, ncol: usize) -> Self
    where
        Self: Sized;
    fn nrow(&self) -> usize;
    fn ncol(&self) -> usize;
    fn assign(&mut self, rhs: &Self);
    fn as_dmatrix(&self) -> &DMatrix<T>;
    fn as_dmatrix_mut(&mut self) -> &mut DMatrix<T>;
    fn set_zeros(&mut self);
    fn set_col(&mut self, icol: usize, v: &[T]);
    fn get_col(&self, icol: usize) -> &[T];
    fn get_mut_col(&mut self, icol: usize) -> &mut [T];
}

impl<T> MatrixExt<T> for Matrix<T>
where
    T: nalgebra::Scalar + Zero + Default + Copy + Clone,
{
    fn new(nrow: usize, ncol: usize) -> Self {
        DMatrix::<T>::from_element(nrow, ncol, T::default())
    }

    fn nrow(&self) -> usize {
        self.nrows()
    }

    fn ncol(&self) -> usize {
        self.ncols()
    }

    fn assign(&mut self, rhs: &Self) {
        self.copy_from(rhs);
    }

    fn as_dmatrix(&self) -> &DMatrix<T> {
        self
    }

    fn as_dmatrix_mut(&mut self) -> &mut DMatrix<T> {
        self
    }

    fn set_zeros(&mut self) {
        self.iter_mut().for_each(|x| *x = T::zero());
    }

    fn set_col(&mut self, icol: usize, v: &[T]) {
        let n1 = icol * self.nrows();
        let n2 = n1 + self.nrows();
        self.as_mut_slice()[n1..n2].copy_from_slice(v);
    }

    fn get_col(&self, icol: usize) -> &[T] {
        let n1 = icol * self.nrows();
        let n2 = n1 + self.nrows();
        &self.as_slice()[n1..n2]
    }

    fn get_mut_col(&mut self, icol: usize) -> &mut [T] {
        let n1 = icol * self.nrows();
        let n2 = n1 + self.nrows();
        &mut self.as_mut_slice()[n1..n2]
    }
}

pub trait MatrixF64Ext {
    fn identity(n: usize) -> Self
    where
        Self: Sized;
    fn symmetrize(&mut self);
    fn inv(&mut self);
    fn pinv(&mut self);
}

impl MatrixF64Ext for Matrix<f64> {
    fn identity(n: usize) -> Self {
        DMatrix::<f64>::identity(n, n)
    }

    fn symmetrize(&mut self) {
        for i in 0..self.ncols() {
            for j in i..self.nrows() {
                let a = self[(j, i)];
                let b = self[(i, j)];
                self[(i, j)] = 0.5 * (a + b);
            }
        }
        for i in 0..self.ncols() {
            for j in i..self.nrows() {
                self[(j, i)] = self[(i, j)];
            }
        }
    }

    fn inv(&mut self) {
        assert_eq!(self.nrows(), self.ncols(), "Matrix::inv requires a square matrix");
        if let Some(inv) = self.clone().try_inverse() {
            self.copy_from(&inv);
        } else {
            self.pinv();
        }
    }

    fn pinv(&mut self) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "Matrix::pinv requires a square matrix"
        );
        let pinv = self
            .clone()
            .svd(true, true)
            .pseudo_inverse(1.0e-30)
            .expect("nalgebra SVD pseudo-inverse failed");
        self.copy_from(&pinv);
    }
}

pub trait MatrixC64Ext {
    fn identity(n: usize) -> Self
    where
        Self: Sized;
    fn inv(&mut self);
    fn pinv(&mut self);
}

impl MatrixC64Ext for Matrix<c64> {
    fn identity(n: usize) -> Self {
        DMatrix::<c64>::identity(n, n)
    }

    fn inv(&mut self) {
        assert_eq!(self.nrows(), self.ncols(), "Matrix::inv requires a square matrix");
        if let Some(inv) = self.clone().try_inverse() {
            self.copy_from(&inv);
        } else {
            self.pinv();
        }
    }

    fn pinv(&mut self) {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "Matrix::pinv requires a square matrix"
        );
        let pinv = self
            .clone()
            .svd(true, true)
            .pseudo_inverse(1.0e-30)
            .expect("nalgebra SVD pseudo-inverse failed");
        self.copy_from(&pinv);
    }
}
