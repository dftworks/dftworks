// column-major memory layout
// [i,j] : i + j * nrow
//   0,0 0,1 0,2        0 2 4
//   1,0 1,1 1,2        1 3 5

#[path = "matrix_c64.rs"]
mod matrix_c64;
pub use matrix_c64::*;

#[path = "matrix_f64.rs"]
mod matrix_f64;
pub use matrix_f64::*;

use crate::c64;
use itertools::multizip;
use nalgebra::DMatrix;
use std::ops::*;
use std::{
    fmt,
    fmt::{Debug, Display},
};

pub trait Dot<RHS = Self> {
    type Output;

    fn dot(&self, other: &RHS) -> Self::Output;
}

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: DMatrix<T>,
}

impl<T: nalgebra::Scalar + Default + Clone> Default for Matrix<T> {
    fn default() -> Self {
        Matrix {
            data: DMatrix::<T>::from_element(0, 0, T::default()),
        }
    }
}

impl Dot<Matrix<f64>> for Matrix<f64> {
    type Output = Self;

    fn dot(&self, rhs: &Matrix<f64>) -> Self::Output {
        assert!(self.ncol() == rhs.nrow());
        let prod = &self.data * &rhs.data;
        Matrix { data: prod }
    }
}

impl Dot<Matrix<c64>> for Matrix<c64> {
    type Output = Self;

    fn dot(&self, rhs: &Matrix<c64>) -> Self::Output {
        assert!(self.ncol() == rhs.nrow());
        let prod = &self.data * &rhs.data;
        Matrix { data: prod }
    }
}

impl<
        T: nalgebra::Scalar
            + num_traits::identities::Zero
            + Default
            + Copy
            + Clone
            + Mul
            + std::ops::AddAssign
            + std::ops::Mul<Output = T>,
    > Dot<Vec<T>> for Matrix<T>
{
    type Output = Vec<T>;

    fn dot(&self, rhs: &Vec<T>) -> Self::Output {
        assert!(self.ncol() == rhs.len());

        let mut v = vec![T::default(); self.nrow()];

        for i in 0..self.ncol() {
            let fact = rhs[i];
            let col = self.get_col(i);

            for j in 0..self.nrow() {
                v[j] += fact * col[j];
            }
        }

        v
    }
}

impl<T: nalgebra::Scalar + num_traits::identities::Zero + Default + Copy + Clone + AddAssign>
    AddAssign for Matrix<T>
{
    fn add_assign(&mut self, rhs: Matrix<T>) {
        for (s, d) in multizip((rhs.as_slice().iter(), self.as_mut_slice().iter_mut())) {
            *d += *s;
        }
    }
}

impl<
        T: nalgebra::Scalar
            + num_traits::identities::Zero
            + Default
            + Copy
            + Clone
            + Add
            + std::ops::Add<Output = T>,
    > Add for Matrix<T>
{
    type Output = Self;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.nrow() == rhs.nrow() && self.ncol() == rhs.ncol());

        let mut m = Matrix::<T>::new(self.nrow(), self.ncol());

        for (d, s1, s2) in multizip((
            m.as_mut_slice().iter_mut(),
            self.as_slice().iter(),
            rhs.as_slice().iter(),
        )) {
            *d = *s1 + *s2;
        }

        m
    }
}

impl<
        T: nalgebra::Scalar
            + num_traits::identities::Zero
            + Default
            + Copy
            + Clone
            + Sub
            + std::ops::Sub<Output = T>,
    > Sub for Matrix<T>
{
    type Output = Self;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.nrow() == rhs.nrow() && self.ncol() == rhs.ncol());

        let mut m = Matrix::<T>::new(self.nrow(), self.ncol());

        for (d, s1, s2) in multizip((
            m.as_mut_slice().iter_mut(),
            self.as_slice().iter(),
            rhs.as_slice().iter(),
        )) {
            *d = *s1 - *s2;
        }

        m
    }
}

impl<T: nalgebra::Scalar + num_traits::identities::Zero + Default + Copy + Clone> Matrix<T> {
    pub fn new(nrow: usize, ncol: usize) -> Matrix<T> {
        Matrix {
            data: DMatrix::<T>::from_element(nrow, ncol, T::default()),
        }
    }

    pub fn assign(&mut self, rhs: &Matrix<T>) {
        let src = rhs.as_slice();
        let dst = self.as_mut_slice();
        dst[..src.len()].copy_from_slice(src);
    }

    pub fn nrow(&self) -> usize {
        self.data.nrows()
    }

    pub fn ncol(&self) -> usize {
        self.data.ncols()
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn as_dmatrix(&self) -> &DMatrix<T> {
        &self.data
    }

    pub fn as_dmatrix_mut(&mut self) -> &mut DMatrix<T> {
        &mut self.data
    }

    pub fn from_dmatrix(data: DMatrix<T>) -> Matrix<T> {
        Matrix { data }
    }

    pub fn into_dmatrix(self) -> DMatrix<T> {
        self.data
    }

    pub fn set_zeros(&mut self) {
        self.data.iter_mut().for_each(|x| *x = T::zero());
    }

    pub fn from_row_slice(nrow: usize, ncol: usize, s: &[T]) -> Matrix<T> {
        Matrix {
            data: DMatrix::<T>::from_row_slice(nrow, ncol, s),
        }
    }

    pub fn from_column_slice(nrow: usize, ncol: usize, s: &[T]) -> Matrix<T> {
        Matrix {
            data: DMatrix::<T>::from_column_slice(nrow, ncol, s),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    pub fn set_col(&mut self, icol: usize, v: &[T]) {
        let n1 = icol * self.nrow();
        let n2 = n1 + self.nrow();
        self.as_mut_slice()[n1..n2].copy_from_slice(v);
    }

    pub fn get_col(&self, icol: usize) -> &[T] {
        let n1 = icol * self.nrow();
        let n2 = n1 + self.nrow();
        &self.as_slice()[n1..n2]
    }

    pub fn get_col_to(&self, v: &mut [T], icol: usize) {
        let n1 = icol * self.nrow();
        let n2 = n1 + self.nrow();
        let v_src = &self.as_slice()[n1..n2];
        v[..v_src.len()].copy_from_slice(v_src);
    }

    pub fn get_mut_col(&mut self, icol: usize) -> &mut [T] {
        let n1 = icol * self.nrow();
        let n2 = n1 + self.nrow();
        &mut self.as_mut_slice()[n1..n2]
    }

    pub fn transpose(&self) -> Matrix<T> {
        Matrix {
            data: self.data.transpose(),
        }
    }
}

impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[(idx[0], idx[1])]
    }
}

impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        &mut self.data[(idx[0], idx[1])]
    }
}

impl<
        T: Debug
            + Display
            + nalgebra::Scalar
            + num_traits::identities::Zero
            + Default
            + Copy
            + Clone,
    > fmt::Display for Matrix<T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.nrow() {
            write!(f, " | ")?;
            for j in 0..self.ncol() {
                write!(f, "{:+8.3} ", self[[i, j]])?;
            }
            writeln!(f, "|")?;
        }
        write!(f, "")
    }
}
