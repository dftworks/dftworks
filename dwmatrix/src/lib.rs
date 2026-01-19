#![allow(warnings)]
// column-major memory layout
// [i,j] : i + j * nrow
//   0,0 0,1 0,2        0 2 4
//   1,0 1,1 1,2        1 3 5

mod dwmatrix_c64;
pub use dwmatrix_c64::*;

mod dwmatrix_f64;
pub use dwmatrix_f64::*;

//////////////////////////////////////////

use dwconsts::*;
use itertools::multizip;
use lapack_sys::*;
use std::ops::*;
use std::{
    fmt,
    fmt::{Debug, Display},
};
use types::c64;

pub trait Dot<RHS = Self> {
    type Output;

    fn dot(&self, other: &RHS) -> Self::Output;
}

#[derive(Debug, Clone, Default)]
pub struct Matrix<T> {
    nrow: usize,
    ncol: usize,
    data: Vec<T>,
}

impl<
        T: num_traits::identities::Zero
            + Default
            + Copy
            + Clone
            + Mul
            + std::ops::AddAssign
            + std::ops::Mul<Output = T>,
    > Dot<Matrix<T>> for Matrix<T>
{
    type Output = Self;

    fn dot(&self, rhs: &Matrix<T>) -> Self::Output {
        assert!(self.ncol() == rhs.nrow());

        let nr_lhs = self.nrow();
        let nc_lhs = self.ncol();

        let nc_rhs = rhs.ncol();

        let mut mdot = Matrix::<T>::new(nr_lhs, nc_rhs);

        for i in 0..nr_lhs {
            for j in 0..nc_rhs {
                for k in 0..nc_lhs {
                    mdot[[i, j]] += self[[i, k]] * rhs[[k, j]];
                }
            }
        }

        mdot
    }
}

impl<
        T: num_traits::identities::Zero
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

impl<T: num_traits::identities::Zero + Default + Copy + Clone + AddAssign> AddAssign for Matrix<T> {
    fn add_assign(&mut self, rhs: Matrix<T>) {
        for (s, d) in multizip((rhs.as_slice_memory_order().unwrap().iter(), self.as_slice_memory_order_mut().unwrap().iter_mut())) {
            *d += *s;
        }
    }
}

impl<
        T: num_traits::identities::Zero + Default + Copy + Clone + Add + std::ops::Add<Output = T>,
    > Add for Matrix<T>
{
    type Output = Self;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.nrow() == rhs.nrow() && self.ncol() == rhs.ncol());

        let mut m = Matrix::<T>::new(self.nrow(), self.ncol());

        for (d, s1, s2) in multizip((
            m.as_slice_memory_order_mut().unwrap().iter_mut(),
            self.as_slice_memory_order().unwrap().iter(),
            rhs.as_slice_memory_order().unwrap().iter(),
        )) {
            *d = *s1 + *s2;
        }

        m
    }
}

impl<
        T: num_traits::identities::Zero + Default + Copy + Clone + Sub + std::ops::Sub<Output = T>,
    > Sub for Matrix<T>
{
    type Output = Self;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.nrow() == rhs.nrow() && self.ncol() == rhs.ncol());

        let mut m = Matrix::<T>::new(self.nrow(), self.ncol());

        for (d, s1, s2) in multizip((
            m.as_slice_memory_order_mut().unwrap().iter_mut(),
            self.as_slice_memory_order().unwrap().iter(),
            rhs.as_slice_memory_order().unwrap().iter(),
        )) {
            *d = *s1 - *s2;
        }

        m
    }
}

impl<T: num_traits::identities::Zero + Default + Copy + Clone> Matrix<T> {
    pub fn new(nrow: usize, ncol: usize) -> Matrix<T> {
        Matrix {
            nrow,
            ncol,
            data: vec![T::default(); nrow * ncol],
        }
    }

    pub fn assign(&mut self, rhs: &Matrix<T>) {
        let src = rhs.as_slice_memory_order().unwrap();
        let dst = self.as_slice_memory_order_mut().unwrap();

        dst[..src.len()].copy_from_slice(src);
    }

    pub fn nrow(&self) -> usize {
        self.nrow
    }

    pub fn ncol(&self) -> usize {
        self.ncol
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn set_zeros(&mut self) {
        self.data.iter_mut().for_each(|x| *x = T::zero());
    }

    pub fn from_row_slice(nrow: usize, ncol: usize, s: &[T]) -> Matrix<T> {
        let mut data: Vec<T> = vec![T::default(); nrow * ncol];
        let mut n = 0;
        for i in 0..nrow {
            for j in 0..ncol {
                data[i + j * nrow] = s[n];
                n += 1;
            }
        }
        Matrix { nrow, ncol, data }
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    pub fn set_col(&mut self, icol: usize, v: &[T]) {
        let n1 = icol * self.nrow;
        let n2 = n1 + self.nrow;

        self.data[n1..n2].copy_from_slice(v);

        // for i in 0..self.nrow {
        //     self.data[icol * self.nrow + i] = v[i].clone();
        // }
    }

    pub fn get_col(&self, icol: usize) -> &[T] {
        let n1 = icol * self.nrow;
        let n2 = n1 + self.nrow;

        &self.data[n1..n2]
    }

    pub fn get_col_to(&self, v: &mut [T], icol: usize) {
        let n1 = icol * self.nrow;
        let n2 = n1 + self.nrow;

        let v_src = &self.data[n1..n2];

        v[..v_src.len()].copy_from_slice(v_src);
    }

    pub fn get_mut_col(&mut self, icol: usize) -> &mut [T] {
        let n1 = icol * self.nrow;
        let n2 = n1 + self.nrow;

        &mut self.data[n1..n2]
    }

    pub fn transpose(&self) -> Matrix<T> {
        let mut data = Vec::with_capacity(self.nrow * self.ncol);
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                data.push(self[[i, j]].clone())
            }
        }
        Matrix {
            nrow: self.ncol,
            ncol: self.nrow,
            data,
        }
    }
}

impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] + idx[1] * self.nrow]
    }
}

impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        &mut self.data[idx[0] + idx[1] * self.nrow]
    }
}

impl<T: Debug + Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.nrow {
            write!(f, " | ")?;
            for j in 0..self.ncol {
                write!(f, "{:+8.3} ", self[[i, j]])?;
            }
            writeln!(f, "|")?;
        }
        write!(f, "")
    }
}

// TODO: Implement save_hdf5 and load_hdf5 here if the support of complex types in the hdf5 crate is released.
//       See: https://github.com/aldanor/hdf5-rust/pull/210 and https://github.com/aldanor/hdf5-rust/issues/262

#[test]
fn test_matrix() {
    let mut m: Matrix<f64> = Matrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    //    ma.set_col(0, &vec![1.0; n]);
    //    ma.set_col(1, &vec![2.0; n]);
    //m.output();
    println!("{}", m);
    m[[1, 1]] = 8.0;
    println!("{}", m);
    println!("{:?}", m.as_slice_memory_order().unwrap());
    let mtrans = m.transpose();
    println!("{}", mtrans);
    println!("{:?}", mtrans.as_slice_memory_order().unwrap());

    let vin = vec![1.0, 2.0, 3.0];
    let mut vout = vec![0.0; 2];

    m.action(vin.as_slice_memory_order().unwrap(), vout.as_slice_memory_order_mut().unwrap());
    println!("m = \n{}", m);
    println!("vin  = {:?}", vin);
    println!("vout = {:?}", vout);

    let cm = Matrix::<c64>::from_row_slice(
        2,
        2,
        &[
            c64 { re: 1.0, im: 0.1 },
            c64 { re: 2.0, im: 0.01 },
            c64 { re: 3.0, im: 0.0 },
            c64 { re: 4.0, im: 0.001 },
        ],
    );

    println!("{}", cm);
    println!("{:?}", cm.adjoint());

    let mut m: Matrix<f64> = Matrix::<f64>::from_row_slice(2, 2, &[1E-2, 0.0, 0.0, 6.0]);
    let m2 = m.clone();
    println!("{}", m);
    m.inv();
    println!("{}", m);
    println!("m * minv = \n{}", m.dot(&m2));

    let mut m: Matrix<c64> = Matrix::<c64>::from_row_slice(
        2,
        2,
        &[
            c64 { re: 0.0, im: 0.1 },
            c64 { re: 0.01, im: 0.01 },
            c64 { re: 0.0, im: 0.0 },
            c64 { re: 6.0, im: 0.0 },
        ],
    );
    let mut m2 = m.clone();

    m2.pinv();
    println!("pinv = \n{}", m2);

    println!("m * minv = \n{}", m.dot(&m2));
}
