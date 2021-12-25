// column-major memory layout
#![allow(warnings)]
use dwconsts::*;
use types::c64;

use std::ops::{Index, IndexMut, MulAssign};

#[derive(Default, Clone, Debug)]
pub struct Array2<T> {
    n0: usize,
    n1: usize,
    data: Vec<T>,
}

impl<T: Default + Copy + Clone + MulAssign> Array2<T> {
    pub fn new(n0: usize, n1: usize) -> Array2<T> {
        Array2 {
            n0,
            n1,
            data: vec![T::default(); n0 * n1],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn set_value(&mut self, value: T)
    where
        T: Clone + Copy,
    {
        for v in self.data.iter_mut() {
            *v = value;
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n0, self.n1)
    }

    pub fn as_slice(&self) -> &Vec<T> {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    pub fn get_col_slice(&self, icol: usize) -> &[T] {
        let n1 = icol * self.n0;
        let n2 = n1 + self.n0;
        &self.data[n1..n2]
    }

    pub fn get_col_ends(&self, icol: usize) -> (usize, usize) {
        let n1 = icol * self.n0;
        let n2 = n1 + self.n0;

        (n1, n2)
    }

    pub fn get_col_mut_slice(&mut self, icol: usize) -> &mut [T] {
        let n1 = icol * self.n0;
        let n2 = n1 + self.n0;
        &mut self.data[n1..n2]
    }

    pub fn scale(&mut self, f: T) {
        for v in self.data.iter_mut() {
            *v *= f;
        }
    }
}

impl<T> Index<[usize; 2]> for Array2<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        let pos = idx[0] + idx[1] * self.n0;
        &self.data[pos]
    }
}

impl<T> IndexMut<[usize; 2]> for Array2<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        let pos = idx[0] + idx[1] * self.n0;
        &mut self.data[pos]
    }
}

#[test]
fn test_array2_c64() {
    let mut m: Array2<c64> = Array2::<c64>::new(4, 3);
    for j in 0..3 {
        for k in 0..4 {
            m[[k, j]] = c64 {
                re: (j * k) as f64,
                im: 0.0,
            };
        }
    }

    for j in 0..3 {
        for k in 0..4 {
            // check if memory is contiguous and value is correct
            println!("{:p} {} {}", &m[[k, j]], m[[k, j]], j * k);
        }
    }
}

#[test]
fn test_array2_f64() {
    let mut m: Array2<f64> = Array2::<f64>::new(4, 3);
    for j in 0..3 {
        for k in 0..4 {
            m[[k, j]] = (j * k) as f64;
        }
    }

    for j in 0..3 {
        for k in 0..4 {
            // check if memory is contiguous and value is correct
            println!("{:p} {} {}", &m[[k, j]], m[[k, j]], j * k);
        }
    }
}
