#![allow(warnings)]

use num::complex::Complex;
use nalgebra::DMatrix;

pub type c64 = Complex<f64>;
pub type Vector3<T> = nalgebra::Vector3<T>;
pub type Vector3f64 = Vector3<f64>;
pub type Vector3i32 = Vector3<i32>;
pub type Vector3c64 = Vector3<c64>;
pub type Matrix<T> = DMatrix<T>;

mod matrix_ext;
pub use matrix_ext::*;

mod matrix_codec;
pub use matrix_codec::*;
