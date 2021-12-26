#![allow(warnings)]
use ndarray::*;
use types::c64;
use std::os::raw::*;
//https://doc.rust-lang.org/src/std/os/raw/mod.rs.html#68
// c_int : i32
// c_uint : u32
// c_long : i64
// c_ulong : u64
// c_float : f32
// c_double : f64
// c_char : i8
// c_uchar : u8
// c_short : i16
// c_ushort : u16

pub static FFTW_FORWARD: c_int = -1;
pub static FFTW_BACKWARD: c_int = 1;
pub static FFTW_ESTIMATE: c_uint = 64;
pub static FFTW_MEASURE: c_uint = 0;

extern "C" {
    pub fn fftw_plan_dft_1d(n: c_int, in_: *mut c64, out: *mut c64, sign: c_int, flags: c_uint) -> *mut c_void;

    pub fn fftw_plan_dft_2d(n0: c_int, n1: c_int, in_: *mut c64, out: *mut c64, sign: c_int, flags: c_uint) -> *mut c_void;

    pub fn fftw_plan_dft_3d(n0: c_int, n1: c_int, n2: c_int, in_: *const c64, out: *mut c64, sign: c_int, flags: c_uint) -> *mut c_void;

    pub fn fftw_plan_dft_r2c_3d(n0: c_int, n1: c_int, n2: c_int, in_: *mut f64, out: *mut c64, flags: c_uint) -> *mut c_void;

    pub fn fftw_execute_dft(p: *const c_void, in_: *const c64, out: *mut c64);

    pub fn fftw_destroy_plan(p: *const c_void);

    pub fn fftw_cleanup();

    pub fn fftw_init_threads(p: *const c_void) -> c_int;
    
    pub fn fftw_plan_with_nthreads(nthreads: c_int) -> *mut c_void;
}
