#![allow(warnings)]
use std::os::raw::*;
use std::ptr;
use types::c64;

// ---------------------------
// FFTW-like constants
// ---------------------------
pub static FFTW_FORWARD: c_int = -1;
pub static FFTW_BACKWARD: c_int = 1;
pub static FFTW_ESTIMATE: c_uint = 64;

// ---------------------------
// cuFFT types and constants
// ---------------------------
type CufftHandle = c_int;

const CUFFT_SUCCESS: c_int = 0;
//const CUFFT_Z2Z: c_int = 0x6a; // double complex to double complex
const CUFFT_Z2Z: c_int = 0x69; // double complex to double complex

// ---------------------------
// cuFFT FFI bindings
// ---------------------------
#[link(name = "cufft")]
extern "C" {
    fn cufftPlan3d(
        plan: *mut CufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        fft_type: c_int,
    ) -> c_int;

    fn cufftDestroy(plan: CufftHandle) -> c_int;

    fn cufftExecZ2Z(
        plan: CufftHandle,
        idata: *mut c64,
        odata: *mut c64,
        direction: c_int,
    ) -> c_int;
}

// ---------------------------
// FFTW API wrappers -> cuFFT
// ---------------------------

// We don't really use threads in cuFFT the same way
pub unsafe fn fftw_init_threads(_: *const c_void) -> c_int {
    0
}

pub unsafe fn fftw_plan_with_nthreads(_: c_int) -> *mut c_void {
    ptr::null_mut()
}

pub unsafe fn fftw_plan_dft_3d(
    n0: c_int,
    n1: c_int,
    n2: c_int,
    _in: *const c64,
    _out: *mut c64,
    _sign: c_int,
    _flags: c_uint,
) -> *mut c_void {
    let mut plan: CufftHandle = 0;
    let res = cufftPlan3d(&mut plan, n0, n1, n2, CUFFT_Z2Z);
    if res != CUFFT_SUCCESS {
        eprintln!("cufftPlan3d failed with error code {}", res);
        return ptr::null_mut();
    }
    // Store plan as void* so it matches FFTW type
    Box::into_raw(Box::new(plan)) as *mut c_void
}

pub unsafe fn fftw_execute_dft(
    p: *const c_void,
    in_: *const c64,
    out: *mut c64,
) {
    if p.is_null() {
        eprintln!("FFTW/FFT plan is null");
        return;
    }
    let plan = *(p as *mut CufftHandle);
    let res = cufftExecZ2Z(plan, in_ as *mut c64, out, FFTW_FORWARD);
    if res != CUFFT_SUCCESS {
        eprintln!("cufftExecZ2Z failed with error code {}", res);
    }
}

pub unsafe fn fftw_destroy_plan(p: *const c_void) {
    if p.is_null() {
        return;
    }
    let plan = *(p as *mut CufftHandle);
    cufftDestroy(plan);
    drop(Box::from_raw(p as *mut CufftHandle));
}

pub unsafe fn fftw_cleanup() {}

