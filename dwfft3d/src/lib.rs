mod fftw;
use fftw::*;

use ndarray::*;
use std::os::raw::*;
use types::*;

pub struct DWFFT3D {
    plan_fwd: *const c_void,
    plan_bwd: *const c_void,
}

impl DWFFT3D {
    pub fn new(n1: usize, n2: usize, n3: usize) -> DWFFT3D {
        unsafe {
            fftw_init_threads(std::ptr::null());
            fftw_plan_with_nthreads(1);
        }

        let arr_in = Array3::<c64>::new([n1, n2, n3]);
        let mut arr_out = Array3::<c64>::new([n1, n2, n3]);

        let slice_in = arr_in.as_slice();
        let slice_out = arr_out.as_mut_slice();

        let plan_fwd: *const c_void;

        unsafe {
            plan_fwd = fftw_plan_dft_3d(
                n3 as i32,
                n2 as i32,
                n1 as i32,
                slice_in.as_ptr(),
                slice_out.as_mut_ptr(),
                FFTW_FORWARD,
                FFTW_ESTIMATE,
            );
        }

        let plan_bwd: *const c_void;

        unsafe {
            plan_bwd = fftw_plan_dft_3d(
                n3 as i32,
                n2 as i32,
                n1 as i32,
                slice_in.as_ptr(),
                slice_out.as_mut_ptr(),
                FFTW_BACKWARD,
                FFTW_ESTIMATE,
            );
        }

        DWFFT3D { plan_fwd, plan_bwd }
    }

    //pub fn fft3d(&self, in_arr: &Array3<c64>, out_arr: &mut Array3<c64>) {
    pub fn fft3d(&self, slice_in: &[c64], slice_out: &mut [c64]) {
        //let slice_in = in_arr.as_slice();
        //let slice_out = out_arr.as_mut_slice();

        unsafe {
            fftw_execute_dft(self.plan_fwd, slice_in.as_ptr(), slice_out.as_mut_ptr());
        }
    }

    //pub fn ifft3d(&self, in_arr: &Array3<c64>, out_arr: &mut Array3<c64>) {
    pub fn ifft3d(&self, slice_in: &[c64], slice_out: &mut [c64]) {
        //let slice_in = in_arr.as_slice();
        //let slice_out = out_arr.as_mut_slice();

        unsafe {
            fftw_execute_dft(self.plan_bwd, slice_in.as_ptr(), slice_out.as_mut_ptr());
        }
    }
}

impl Drop for DWFFT3D {
    fn drop(&mut self) {
        let plans = [self.plan_fwd, self.plan_bwd];

        for (_, plan) in plans.iter().enumerate() {
            if !plan.is_null() {
                unsafe {
                    fftw_destroy_plan(*plan);
                }
            }
        }
    }
}
