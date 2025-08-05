/////////////////////////////////////////////////////

#[cfg(feature = "cpu")]
mod fftw_cpu;

#[cfg(feature = "gpu")]
mod fftw_gpu;

// Only one of these will be compiled depending on the feature

cfg_if::cfg_if! {
    if #[cfg(feature = "cpu")] {
        pub mod fftw {
            pub use crate::fftw_cpu::*;
        }
    } else if #[cfg(feature = "gpu")] {
        pub mod fftw {
            pub use crate::fftw_gpu::*;
        }
    } else {
        compile_error!("You must enable exactly one of the `cpu` or `gpu` features.");
    }
}

use ndarray::*;
use std::os::raw::*;
use types::c64; // This is num_complex::Complex<f64>

use crate::fftw::*;

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

        // Create zero-initialized arrays
        let mut arr_in = Array3::<c64>::new([n1, n2, n3]);
        let mut arr_out = Array3::<c64>::new([n1, n2, n3]);

        for v in arr_in.as_mut_slice().iter_mut() {
            *v = c64::new(0.0, 0.0);
        }
        for v in arr_out.as_mut_slice().iter_mut() {
            *v = c64::new(0.0, 0.0);
        }

        let slice_in = arr_in.as_slice();
        let slice_out = arr_out.as_mut_slice();

        let plan_fwd: *const c_void;
        let plan_bwd: *const c_void;
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

    pub fn fft3d(&self, slice_in: &[c64], slice_out: &mut [c64]) {
        unsafe {
            fftw_execute_dft(self.plan_fwd, slice_in.as_ptr(), slice_out.as_mut_ptr());
        }
    }

    pub fn ifft3d(&self, slice_in: &[c64], slice_out: &mut [c64]) {
        unsafe {
            fftw_execute_dft(self.plan_bwd, slice_in.as_ptr(), slice_out.as_mut_ptr());
        }
    }
}

impl Drop for DWFFT3D {
    fn drop(&mut self) {
        let plans = [self.plan_fwd, self.plan_bwd];
        for plan in plans.iter() {
            if !plan.is_null() {
                unsafe { fftw_destroy_plan(*plan) };
            }
        }
    }
}

