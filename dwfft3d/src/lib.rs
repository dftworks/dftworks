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
use std::ffi::CString;
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};
use types::c64; // This is num_complex::Complex<f64>

use crate::fftw::*;

pub struct DWFFT3D {
    plan_fwd: *const c_void,
    plan_bwd: *const c_void,
}

use cfg_if::cfg_if;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftPlanningMode {
    Estimate,
    Measure,
}

impl Default for FftPlanningMode {
    fn default() -> Self {
        FftPlanningMode::Estimate
    }
}

impl FftPlanningMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FftPlanningMode::Estimate => "estimate",
            FftPlanningMode::Measure => "measure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOptions {
    pub threads: usize,
    pub planning_mode: FftPlanningMode,
    pub wisdom_file: Option<String>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            planning_mode: FftPlanningMode::Estimate,
            wisdom_file: None,
        }
    }
}

impl BackendOptions {
    fn normalized(mut self) -> Self {
        if self.threads == 0 {
            self.threads = 1;
        }
        self.wisdom_file = self
            .wisdom_file
            .and_then(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
        self
    }
}

fn backend_options_cell() -> &'static Mutex<BackendOptions> {
    static CELL: OnceLock<Mutex<BackendOptions>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(BackendOptions::default()))
}

pub fn init_backend() {
    // Keep backward-compatible no-arg initialization semantics.
    let _ = backend_options();
}

pub fn configure_runtime(options: BackendOptions) {
    let normalized = options.normalized();
    if let Some(path) = normalized.wisdom_file.as_deref() {
        try_import_wisdom(path);
    }

    let mut guard = backend_options_cell()
        .lock()
        .expect("dwfft3d backend options mutex poisoned");
    *guard = normalized;
}

pub fn backend_options() -> BackendOptions {
    backend_options_cell()
        .lock()
        .expect("dwfft3d backend options mutex poisoned")
        .clone()
}

pub fn backend_name() -> &'static str {
    cfg_if! {
        if #[cfg(feature = "gpu")] {
            "GPU"
        } else if #[cfg(feature = "cpu")] {
            "CPU"
        } else {
            compile_error!("No backend feature enabled. Enable either 'cpu' or 'gpu'.");
        }
    }
}

impl DWFFT3D {
    pub fn new(n1: usize, n2: usize, n3: usize) -> DWFFT3D {
        let options = backend_options();
        let nthreads = options.threads.max(1).min(i32::MAX as usize) as c_int;
        let planner_flag = match options.planning_mode {
            FftPlanningMode::Estimate => FFTW_ESTIMATE,
            FftPlanningMode::Measure => FFTW_MEASURE,
        };

        unsafe {
            fftw_init_threads(std::ptr::null());
            fftw_plan_with_nthreads(nthreads);
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
                planner_flag,
            );

            plan_bwd = fftw_plan_dft_3d(
                n3 as i32,
                n2 as i32,
                n1 as i32,
                slice_in.as_ptr(),
                slice_out.as_mut_ptr(),
                FFTW_BACKWARD,
                planner_flag,
            );
        }

        if let Some(path) = options.wisdom_file.as_deref() {
            try_export_wisdom(path);
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

fn try_import_wisdom(path: &str) {
    if let Ok(cpath) = CString::new(path) {
        unsafe {
            let _ = fftw_import_wisdom_from_filename(cpath.as_ptr());
        }
    }
}

fn try_export_wisdom(path: &str) {
    if let Ok(cpath) = CString::new(path) {
        unsafe {
            let _ = fftw_export_wisdom_to_filename(cpath.as_ptr());
        }
    }
}
