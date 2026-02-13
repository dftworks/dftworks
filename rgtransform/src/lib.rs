use dwfft3d::DWFFT3D;
use gvector::*;
use ndarray::*;
use pwdensity::*;
use types::c64;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;

struct ThreadWorkspace {
    pfft: DWFFT3D,
    fft_work: Array3<c64>,
}

thread_local! {
    static THREAD_WORKSPACE: RefCell<HashMap<[usize; 3], ThreadWorkspace>> = RefCell::new(HashMap::new());
}

static FFTW_PLAN_BUILD_LOCK: Mutex<()> = Mutex::new(());

pub struct RGTransform {
    fftmesh: [usize; 3],
}

impl RGTransform {
    pub fn new(n1: usize, n2: usize, n3: usize) -> RGTransform {
        RGTransform {
            fftmesh: [n1, n2, n3],
        }
    }

    fn with_workspace<R>(&self, f: impl FnOnce(&DWFFT3D, &mut Array3<c64>) -> R) -> R {
        THREAD_WORKSPACE.with(|workspaces| {
            let mut workspaces = workspaces.borrow_mut();

            let workspace = workspaces.entry(self.fftmesh).or_insert_with(|| {
                // FFTW planning is not thread-safe; serialize plan creation.
                let _guard = FFTW_PLAN_BUILD_LOCK
                    .lock()
                    .expect("FFTW plan creation lock poisoned");
                ThreadWorkspace {
                    pfft: DWFFT3D::new(self.fftmesh[0], self.fftmesh[1], self.fftmesh[2]),
                    fft_work: Array3::<c64>::new(self.fftmesh),
                }
            });

            f(&workspace.pfft, &mut workspace.fft_work)
        })
    }

    pub fn r3d_to_g1d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        rhog_1d: &mut [c64],
    ) {
        self.with_workspace(|pfft, fft_work| {
            forward(pfft, rho_3d, fft_work.as_mut_slice());

            utility::map_3d_to_1d(
                gvec.get_miller(),
                pwden.get_gindex(),
                self.fftmesh[0],
                self.fftmesh[1],
                self.fftmesh[2],
                fft_work,
                rhog_1d,
            );
        });
    }

    pub fn g1d_to_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rhog_1d: &[c64],
        rho_3d: &mut [c64],
    ) {
        self.with_workspace(|pfft, fft_work| {
            utility::map_1d_to_3d(
                gvec.get_miller(),
                pwden.get_gindex(),
                self.fftmesh[0],
                self.fftmesh[1],
                self.fftmesh[2],
                rhog_1d,
                fft_work,
            );

            backward(pfft, fft_work.as_slice(), rho_3d);
        });
    }

    pub fn r3d_to_g3d(&self, r: &[c64], g: &mut [c64]) {
        self.with_workspace(|pfft, _| forward(pfft, r, g));
    }

    pub fn g3d_to_r3d(&self, g: &[c64], r: &mut [c64]) {
        self.with_workspace(|pfft, _| backward(pfft, g, r));
    }
}

fn forward(pfft: &DWFFT3D, r: &[c64], g: &mut [c64]) {
    pfft.fft3d(r, g);

    let ng_f64 = g.len() as f64;

    g.iter_mut().for_each(|x| *x /= ng_f64);
}

fn backward(pfft: &DWFFT3D, g: &[c64], r: &mut [c64]) {
    pfft.ifft3d(g, r);
}
