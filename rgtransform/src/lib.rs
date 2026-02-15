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

    pub fn gradient_norm_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        grad_norm_3d: &mut [c64],
    ) {
        let nfft = rho_3d.len();
        let mut drho_x = vec![c64::new(0.0, 0.0); nfft];
        let mut drho_y = vec![c64::new(0.0, 0.0); nfft];
        let mut drho_z = vec![c64::new(0.0, 0.0); nfft];

        self.gradient_r3d(
            gvec,
            pwden,
            rho_3d,
            &mut drho_x,
            &mut drho_y,
            &mut drho_z,
        );

        for i in 0..nfft {
            let grad_norm = (drho_x[i].norm_sqr() + drho_y[i].norm_sqr() + drho_z[i].norm_sqr())
                .sqrt();
            grad_norm_3d[i] = c64::new(grad_norm, 0.0);
        }
    }

    pub fn gradient_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        grad_x_3d: &mut [c64],
        grad_y_3d: &mut [c64],
        grad_z_3d: &mut [c64],
    ) {
        let npw = pwden.get_n_plane_waves();

        let mut rhog = vec![c64::new(0.0, 0.0); npw];
        self.r3d_to_g1d(gvec, pwden, rho_3d, &mut rhog);

        let mut drhog_x = vec![c64::new(0.0, 0.0); npw];
        let mut drhog_y = vec![c64::new(0.0, 0.0); npw];
        let mut drhog_z = vec![c64::new(0.0, 0.0); npw];

        let gindex = pwden.get_gindex();
        let gcart = gvec.get_cart();

        for ipw in 0..npw {
            let g = gcart[gindex[ipw]];
            let v = rhog[ipw];

            // i * G * rho(G)
            drhog_x[ipw] = c64::new(-g.x * v.im, g.x * v.re);
            drhog_y[ipw] = c64::new(-g.y * v.im, g.y * v.re);
            drhog_z[ipw] = c64::new(-g.z * v.im, g.z * v.re);
        }

        self.g1d_to_r3d(gvec, pwden, &drhog_x, grad_x_3d);
        self.g1d_to_r3d(gvec, pwden, &drhog_y, grad_y_3d);
        self.g1d_to_r3d(gvec, pwden, &drhog_z, grad_z_3d);
    }

    pub fn divergence_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        vec_x_3d: &[c64],
        vec_y_3d: &[c64],
        vec_z_3d: &[c64],
        div_3d: &mut [c64],
    ) {
        let npw = pwden.get_n_plane_waves();
        let mut vg_x = vec![c64::new(0.0, 0.0); npw];
        let mut vg_y = vec![c64::new(0.0, 0.0); npw];
        let mut vg_z = vec![c64::new(0.0, 0.0); npw];

        self.r3d_to_g1d(gvec, pwden, vec_x_3d, &mut vg_x);
        self.r3d_to_g1d(gvec, pwden, vec_y_3d, &mut vg_y);
        self.r3d_to_g1d(gvec, pwden, vec_z_3d, &mut vg_z);

        let mut div_g = vec![c64::new(0.0, 0.0); npw];
        let gindex = pwden.get_gindex();
        let gcart = gvec.get_cart();

        for ipw in 0..npw {
            let g = gcart[gindex[ipw]];
            let dot = vg_x[ipw] * g.x + vg_y[ipw] * g.y + vg_z[ipw] * g.z;
            div_g[ipw] = c64::new(-dot.im, dot.re);
        }

        self.g1d_to_r3d(gvec, pwden, &div_g, div_3d);
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
