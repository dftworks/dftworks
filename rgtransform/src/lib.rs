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
    rhog: Vec<c64>,
    spectral_g_work: Vec<c64>,
    div_g: Vec<c64>,
    real_work: Vec<c64>,
}

impl ThreadWorkspace {
    fn ensure_npw(&mut self, npw: usize) {
        if self.rhog.len() != npw {
            self.rhog.resize(npw, c64::new(0.0, 0.0));
        }
        if self.spectral_g_work.len() != npw {
            self.spectral_g_work.resize(npw, c64::new(0.0, 0.0));
        }
        if self.div_g.len() != npw {
            self.div_g.resize(npw, c64::new(0.0, 0.0));
        }
    }

    fn ensure_nfft(&mut self, nfft: usize) {
        if self.real_work.len() != nfft {
            self.real_work.resize(nfft, c64::new(0.0, 0.0));
        }
    }
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

    fn with_workspace<R>(&self, f: impl FnOnce(&mut ThreadWorkspace) -> R) -> R {
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
                    rhog: Vec::new(),
                    spectral_g_work: Vec::new(),
                    div_g: Vec::new(),
                    real_work: Vec::new(),
                }
            });

            f(workspace)
        })
    }

    fn r3d_to_g1d_raw(
        &self,
        pfft: &DWFFT3D,
        fft_work: &mut Array3<c64>,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        rhog_1d: &mut [c64],
    ) {
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
    }

    fn g1d_to_r3d_raw(
        &self,
        pfft: &DWFFT3D,
        fft_work: &mut Array3<c64>,
        gvec: &GVector,
        pwden: &PWDensity,
        rhog_1d: &[c64],
        rho_3d: &mut [c64],
    ) {
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
    }

    pub fn r3d_to_g1d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        rhog_1d: &mut [c64],
    ) {
        self.with_workspace(|workspace| {
            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                rho_3d,
                rhog_1d,
            );
        })
    }

    pub fn g1d_to_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rhog_1d: &[c64],
        rho_3d: &mut [c64],
    ) {
        self.with_workspace(|workspace| {
            self.g1d_to_r3d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                rhog_1d,
                rho_3d,
            );
        })
    }

    pub fn gradient_norm_r3d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        grad_norm_3d: &mut [c64],
    ) {
        // Convenience wrapper used by code paths that only need |grad rho|.
        // Internally we still use spectral derivatives for each Cartesian component.
        let nfft = rho_3d.len();
        let npw = pwden.get_n_plane_waves();
        let gindex = pwden.get_gindex();
        let gcart = gvec.get_cart();

        self.with_workspace(|workspace| {
            workspace.ensure_npw(npw);
            workspace.ensure_nfft(nfft);

            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                rho_3d,
                &mut workspace.rhog,
            );

            for v in grad_norm_3d.iter_mut() {
                *v = c64::new(0.0, 0.0);
            }

            for idir in 0..3 {
                for ipw in 0..npw {
                    let g = gcart[gindex[ipw]];
                    let gcomp = match idir {
                        0 => g.x,
                        1 => g.y,
                        _ => g.z,
                    };
                    let v = workspace.rhog[ipw];
                    workspace.spectral_g_work[ipw] = c64::new(-gcomp * v.im, gcomp * v.re);
                }

                self.g1d_to_r3d_raw(
                    &workspace.pfft,
                    &mut workspace.fft_work,
                    gvec,
                    pwden,
                    workspace.spectral_g_work.as_slice(),
                    workspace.real_work.as_mut_slice(),
                );

                for i in 0..nfft {
                    grad_norm_3d[i].re += workspace.real_work[i].norm_sqr();
                }
            }

            for v in grad_norm_3d.iter_mut() {
                v.re = v.re.sqrt();
                v.im = 0.0;
            }
        });
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
        // Spectral derivative workflow:
        //   rho(r) --FFT--> rho(G)
        //   d/dx rho <-> i Gx rho(G), similarly for y/z
        //   iG*rho(G) --iFFT--> grad components in real space
        let npw = pwden.get_n_plane_waves();
        let gindex = pwden.get_gindex();
        let gcart = gvec.get_cart();

        self.with_workspace(|workspace| {
            workspace.ensure_npw(npw);
            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                rho_3d,
                &mut workspace.rhog,
            );

            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let v = workspace.rhog[ipw];
                // i * Gx * rho(G)
                workspace.spectral_g_work[ipw] = c64::new(-g.x * v.im, g.x * v.re);
            }
            self.g1d_to_r3d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                workspace.spectral_g_work.as_slice(),
                grad_x_3d,
            );

            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let v = workspace.rhog[ipw];
                // i * Gy * rho(G)
                workspace.spectral_g_work[ipw] = c64::new(-g.y * v.im, g.y * v.re);
            }
            self.g1d_to_r3d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                workspace.spectral_g_work.as_slice(),
                grad_y_3d,
            );

            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let v = workspace.rhog[ipw];
                // i * Gz * rho(G)
                workspace.spectral_g_work[ipw] = c64::new(-g.z * v.im, g.z * v.re);
            }
            self.g1d_to_r3d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                workspace.spectral_g_work.as_slice(),
                grad_z_3d,
            );
        });
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
        // Spectral divergence workflow:
        //   F(r) --FFT--> F(G)
        //   div F <-> i G dot F(G)
        //   i(G·F(G)) --iFFT--> div F(r)
        let npw = pwden.get_n_plane_waves();
        let gindex = pwden.get_gindex();
        let gcart = gvec.get_cart();

        self.with_workspace(|workspace| {
            workspace.ensure_npw(npw);

            for ipw in 0..npw {
                workspace.div_g[ipw] = c64::new(0.0, 0.0);
            }

            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                vec_x_3d,
                &mut workspace.spectral_g_work,
            );
            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let dot = workspace.spectral_g_work[ipw] * g.x;
                workspace.div_g[ipw] += c64::new(-dot.im, dot.re);
            }

            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                vec_y_3d,
                &mut workspace.spectral_g_work,
            );
            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let dot = workspace.spectral_g_work[ipw] * g.y;
                workspace.div_g[ipw] += c64::new(-dot.im, dot.re);
            }

            self.r3d_to_g1d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                vec_z_3d,
                &mut workspace.spectral_g_work,
            );
            for ipw in 0..npw {
                let g = gcart[gindex[ipw]];
                let dot = workspace.spectral_g_work[ipw] * g.z;
                workspace.div_g[ipw] += c64::new(-dot.im, dot.re);
            }

            self.g1d_to_r3d_raw(
                &workspace.pfft,
                &mut workspace.fft_work,
                gvec,
                pwden,
                &workspace.div_g,
                div_3d,
            );
        });
    }

    pub fn r3d_to_g3d(&self, r: &[c64], g: &mut [c64]) {
        self.with_workspace(|workspace| forward(&workspace.pfft, r, g));
    }

    pub fn g3d_to_r3d(&self, g: &[c64], r: &mut [c64]) {
        self.with_workspace(|workspace| backward(&workspace.pfft, g, r));
    }
}

fn forward(pfft: &DWFFT3D, r: &[c64], g: &mut [c64]) {
    pfft.fft3d(r, g);

    let ng_f64 = g.len() as f64;

    // Normalize forward transform so r<->g mappings remain consistent with
    // energy/derivative formulas used throughout the codebase.
    g.iter_mut().for_each(|x| *x /= ng_f64);
}

fn backward(pfft: &DWFFT3D, g: &[c64], r: &mut [c64]) {
    pfft.ifft3d(g, r);
}
