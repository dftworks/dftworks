use dwfft3d::DWFFT3D;
use gvector::*;
use ndarray::*;
use pwdensity::*;
use types::c64;

use std::cell::RefCell;

pub struct RGTransform {
    fftmesh: [usize; 3],
    pfft: DWFFT3D,
    fft_work: RefCell<Array3<c64>>,
}

impl RGTransform {
    pub fn new(n1: usize, n2: usize, n3: usize) -> RGTransform {
        let pfft = DWFFT3D::new(n1, n2, n3);

        let fftmesh = [n1, n2, n3];
        let fft_work = RefCell::new(Array3::<c64>::new(fftmesh));

        RGTransform {
            fftmesh,
            pfft,
            fft_work,
        }
    }

    pub fn r3d_to_g1d(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rho_3d: &[c64],
        rhog_1d: &mut [c64],
    ) {
        forward(
            &self.pfft,
            rho_3d,
            &mut self.fft_work.borrow_mut().as_mut_slice(),
        );

        utility::map_3d_to_1d(
            gvec.get_miller(),
            pwden.get_gindex(),
            self.fftmesh[0],
            self.fftmesh[1],
            self.fftmesh[2],
            &self.fft_work.borrow(),
            rhog_1d,
        );
    }

    pub fn g1d_to_r3d(
        &self,
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
            &mut self.fft_work.borrow_mut(),
        );

        backward(&self.pfft, &self.fft_work.borrow().as_slice(), rho_3d);
    }

    pub fn r3d_to_g3d(&self, r: &[c64], g: &mut [c64]) {
        forward(&self.pfft, r, g);
    }

    pub fn g3d_to_r3d(&self, g: &[c64], r: &mut [c64]) {
        backward(&self.pfft, g, r);
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
