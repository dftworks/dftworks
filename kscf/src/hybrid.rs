use control::Control;
use dwconsts::{EPS20, FOURPI};
use gvector::GVector;
use matrix::Matrix;
use ndarray::Array3;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::c64;

fn effective_occ_scale(is_spin: bool) -> f64 {
    if is_spin {
        1.0
    } else {
        0.5
    }
}

fn screened_exchange_kernel_value(g: f64, omega: f64) -> f64 {
    if g <= 1.0E-12 {
        return 0.0;
    }

    let x = -(g * g) / (4.0 * omega * omega);

    FOURPI * (1.0 - x.exp()) / (g * g)
}

pub struct HybridPotential {
    enabled: bool,
    alpha: f64,
    kernel_sr: Vec<f64>,
}

pub struct HybridPrepared {
    occ_eff: Vec<f64>,
    psi_occ_r: Vec<Vec<c64>>,
    exchange_energy: f64,
}

impl HybridPrepared {
    pub fn exchange_energy(&self) -> f64 {
        self.exchange_energy
    }

    pub fn is_empty(&self) -> bool {
        self.psi_occ_r.is_empty()
    }
}

pub struct HybridWorkspace {
    vin_r: Array3<c64>,
    pair_r: Array3<c64>,
    pair_g: Vec<c64>,
    potential_g: Vec<c64>,
    potential_r: Array3<c64>,
    contrib_r: Array3<c64>,
    contrib_g: Array3<c64>,
    fft_workspace: Array3<c64>,
}

impl HybridWorkspace {
    pub fn new(fft_shape: [usize; 3], npw_rho: usize) -> Self {
        Self {
            vin_r: Array3::<c64>::new(fft_shape),
            pair_r: Array3::<c64>::new(fft_shape),
            pair_g: vec![c64::new(0.0, 0.0); npw_rho],
            potential_g: vec![c64::new(0.0, 0.0); npw_rho],
            potential_r: Array3::<c64>::new(fft_shape),
            contrib_r: Array3::<c64>::new(fft_shape),
            contrib_g: Array3::<c64>::new(fft_shape),
            fft_workspace: Array3::<c64>::new(fft_shape),
        }
    }
}

impl HybridPotential {
    pub fn new(control: &Control, pwwfc: &PWBasis, pwden: &PWDensity) -> Self {
        if control.get_xc_scheme() != "hse06" {
            return Self {
                enabled: false,
                alpha: 0.0,
                kernel_sr: Vec::new(),
            };
        }

        let k_cart = pwwfc.get_k_cart();

        if k_cart.norm2() > 1.0E-10 {
            panic!("xc_scheme='hse06' currently supports only Gamma-only k-points (k_cart=0).");
        }

        let alpha = control.get_hse06_alpha();
        let omega = control.get_hse06_omega();

        let kernel_sr = pwden
            .get_g()
            .iter()
            .map(|&g| screened_exchange_kernel_value(g, omega))
            .collect();

        Self {
            enabled: true,
            alpha,
            kernel_sr,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn make_workspace(&self, fft_shape: [usize; 3], npw_rho: usize) -> HybridWorkspace {
        HybridWorkspace::new(fft_shape, npw_rho)
    }

    pub fn prepare(
        &self,
        rgtrans: &RGTransform,
        gvec: &GVector,
        pwden: &PWDensity,
        volume: f64,
        fft_shape: [usize; 3],
        fft_linear_index: &[usize],
        evecs: &Matrix<c64>,
        occ: &[f64],
        is_spin: bool,
    ) -> HybridPrepared {
        if !self.enabled {
            return HybridPrepared {
                occ_eff: Vec::new(),
                psi_occ_r: Vec::new(),
                exchange_energy: 0.0,
            };
        }

        let occ_scale = effective_occ_scale(is_spin);

        let mut psi_tmp = Array3::<c64>::new(fft_shape);
        let mut fft_workspace = Array3::<c64>::new(fft_shape);

        let mut occ_eff = Vec::<f64>::new();
        let mut psi_occ_r = Vec::<Vec<c64>>::new();

        for ibnd in 0..occ.len() {
            let occ_weight = occ[ibnd] * occ_scale;
            if occ_weight < EPS20 {
                continue;
            }

            hpsi::compute_unk_3d_with_cached_fft_index(
                rgtrans,
                volume,
                fft_linear_index,
                evecs.get_col(ibnd),
                &mut psi_tmp,
                &mut fft_workspace,
            );

            occ_eff.push(occ_weight);
            psi_occ_r.push(psi_tmp.as_slice().to_vec());
        }

        let exchange_energy = self.compute_exchange_energy(
            gvec, pwden, rgtrans, volume, &occ_eff, &psi_occ_r, fft_shape,
        );

        HybridPrepared {
            occ_eff,
            psi_occ_r,
            exchange_energy,
        }
    }

    fn compute_exchange_energy(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rgtrans: &RGTransform,
        volume: f64,
        occ_eff: &[f64],
        psi_occ_r: &[Vec<c64>],
        fft_shape: [usize; 3],
    ) -> f64 {
        if occ_eff.is_empty() {
            return 0.0;
        }

        let n_occ = occ_eff.len();

        let mut pair_r = Array3::<c64>::new(fft_shape);
        let mut pair_g = vec![c64::new(0.0, 0.0); pwden.get_n_plane_waves()];

        let mut sum_weighted = 0.0;

        for i in 0..n_occ {
            for j in 0..=i {
                let psi_i = &psi_occ_r[i];
                let psi_j = &psi_occ_r[j];
                let pair_r_slice = pair_r.as_mut_slice();

                for ip in 0..pair_r_slice.len() {
                    pair_r_slice[ip] = psi_j[ip].conj() * psi_i[ip];
                }

                rgtrans.r3d_to_g1d(gvec, pwden, pair_r.as_slice(), &mut pair_g);

                let mut integral_g = 0.0;
                for ig in 1..pair_g.len() {
                    integral_g += self.kernel_sr[ig] * pair_g[ig].norm_sqr();
                }

                let pref = if i == j { 1.0 } else { 2.0 };
                sum_weighted += pref * occ_eff[i] * occ_eff[j] * integral_g;
            }
        }

        -0.5 * self.alpha * volume * sum_weighted
    }

    pub fn apply_on_psi(
        &self,
        rgtrans: &RGTransform,
        gvec: &GVector,
        pwden: &PWDensity,
        volume: f64,
        fft_linear_index: &[usize],
        prepared: &HybridPrepared,
        workspace: &mut HybridWorkspace,
        vin: &[c64],
        vout: &mut [c64],
    ) {
        if !self.enabled || prepared.is_empty() {
            return;
        }

        hpsi::compute_unk_3d_with_cached_fft_index(
            rgtrans,
            volume,
            fft_linear_index,
            vin,
            &mut workspace.vin_r,
            &mut workspace.fft_workspace,
        );

        for x in workspace.contrib_r.as_mut_slice().iter_mut() {
            *x = c64::new(0.0, 0.0);
        }

        let vin_r = workspace.vin_r.as_slice();

        for (j, psi_j_r) in prepared.psi_occ_r.iter().enumerate() {
            let occ_j = prepared.occ_eff[j];
            if occ_j < EPS20 {
                continue;
            }

            let pair_r = workspace.pair_r.as_mut_slice();
            for ip in 0..pair_r.len() {
                pair_r[ip] = psi_j_r[ip].conj() * vin_r[ip];
            }

            rgtrans.r3d_to_g1d(
                gvec,
                pwden,
                workspace.pair_r.as_slice(),
                &mut workspace.pair_g,
            );

            for ig in 0..workspace.pair_g.len() {
                workspace.potential_g[ig] = workspace.pair_g[ig] * self.kernel_sr[ig];
            }

            rgtrans.g1d_to_r3d(
                gvec,
                pwden,
                &workspace.potential_g,
                workspace.potential_r.as_mut_slice(),
            );

            let scale = -self.alpha * occ_j;
            let pot_r = workspace.potential_r.as_slice();
            let contrib_r = workspace.contrib_r.as_mut_slice();

            for ip in 0..contrib_r.len() {
                contrib_r[ip] += scale * psi_j_r[ip] * pot_r[ip];
            }
        }

        rgtrans.r3d_to_g3d(
            workspace.contrib_r.as_slice(),
            workspace.contrib_g.as_mut_slice(),
        );
        workspace.contrib_g.scale(volume.sqrt());

        let contrib_g = workspace.contrib_g.as_slice();
        for (ipw, &idx) in fft_linear_index.iter().enumerate() {
            vout[ipw] += contrib_g[idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_occ_scale, screened_exchange_kernel_value};

    #[test]
    fn test_effective_occ_scale() {
        assert!((effective_occ_scale(false) - 0.5).abs() < 1.0E-12);
        assert!((effective_occ_scale(true) - 1.0).abs() < 1.0E-12);
    }

    #[test]
    fn test_screened_exchange_kernel_limits() {
        assert!(screened_exchange_kernel_value(0.0, 0.11).abs() < 1.0E-14);
        let k = screened_exchange_kernel_value(1.0, 0.11);
        assert!(k.is_finite());
        assert!(k > 0.0);
    }
}
