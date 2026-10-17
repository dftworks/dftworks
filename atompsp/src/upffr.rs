use crate::upfread::{check_len, read_upf, UpfData};
use crate::AtomPSP;

use dwconsts::*;
use types::*;

use std::fmt;

/// Fully relativistic UPF v2 pseudopotential (`pot_scheme = upf-fr`). The file has
/// one projector per (l, j); each j = l -/+ 1/2 pair is combined into one
/// scalar-relativistic projector (the `AtomPSP` getters) and one spin-orbit
/// projector (`get_*_soc`).
pub struct AtomPSPUPFFR {
    /// the file as read: (l, j) projectors
    upf: UpfData,

    // scalar-relativistic projectors
    lbeta: Vec<usize>,
    vbeta: Vec<Vec<f64>>,
    dij: Matrix<f64>,

    // spin-orbit projectors (for future noncollinear runs; not used yet)
    lbeta_soc: Vec<usize>,
    vbeta_soc: Vec<Vec<f64>>,
    dij_soc: Matrix<f64>,
}

impl AtomPSPUPFFR {
    /// Read a fully relativistic UPF v2 (XML) file.
    pub fn from_file(pspfile: &str) -> Result<AtomPSPUPFFR, String> {
        let upf = read_upf(pspfile)?;
        check_relbeta(&upf)?;

        let mut psp = AtomPSPUPFFR {
            upf,
            lbeta: Vec::new(),
            vbeta: Vec::new(),
            dij: Matrix::<f64>::new(0, 0),
            lbeta_soc: Vec::new(),
            vbeta_soc: Vec::new(),
            dij_soc: Matrix::<f64>::new(0, 0),
        };
        psp.build_scalar_beta();
        psp.build_soc_beta();

        Ok(psp)
    }

    pub fn get_nbeta_soc(&self) -> usize {
        self.lbeta_soc.len()
    }

    pub fn get_beta_soc(&self, ibeta: usize) -> &[f64] {
        &self.vbeta_soc[ibeta]
    }

    pub fn get_lbeta_soc(&self, ibeta: usize) -> usize {
        self.lbeta_soc[ibeta]
    }

    pub fn get_dfact_soc(&self, ibeta: usize) -> f64 {
        self.dij_soc[(ibeta, ibeta)]
    }

    fn build_scalar_beta(&mut self) {
        let upf = &self.upf;
        let mmax = upf.mmax;

        // one scalar projector per l = 0 projector and per (l - 1/2, l + 1/2) pair
        let mut nbeta = 0;
        for (&l, &j) in upf.lll.iter().zip(upf.jjj.iter()) {
            nbeta += 1;

            if l > 0 && (j - (l as f64 + 0.5)).abs() < EPS5 {
                nbeta -= 1;
            }
        }

        self.lbeta = vec![0; nbeta];
        self.dij = Matrix::<f64>::new(nbeta, nbeta);

        let mut jbeta = 0;
        for ibeta in 0..nbeta {
            let l = upf.lll[jbeta];

            if l == 0 {
                self.vbeta.push(upf.beta[jbeta].clone());
                self.dij[(ibeta, ibeta)] = upf.dij[(jbeta, jbeta)];
                self.lbeta[ibeta] = l;

                jbeta += 1;
            } else {
                let l_f64 = l as f64;
                let dminus = upf.dij[(jbeta, jbeta)];
                let dplus = upf.dij[(jbeta + 1, jbeta + 1)];

                let dpm = ((l_f64 + 1.0) * dplus + l_f64 * dminus) / (2.0 * l_f64 + 1.0);

                let fm = (dminus / dpm).sqrt();
                let fp = (dplus / dpm).sqrt();

                let mut beta = vec![0.0; mmax];
                for i in 0..mmax {
                    beta[i] = l_f64 * upf.beta[jbeta][i] * fm
                        + (l_f64 + 1.0) * upf.beta[jbeta + 1][i] * fp;

                    beta[i] /= 2.0 * l_f64 + 1.0;
                }
                self.vbeta.push(beta);

                self.dij[(ibeta, ibeta)] = dpm;
                self.lbeta[ibeta] = l;

                // skip the (l, l - 1/2) and (l, l + 1/2) projectors
                jbeta += 2;
            }
        }
    }

    fn build_soc_beta(&mut self) {
        let upf = &self.upf;
        let mmax = upf.mmax;

        let mut dij_soc = Vec::new();

        let mut jbeta = 0;
        for _ in 0..self.lbeta.len() {
            let l = upf.lll[jbeta];

            if l == 0 {
                jbeta += 1;
            } else {
                let l_f64 = l as f64;
                let dminus = upf.dij[(jbeta, jbeta)];
                let dplus = upf.dij[(jbeta + 1, jbeta + 1)];

                let dpm = ((l_f64 + 1.0) * dplus + l_f64 * dminus) / (2.0 * l_f64 + 1.0);

                let mut vsoc = vec![0.0; mmax];
                for i in 0..mmax {
                    vsoc[i] = -upf.beta[jbeta][i] * (dminus / dpm).sqrt()
                        + upf.beta[jbeta + 1][i] * (dplus / dpm).sqrt();

                    vsoc[i] *= 2.0 / (2.0 * l_f64 + 1.0);
                }
                self.vbeta_soc.push(vsoc);

                dij_soc.push(dpm);
                self.lbeta_soc.push(l);

                // skip the (l, l - 1/2) and (l, l + 1/2) projectors
                jbeta += 2;
            }
        }

        let nbeta = self.lbeta_soc.len();
        self.dij_soc = Matrix::<f64>::new(nbeta, nbeta);
        for ibeta in 0..nbeta {
            self.dij_soc[(ibeta, ibeta)] = dij_soc[ibeta];
        }
    }
}

// The projectors must be listed as: l = 0 with j = 1/2, or, for l > 0, j = l - 1/2
// immediately followed by j = l + 1/2. build_scalar_beta and build_soc_beta rely on it.
fn check_relbeta(upf: &UpfData) -> Result<(), String> {
    if upf.lll.is_empty() && upf.nbeta > 0 {
        return Err(
            "no PP_RELBETA (j) data: pot_scheme = upf-fr needs a fully relativistic file"
                .to_string(),
        );
    }
    check_len("PP_RELBETA lll", upf.lll.len(), upf.nbeta)?;
    check_len("PP_RELBETA jjj", upf.jjj.len(), upf.nbeta)?;

    let has_j = |ibeta: usize, j: f64| (upf.jjj[ibeta] - j).abs() < EPS5;

    let mut ibeta = 0;
    while ibeta < upf.nbeta {
        let l = upf.lll[ibeta];
        let lf = l as f64;

        if l == 0 {
            if !has_j(ibeta, 0.5) {
                return Err(format!(
                    "PP_RELBETA.{}: l = 0 needs j = 0.5, got {}",
                    ibeta + 1,
                    upf.jjj[ibeta]
                ));
            }
            ibeta += 1;
            continue;
        }

        let is_pair = ibeta + 1 < upf.nbeta
            && upf.lll[ibeta + 1] == l
            && has_j(ibeta, lf - 0.5)
            && has_j(ibeta + 1, lf + 0.5);
        if !is_pair {
            return Err(format!(
                "PP_RELBETA.{}: l = {} projectors must come as j = {} followed by j = {}",
                ibeta + 1,
                l,
                lf - 0.5,
                lf + 0.5
            ));
        }
        ibeta += 2;
    }

    Ok(())
}

impl AtomPSP for AtomPSPUPFFR {
    fn get_nlcc(&self) -> bool {
        self.upf.core_correction
    }

    fn get_nbeta(&self) -> usize {
        self.lbeta.len()
    }

    fn get_lbeta(&self, ibeta: usize) -> usize {
        self.lbeta[ibeta]
    }

    fn get_beta(&self, ibeta: usize) -> &[f64] {
        &self.vbeta[ibeta]
    }

    fn get_dfact(&self, ibeta: usize) -> f64 {
        self.dij[(ibeta, ibeta)]
    }

    fn get_zion(&self) -> f64 {
        self.upf.zion
    }

    fn get_lloc(&self) -> i32 {
        self.upf.lloc
    }

    fn get_lmax(&self) -> usize {
        self.upf.lmax
    }

    fn get_mmax(&self) -> usize {
        self.upf.mmax
    }

    fn get_rad(&self) -> &[f64] {
        &self.upf.rad
    }

    fn get_rab(&self) -> &[f64] {
        &self.upf.rab
    }

    fn get_rho(&self) -> &[f64] {
        &self.upf.rhoatom
    }

    fn get_rhocore(&self) -> &[f64] {
        &self.upf.rhocore
    }

    fn get_wfc(&self, l: usize) -> &[f64] {
        self.upf.chi.get(&l).unwrap()
    }

    fn has_wfc(&self, l: usize) -> bool {
        self.upf.chi.contains_key(&l)
    }

    fn get_wfc_channels(&self) -> Vec<usize> {
        let mut ls: Vec<usize> = self.upf.chi.keys().copied().collect();
        ls.sort_unstable();
        ls
    }

    fn get_vloc(&self) -> &[f64] {
        &self.upf.vloc
    }
}

impl fmt::Display for AtomPSPUPFFR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            " element = {:?} zion = {} lmax = {} lloc = {}",
            self.upf.element, self.upf.zion, self.upf.lmax, self.upf.lloc
        )?;
        writeln!(f, " mmax = {} nbeta = {}", self.upf.mmax, self.lbeta.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_path(name: &str) -> String {
        format!("{}/src/{}", env!("CARGO_MANIFEST_DIR"), name)
    }

    #[test]
    fn test_load_si_fr() {
        let psp = AtomPSPUPFFR::from_file(&si_path("Si-fr.upf")).unwrap();

        // 6 (l, j) projectors: two with l = 0 and two (j = 1/2, 3/2) pairs with l = 1.
        assert_eq!(psp.get_nbeta(), 4);
        assert_eq!(psp.get_nbeta_soc(), 2);
        assert_eq!(psp.get_zion(), 4.0);
        for ibeta in 0..psp.get_nbeta() {
            assert_eq!(psp.get_beta(ibeta).len(), psp.get_mmax());
        }
    }

    #[test]
    fn test_scalar_file_is_an_error() {
        // A scalar-relativistic file has no PP_RELBETA data.
        let err = AtomPSPUPFFR::from_file(&si_path("Si-sr.upf"))
            .err()
            .unwrap();
        assert!(err.contains("PP_RELBETA"), "{}", err);
    }

    #[test]
    fn test_wrong_j_order_is_an_error() {
        // Give the first l = 1 projector j = 3/2: the pair is then (3/2, 3/2).
        let text = std::fs::read_to_string(si_path("Si-fr.upf")).unwrap();
        let text = text.replacen("lll=\"1\" jjj=\"0.5\"", "lll=\"1\" jjj=\"1.5\"", 1);
        let path =
            std::env::temp_dir().join(format!("dftworks-upffr-order-{}.upf", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let result = AtomPSPUPFFR::from_file(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();

        let err = result.err().unwrap();
        assert!(err.contains("PP_RELBETA.3"), "{}", err);
    }
}
