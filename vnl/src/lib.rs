use atompsp::AtomPSP;
use crystal::Crystal;
use dwconsts::*;
use integral;
use pspot::PSPot;
use pwbasis::PWBasis;
use special;

use std::collections::HashMap;

pub struct VNL {
    _ik: usize,

    // HashMap<specie, Vec<projector_index -> beta_l(|k+G|)>>
    kgbeta_all: HashMap<String, Vec<Vec<f64>>>,
    // Derivative w.r.t. |k+G|^2 used by stress/nonlocal derivatives.
    dkgbeta_all: HashMap<String, Vec<Vec<f64>>>,
}

impl VNL {
    pub fn new(ik: usize, pots: &PSPot, pwwfc: &PWBasis, crystal: &Crystal) -> VNL {
        let latt = crystal.get_latt();

        let mut kgbeta_all = HashMap::new();

        let mut dkgbeta_all = HashMap::new();

        // Precompute projector radial transforms for all species at this k-point.
        let species = crystal.get_unique_species();

        for specie in species.iter() {
            let t_atpsp = pots.get_psp(specie);

            let kgbeta_one = atomic_super_position(t_atpsp, pwwfc, latt.volume());

            let dkgbeta_one = atomic_super_position_diff(t_atpsp, pwwfc, latt.volume());

            kgbeta_all.insert(specie.clone(), kgbeta_one);

            dkgbeta_all.insert(specie.clone(), dkgbeta_one);
        }

        VNL {
            _ik: ik,
            kgbeta_all,
            dkgbeta_all,
        }
    }

    pub fn get_kgbeta_all(&self) -> &HashMap<String, Vec<Vec<f64>>> {
        &self.kgbeta_all
    }

    pub fn get_dkgbeta_all(&self) -> &HashMap<String, Vec<Vec<f64>>> {
        &self.dkgbeta_all
    }
}

// Return one radial table per projector beta.
pub fn atomic_super_position(atpsp: &dyn AtomPSP, pwwfc: &PWBasis, volume: f64) -> Vec<Vec<f64>> {
    let rad = atpsp.get_rad();
    let rab = atpsp.get_rab();

    let mut kgbeta = Vec::new();

    let nbeta = atpsp.get_nbeta();

    let kg = pwwfc.get_kg();

    for i in 0..nbeta {
        let l = atpsp.get_lbeta(i);

        let beta = atpsp.get_beta(i);

        let t = compute_vnl_of_kg(kg, l, beta, rad, rab, volume);

        kgbeta.push(t);
    }

    kgbeta
}

// PRB, 51, 14697 (1995), Eq.11
fn compute_vnl_of_kg(
    kg: &[f64],
    l: usize,
    beta: &[f64],
    rad: &[f64],
    rab: &[f64],
    volume: f64,
) -> Vec<f64> {
    let npw = kg.len();

    let mut vg = vec![0.0; npw];

    let mmax = rad.len();

    let mut work = vec![0.0; mmax];

    // Normalization for radial transform to reciprocal projector form.
    let fact = FOURPI / volume.sqrt();

    for iw in 0..npw {
        for i in 0..mmax {
            let r = rad[i];

            work[i] = beta[i] * r * special::spherical_bessel_jn(l, kg[iw] * r);
        }

        //vg[iw] = fact * integral::simpson_log(&work, rad);
        vg[iw] = fact * integral::simpson_rab(&work, rab);
    }

    vg
}

// Derivative counterpart of `atomic_super_position`.
pub fn atomic_super_position_diff(
    atpsp: &dyn AtomPSP,
    pwwfc: &PWBasis,
    volume: f64,
) -> Vec<Vec<f64>> {
    let rad = atpsp.get_rad();
    let rab = atpsp.get_rab();

    let mut kgbeta = Vec::new();

    let nbeta = atpsp.get_nbeta();

    let kg = pwwfc.get_kg();

    for i in 0..nbeta {
        let l = atpsp.get_lbeta(i);

        let beta = atpsp.get_beta(i);

        let t = compute_dvnl_of_kg(kg, l, beta, rad, rab, volume);

        kgbeta.push(t);
    }

    kgbeta
}

// PRB, 51, 14697 (1995), Eq.13
fn compute_dvnl_of_kg(
    kg: &[f64],
    l: usize,
    beta: &[f64],
    rad: &[f64],
    rab: &[f64],
    volume: f64,
) -> Vec<f64> {
    let npw = kg.len();

    let mut vg = vec![0.0; npw];

    let mmax = rad.len();

    let mut work = vec![0.0; mmax];

    // Prefactor from analytical derivative identities of spherical Bessel basis.
    let fact = FOURPI / volume.sqrt() / (2.0 * l as f64 + 1.0);
    if l == 0 {
        for iw in 0..npw {
            for i in 0..mmax {
                let r = rad[i];
                // j_{l+1}
                let j1 = special::spherical_bessel_jn(l + 1, kg[iw] * r);

                work[i] = beta[i] * r * r * (l as f64 + 1.0) * j1;
            }

            vg[iw] = fact * integral::simpson_rab(&work, rab);
        }
    } else {
        for iw in 0..npw {
            for i in 0..mmax {
                let r = rad[i];
                // j_{l+1}
                let j1 = special::spherical_bessel_jn(l + 1, kg[iw] * r);

                // j_{l-1}
                let j2 = special::spherical_bessel_jn(l - 1, kg[iw] * r);

                work[i] = beta[i] * r * r * ((l as f64 + 1.0) * j1 - (l as f64) * j2);
            }

            vg[iw] = fact * integral::simpson_rab(&work, rab);
        }
    }

    vg
}
