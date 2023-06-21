#![allow(warnings)]

use crate::FermiLevel;
use dfttypes::*;
use dwconsts::*;
use kscf::KSCF;

pub struct FermiLevelNonspin {}

impl FermiLevelNonspin {
    pub fn new() -> FermiLevelNonspin {
        FermiLevelNonspin {}
    }
}

impl FermiLevel for FermiLevelNonspin {
    fn get_fermi_level(&self, vkscf: &mut VKSCF, nelec: f64, vevals: &VKEigenValue) -> f64 {
        let vevals = vevals.as_non_spin().unwrap();
        let vkscf = vkscf.as_non_spin_mut().unwrap();

        let mut fermi_level = get_initial_fermi_level(nelec, vevals);

        let mut total_electrons = |fermi| get_total_electrons(vkscf, vevals, fermi);

        let mut ntot = total_electrons(fermi_level);

        let mut upper = fermi_level;
        let mut lower = fermi_level;

        while ntot < nelec {
            upper += EPS2 * EV_TO_HA; // convert ev to ha
            ntot = total_electrons(upper);
        }

        while ntot > nelec {
            lower -= EPS2 * EV_TO_HA; // convert ev to ha
            ntot = total_electrons(lower);
        }

        while (ntot - nelec).abs() > EPS12 {
            fermi_level = (upper + lower) / 2.0;
            ntot = total_electrons(fermi_level);

            if ntot > nelec {
                upper = fermi_level;
            }

            if ntot < nelec {
                lower = fermi_level;
            }

            //println!("Ef = {}, ntot = {}", fermi_level, ntot);
        }

        fermi_level
    }

    fn set_occ(
        &self,
        vkscf: &mut VKSCF,
        nelec: f64,
        vevals: &VKEigenValue,
        fermi_level: f64,
        occ_inversion: f64,
    ) {
        if occ_inversion < EPS10 {
            return;
        }

        let vevals = vevals.as_non_spin().unwrap();
        let vkscf = vkscf.as_non_spin_mut().unwrap();

        let total_electrons_below = |energy_level| {
            let mut ntot_local = 0.0;

            for (ik, kscf) in vkscf.iter().enumerate() {
                let evals = &vevals[ik];

                ntot_local += kscf.get_total_occ_below(evals, energy_level) * kscf.get_k_weight();
            }

            let ntot = ntot_local;

            ntot
        };

        let mut ntot = total_electrons_below(fermi_level);

        let mut upper = fermi_level;
        let mut lower = fermi_level;

        // valence bands

        let nelec_occ = nelec * (1.0 - occ_inversion);

        while ntot < nelec_occ {
            upper += EPS2 * EV_TO_HA;
            ntot = total_electrons_below(upper);
        }

        while ntot > nelec_occ {
            lower -= EPS2 * EV_TO_HA;
            ntot = total_electrons_below(lower);
        }

        let mut vbm_level = 0.0;

        while (ntot - nelec_occ).abs() > EPS12 {
            vbm_level = (upper + lower) / 2.0;

            ntot = total_electrons_below(vbm_level);

            if ntot > nelec_occ {
                upper = vbm_level;
            }

            if ntot < nelec_occ {
                lower = vbm_level;
            }
        }

        // conduction bands

	// set occupation numbers
    }
}

fn get_initial_fermi_level(nelec: f64, vevals: &Vec<Vec<f64>>) -> f64 {
    let nvbands = (nelec / 2.0) as usize;

    let mut homo_local = -std::f64::INFINITY;
    let mut lumo_local = std::f64::INFINITY;

    let ib_homo = nvbands - 1;
    let ib_lumo = nvbands;

    for ik in 0..vevals.len() {
        let evals = &vevals[ik];

        if homo_local < evals[ib_homo] {
            homo_local = evals[ib_homo];
        }

        if lumo_local > evals[ib_lumo] {
            lumo_local = evals[ib_lumo];
        }
    }

    let homo = homo_local;
    let lumo = lumo_local;

    let fermi = (homo + lumo) / 2.0;

    fermi
}

pub fn get_total_electrons(vkscf: &mut [KSCF], vevals: &Vec<Vec<f64>>, fermi: f64) -> f64 {
    let mut ntot_local = 0.0;

    for (ik, kscf) in vkscf.iter_mut().enumerate() {
        let evals = &vevals[ik];

        kscf.compute_occ(fermi, evals);

        ntot_local += kscf.get_total_occ() * kscf.get_k_weight();
    }

    let ntot = ntot_local;

    ntot
}
