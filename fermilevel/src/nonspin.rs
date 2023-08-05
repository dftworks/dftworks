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

        // valence bands

        let nelec_ref = nelec * (1.0 - occ_inversion);

        let mut nelec_below = total_electrons_below(vkscf, vevals, fermi_level);

        let mut upper = fermi_level;
        let mut lower = fermi_level;

        while nelec_below < nelec_ref {
            upper += EPS2 * EV_TO_HA;
            nelec_below = total_electrons_below(vkscf, vevals, upper);
        }

        while nelec_below > nelec_ref {
            lower -= EPS2 * EV_TO_HA;
            nelec_below = total_electrons_below(vkscf, vevals, lower);
        }

        let mut vb_level = 0.0;

        while (nelec_below - nelec_ref).abs() > EPS12 {
            vb_level = (upper + lower) / 2.0;

            nelec_below = total_electrons_below(vkscf, vevals, vb_level);

            if nelec_below > nelec_ref {
                upper = vb_level;
            }

            if nelec_below < nelec_ref {
                lower = vb_level;
            }
        }

        // conduction bands

        let nelec_ref = nelec * occ_inversion;

        let mut upper = fermi_level;
        let mut lower = fermi_level;

        let mut nelec_between = 0.0;

        while nelec_between < nelec_ref {
            upper += EPS2 * EV_TO_HA;
            nelec_between = total_conduction_electrons_between(vkscf, vevals, upper, fermi_level);
        }

        let mut cb_level = 0.0;

        while (nelec_between - nelec_ref).abs() > EPS12 {
            cb_level = (upper + lower) / 2.0;

            nelec_between =
                total_conduction_electrons_between(vkscf, vevals, cb_level, fermi_level);

            if nelec_between > nelec_ref {
                upper = cb_level;
            }

            if nelec_between < nelec_ref {
                lower = cb_level;
            }
        }

        // set occupation numbers
        // For all energy states below vb_level and between fermi_level and cb_level,
        // it should be fully occupied (2 for nonspin, 1 for spin).

        for (ik, kscf) in vkscf.iter_mut().enumerate() {
            let evals = &vevals[ik];

            kscf.set_occ_inversion(evals, vb_level, fermi_level, cb_level);
        }
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

fn total_electrons_below(vkscf: &[KSCF], vevals: &Vec<Vec<f64>>, energy_level: f64) -> f64 {
    let mut ntot_local = 0.0;

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evals = &vevals[ik];

        ntot_local += kscf.get_total_valence_occ_below(evals, energy_level) * kscf.get_k_weight();
    }

    let ntot = ntot_local;

    ntot
}

fn total_conduction_electrons_between(
    vkscf: &[KSCF],
    vevals: &Vec<Vec<f64>>,
    upper_level: f64,
    lower_level: f64,
) -> f64 {
    let mut ntot_local = 0.0;

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evals = &vevals[ik];

        ntot_local += kscf.get_total_conduction_occ_between(evals, upper_level, lower_level)
            * kscf.get_k_weight();
    }

    let ntot = ntot_local;

    ntot
}
