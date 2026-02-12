#![allow(warnings)]

use crate::FermiLevel;
use dfttypes::*;
use dwconsts::*;
use kscf::KSCF;
use mpi_sys::MPI_COMM_WORLD;

pub struct FermiLevelSpin {}

impl FermiLevelSpin {
    pub fn new() -> FermiLevelSpin {
        FermiLevelSpin {}
    }
}

impl FermiLevel for FermiLevelSpin {
    fn get_fermi_level(&self, vkscf: &mut VKSCF, nelec: f64, vevals: &VKEigenValue) -> f64 {
        let (vevals_up, vevals_dn) = vevals.as_spin().unwrap();
        let (vkscf_up, vkscf_dn) = vkscf.as_spin_mut().unwrap();

        let fermi_level_up = get_initial_fermi_level(nelec, vevals_up);
        let fermi_level_dn = get_initial_fermi_level(nelec, vevals_dn);

        let mut fermi_level = (fermi_level_up + fermi_level_dn) / 2.0;

        let mut total_electrons_up = |fermi| get_total_electrons(vkscf_up, vevals_up, fermi);
        let mut total_electrons_dn = |fermi| get_total_electrons(vkscf_dn, vevals_dn, fermi);

        let mut ntot_up = total_electrons_up(fermi_level);
        let mut ntot_dn = total_electrons_dn(fermi_level);

        let mut ntot = ntot_up + ntot_dn;

        let mut upper = fermi_level;
        let mut lower = fermi_level;

        while ntot < nelec {
            upper += EPS2 * EV_TO_HA; // convert ev to ha
            ntot_up = total_electrons_up(upper);
            ntot_dn = total_electrons_dn(upper);
            ntot = ntot_up + ntot_dn;
        }

        while ntot > nelec {
            lower -= EPS2 * EV_TO_HA; // convert ev to ha
            ntot_up = total_electrons_up(lower);
            ntot_dn = total_electrons_dn(lower);
            ntot = ntot_up + ntot_dn;
        }

        while (ntot - nelec).abs() > EPS12 {
            fermi_level = (upper + lower) / 2.0;
            ntot_up = total_electrons_up(fermi_level);
            ntot_dn = total_electrons_dn(fermi_level);
            ntot = ntot_up + ntot_dn;

            if ntot > nelec {
                upper = fermi_level;
            }

            if ntot < nelec {
                lower = fermi_level;
            }
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
    ) -> Option<f64> {
        if occ_inversion < EPS10 {
            return None;
        }

        let _vkscf = vkscf.as_spin_mut().unwrap();

        Some(nelec)
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

    let mut homo = 0.0;
    let mut lumo = 0.0;

    dwmpi::reduce_scalar_max(&homo_local, &mut homo, MPI_COMM_WORLD);
    dwmpi::reduce_scalar_min(&lumo_local, &mut lumo, MPI_COMM_WORLD);

    let mut fermi = (homo + lumo) / 2.0;
    dwmpi::bcast_scalar(&mut fermi, MPI_COMM_WORLD);

    fermi
}

pub fn get_total_electrons(vkscf: &mut [KSCF], vevals: &Vec<Vec<f64>>, fermi: f64) -> f64 {
    let mut ntot_local = 0.0;

    for (ik, kscf) in vkscf.iter_mut().enumerate() {
        let evals = &vevals[ik];

        kscf.compute_occ(fermi, evals);

        ntot_local += kscf.get_total_occ() * kscf.get_k_weight();
    }

    let mut ntot = 0.0;
    dwmpi::reduce_scalar_sum(&ntot_local, &mut ntot, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut ntot, MPI_COMM_WORLD);

    ntot
}
