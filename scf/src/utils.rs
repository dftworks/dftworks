#![allow(warnings)]

use super::hartree;
use control::*;
use crystal::Crystal;
use density::*;
use dfttypes::*;
use dwconsts::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::*;
use itertools::multizip;
use kpts::KPTS;
use kscf::KSCF;
use matrix::Matrix;
use mixing::Mixing;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::*;
use num_traits::Zero;
use pspot::PSPot;
use pwbasis::*;
use pwdensity::*;
use rayon::prelude::*;
use rgtransform::RGTransform;
use std::io::Write;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;
use xc::*;

const PARALLEL_MIN_LEN: usize = 8192;

#[inline]
fn use_parallel_for_len(len: usize) -> bool {
    len >= PARALLEL_MIN_LEN && rayon::current_num_threads() > 1
}

pub fn compute_v_hartree(pwden: &PWDensity, rhog: &RHOG, vhg: &mut [c64]) {
    if let RHOG::NonSpin(rhog) = rhog {
        hartree::potential(pwden.get_g(), rhog, vhg);
    }
}

pub fn display_parallel_runtime_info() {
    if !dwmpi::is_root() {
        return;
    }

    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let mpi_ranks = dwmpi::get_comm_world_size();

    println!(
        "     {:<width1$} = {:>4} (RAYON_NUM_THREADS={}, host_threads={}, mpi_ranks={})",
        "rayon_threads",
        rayon_threads,
        rayon_env,
        host_threads,
        mpi_ranks,
        width1 = OUT_WIDTH1
    );
}

// v_xc in r space first and then transform to G space; this will change with the density

pub fn compute_v_e_xc_of_r(
    xc: &dyn XC,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &mut VXCR,
    exc_3d: &mut Array3<c64>,
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        // rho_3d <-- rho_3d + rhocore_3d
        rho_3d.add_from(rhocore_3d);
    }

    // for lda, drho_3d = None
    let drho_3d = None;
    xc.potential_and_energy(rho_3d, drho_3d, vxc_3d, exc_3d);

    if let RHOR::NonSpin(rho_3d) = rho_3d {
        // rho_3d <-- rho_3d - rhocore_3d
        rho_3d.substract(rhocore_3d);
    }
}

pub fn compute_v_xc_of_g(
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    vxc_3d: &VXCR,
    vxcg: &mut VXCG,
) {
    let vxc_3d = vxc_3d.as_non_spin().unwrap();
    let vxcg = vxcg.as_non_spin_mut().unwrap();

    rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d.as_slice(), vxcg);
}

// v_xc + v_h + v_psloc in G space

pub fn add_up_v(vpslocg: &[c64], vhg: &[c64], vxcg: &VXCG, vlocg: &mut [c64]) {
    let vxcg = vxcg.as_non_spin().unwrap();
    debug_assert_eq!(vpslocg.len(), vlocg.len());
    debug_assert_eq!(vhg.len(), vlocg.len());
    debug_assert_eq!(vxcg.len(), vlocg.len());

    if use_parallel_for_len(vlocg.len()) {
        vlocg
            .par_iter_mut()
            .zip(vxcg.par_iter())
            .zip(vhg.par_iter())
            .zip(vpslocg.par_iter())
            .for_each(|(((v_loc, v_xc), v_ha), v_psloc)| {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            });
    } else {
        for (v_loc, v_xc, v_ha, v_psloc) in
            multizip((vlocg.iter_mut(), vxcg.iter(), vhg.iter(), vpslocg.iter()))
        {
            *v_loc = *v_xc + *v_ha + *v_psloc;
        }
    }
}

pub fn compute_rho_of_g(
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    rho_3d: &mut RHOR,
    rhog_out: &mut [c64],
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        rgtrans.r3d_to_g1d(gvec, pwden, rho_3d.as_slice(), rhog_out);
    }
}

pub fn compute_next_density(
    pwden: &PWDensity,
    mixing: &mut dyn Mixing,
    rhog_out: &[c64],
    rhog_diff: &mut [c64],
    rhog: &mut RHOG,
) {
    if let RHOG::NonSpin(rhog) = rhog {
        debug_assert_eq!(rhog_out.len(), rhog_diff.len());
        debug_assert_eq!(rhog.len(), rhog_diff.len());

        // mix old and new densities to get the density for the next iteration
        if use_parallel_for_len(rhog_diff.len()) {
            rhog_diff
                .par_iter_mut()
                .zip(rhog_out.par_iter())
                .zip(rhog.par_iter())
                .for_each(|((d, out), old)| {
                    *d = *out - *old;
                });
        } else {
            for ipw in 0..rhog_diff.len() {
                rhog_diff[ipw] = rhog_out[ipw] - rhog[ipw];
            }
        }

        mixing.compute_next_density(pwden.get_g(), rhog, rhog_diff);
    }
}

pub fn display_eigen_values(
    crystal: &Crystal,
    kpts: &dyn KPTS,
    vpwwfc: &[PWBasis],
    vkscf: &VKSCF,
    vkevals: &VKEigenValue,
) {
    let blatt = crystal.get_latt().reciprocal();

    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevals = vkevals.as_non_spin().unwrap();

    let rank = dwmpi::get_comm_world_rank();

    for irank in 0..dwmpi::get_comm_world_size() {
        dwmpi::barrier(MPI_COMM_WORLD);

        if irank == rank {
            for (ik, evals) in t_vkevals.iter().enumerate() {
                let g_ik = t_vkscf[ik].get_ik();
                let k_frac = kpts.get_k_frac(g_ik);
                let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                let npw_wfc = vpwwfc[ik].get_n_plane_waves();

                print_k_point(g_ik, k_frac, k_cart, npw_wfc);

                let occ = t_vkscf[ik].get_occ();

                print_eigen_values(evals, occ);
            }

            std::io::stdout().flush();
        }
    }

    dwmpi::barrier(MPI_COMM_WORLD);
}

pub fn print_eigen_values(v: &[f64], occ: &[f64]) {
    println!();

    for (i, elem) in v.iter().enumerate() {
        println!(
            "       {:<6} {:16.6} {:12.6}",
            i + 1,
            elem * HA_TO_EV,
            occ[i]
        );
    }
}

pub fn print_k_point(ik: usize, xk_frac: Vector3f64, xk_cart: Vector3f64, npw_wfc: usize) {
    println!();

    println!("   kpoint-{} npws = {}", ik + 1, npw_wfc);

    println!(
        "     k_frac = [ {:.8}, {:.8}, {:.8} ]",
        xk_frac.x, xk_frac.y, xk_frac.z
    );

    println!(
        "     k_cart = [ {:.8}, {:.8}, {:.8} ] (1/a0)",
        xk_cart.x, xk_cart.y, xk_cart.z
    );
}

pub fn compute_force(
    control: &Control,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    pots: &PSPot,
    ewald: &Ewald,
    vkscf: &VKSCF,
    vkevecs: &VKEigenVector,
    rhog: &RHOG,
    vxcg: &VXCG,
    symdrv: &dyn SymmetryDriver,
    force_total: &mut Vec<Vector3f64>,
) {
    let natoms = crystal.get_n_atoms();

    let mut force_loc = vec![Vector3f64::zeros(); natoms];
    let mut force_vnl_local = vec![Vector3f64::zeros(); natoms];
    let mut force_vnl = vec![Vector3f64::zeros(); natoms];

    force::vpsloc(
        pots,
        crystal,
        gvec,
        pwden,
        rhog.as_non_spin().unwrap(),
        &mut force_loc,
    );

    force::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
        &mut force_vnl_local,
    );

    dwmpi::reduce_slice_sum(
        vector3::as_slice_of_element(&force_vnl_local),
        vector3::as_mut_slice_of_element(&mut force_vnl),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(
        vector3::as_mut_slice_of_element(&mut force_vnl),
        MPI_COMM_WORLD,
    );

    let force_ewald = ewald.get_force();

    let mut force_nlcc = vec![Vector3f64::zeros(); natoms];

    force::nlcc_xc(
        pots,
        crystal,
        gvec,
        pwden,
        vxcg.as_non_spin().unwrap(),
        &mut force_nlcc,
    );

    for iat in 0..natoms {
        force_total[iat] = force_ewald[iat] + force_loc[iat] + force_vnl[iat] + force_nlcc[iat];
    }

    if dwmpi::is_root() {
        force::display(
            crystal,
            &force_total,
            &force_ewald,
            &force_loc,
            &force_vnl,
            &force_nlcc,
        );
    }
}

pub fn compute_stress(
    control: &Control,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    pots: &PSPot,
    ewald: &Ewald,
    vkscf: &VKSCF,
    vkevecs: &VKEigenVector,
    rhog: &RHOG,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxcg: &VXCG,
    vxc_3d: &VXCR,
    exc_3d: &Array3<c64>,
    symdrv: &dyn SymmetryDriver,
    stress_total: &mut Matrix<f64>,
) {
    let mut stress_kin_local = stress::kinetic(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_kin = Matrix::new(3, 3);

    dwmpi::reduce_slice_sum(
        stress_kin_local.as_slice(),
        stress_kin.as_mut_slice(),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(stress_kin.as_mut_slice(), MPI_COMM_WORLD);

    let mut stress_vnl_local = stress::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_vnl = Matrix::new(3, 3);

    dwmpi::reduce_slice_sum(
        stress_vnl_local.as_slice(),
        stress_vnl.as_mut_slice(),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(stress_vnl.as_mut_slice(), MPI_COMM_WORLD);

    let stress_hartree = stress::hartree(gvec, pwden, rhog.as_non_spin().unwrap());
    let stress_loc = stress::vpsloc(pots, crystal, gvec, pwden, rhog.as_non_spin().unwrap());

    let stress_xc = stress::xc(
        crystal.get_latt(),
        rho_3d.as_non_spin_mut().unwrap(),
        rhocore_3d,
        vxc_3d.as_non_spin().unwrap(),
        &exc_3d,
    );

    let stress_xc_nlcc = stress::nlcc_xc(pots, crystal, gvec, pwden, vxcg.as_non_spin().unwrap());

    let stress_ewald = ewald.get_stress();

    for i in 0..3 {
        for j in 0..3 {
            stress_total[[i, j]] = stress_kin[[i, j]]
                + stress_hartree[[i, j]]
                + stress_xc[[i, j]]
                + stress_xc_nlcc[[i, j]]
                + stress_loc[[i, j]]
                + stress_vnl[[i, j]]
                + stress_ewald[[i, j]];
        }
    }

    if dwmpi::is_root() {
        stress::display_stress_by_parts(
            &stress_kin,
            &stress_hartree,
            &stress_xc,
            &stress_xc_nlcc,
            &stress_loc,
            &stress_vnl,
            &stress_ewald,
            &stress_total,
        );
    }
}

pub fn get_n_plane_waves_max(vpwwfc: &[PWBasis]) -> usize {
    let mut npw_max = 0;
    for pwwfc in vpwwfc.iter() {
        let npw = pwwfc.get_n_plane_waves();

        if npw > npw_max {
            npw_max = npw;
        }
    }

    npw_max
}

pub fn solve_eigen_equations(
    rgtrans: &RGTransform,
    vloc_3d: &Array3<c64>,
    eigvalue_epsilon: f64,
    geom_iter: usize,
    scf_iter: usize,
    vkscf: &VKSCF,
    vkevals: &mut VKEigenValue,
    vkevecs: &mut VKEigenVector,
) {
    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevecs = vkevecs.as_non_spin_mut().unwrap();
    let t_vkevals = vkevals.as_non_spin_mut().unwrap();

    for (ik, kscf) in t_vkscf.iter().enumerate() {
        kscf.run(
            rgtrans,
            vloc_3d,
            eigvalue_epsilon,
            geom_iter,
            scf_iter,
            &mut t_vkevals[ik],
            &mut t_vkevecs[ik],
        );
    }
}

pub fn get_eigvalue_epsilon(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
) -> f64 {
    let mut eig_epsilon: f64;

    //if control.is_band() {
    if control.get_scf_max_iter() <= 1 {
        eig_epsilon = control.get_eigval_epsilon();
    } else {
        if geom_iter == 1 {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS3 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon =
                        (EPS3 * EV_TO_HA).min(0.0001 * energy_diff / (1.0_f64).max(ntot_elec));

                    eig_epsilon = eig_epsilon
                        .max(EPS13 * EV_TO_HA)
                        .min(control.get_eigval_epsilon());
                }
            }
        } else {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS6 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon = (EPS11 * EV_TO_HA)
                        .min(energy_diff / (npw_wfc as f64).powf(1.0) / (1.0_f64).max(ntot_elec));
                }
            }
        }

        eig_epsilon = eig_epsilon.max(EPS16 * EV_TO_HA);
    }

    eig_epsilon
}

pub fn compute_total_energy(
    pwden: &PWDensity,
    crystal: &Crystal,
    rhog: &[c64],
    vkscf: &[KSCF],
    vevals: &[Vec<f64>],
    rho_3d: &mut Array3<c64>,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
    vxc_3d: &Array3<c64>,
    ew_total: f64,
) -> f64 {
    let latt = crystal.get_latt();

    let etot_hartree = energy::hartree(pwden, latt, rhog);

    // bands energy

    let etot_bands_local = get_bands_energy(vkscf, vevals);

    let mut etot_bands = 0.0;

    dwmpi::reduce_scalar_sum(&etot_bands_local, &mut etot_bands, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut etot_bands, MPI_COMM_WORLD);

    //

    let etot_xc = energy::exc(latt, &rho_3d, &rhocore_3d, &exc_3d);
    let etot_vxc = energy::vxc(
        latt,
        rho_3d.as_slice(),
        rhocore_3d.as_slice(),
        vxc_3d.as_slice(),
    );

    let etot_one = etot_bands - etot_vxc - 2.0 * etot_hartree;

    let etot = etot_one + etot_xc + etot_hartree + ew_total;

    etot
}

// In QE
// hwf_energy = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband_hwf
// etot       = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband + descf

fn get_bands_energy(vkscf: &[KSCF], vevals: &[Vec<f64>]) -> f64 {
    let etot_bands = energy::band_structure(vkscf, vevals);

    etot_bands
}
