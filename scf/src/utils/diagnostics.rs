#![allow(warnings)]

use control::{Control, VerbosityLevel};
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use kpts::KPTS;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::Array3;
use pwbasis::PWBasis;
use rgtransform::RGTransform;
use std::io::Write;
use types::c64;
use vector3::Vector3f64;

pub fn display_eigen_values(
    verbosity: VerbosityLevel,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    vpwwfc: &[PWBasis],
    vkscf: &VKSCF,
    vkevals: &VKEigenValue,
) {
    if matches!(verbosity, VerbosityLevel::Quiet) {
        return;
    }

    let blatt = crystal.get_latt().reciprocal();

    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevals = vkevals.as_non_spin().unwrap();

    let rank = dwmpi::get_comm_world_rank();
    let ordered_rank_output = verbosity >= VerbosityLevel::Verbose;

    if ordered_rank_output {
        for irank in 0..dwmpi::get_comm_world_size() {
            dwmpi::barrier(MPI_COMM_WORLD);

            if irank == rank {
                debug_assert_eq!(t_vkscf.len(), t_vkevals.len());
                debug_assert_eq!(t_vkscf.len(), vpwwfc.len());

                for (kscf_k, evals, pwwfc_k) in itertools::multizip((
                    t_vkscf.iter(),
                    t_vkevals.iter(),
                    vpwwfc.iter(),
                )) {
                    let g_ik = kscf_k.get_ik();
                    let k_frac = kpts.get_k_frac(g_ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let npw_wfc = pwwfc_k.get_n_plane_waves();

                    print_k_point(g_ik, k_frac, k_cart, npw_wfc);

                    let occ = kscf_k.get_occ();
                    print_eigen_values(evals, occ);
                }

                std::io::stdout().flush();
            }
        }

        dwmpi::barrier(MPI_COMM_WORLD);
    } else if dwmpi::is_root() {
        // Production path: avoid rank-serialized eigenvalue output.
        // Ordered rank-by-rank dumps are available with verbosity=verbose/debug.
        debug_assert_eq!(t_vkscf.len(), t_vkevals.len());
        debug_assert_eq!(t_vkscf.len(), vpwwfc.len());

        for (kscf_k, evals, pwwfc_k) in itertools::multizip((
            t_vkscf.iter(),
            t_vkevals.iter(),
            vpwwfc.iter(),
        )) {
            let g_ik = kscf_k.get_ik();
            let k_frac = kpts.get_k_frac(g_ik);
            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
            let npw_wfc = pwwfc_k.get_n_plane_waves();

            print_k_point(g_ik, k_frac, k_cart, npw_wfc);

            let occ = kscf_k.get_occ();
            print_eigen_values(evals, occ);
        }

        std::io::stdout().flush();
    }
}

pub fn display_spin_eigen_values(
    verbosity: VerbosityLevel,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    vpwwfc: &[PWBasis],
    vkscf: &VKSCF,
    vkevals: &VKEigenValue,
) {
    if matches!(verbosity, VerbosityLevel::Quiet) {
        return;
    }

    let blatt = crystal.get_latt().reciprocal();
    let ordered_rank_output = verbosity >= VerbosityLevel::Verbose;
    let rank = dwmpi::get_comm_world_rank();

    if ordered_rank_output {
        for irank in 0..dwmpi::get_comm_world_size() {
            dwmpi::barrier(MPI_COMM_WORLD);

            if irank == rank {
                if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
                    if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
                        debug_assert_eq!(vkscf_up.len(), vkscf_dn.len());
                        debug_assert_eq!(vkscf_up.len(), vkevals_up.len());
                        debug_assert_eq!(vkscf_up.len(), vkevals_dn.len());
                        debug_assert_eq!(vkscf_up.len(), vpwwfc.len());

                        for (kscf_up_k, kscf_dn_k, evals_up, evals_dn, pwwfc_k) in
                            itertools::multizip((
                                vkscf_up.iter(),
                                vkscf_dn.iter(),
                                vkevals_up.iter(),
                                vkevals_dn.iter(),
                                vpwwfc.iter(),
                            ))
                        {
                            let ik_global = kscf_up_k.get_ik();
                            let k_frac = kpts.get_k_frac(ik_global);
                            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                            let npw_wfc = pwwfc_k.get_n_plane_waves();

                            print_k_point(ik_global, k_frac, k_cart, npw_wfc);

                            let occ_up = kscf_up_k.get_occ();
                            let occ_dn = kscf_dn_k.get_occ();

                            print_spin_eigen_values(evals_up, occ_up, evals_dn, occ_dn);
                        }
                    }
                }

                std::io::stdout().flush();
            }
        }

        dwmpi::barrier(MPI_COMM_WORLD);
    } else if dwmpi::is_root() {
        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
                debug_assert_eq!(vkscf_up.len(), vkscf_dn.len());
                debug_assert_eq!(vkscf_up.len(), vkevals_up.len());
                debug_assert_eq!(vkscf_up.len(), vkevals_dn.len());
                debug_assert_eq!(vkscf_up.len(), vpwwfc.len());

                for (kscf_up_k, kscf_dn_k, evals_up, evals_dn, pwwfc_k) in itertools::multizip((
                    vkscf_up.iter(),
                    vkscf_dn.iter(),
                    vkevals_up.iter(),
                    vkevals_dn.iter(),
                    vpwwfc.iter(),
                )) {
                    let ik_global = kscf_up_k.get_ik();
                    let k_frac = kpts.get_k_frac(ik_global);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let npw_wfc = pwwfc_k.get_n_plane_waves();

                    print_k_point(ik_global, k_frac, k_cart, npw_wfc);

                    let occ_up = kscf_up_k.get_occ();
                    let occ_dn = kscf_dn_k.get_occ();

                    print_spin_eigen_values(evals_up, occ_up, evals_dn, occ_dn);
                }

                std::io::stdout().flush();
            }
        }
    }
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

pub fn print_spin_eigen_values(v_up: &[f64], occ_up: &[f64], v_dn: &[f64], occ_dn: &[f64]) {
    println!();

    for (i, _elem) in v_up.iter().enumerate() {
        println!(
            "       {:<6} {:16.6} {:12.6} {:16.6} {:12.6}",
            i + 1,
            v_up[i] * HA_TO_EV,
            occ_up[i],
            v_dn[i] * HA_TO_EV,
            occ_dn[i]
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

#[derive(Clone, Copy)]
enum EigvalueEpsilonFlavor {
    NonSpin,
    Spin,
}

fn get_eigvalue_epsilon_by_flavor(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
    flavor: EigvalueEpsilonFlavor,
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
                    let geom1_floor = match flavor {
                        EigvalueEpsilonFlavor::NonSpin => EPS3,
                        EigvalueEpsilonFlavor::Spin => EPS4,
                    };
                    eig_epsilon = (geom1_floor * EV_TO_HA)
                        .min(0.0001 * energy_diff / (1.0_f64).max(ntot_elec));

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

pub fn get_eigvalue_epsilon(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
) -> f64 {
    get_eigvalue_epsilon_by_flavor(
        geom_iter,
        scf_iter,
        control,
        ntot_elec,
        energy_diff,
        npw_wfc,
        EigvalueEpsilonFlavor::NonSpin,
    )
}

pub fn get_eigvalue_epsilon_spin(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
) -> f64 {
    get_eigvalue_epsilon_by_flavor(
        geom_iter,
        scf_iter,
        control,
        ntot_elec,
        energy_diff,
        npw_wfc,
        EigvalueEpsilonFlavor::Spin,
    )
}
