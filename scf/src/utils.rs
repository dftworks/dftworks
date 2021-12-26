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
use ndarray::*;
use num_traits::Zero;
use pspot::PSPot;
use pwbasis::*;
use pwdensity::*;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;
use xc::*;

pub fn compute_v_hartree(pwden: &PWDensity, rhog: &RHOG, vhg: &mut [c64]) {
    if let RHOG::NonSpin(rhog) = rhog {
        hartree::potential(pwden.get_g(), rhog, vhg);
    }
}

// v_xc in r space first and then transform to G space; this will change with the density

pub fn compute_v_e_xc_of_r(
    xc: &Box<dyn XC>,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &mut VXCR,
    exc_3d: &mut Array3<c64>,
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        // rho_3d <-- rho_3d + rhocore_3d
        rho_3d.add_from(rhocore_3d);
    }

    xc.potential_and_energy(rho_3d, vxc_3d, exc_3d);

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
    for (v_loc, v_xc, v_ha, v_psloc) in multizip((
        vlocg.iter_mut(),
        vxcg.as_non_spin().unwrap().iter(),
        vhg.iter(),
        vpslocg.iter(),
    )) {
        *v_loc = *v_xc + *v_ha + *v_psloc;
    }
}

pub fn compute_and_symmetrize_rho_of_g(
    control: &Control,
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    kpts: &Box<dyn KPTS>,
    density_driver: &Box<dyn Density>,
    symdrv: &Box<dyn SymmetryDriver>,
    rho_3d: &mut RHOR,
    rhog_out: &mut [c64],
    fftgrid: &FFTGrid,
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        rgtrans.r3d_to_g1d(gvec, pwden, rho_3d.as_slice(), rhog_out);
    }
}

pub fn compute_next_density(
    pwden: &PWDensity,
    mixing: &mut Box<dyn Mixing>,
    rhog_out: &[c64],
    rhog: &mut RHOG,
) {
    let npw_rho = rhog.as_non_spin().unwrap().len();
    let mut rhog_diff = vec![c64::zero(); npw_rho];

    if let RHOG::NonSpin(rhog) = rhog {
        // mix old and new densities to get the density for the next iteration
        for ipw in 0..npw_rho {
            rhog_diff[ipw] = rhog_out[ipw] - rhog[ipw];
        }

        mixing.compute_next_density(pwden.get_g(), rhog, &rhog_diff);
    }
}

pub fn display_eigen_values(
    crystal: &Crystal,
    kpts: &Box<dyn KPTS>,
    vpwwfc: &[PWBasis],
    vkscf: &VKSCF,
    vkevals: &VKEigenValue,
) {
    let blatt = crystal.get_latt().reciprocal();

    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevals = vkevals.as_non_spin().unwrap();

    for (ik, evals) in t_vkevals.iter().enumerate() {
        let k_frac = kpts.get_k_frac(ik);
        let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
        let npw_wfc = vpwwfc[ik].get_n_plane_waves();

        print_k_point(ik, k_frac, k_cart, npw_wfc);

        let occ = t_vkscf[ik].get_occ();

        print_eigen_values(evals, occ);
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
    symdrv: &Box<dyn SymmetryDriver>,
    force_total: &mut Vec<Vector3f64>,
) {
    let natoms = crystal.get_n_atoms();

    let mut force_loc = vec![Vector3f64::zeros(); natoms];
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
        &mut force_vnl,
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

    force::display(
        crystal,
        &force_total,
        &force_ewald,
        &force_loc,
        &force_vnl,
        &force_nlcc,
    );
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
    symdrv: &Box<dyn SymmetryDriver>,
    stress_total: &mut Matrix<f64>,
) {
    let mut stress_kin = stress::kinetic(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_vnl = stress::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

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
    crystal: &Crystal,
    fftgrid: &FFTGrid,
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

    let nkpt = t_vkscf.len();

    let mut vk_n_band_converged = vec![0; nkpt];
    let mut vk_n_hpsi = vec![0; nkpt];

    for (ik, kscf) in t_vkscf.iter().enumerate() {
        let (n_band_converged, n_hpsi) = kscf.run(
            crystal,
            &fftgrid,
            rgtrans,
            &vloc_3d,
            eigvalue_epsilon,
            geom_iter,
            scf_iter,
            &mut t_vkevals[ik],
            &mut t_vkevecs[ik],
        );

        vk_n_band_converged[ik] = n_band_converged;
        vk_n_hpsi[ik] = n_hpsi;
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
                        (EPS2 * EV_TO_HA).min(0.001 * energy_diff / (1.0_f64).max(ntot_elec));

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
    vevals: &Vec<Vec<f64>>,
    rho_3d: &mut Array3<c64>,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
    vxc_3d: &Array3<c64>,
    ew_total: f64,
) -> f64 {
    let latt = crystal.get_latt();

    let etot_hartree = energy::hartree(pwden, latt, rhog);

    let etot_bands = get_bands_energy(vkscf, vevals);

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

fn get_bands_energy(vkscf: &[KSCF], vevals: &Vec<Vec<f64>>) -> f64 {
    let etot_bands = energy::band_structure(vkscf, vevals);

    etot_bands
}
