//#![allow(warnings)]

use super::utils;
use crate::SCF;
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;

//use rayon::prelude::*;

pub struct SCFNonspin {}

impl SCFNonspin {
    pub fn new() -> SCFNonspin {
        SCFNonspin {}
    }
}

impl SCF for SCFNonspin {
    fn run(
        &self,
        geom_iter: usize,
        control: &Control,
        crystal: &Crystal,
        gvec: &GVector,
        pwden: &PWDensity,
        pots: &PSPot,
        rgtrans: &RGTransform,
        kpts: &Box<dyn KPTS>,
        ewald: &Ewald,
        vpwwfc: &[PWBasis],
        vkscf: &mut VKSCF,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
        rhocore_3d: &Array3<c64>,
        vkevals: &mut VKEigenValue,
        vkevecs: &mut VKEigenVector,
        symdrv: &Box<dyn SymmetryDriver>,
        stress_total: &mut Matrix<f64>,
        force_total: &mut Vec<Vector3f64>,
    ) {
        //println!("");
        //println!("   {:*^120}", " self-consistent field ");

        let density_driver = density::new(control.get_spin_scheme());

        //

        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());

        let fft_shape = fftgrid.get_size();
        let npw_rho = pwden.get_n_plane_waves();

        //

        let mut vhg = vec![c64::zero(); npw_rho];

        //

        let xc = xc::new(control.get_xc_scheme());

        let mut vxcg = VXCG::NonSpin(vec![c64::zero(); npw_rho]);

        let mut vxc_3d = VXCR::NonSpin(Array3::<c64>::new(fft_shape));

        let mut exc_3d = Array3::<c64>::new(fft_shape);

        // v_psloc in G space; this will not change for a fixed set of ion positions

        let mut vpslocg = vec![c64::zero(); npw_rho];
        vloc::from_atomic_super_position(pots, crystal, gvec, pwden, &mut vpslocg);

        // v_ha and v_xc change with density

        utils::compute_v_hartree(pwden, rhog, &mut vhg);

        utils::compute_v_e_xc_of_r(&xc, rho_3d, rhocore_3d, &mut vxc_3d, &mut exc_3d);

        utils::compute_v_xc_of_g(gvec, pwden, rgtrans, &vxc_3d, &mut vxcg);

        //

        let mut vlocg = vec![c64::zero(); npw_rho];
        utils::add_up_v(&vpslocg, &vhg, &vxcg, &mut vlocg);

        // density mixing

        let mut mixing = mixing::new(control);

        //

        let mut vloc_3d = Array3::<c64>::new(fft_shape);

        let ntot_elec = crystal.get_n_total_electrons(pots);

        let mut rhog_out = vec![c64::zero(); npw_rho];

        let mut energy_diff = 0.0;

        let npw_wfc_max = utils::get_n_plane_waves_max(vpwwfc);

        let fermi_driver = fermilevel::new(control.get_spin_scheme());

        //

        let mut scf_iter = 1;

        println!(
            "    {:>3}  {:>10} {:>10} {:>16} {:>25} {:>25} {:>12}",
            "", "eps(eV)", "Fermi(eV)", "charge", "Eharris(Ry)", "Escf(Ry)", "dE(eV)"
        );

        loop {
            // transform v_loc from G to r

            rgtrans.g1d_to_r3d(gvec, pwden, &vlocg, vloc_3d.as_mut_slice());

            //

            let eigvalue_epsilon = utils::get_eigvalue_epsilon(
                geom_iter,
                scf_iter,
                control,
                ntot_elec,
                energy_diff,
                npw_wfc_max,
            );

            utils::solve_eigen_equations(
                crystal,
                &fftgrid,
                rgtrans,
                &vloc_3d,
                eigvalue_epsilon,
                geom_iter,
                scf_iter,
                vkscf,
                vkevals,
                vkevecs,
            );

            // calculate Fermi level; vkscf has to be &mut since occ will be modified

            let fermi_level = fermi_driver.get_fermi_level(vkscf, ntot_elec, &vkevals);

            // calculate Harris energy

            let energy_harris = utils::compute_total_energy(
                pwden,
                crystal,
                rhog.as_non_spin().unwrap(),
                &vkscf.as_non_spin().unwrap(),
                &vkevals.as_non_spin().unwrap(),
                rho_3d.as_non_spin_mut().unwrap(),
                rhocore_3d,
                &exc_3d,
                vxc_3d.as_non_spin().unwrap(),
                ewald.get_energy(),
            );

            // compute rho r by building the density based on the new wavefunctions

            density_driver.compute_charge_density(
                &vkscf,
                rgtrans,
                &vkevecs,
                crystal.get_latt().volume(),
                rho_3d,
            );

            let charge = rho_3d.as_non_spin().unwrap().sum().re * crystal.get_latt().volume()
                / fftgrid.get_ntotf64();

            // rho r -> G

            utils::compute_and_symmetrize_rho_of_g(
                control,
                gvec,
                pwden,
                rgtrans,
                kpts,
                &density_driver,
                symdrv,
                rho_3d,
                &mut rhog_out,
                &fftgrid,
            );

            // up to here
            // rhog_out: out rho in G
            // rho_3d:   out rho in r

            utils::compute_v_e_xc_of_r(&xc, rho_3d, rhocore_3d, &mut vxc_3d, &mut exc_3d);

            // calculate scf energy

            let energy_scf = utils::compute_total_energy(
                pwden,
                crystal,
                &rhog_out,
                &vkscf.as_non_spin().unwrap(),
                &vkevals.as_non_spin().unwrap(),
                rho_3d.as_non_spin_mut().unwrap(),
                rhocore_3d,
                &exc_3d,
                vxc_3d.as_non_spin().unwrap(),
                ewald.get_energy(),
            );

            energy_diff = (energy_scf - energy_harris).abs();

            //

            println!(
                "    {:>3}: {:>10.3E} {:>10.3E} {:>16.6E} {:>25.12E} {:>25.12E} {:>12.3E}",
                scf_iter,
                eigvalue_epsilon * HA_TO_EV,
                fermi_level * HA_TO_EV,
                charge,
                energy_harris * HA_TO_RY,
                energy_scf * HA_TO_RY,
                energy_diff * HA_TO_EV
            );

            /////////////////////////////////////////////////
            // check convergence

            // if converged, then exit

            if energy_diff < control.get_energy_epsilon() {
                println!(
                    "\n     {:<width1$}",
                    "scf_convergence_success",
                    width1 = OUT_WIDTH1
                );

                break;
            }

            // if not converged, but exceed the max_scf, then exit

            if scf_iter == control.get_scf_max_iter() {
                println!(
                    "\n     {:<width1$}",
                    "scf_convergence_failure",
                    width1 = OUT_WIDTH1
                );

                break;
            }

            /////////////////////////////////////////////////

            // rho in G space

            utils::compute_next_density(pwden, &mut mixing, &rhog_out, rhog);

            // rho  in r space

            rgtrans.g1d_to_r3d(
                gvec,
                pwden,
                rhog.as_non_spin().unwrap(),
                rho_3d.as_non_spin_mut().unwrap().as_mut_slice(),
            );

            ///////////////////////////////////////////////////
            // build the local potential for the next iteration

            // v_h

            utils::compute_v_hartree(pwden, rhog, &mut vhg);

            // v_xc

            // v_xc in r space

            utils::compute_v_e_xc_of_r(&xc, rho_3d, rhocore_3d, &mut vxc_3d, &mut exc_3d);

            // v_xc in G space

            utils::compute_v_xc_of_g(gvec, pwden, rgtrans, &vxc_3d, &mut vxcg);

            // v_xc + v_h + v_psloc in G space

            utils::add_up_v(&vpslocg, &vhg, &vxcg, &mut vlocg);

            ///////////////////////////////////////////////////

            scf_iter += 1;
        }

        // after the SCF iterations

        // display eigenvalues

        utils::display_eigen_values(crystal, kpts, vpwwfc, vkscf, vkevals);

        // force

        utils::compute_force(
            control,
            crystal,
            gvec,
            pwden,
            pots,
            ewald,
            vkscf,
            vkevecs,
            rhog,
            &vxcg,
            symdrv,
            force_total,
        );

        // stress

        utils::compute_stress(
            control,
            crystal,
            gvec,
            pwden,
            pots,
            ewald,
            vkscf,
            vkevecs,
            rhog,
            rho_3d,
            rhocore_3d,
            &vxcg,
            &vxc_3d,
            &exc_3d,
            symdrv,
            stress_total,
        );
    }
}
