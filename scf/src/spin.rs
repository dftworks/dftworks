#![allow(warnings)]

use super::hartree;
use super::utils;
use crate::SCF;
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::GVector;
use itertools::multizip;
use kpts::KPTS;
use kscf::KSCF;
use lattice::Lattice;
use matrix::Matrix;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::c64;
use vector3::Vector3f64;

//use rayon::prelude::*;

pub struct SCFSpin {}

impl SCFSpin {
    pub fn new() -> SCFSpin {
        SCFSpin {}
    }
}

impl SCF for SCFSpin {
    fn run(
        &self,
        geom_iter: usize,
        control: &Control,
        crystal: &Crystal,
        gvec: &GVector,
        pwden: &PWDensity,
        pots: &PSPot,
        rgtrans: &RGTransform,
        kpts: &dyn KPTS,
        ewald: &Ewald,
        vpwwfc: &[PWBasis],
        vkscf: &mut VKSCF,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
        rhocore_3d: &Array3<c64>,
        vkevals: &mut VKEigenValue,
        vkevecs: &mut VKEigenVector,
        symdrv: &dyn SymmetryDriver,
        stress_total: &mut Matrix<f64>,
        force_total: &mut Vec<Vector3f64>,
    ) {
        println!("");
        println!("   {:*^60}", " self-consistent field ");
        utils::display_parallel_runtime_info();

        let density_driver = density::new(control.get_spin_scheme_enum());

        let blatt = crystal.get_latt().reciprocal();

        //
        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());

        let fft_shape = fftgrid.get_size();

        let nfft = fftgrid.get_ntot();

        let [n1, n2, n3] = fft_shape;

        let npw_rho = pwden.get_n_plane_waves();

        //

        let mut rhog_tot = vec![c64::zero(); npw_rho];

        if let RHOG::Spin(rhog_up, rhog_dn) = rhog {
            for i in 0..npw_rho {
                rhog_tot[i] = rhog_up[i] + rhog_dn[i];
            }
        }

        //

        let mut vpslocg = vec![c64::zero(); npw_rho];

        //

        let mut vhg = vec![c64::zero(); npw_rho];

        //

        let mut vxcg = VXCG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]);

        let mut vxc_3d = VXCR::Spin(
            Array3::<c64>::new([n1, n2, n3]),
            Array3::<c64>::new([n1, n2, n3]),
        );

        let mut exc_3d = Array3::<c64>::new(fft_shape);

        //
        let mut vlocg_up = vec![c64::zero(); npw_rho];
        let mut vlocg_dn = vec![c64::zero(); npw_rho];

        // v_psloc in G space; this will not change for a fixed set of ion positions

        vloc::from_atomic_super_position(pots, crystal, gvec, pwden, &mut vpslocg);

        // v_h in G space; this will change with the density

        hartree::potential(pwden.get_g(), &rhog_tot, &mut vhg);

        // v_xc in r space first and then transform to G space; this changes
        // with density every SCF iteration.

        let xc = xc::new(control.get_xc_scheme());

        // NLCC: evaluate XC using total charge seen by the functional.
        // In spin-collinear mode we split rhocore equally across up/down.
        if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
            for i in 0..nfft {
                rho_3d_up.as_mut_slice()[i] += rhocore_3d.as_slice()[i];
                rho_3d_dn.as_mut_slice()[i] += rhocore_3d.as_slice()[i];
            }
        }

        // XC call includes full GGA derivative logic internally.
        xc.potential_and_energy(gvec, pwden, rgtrans, rho_3d, &mut vxc_3d, &mut exc_3d);

        // Restore valence-only rho in SCF state.
        if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
            for i in 0..nfft {
                rho_3d_up.as_mut_slice()[i] -= rhocore_3d.as_slice()[i];
                rho_3d_dn.as_mut_slice()[i] -= rhocore_3d.as_slice()[i];
            }
        }

        {
            let (vxc_3d_up, vxc_3d_dn) = vxc_3d.as_spin().unwrap();
            let (vxcg_up, vxcg_dn) = vxcg.as_spin_mut().unwrap();

            rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_up.as_slice(), vxcg_up);
            rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_dn.as_slice(), vxcg_dn);
        }

        // v_xc + v_h + v_psloc in G space

        {
            let (vxcg_up, vxcg_dn) = vxcg.as_spin().unwrap();

            // spin up

            for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                vlocg_up.iter_mut(),
                vxcg_up.iter(),
                vhg.iter(),
                vpslocg.iter(),
            )) {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            }

            // spin dn

            for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                vlocg_dn.iter_mut(),
                vxcg_dn.iter(),
                vhg.iter(),
                vpslocg.iter(),
            )) {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            }
        }
        //

        // density mixing

        let mut mixing_rho = mixing::new(control);

        let mut mixing_spin = mixing::new(control);

        //
        let nkpt = kpts.get_n_kpts();

        let mut vloc_3d_up = Array3::<c64>::new([n1, n2, n3]);
        let mut vloc_3d_dn = Array3::<c64>::new([n1, n2, n3]);

        let ntot_elec = crystal.get_n_total_electrons(pots);

        let mut rhog_out = RHOG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]);

        let mut rhog_diff = vec![c64::zero(); npw_rho];
        let mut rhog_total = vec![c64::zero(); npw_rho];
        let mut rhog_spin = vec![c64::zero(); npw_rho];
        let mut vk_n_band_converged = vec![0; nkpt];
        let mut vk_n_hpsi = vec![0; nkpt];

        let mut energy_diff = 0.0;

        let npw_wfc_max = get_n_plane_waves_max(&vpwwfc);

        // Reuse one Fermi-level driver across SCF iterations to avoid repeated allocations.
        let fermi_driver = fermilevel::new(control.get_spin_scheme_enum());

        let mut scf_iter = 1;

        loop {
            println!("\n   #step: geom-{}-scf-{}\n", geom_iter, scf_iter);
            //println!("\n   {} {:<5}\n", "#scf_iteration", format!("{}", scf_iter));

            // transform v_loc from G to r

            {
                rgtrans.g1d_to_r3d(gvec, pwden, &vlocg_up, vloc_3d_up.as_mut_slice());
                rgtrans.g1d_to_r3d(gvec, pwden, &vlocg_dn, vloc_3d_dn.as_mut_slice());
            }

            // set the epsilon for eigenvalue for this iteration
            // use the Hartree energy as delta E

            let eigvalue_epsilon = get_eigvalue_epsilon(
                geom_iter,
                scf_iter,
                control,
                ntot_elec,
                energy_diff,
                npw_wfc_max,
            );
            println!(
                "     {:<width1$} = {:>width2$.3E}",
                "eigval_epsilon (eV)",
                eigvalue_epsilon * HA_TO_EV,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            //

            vk_n_band_converged.fill(0);
            vk_n_hpsi.fill(0);
            {
                //            let (vkscf_up, vkscf_dn) = utility::get_slice_up_dn(vkscf);
                //                let (vkevals_up, vkevals_dn) = utility::get_mut_slice_up_dn(vkevals);
                //              let (vkevecs_up, vkevecs_dn) = utility::get_mut_slice_up_dn(vkevecs);

                if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
                    if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
                        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
                            // spin up

                            for (ik, kscf) in vkscf_up.iter().enumerate() {
                                let (n_band_converged, n_hpsi) = kscf.run(
                                    rgtrans,
                                    &vloc_3d_up,
                                    eigvalue_epsilon,
                                    geom_iter,
                                    scf_iter,
                                    &mut vkevals_up[ik],
                                    &mut vkevecs_up[ik],
                                );

                                vk_n_band_converged[ik] = n_band_converged;
                                vk_n_hpsi[ik] = n_hpsi;
                            }

                            println!(
                                "     {:<width1$} =      {:3?}",
                                "n_band_converged",
                                vk_n_band_converged,
                                width1 = OUT_WIDTH1,
                            );

                            println!(
                                "     {:<width1$} =      {:3?}",
                                "n_ham_on_psi",
                                vk_n_hpsi,
                                width1 = OUT_WIDTH1,
                            );

                            // spin down

                            for (ik, kscf) in vkscf_dn.iter().enumerate() {
                                let (n_band_converged, n_hpsi) = kscf.run(
                                    rgtrans,
                                    &vloc_3d_dn,
                                    eigvalue_epsilon,
                                    geom_iter,
                                    scf_iter,
                                    &mut vkevals_dn[ik],
                                    &mut vkevecs_dn[ik],
                                );

                                vk_n_band_converged[ik] = n_band_converged;
                                vk_n_hpsi[ik] = n_hpsi;
                            }

                            println!(
                                "     {:<width1$} =      {:3?}",
                                "n_band_converged",
                                vk_n_band_converged,
                                width1 = OUT_WIDTH1,
                            );

                            println!(
                                "     {:<width1$} =      {:3?}",
                                "n_ham_on_psi",
                                vk_n_hpsi,
                                width1 = OUT_WIDTH1,
                            );
                        }
                    }
                }
            }
            // calculate Fermi level; vkscf has to be &mut since occ will be modified

            let fermi_level = fermi_driver.get_fermi_level(vkscf, ntot_elec, &vkevals);

            println!(
                "     {:<width1$} = {:>width2$.5}",
                "Fermi_level",
                fermi_level * HA_TO_EV,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            // calculate Harris energy

            let energy_harris = compute_total_energy(
                pwden,
                crystal.get_latt(),
                rhog,
                &vkscf,
                &vkevals,
                rho_3d,
                rhocore_3d,
                &exc_3d,
                &vxc_3d,
                ewald.get_energy(),
            );

            // build the density based on the new wavefunctions

            density_driver.compute_charge_density(
                &vkscf,
                rgtrans,
                &vkevecs,
                crystal.get_latt().volume(),
                rho_3d,
            );

            let mut charge = 0.0;

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                let charge_up =
                    rho_3d_up.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();
                let charge_dn =
                    rho_3d_dn.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();

                charge = charge_up + charge_dn;
            }

            println!(
                "     {:<width1$} = {:>width2$.5}",
                "charge",
                charge,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            // rho r -> G

            if let RHOG::Spin(ref mut rhog_out_up, ref mut rhog_out_dn) = &mut rhog_out {
                if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                    rgtrans.r3d_to_g1d(gvec, pwden, rho_3d_up.as_slice(), rhog_out_up);
                    rgtrans.r3d_to_g1d(gvec, pwden, rho_3d_dn.as_slice(), rhog_out_dn);
                }
            }

            // if symmetry enabled, the charge density need to be symmetrized

            //
            // rhog_out: out rho in G
            // rho_3d:   out rho in r

            // NLCC: add core density before XC evaluation.

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                for i in 0..nfft {
                    rho_3d_up.as_mut_slice()[i] += rhocore_3d.as_slice()[i] / 2.0;
                    rho_3d_dn.as_mut_slice()[i] += rhocore_3d.as_slice()[i] / 2.0;
                }
            }

            // Full GGA-consistent v_xc and eps_xc.
            xc.potential_and_energy(gvec, pwden, rgtrans, rho_3d, &mut vxc_3d, &mut exc_3d);

            // Restore valence-only rho after XC.

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                for i in 0..nfft {
                    rho_3d_up.as_mut_slice()[i] -= rhocore_3d.as_slice()[i] / 2.0;
                    rho_3d_dn.as_mut_slice()[i] -= rhocore_3d.as_slice()[i] / 2.0;
                }
            }

            // calculate scf energy

            let energy_scf = compute_total_energy(
                pwden,
                crystal.get_latt(),
                &rhog_out,
                &vkscf,
                &vkevals,
                rho_3d,
                rhocore_3d,
                &exc_3d,
                &vxc_3d,
                ewald.get_energy(),
            );

            println!(
                "     {:<width1$} = {:>width2$.12}",
                "harris_energy (Ry)",
                energy_harris * HA_TO_RY,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            println!(
                "     {:<width1$} = {:>width2$.12}",
                "scf_energy (Ry)",
                energy_scf * HA_TO_RY,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            energy_diff = (energy_scf - energy_harris).abs();

            println!(
                "     {:<width1$} = {:>width2$.3E}",
                "delta_energy (eV)",
                energy_diff * HA_TO_EV,
                width1 = OUT_WIDTH1,
                width2 = OUT_WIDTH3
            );

            //// check convergence

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

            // mix old and new densities to get the density for the next iteration

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                let rho_3d_up = rho_3d_up.as_mut_slice();
                let rho_3d_dn = rho_3d_dn.as_mut_slice();

                if let RHOG::Spin(ref rhog_out_up, ref rhog_out_dn) = &rhog_out {
                    if let RHOG::Spin(rhog_up, rhog_dn) = rhog {
                        for ipw in 0..npw_rho {
                            rhog_total[ipw] = rhog_up[ipw] + rhog_dn[ipw];
                            rhog_spin[ipw] = rhog_up[ipw] - rhog_dn[ipw];
                        }

                        // total rho

                        for ipw in 0..npw_rho {
                            rhog_diff[ipw] = (rhog_out_up[ipw] + rhog_out_dn[ipw])
                                - (rhog_up[ipw] + rhog_dn[ipw]);
                        }

                        mixing_rho.compute_next_density(pwden.get_g(), &mut rhog_total, &rhog_diff);

                        // spin rho

                        for ipw in 0..npw_rho {
                            rhog_diff[ipw] = (rhog_out_up[ipw] - rhog_out_dn[ipw])
                                - (rhog_up[ipw] - rhog_dn[ipw]);
                        }

                        mixing_spin.compute_next_density(pwden.get_g(), &mut rhog_spin, &rhog_diff);

                        for ipw in 0..npw_rho {
                            rhog_up[ipw] = (rhog_total[ipw] + rhog_spin[ipw]) / 2.0;
                        }

                        for ipw in 0..npw_rho {
                            rhog_dn[ipw] = (rhog_total[ipw] - rhog_spin[ipw]) / 2.0;
                        }

                        rgtrans.g1d_to_r3d(gvec, pwden, rhog_up, rho_3d_up);
                        rgtrans.g1d_to_r3d(gvec, pwden, rhog_dn, rho_3d_dn);
                    }
                }
            }

            // build the local potential for the next iteration

            // v_h

            if let RHOG::Spin(rhog_up, rhog_dn) = rhog {
                for i in 0..npw_rho {
                    rhog_tot[i] = rhog_up[i] + rhog_dn[i];
                }
            }

            hartree::potential(pwden.get_g(), &rhog_tot, &mut vhg);

            // v_xc

            // NLCC: include core density during XC update.

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                for i in 0..nfft {
                    rho_3d_up.as_mut_slice()[i] += rhocore_3d.as_slice()[i] / 2.0;
                }

                for i in 0..nfft {
                    rho_3d_dn.as_mut_slice()[i] += rhocore_3d.as_slice()[i] / 2.0;
                }
            }

            // Full GGA-consistent XC refresh for next SCF iteration.
            xc.potential_and_energy(gvec, pwden, rgtrans, rho_3d, &mut vxc_3d, &mut exc_3d);

            // Restore valence-only rho for SCF state/mixing.

            if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
                for i in 0..nfft {
                    rho_3d_up.as_mut_slice()[i] -= rhocore_3d.as_slice()[i] / 2.0;
                }

                for i in 0..nfft {
                    rho_3d_dn.as_mut_slice()[i] -= rhocore_3d.as_slice()[i] / 2.0;
                }
            }

            {
                let (vxc_3d_up, vxc_3d_dn) = vxc_3d.as_spin().unwrap();
                let (vxcg_up, vxcg_dn) = vxcg.as_spin_mut().unwrap();

                rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_up.as_slice(), vxcg_up);
                rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_dn.as_slice(), vxcg_dn);
            }

            // v_xc + v_h + v_psloc in G space
            {
                let (vxcg_up, vxcg_dn) = vxcg.as_spin().unwrap();

                // spin up

                for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                    vlocg_up.iter_mut(),
                    vxcg_up.iter(),
                    vhg.iter(),
                    vpslocg.iter(),
                )) {
                    *v_loc = *v_xc + *v_ha + *v_psloc;
                }

                // spin dn

                for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                    vlocg_dn.iter_mut(),
                    vxcg_dn.iter(),
                    vhg.iter(),
                    vpslocg.iter(),
                )) {
                    *v_loc = *v_xc + *v_ha + *v_psloc;
                }
            }

            scf_iter += 1;
        }

        // display eigenvalues

        //let (vkscf_up, vkscf_dn) = utility::get_slice_up_dn(vkscf);
        //let (vkevals_up, vkevals_dn) = utility::get_slice_up_dn(vkevals);

        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
                for ik in 0..nkpt {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let npw_wfc = vpwwfc[ik].get_n_plane_waves();

                    print_k_point(ik, k_frac, k_cart, npw_wfc);

                    let occ_up = vkscf_up[ik].get_occ();
                    let occ_dn = vkscf_dn[ik].get_occ();

                    let evals_up = &vkevals_up[ik];
                    let evals_dn = &vkevals_dn[ik];

                    print_eigen_values(evals_up, occ_up, evals_dn, occ_dn);
                }
            }
        }

        // force

        let natoms = crystal.get_n_atoms();

        let mut force_loc = vec![Vector3f64::zeros(); natoms];
        let mut force_vnl = vec![Vector3f64::zeros(); natoms];

        force::vpsloc(pots, crystal, gvec, pwden, &rhog_tot, &mut force_loc);

        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
                force::vnl(crystal, &vkscf_up, &vkevecs_up, &mut force_vnl);
                force::vnl(crystal, &vkscf_dn, &vkevecs_dn, &mut force_vnl);
            }
        }

        let force_ewald = ewald.get_force();

        let mut force_nlcc = vec![Vector3f64::zeros(); natoms];

        {
            let (vxcg_up, vxcg_dn) = vxcg.as_spin().unwrap();

            let mut force_nlcc_up = vec![Vector3f64::zeros(); natoms];
            let mut force_nlcc_dn = vec![Vector3f64::zeros(); natoms];

            force::nlcc_xc(pots, crystal, gvec, pwden, vxcg_up, &mut force_nlcc_up);
            force::nlcc_xc(pots, crystal, gvec, pwden, vxcg_dn, &mut force_nlcc_dn);

            for iat in 0..natoms {
                force_nlcc[iat] = (force_nlcc_up[iat] + force_nlcc_dn[iat]) / 2.0;
            }
        }
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

        // stress

        let mut stress_kin = Matrix::<f64>::new(3, 3);
        let mut stress_vnl = Matrix::<f64>::new(3, 3);

        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
                let stress_kin_up = stress::kinetic(crystal, &vkscf_up, &vkevecs_up);
                let stress_kin_dn = stress::kinetic(crystal, &vkscf_dn, &vkevecs_dn);

                stress_kin = stress_kin_up + stress_kin_dn;

                let stress_vnl_up = stress::vnl(crystal, &vkscf_up, &vkevecs_up);
                let stress_vnl_dn = stress::vnl(crystal, &vkscf_dn, &vkevecs_dn);

                stress_vnl = stress_vnl_up + stress_vnl_dn;
            }
        }

        let stress_hartree = stress::hartree(gvec, pwden, &rhog_tot);
        let stress_xc = stress::xc_spin(crystal.get_latt(), rho_3d, rhocore_3d, &vxc_3d, &exc_3d);

        let (vxcg_up, vxcg_dn) = vxcg.as_spin().unwrap();

        let stress_xc_nlcc_up = stress::nlcc_xc(pots, crystal, gvec, pwden, vxcg_up);
        let stress_xc_nlcc_dn = stress::nlcc_xc(pots, crystal, gvec, pwden, vxcg_dn);

        let mut stress_xc_nlcc = Matrix::<f64>::new(3, 3);

        for i in 0..3 {
            for j in 0..3 {
                stress_xc_nlcc[[i, j]] =
                    (stress_xc_nlcc_up[[i, j]] + stress_xc_nlcc_dn[[i, j]]) / 2.0;
            }
        }

        let stress_loc = stress::vpsloc(pots, crystal, gvec, pwden, &rhog_tot);

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
}

pub fn compute_total_energy(
    pwden: &PWDensity,
    latt: &Lattice,
    rhog: &RHOG,
    vkscf: &VKSCF,
    vevals: &VKEigenValue,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
    vxc_3d: &VXCR,
    ew_total: f64,
) -> f64 {
    let npw_rho = pwden.get_n_plane_waves();

    // hartree energy

    let mut rhog_tot = vec![c64::zero(); npw_rho];

    let (rhog_up, rhog_dn) = rhog.as_spin().unwrap();
    for i in 0..npw_rho {
        rhog_tot[i] = rhog_up[i] + rhog_dn[i];
    }

    let etot_hartree = energy::hartree(pwden, latt, &rhog_tot);

    //

    let mut etot_bands = 0.0;
    if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
        if let VKEigenValue::Spin(vevals_up, vevals_dn) = vevals {
            let etot_bands_up = get_bands_energy(vkscf_up, vevals_up);
            let etot_bands_dn = get_bands_energy(vkscf_dn, vevals_dn);

            etot_bands = etot_bands_up + etot_bands_dn;
        }
    }

    let etot_vxc = energy::vxc_spin(latt, rho_3d, rhocore_3d.as_slice(), vxc_3d);

    let etot_xc = energy::exc_spin(latt, &rho_3d, &rhocore_3d, &exc_3d);

    let etot_one = etot_bands - etot_vxc - 2.0 * etot_hartree;

    let etot = etot_one + etot_xc + etot_hartree + ew_total;

    etot
}

//hwf_energy = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband_hwf
//etot       = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband + descf

pub fn get_bands_energy(vkscf: &[KSCF], vevals: &Vec<Vec<f64>>) -> f64 {
    let etot_bands = energy::band_structure(vkscf, vevals);

    //mpi::reduce_scalar_sum(&etot_bands_local, &mut etot_bands, MPI_COMM_WORLD);
    //mpi::bcast_scalar(&etot_bands, MPI_COMM_WORLD);

    etot_bands
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
                        (EPS4 * EV_TO_HA).min(0.0001 * energy_diff / (1.0_f64).max(ntot_elec));

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

fn get_n_plane_waves_max(vpwwfc: &[PWBasis]) -> usize {
    let mut npw_max = 0;

    for pwwfc in vpwwfc.iter() {
        let npw = pwwfc.get_n_plane_waves();

        if npw > npw_max {
            npw_max = npw;
        }
    }

    npw_max
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

pub fn print_eigen_values(v_up: &[f64], occ_up: &[f64], v_dn: &[f64], occ_dn: &[f64]) {
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
