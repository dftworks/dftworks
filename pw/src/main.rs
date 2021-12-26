#![allow(warnings)]
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use fftgrid::FFTGrid;
use gvector::GVector;
use kscf::KSCF;
use matrix::*;
use ndarray::*;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::*;
use types::*;
use vector3::Vector3f64;
use vnl::VNL;

fn main() {
    // first statement

    // start the timer-main

    let stopwatch_main = std::time::Instant::now();

    // read in control parameters

    let mut control = Control::new();
    control.read_file("in.ctrl");

    control.display();

    // read in crystal

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let latt0 = crystal.get_latt().clone();
    let geom_optim_mask_cell = crystal.get_cell_mask().clone();

    // read in pots

    let pots = PSPot::new(control.get_pot_scheme());

    pots.display();

    // zions

    let zions = crystal.get_zions(&pots);

    // println!("zions = {:?}", zions);

    // read in kpts

    let kpts = kpts::new(control.get_kpts_scheme(), &crystal, control.get_symmetry());

    kpts.display();

    //

    let mut stress_total = Matrix::<f64>::new(3, 3);
    let mut force_total = vec![Vector3f64::zeros(); crystal.get_n_atoms()];

    //let mut vkevals: Vec<Vec<f64>>;
    let mut vkevals: VKEigenValue;

    let mut vkevecs: VKEigenVector;

    // geometry optimization

    let mut geom_iter = 1;

    let mut geom_driver = geom::new(
        control.get_geom_optim_scheme(),
        control.get_geom_optim_alpha(),
        control.get_geom_optim_history_steps(),
    );

    // self-consistent field

    let scf_driver = scf::new(control.get_spin_scheme());

    let density_driver = density::new(control.get_spin_scheme());

    // crystal.display();

    loop {
        // FFT Grid

        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
        let [n1, n2, n3] = fftgrid.get_size();

        println!("FFTGrid : {}", fftgrid);

        // RGTransform

        let rgtrans = rgtransform::RGTransform::new(n1, n2, n3);

        // G vectors

        let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);

        // G vectors for density expansion

        let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
        let npw_rho = pwden.get_n_plane_waves();

        println!("npw_rho = {}", npw_rho);

        //loop {
        println!();
        crystal.display();

        // Ewald

        let ewald = ewald::Ewald::new(&crystal, &zions, &gvec, &pwden);

        let nspin = if control.is_spin() { 2 } else { 1 };

        let mut rhog = if control.is_spin() {
            RHOG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho])
        } else {
            RHOG::NonSpin(vec![c64::zero(); npw_rho])
        };

        let mut rho_3d = if control.is_spin() {
            RHOR::Spin(
                Array3::<c64>::new([n1, n2, n3]),
                Array3::<c64>::new([n1, n2, n3]),
            )
        } else {
            RHOR::NonSpin(Array3::<c64>::new([n1, n2, n3]))
        };

        // set rhog and rho_3d to be the atomic super position

        if geom_iter == 1 && std::path::Path::new("out.scf.rho").exists() {
            if let RHOG::NonSpin(ref mut rhog) = &mut rhog {
                if let RHOR::NonSpin(ref mut rho_3d) = &mut rho_3d {
                    rho_3d.load("out.scf.rho");
                    rgtrans.r3d_to_g1d(&gvec, &pwden, rho_3d.as_slice(), rhog);
                }
            }

            println!("   load charge density from out.scf.rho");
        } else {
            density_driver.from_atomic_super_position(
                &pots,
                &crystal,
                &rgtrans,
                &gvec,
                &pwden,
                &mut rhog,
                &mut rho_3d,
            );

            println!();
            println!("   construct charge density from constituent atoms");
        }

        let mut total_rho = 0.0;

        if let RHOR::NonSpin(ref rho_3d) = &rho_3d {
            total_rho = rho_3d.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();
        } else if let RHOR::Spin(ref rho_3d_up, ref rho_3d_dn) = rho_3d {
            let total_rho_up =
                rho_3d_up.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();
            let total_rho_dn =
                rho_3d_dn.sum().re * crystal.get_latt().volume() / fftgrid.get_ntotf64();

            total_rho = total_rho_up + total_rho_dn;
        }

        println!("   initial_charge = {}", total_rho);

        // core charge

        let mut rhocoreg = vec![c64::zero(); npw_rho];
        let mut rhocore_3d = Array3::<c64>::new([n1, n2, n3]);
        rhocore_3d.set_value(c64::zero());

        nlcc::from_atomic_super_position(
            &pots,
            &crystal,
            &rgtrans,
            &gvec,
            &pwden,
            &mut rhocoreg,
            &mut rhocore_3d,
        );

        // symmetry

        let mut atom_positions = vec![[0.0; 3]; crystal.get_n_atoms()];

        let vatoms = crystal.get_atom_positions();
        for iat in 0..crystal.get_n_atoms() {
            atom_positions[iat][0] = vatoms[iat].x;
            atom_positions[iat][1] = vatoms[iat].y;
            atom_positions[iat][2] = vatoms[iat].z;
        }

        let symdrv = symmetry::new(
            &crystal.get_latt().as_2d_array_row_major(),
            &atom_positions,
            &crystal.get_atom_types(),
            EPS6,
        );

        //symdrv.display();

        //

        let nband = control.get_nband();
        let nkpt = kpts.get_n_kpts();

        let blatt = crystal.get_latt().reciprocal();

        // vpwwfc

        let mut vpwwfc = Vec::<PWBasis>::with_capacity(nkpt);

        (0..nkpt).into_iter().for_each(|ik| {
            let k_frac = kpts.get_k_frac(ik);
            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);

            let pwwfc: PWBasis = PWBasis::new(k_cart, ik, control.get_ecut(), &gvec);

            vpwwfc.push(pwwfc);
        });

        // vvnl

        let mut vvnl = Vec::<VNL>::with_capacity(nkpt);

        for ik in 0..nkpt {
            let vnl: VNL = VNL::new(ik, &pots, &vpwwfc[ik], &crystal);
            vvnl.push(vnl);
        }

        // vkscf

        let mut vkscf: VKSCF;

        if !control.is_spin() {
            vkscf = VKSCF::NonSpin(Vec::<KSCF>::new());
            if let VKSCF::NonSpin(ref mut vkscf) = vkscf {
                for ik in 0..nkpt {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik],
                        &vvnl[ik],
                        ik,
                        k_cart,
                        k_weight,
                    );

                    vkscf.push(kscf);
                }
            }
        } else {
            vkscf = VKSCF::Spin(Vec::<KSCF>::new(), Vec::<KSCF>::new());
            if let VKSCF::Spin(ref mut vkscf_up, ref mut vkscf_dn) = vkscf {
                for ik in 0..nkpt {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik],
                        &vvnl[ik],
                        ik,
                        k_cart,
                        k_weight,
                    );

                    vkscf_up.push(kscf);
                }

                for ik in 0..nkpt {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik],
                        &vvnl[ik],
                        ik,
                        k_cart,
                        k_weight,
                    );

                    vkscf_dn.push(kscf);
                }
            }
        }

        if !control.is_spin() {
            vkevals = VKEigenValue::NonSpin(vec![vec![0.0; nband]; nkpt]);
            vkevecs = VKEigenVector::NonSpin(vec![Matrix::new(0, 0); 0]);

            if let VKEigenVector::NonSpin(ref mut vkevecs) = vkevecs {
                for (ik, pwwfc) in vpwwfc.iter().enumerate() {
                    let npw = pwwfc.get_n_plane_waves();
                    vkevecs.push(Matrix::new(npw, nband));
                }
            }
        } else {
            vkevals =
                VKEigenValue::Spin(vec![vec![0.0; nband]; nkpt], vec![vec![0.0; nband]; nkpt]);
            vkevecs = VKEigenVector::Spin(vec![Matrix::new(0, 0); 0], vec![Matrix::new(0, 0); 0]);

            if let VKEigenVector::Spin(ref mut vkevc_up, ref mut vkevc_dn) = vkevecs {
                for (ik, pwwfc) in vpwwfc.iter().enumerate() {
                    let npw = pwwfc.get_n_plane_waves();
                    vkevc_up.push(Matrix::new(npw, nband));
                }

                for (ik, pwwfc) in vpwwfc.iter().enumerate() {
                    let npw = pwwfc.get_n_plane_waves();
                    vkevc_dn.push(Matrix::new(npw, nband));
                }
            }
        }

        // ions optimization

        println!("\n   #step: geom-{}\n", geom_iter);

        //loop {
        scf_driver.run(
            geom_iter,
            &control,
            &crystal,
            &gvec,
            &pwden,
            &pots,
            &rgtrans,
            &kpts,
            &ewald,
            &vpwwfc,
            &mut vkscf,
            &mut rhog,
            &mut rho_3d,
            &rhocore_3d,
            &mut vkevals,
            &mut vkevecs,
            &symdrv,
            &mut stress_total,
            &mut force_total,
        );

        // save rho

        if control.get_save_rho() {
            if let RHOR::NonSpin(ref rho_3d) = &rho_3d {
                rho_3d.save("out.scf.rho");
            } else if let RHOR::Spin(ref rho_3d_up, ref rho_3d_dn) = &rho_3d {
                rho_3d_up.save("out.scf.rho.up");
                rho_3d_dn.save("out.scf.rho.dn");
            }
        }

        crystal.output();

        // if converged, then exit

        let force_max = force::get_max_force(&force_total);

        let stress_max = stress::get_max_stress(&stress_total);

        if control.get_geom_optim_cell() {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA
                && stress_max < control.get_geom_optim_stress_tolerance() * STRESS_KB_TO_HA
            {
                println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                post_processing(&control, &vkevals, &vkevecs, &vkscf);

                break;
            }
        } else {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
                println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                post_processing(&control, &vkevals, &vkevecs, &vkscf);

                break;
            }
        }

        // if not converged, but reach the max geometry optimization steps, then exit

        if geom_iter >= control.get_geom_optim_max_steps() {
            println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", geom_iter);

            break;
        }

        // if not converged, get the atom positions and (lattice vectors if cell is also relaxed) for the next optim iteration
        let geom_optim_mask_ions = vec![
            Vector3f64 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            };
            crystal.get_n_atoms()
        ];

        let mut bcell_move = control.get_geom_optim_cell();

        geom_driver.compute_next_configuration(
            &mut crystal,
            &force_total,
            &stress_total,
            &geom_optim_mask_ions,
            &geom_optim_mask_cell,
            &latt0,
            bcell_move,
        );

        // end: geom relaxation

        geom_iter += 1;
    }

    // computing time statistics

    println!();
    println!("   {:-^88}", " statistics ");
    println!();
    let elapsed_main_seconds = stopwatch_main.elapsed().as_secs_f64();
    println!(
        "   {:16}{:5}{:16.2} seconds {:16.2} hours",
        "Total",
        ":",
        elapsed_main_seconds,
        elapsed_main_seconds / 3600.0
    );

    // last statement
}

fn post_processing(
    control: &Control,
    vkevals: &VKEigenValue,
    vkevecs: &VKEigenVector,
    vkscf: &VKSCF,
) {
    // total density of states

    println!("   compute total density of states");

    //dos::compute_total_density_of_states(control, vkevals, vkevecs, vkscf);
}

fn matrix3x3_to_vector3(mat: &Matrix<f64>) -> Vec<Vector3f64> {
    let mut v = vec![Vector3f64::zeros(); 3];

    for i in 0..3 {
        v[i].x = mat[[0, i]];
        v[i].y = mat[[1, i]];
        v[i].z = mat[[2, i]];
    }

    v
}
