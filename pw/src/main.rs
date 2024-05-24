#![allow(warnings)]
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use fftgrid::FFTGrid;
use gvector::GVector;
use kscf::KSCF;
use matrix::*;
use mpi_sys::MPI_COMM_WORLD;
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
    dwmpi::init();

    // start the timer-main

    let stopwatch_main = std::time::Instant::now();

    // read in control parameters

    let mut control = Control::new();
    control.read_file("in.ctrl");

    if dwmpi::is_root() {
        control.display();
    }

    // read in crystal

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let latt0 = crystal.get_latt().clone();
    let geom_optim_mask_cell = crystal.get_cell_mask().clone();

    // read in pots

    let pots = PSPot::new(control.get_pot_scheme());

    if dwmpi::is_root() {
        pots.display();
    }

    // zions

    let zions = crystal.get_zions(&pots);

    // println!("zions = {:?}", zions);

    // read in kpts

    let kpts = kpts::new(control.get_kpts_scheme(), &crystal, control.get_symmetry());

    if dwmpi::is_root() {
        kpts.display();
    }

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

        if dwmpi::is_root() {
            println!("FFTGrid : {}", fftgrid);
        }

        // RGTransform

        let rgtrans = rgtransform::RGTransform::new(n1, n2, n3);

        // G vectors

        let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);

        // G vectors for density expansion

        let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
        let npw_rho = pwden.get_n_plane_waves();

        if dwmpi::is_root() {
            println!("npw_rho = {}", npw_rho);
        }

        //loop {

        if dwmpi::is_root() {
            println!();
            crystal.display();
        }

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

        if dwmpi::is_root() {
            if geom_iter == 1 {
                if std::path::Path::new("out.scf.rho.hdf5").exists() {
                    if let RHOG::NonSpin(ref mut rhog) = &mut rhog {
                        if let RHOR::NonSpin(ref mut rho_3d) = &mut rho_3d {
                            rho_3d.load_hdf5("out.scf.rho.hdf5");
                            rgtrans.r3d_to_g1d(&gvec, &pwden, rho_3d.as_slice(), rhog);
                        }
                    }

                    println!("   load charge density from out.scf.rho.hdf5");
                }
                else if std::path::Path::new("out.scf.rho").exists() {
                    if let RHOG::NonSpin(ref mut rhog) = &mut rhog {
                        if let RHOR::NonSpin(ref mut rho_3d) = &mut rho_3d {
                            rho_3d.load("out.scf.rho");
                            rgtrans.r3d_to_g1d(&gvec, &pwden, rho_3d.as_slice(), rhog);
                        }
                    }

                    println!("   load charge density from out.scf.rho");
                }
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
        }

        if control.is_spin() {
            let (rhog_up, rhog_dn) = rhog.as_spin().unwrap();

            dwmpi::bcast_slice(rhog_up, MPI_COMM_WORLD);
            dwmpi::bcast_slice(rhog_dn, MPI_COMM_WORLD);

            let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin().unwrap();

            dwmpi::bcast_slice(rho_3d_up.as_slice(), MPI_COMM_WORLD);
            dwmpi::bcast_slice(rho_3d_dn.as_slice(), MPI_COMM_WORLD);
        } else {
            dwmpi::bcast_slice(rhog.as_non_spin().unwrap(), MPI_COMM_WORLD);

            dwmpi::bcast_slice(rho_3d.as_non_spin().unwrap().as_slice(), MPI_COMM_WORLD);
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

        if dwmpi::is_root() {
            println!("   initial_charge = {}", total_rho);
        }

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

        let nrank = dwmpi::get_comm_world_size() as usize;
        let my_nkpt = kpts_distribution::get_my_k_total(nkpt, nrank);

        let mut vpwwfc = Vec::<PWBasis>::with_capacity(my_nkpt);

        let ik_first = kpts_distribution::get_my_k_first(nkpt, nrank);
        let ik_last = kpts_distribution::get_my_k_last(nkpt, nrank);

        (ik_first..=ik_last).into_iter().for_each(|ik| {
            let k_frac = kpts.get_k_frac(ik);
            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);

            let pwwfc: PWBasis = PWBasis::new(k_cart, ik, control.get_ecut(), &gvec);

            vpwwfc.push(pwwfc);
        });

        // vvnl

        let mut vvnl = Vec::<VNL>::with_capacity(my_nkpt);

        for ik in ik_first..=ik_last {
            let vnl: VNL = VNL::new(ik, &pots, &vpwwfc[ik - ik_first], &crystal);
            vvnl.push(vnl);
        }

        // vkscf

        let mut vkscf: VKSCF;

        if !control.is_spin() {
            vkscf = VKSCF::NonSpin(Vec::<KSCF>::new());
            if let VKSCF::NonSpin(ref mut vkscf) = vkscf {
                for ik in ik_first..=ik_last {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik - ik_first],
                        &vvnl[ik - ik_first],
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
                for ik in ik_first..=ik_last {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik - ik_first],
                        &vvnl[ik - ik_last],
                        ik,
                        k_cart,
                        k_weight,
                    );

                    vkscf_up.push(kscf);
                }

                for ik in ik_first..=ik_last {
                    let k_frac = kpts.get_k_frac(ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let k_weight = kpts.get_k_weight(ik);

                    let kscf = KSCF::new(
                        &control,
                        &gvec,
                        &pots,
                        &vpwwfc[ik - ik_first],
                        &vvnl[ik - ik_last],
                        ik,
                        k_cart,
                        k_weight,
                    );

                    vkscf_dn.push(kscf);
                }
            }
        }

        if !control.is_spin() {
            vkevals = VKEigenValue::NonSpin(vec![vec![0.0; nband]; my_nkpt]);
            vkevecs = VKEigenVector::NonSpin(vec![Matrix::new(0, 0); 0]);

            if let VKEigenVector::NonSpin(ref mut vkevecs) = vkevecs {
                for (ik, pwwfc) in vpwwfc.iter().enumerate() {
                    let npw = pwwfc.get_n_plane_waves();
                    vkevecs.push(Matrix::new(npw, nband));
                }
            }
        } else {
            vkevals = VKEigenValue::Spin(
                vec![vec![0.0; nband]; my_nkpt],
                vec![vec![0.0; nband]; my_nkpt],
            );
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

        if dwmpi::is_root() {
            println!("\n   #step: geom-{}\n", geom_iter);
        }

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

        if dwmpi::is_root() {
            if control.get_save_rho() {
                if let RHOR::NonSpin(ref rho_3d) = &rho_3d {
                    rho_3d.save("out.scf.rho");
                    rho_3d.save_hdf5("out.scf.rho.hdf5");
                } else if let RHOR::Spin(ref rho_3d_up, ref rho_3d_dn) = &rho_3d {
                    rho_3d_up.save("out.scf.rho.up");
                    rho_3d_up.save_hdf5("out.scf.rho.up.hdf5");
                    rho_3d_dn.save("out.scf.rho.dn");
                    rho_3d_dn.save_hdf5("out.scf.rho.dn.hdf5");
                }
            }

            crystal.output();
        }

        // if converged, then exit

        let force_max = force::get_max_force(&force_total);

        let stress_max = stress::get_max_stress(&stress_total);

        if control.get_geom_optim_cell() {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA
                && stress_max < control.get_geom_optim_stress_tolerance() * STRESS_KB_TO_HA
            {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }

                post_processing(&control, &vkevals, &vkevecs, &vkscf);

                break;
            }
        } else {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }

                post_processing(&control, &vkevals, &vkevecs, &vkscf);

                break;
            }
        }

        // if not converged, but reach the max geometry optimization steps, then exit

        if geom_iter >= control.get_geom_optim_max_steps() {
            if dwmpi::is_root() {
                println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", geom_iter);
            }

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
    let elapsed_main_seconds = stopwatch_main.elapsed().as_secs_f64();

    if dwmpi::is_root() {
        println!();
        println!("   {:-^88}", " statistics ");
        println!();

        println!(
            "   {:16}{:5}{:16.2} seconds {:16.2} hours",
            "Total",
            ":",
            elapsed_main_seconds,
            elapsed_main_seconds / 3600.0
        );
    }

    // last statement

    std::thread::sleep(std::time::Duration::from_millis(200));
    
    dwmpi::barrier(MPI_COMM_WORLD);

    dwmpi::finalize();
}

fn post_processing(
    control: &Control,
    vkevals: &VKEigenValue,
    vkevecs: &VKEigenVector,
    vkscf: &VKSCF,
) {
    // total density of states

    // println!("   compute total density of states");

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
