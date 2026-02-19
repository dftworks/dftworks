#![allow(warnings)]
use control::{Control, SpinScheme};
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
use rayon;
use std::time::{SystemTime, UNIX_EPOCH};
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;
use vnl::VNL;

fn display_program_header() {
    println!();
    println!("   {:=^88}", "");
    println!("   {:^88}", "DFTWorks");
    println!(
        "   {:^88}",
        "Self-Consistent Plane-Wave Density Functional Theory"
    );
    println!("   {:=^88}", "");
    println!();
}

fn format_unix_seconds_as_utc_iso(timestamp_unix_s: u64) -> String {
    // Convert Unix epoch seconds to an ISO-like UTC timestamp without external crates.
    // Algorithm adapted from civil date conversion by Howard Hinnant.
    let days = (timestamp_unix_s / 86_400) as i64;
    let seconds_of_day = timestamp_unix_s % 86_400;

    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = mp + if mp < 10 { 3 } else { -9 }; // [1, 12]
    if month <= 2 {
        year += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

fn display_system_information() {
    const OUT_WIDTH1: usize = 28;
    const OUT_WIDTH2: usize = 18;

    let backend = dwfft3d::backend_name();
    let mpi_rank = dwmpi::get_comm_world_rank();
    let mpi_ranks = dwmpi::get_comm_world_size();
    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp_unix_s = now.as_secs();
    let timestamp_utc = format_unix_seconds_as_utc_iso(timestamp_unix_s);

    println!("   {:-^88}", " system information ");
    println!();
    println!(
        "   {:<width1$} = {:>width2$}",
        "backend",
        backend,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "hostname",
        hostname,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "os",
        std::env::consts::OS,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "arch",
        std::env::consts::ARCH,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_rank",
        mpi_rank,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_ranks",
        mpi_ranks,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {}",
        "timestamp_utc",
        timestamp_utc,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:>4} (RAYON_NUM_THREADS={}, host_threads={})",
        "rayon_threads",
        rayon_threads,
        rayon_env,
        host_threads,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "working_directory",
        cwd,
        width1 = OUT_WIDTH1
    );
    println!();
}

fn display_grid_information(fftgrid: &FFTGrid, pwden: &PWDensity) {
    const OUT_WIDTH1: usize = 28;

    println!("   {:-^88}", " grid information ");
    println!();
    println!("   FFTGrid : {}", fftgrid);
    println!("   npw_rho = {}", pwden.get_n_plane_waves());
    println!(
        "   {:<width1$} = {}",
        "nfft",
        fftgrid.get_ntot(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "rho_gshells",
        pwden.get_n_gshell(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:16.8} 1/bohr ({:16.8} 1/A)",
        "gmax_rho",
        pwden.get_gmax(),
        pwden.get_gmax() * ANG_TO_BOHR,
        width1 = OUT_WIDTH1
    );
}

fn display_symmetry_equivalent_atoms(crystal: &Crystal, symdrv: &dyn SymmetryDriver) {
    let sym_atom = symdrv.get_sym_atom();
    let natoms = sym_atom.len();
    let n_sym = symdrv.get_n_sym_ops();

    println!();
    println!("   {:-^88}", " symmetry-equivalent atoms ");
    println!("   mapping convention: atom(i) --sym_op--> atom(j), 1-based atom index");
    println!("   n_atoms = {}, n_sym_ops = {}", natoms, n_sym);

    let atom_species = crystal.get_atom_species();
    for (iat, mapping_row) in sym_atom.iter().enumerate() {
        let mapped_one_based: Vec<usize> = mapping_row
            .iter()
            .map(|&jat| jat.saturating_add(1))
            .collect();
        let species = atom_species.get(iat).map(|s| s.as_str()).unwrap_or("?");
        println!(
            "   atom {:>4} {:<6} -> {:?}",
            iat + 1,
            format!("({})", species),
            mapped_one_based
        );
    }

    let mut visited = vec![false; natoms];
    let mut classes: Vec<Vec<usize>> = Vec::new();

    for iat in 0..natoms {
        if visited[iat] {
            continue;
        }

        let mut class = vec![iat];
        if let Some(row) = sym_atom.get(iat) {
            for &jat in row.iter() {
                if jat < natoms {
                    class.push(jat);
                }
            }
        }
        class.sort_unstable();
        class.dedup();

        for &jat in class.iter() {
            visited[jat] = true;
        }
        classes.push(class);
    }

    println!("   equivalence classes ({} total)", classes.len());
    for (iclass, class) in classes.iter().enumerate() {
        let one_based: Vec<usize> = class.iter().map(|&iat| iat + 1).collect();
        println!("   class {:>4} -> {:?}", iclass + 1, one_based);
    }
}

fn main() {
    // Top-level program flow:
    // 1) initialize MPI/runtime and read all inputs
    // 2) build per-geometry numerical context (FFT/G-vectors/k-points)
    // 3) run SCF to obtain energies, forces, and stress
    // 4) optionally perform geometry optimization step
    // 5) export requested artifacts (charge/wfc/eig files)
    // first statement
    dwmpi::init();

    // start the timer-main

    let stopwatch_main = std::time::Instant::now();

    // read in control parameters

    let mut control = Control::new();
    control.read_file("in.ctrl");

    // dwfft3d
    dwfft3d::init_backend();

    if dwmpi::is_root() {
        display_program_header();
        display_system_information();
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

    // Geometry optimizer state (BFGS/DIIS depending on in.ctrl).

    let mut geom_iter = 1;

    let mut geom_driver = geom::new(
        control.get_geom_optim_scheme(),
        control.get_geom_optim_alpha(),
        control.get_geom_optim_history_steps(),
    );

    // Self-consistent field drivers selected by spin scheme.

    let spin_scheme = control.get_spin_scheme_enum();

    let scf_driver = scf::new(spin_scheme);

    let density_driver = density::new(spin_scheme);

    // crystal.display();

    loop {
        // Rebuild reciprocal/FFT objects each geometry step because lattice may
        // change during cell optimization.
        // FFT Grid

        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
        let [n1, n2, n3] = fftgrid.get_size();

        // RGTransform

        let rgtrans = rgtransform::RGTransform::new(n1, n2, n3);

        // G vectors

        let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);

        // G vectors for density expansion

        let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
        let npw_rho = pwden.get_n_plane_waves();

        if dwmpi::is_root() {
            display_grid_information(&fftgrid, &pwden);
        }

        //loop {

        if dwmpi::is_root() {
            println!();
            crystal.display();
        }

        // Ewald

        let ewald = ewald::Ewald::new(&crystal, &zions, &gvec, &pwden);

        let nspin = match spin_scheme {
            SpinScheme::NonSpin => 1,
            SpinScheme::Spin => 2,
            SpinScheme::Ncl => {
                panic!("spin_scheme='ncl' is not implemented yet in pw initialization")
            }
        };

        let mut rhog = match spin_scheme {
            SpinScheme::NonSpin => RHOG::NonSpin(vec![c64::zero(); npw_rho]),
            SpinScheme::Spin => RHOG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]),
            SpinScheme::Ncl => {
                panic!("spin_scheme='ncl' is not implemented yet in pw initialization")
            }
        };

        let mut rho_3d = match spin_scheme {
            SpinScheme::NonSpin => RHOR::NonSpin(Array3::<c64>::new([n1, n2, n3])),
            SpinScheme::Spin => RHOR::Spin(
                Array3::<c64>::new([n1, n2, n3]),
                Array3::<c64>::new([n1, n2, n3]),
            ),
            SpinScheme::Ncl => {
                panic!("spin_scheme='ncl' is not implemented yet in pw initialization")
            }
        };

        // Initialize density either from previous converged file (restart) or
        // from atomic superposition.

        if dwmpi::is_root() {
            if geom_iter == 1 && std::path::Path::new("out.scf.rho.hdf5").exists() {
                if let RHOG::NonSpin(ref mut rhog) = &mut rhog {
                    rho_3d = RHOR::load_hdf5(matches!(spin_scheme, SpinScheme::Spin)).1;
                    if let RHOR::NonSpin(data) = &mut rho_3d {
                        rgtrans.r3d_to_g1d(&gvec, &pwden, data.as_slice(), rhog);
                    }
                }

                println!("   load charge density from out.scf.rho.hdf5");
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

        match spin_scheme {
            SpinScheme::Spin => {
                let (rhog_up, rhog_dn) = rhog.as_spin_mut().unwrap();

                dwmpi::bcast_slice(rhog_up.as_mut_slice(), MPI_COMM_WORLD);
                dwmpi::bcast_slice(rhog_dn.as_mut_slice(), MPI_COMM_WORLD);

                let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin_mut().unwrap();

                dwmpi::bcast_slice(rho_3d_up.as_mut_slice(), MPI_COMM_WORLD);
                dwmpi::bcast_slice(rho_3d_dn.as_mut_slice(), MPI_COMM_WORLD);
            }
            SpinScheme::NonSpin => {
                dwmpi::bcast_slice(
                    rhog.as_non_spin_mut().unwrap().as_mut_slice(),
                    MPI_COMM_WORLD,
                );

                dwmpi::bcast_slice(
                    rho_3d.as_non_spin_mut().unwrap().as_mut_slice(),
                    MPI_COMM_WORLD,
                );
            }
            SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in pw broadcast"),
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

        // Core charge used by NLCC terms.

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

        // Symmetry helper used by force/stress post-processing.

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

        if control.get_symmetry() && dwmpi::is_root() {
            println!();
            println!("   {:-^88}", " symmetry analysis ");
            symdrv.display();

            let n_sym = symdrv.get_n_sym_ops();
            let fft_comm_ops =
                symdrv.get_fft_commensurate_ops(fftgrid.get_size(), kpts.get_k_mesh(), EPS6);
            println!(
                "   commensurate_ops (fft+kmesh) = {} / {}",
                fft_comm_ops.len(),
                n_sym
            );

            let kmesh = kpts.get_k_mesh();
            if kmesh[0] > 0 && kmesh[1] > 0 && kmesh[2] > 0 {
                let nk_full = kmesh[0] as usize * kmesh[1] as usize * kmesh[2] as usize;
                println!(
                    "   ir_kpoints (symmetry-reduced) = {} / {}",
                    kpts.get_n_kpts(),
                    nk_full
                );
            }

            println!(
                "   sym_atom mapping dimensions   = {} atoms x {} ops",
                symdrv.get_sym_atom().len(),
                n_sym
            );

            display_symmetry_equivalent_atoms(&crystal, symdrv.as_ref());
        }

        //

        let nband = control.get_nband();
        let nkpt = kpts.get_n_kpts();

        let blatt = crystal.get_latt().reciprocal();

        // vpwwfc

        let nrank = dwmpi::get_comm_world_size() as usize;
        let my_nkpt = kpts_distribution::get_my_k_total(nkpt, nrank);

        let mut vpwwfc = Vec::<PWBasis>::with_capacity(my_nkpt);

        let ik_range = kpts_distribution::get_my_k_range(nkpt, nrank);
        let (ik_first, ik_last) = ik_range.unwrap_or((0, 0));

        if let Some((ik_first, ik_last)) = ik_range {
            (ik_first..=ik_last).into_iter().for_each(|ik| {
                let k_frac = kpts.get_k_frac(ik);
                let k_cart = kpts.frac_to_cart(&k_frac, &blatt);

                let pwwfc: PWBasis = PWBasis::new(k_cart, ik, control.get_ecut(), &gvec);

                vpwwfc.push(pwwfc);
            });
        }

        // vvnl

        let mut vvnl = Vec::<VNL>::with_capacity(my_nkpt);

        if let Some((ik_first, ik_last)) = ik_range {
            for ik in ik_first..=ik_last {
                let vnl: VNL = VNL::new(ik, &pots, &vpwwfc[ik - ik_first], &crystal);
                vvnl.push(vnl);
            }
        }

        // vkscf

        let mut vkscf: VKSCF;

        match spin_scheme {
            SpinScheme::NonSpin => {
                vkscf = VKSCF::NonSpin(Vec::<KSCF>::new());
                if let VKSCF::NonSpin(ref mut vkscf) = vkscf {
                    if let Some((ik_first, ik_last)) = ik_range {
                        for ik in ik_first..=ik_last {
                            let k_frac = kpts.get_k_frac(ik);
                            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                            let k_weight = kpts.get_k_weight(ik);

                            let kscf = KSCF::new(
                                &control,
                                &gvec,
                                &crystal,
                                &pots,
                                &vpwwfc[ik - ik_first],
                                &vvnl[ik - ik_first],
                                fftgrid.get_size(),
                                ik,
                                k_cart,
                                k_weight,
                            );

                            vkscf.push(kscf);
                        }
                    }
                }
            }
            SpinScheme::Spin => {
                vkscf = VKSCF::Spin(Vec::<KSCF>::new(), Vec::<KSCF>::new());
                if let VKSCF::Spin(ref mut vkscf_up, ref mut vkscf_dn) = vkscf {
                    if let Some((ik_first, ik_last)) = ik_range {
                        for ik in ik_first..=ik_last {
                            let k_frac = kpts.get_k_frac(ik);
                            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                            let k_weight = kpts.get_k_weight(ik);

                            let kscf = KSCF::new(
                                &control,
                                &gvec,
                                &crystal,
                                &pots,
                                &vpwwfc[ik - ik_first],
                                &vvnl[ik - ik_first],
                                fftgrid.get_size(),
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
                                &crystal,
                                &pots,
                                &vpwwfc[ik - ik_first],
                                &vvnl[ik - ik_first],
                                fftgrid.get_size(),
                                ik,
                                k_cart,
                                k_weight,
                            );

                            vkscf_dn.push(kscf);
                        }
                    }
                }
            }
            SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in KSCF setup"),
        }

        match spin_scheme {
            SpinScheme::NonSpin => {
                vkevals = VKEigenValue::NonSpin(vec![vec![0.0; nband]; my_nkpt]);
                vkevecs = VKEigenVector::NonSpin(vec![Matrix::new(0, 0); 0]);

                if let VKEigenVector::NonSpin(ref mut vkevecs) = vkevecs {
                    for (ik, pwwfc) in vpwwfc.iter().enumerate() {
                        let npw = pwwfc.get_n_plane_waves();
                        vkevecs.push(Matrix::new(npw, nband));
                    }
                }
            }
            SpinScheme::Spin => {
                vkevals = VKEigenValue::Spin(
                    vec![vec![0.0; nband]; my_nkpt],
                    vec![vec![0.0; nband]; my_nkpt],
                );
                vkevecs =
                    VKEigenVector::Spin(vec![Matrix::new(0, 0); 0], vec![Matrix::new(0, 0); 0]);

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
            SpinScheme::Ncl => panic!("spin_scheme='ncl' is not implemented yet in eigen setup"),
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
            kpts.as_ref(),
            &ewald,
            &vpwwfc,
            &mut vkscf,
            &mut rhog,
            &mut rho_3d,
            &rhocore_3d,
            &mut vkevals,
            &mut vkevecs,
            symdrv.as_ref(),
            &mut stress_total,
            &mut force_total,
        );

        // save rho

        if dwmpi::is_root() {
            if control.get_save_rho() {
                rho_3d.save_hdf5(&blatt);
            }

            crystal.output();
        }

        // save wavefunction
        if control.get_save_wfc() || control.get_wannier90_export() {
            vkevecs.save_hdf5(ik_first, &vpwwfc, &blatt);
        }

        // if converged, then exit

        let force_max = force::get_max_force(&force_total);

        let stress_max = stress::get_max_stress(&stress_total);

        let mut should_exit = false;

        if control.get_geom_optim_cell() {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA
                && stress_max < control.get_geom_optim_stress_tolerance() * STRESS_KB_TO_HA
            {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }
                should_exit = true;
            }
        } else {
            if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
                if dwmpi::is_root() {
                    println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
                }
                should_exit = true;
            }
        }

        // if not converged, but reach the max geometry optimization steps, then exit

        if !should_exit && geom_iter >= control.get_geom_optim_max_steps() {
            if dwmpi::is_root() {
                println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", geom_iter);
            }
            should_exit = true;
        }

        if should_exit {
            if control.get_wannier90_export() {
                match wannier90::write_eig_inputs(&control, &vkevals, ik_first) {
                    Ok(summary) => {
                        if dwmpi::is_root() {
                            println!();
                            println!("   {:-^88}", " wannier90 eig export ");
                            for file in summary.written_files.iter() {
                                println!("   wrote {}", file);
                            }
                            println!("   run `w90-win` and `w90-amn` to generate remaining Wannier90 inputs");
                        }
                    }
                    Err(err) => {
                        if dwmpi::is_root() {
                            eprintln!("   wannier90 eig export failed: {}", err);
                        }
                    }
                }
            }

            post_processing(&control, &vkevals, &vkevecs, &vkscf);

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
