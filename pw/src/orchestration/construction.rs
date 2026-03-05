use control::{Control, SpinScheme, VerbosityLevel};
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use kpts::KPTS;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use symmetry::SymmetryDriver;
use types::c64;

pub(crate) struct GeometryPhaseInput<'a, 'ws> {
    pub control: &'a Control,
    pub crystal: &'a Crystal,
    pub pots: &'a PSPot,
    pub kpts: &'a dyn KPTS,
    pub zions: &'a [f64],
    pub spin_scheme: SpinScheme,
    pub geom_iter: usize,
    pub density_driver: &'a dyn density::Density,
    pub orchestration_workspace: &'ws mut crate::OrchestrationWorkspace,
}

pub(crate) struct GeometryPhaseArtifacts<'a> {
    pub geom_ctx: crate::GeometryStepContext,
    pub runtime_ctx: crate::RuntimeContext<'a>,
    pub expected_checkpoint_meta: CheckpointMeta,
    pub ewald: ewald::Ewald,
    pub rhog: RHOG,
    pub rho_3d: RHOR,
    pub symdrv: Box<dyn SymmetryDriver>,
    pub electronic_ctx: crate::ElectronicStepContext,
}

pub(crate) fn construct_geometry_phase<'a, 'ws>(
    input: GeometryPhaseInput<'a, 'ws>,
) -> Result<GeometryPhaseArtifacts<'a>, String> {
    let verbosity = input.control.get_verbosity_enum();

    let geom_ctx = crate::GeometryStepContext::new(input.crystal, input.control.get_ecutrho());
    let runtime_ctx =
        crate::RuntimeContext::new(input.control, input.crystal, input.pots, input.kpts, input.spin_scheme);
    let expected_checkpoint_meta = runtime_ctx.checkpoint_meta();

    if dwmpi::is_root() && verbosity >= VerbosityLevel::Normal {
        crate::runtime_display::display_grid_information(geom_ctx.fftgrid(), geom_ctx.pwden());
    }

    if dwmpi::is_root() && verbosity >= VerbosityLevel::Verbose {
        println!();
        input.crystal.display();
    }

    let ewald = ewald::Ewald::new(input.crystal, input.zions, geom_ctx.gvec(), geom_ctx.pwden());

    let [n1, n2, n3] = geom_ctx.fft_shape();
    let npw_rho = geom_ctx.pwden().get_n_plane_waves();

    let mut rhog = match input.spin_scheme {
        SpinScheme::NonSpin => RHOG::NonSpin(vec![c64::zero(); npw_rho]),
        SpinScheme::Spin => RHOG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]),
        SpinScheme::Ncl => {
            return Err(
                "unsupported capability: spin_scheme='ncl' is not implemented in pw initialization"
                    .to_string(),
            )
        }
    };

    let mut rho_3d = match input.spin_scheme {
        SpinScheme::NonSpin => RHOR::NonSpin(Array3::<c64>::new([n1, n2, n3])),
        SpinScheme::Spin => RHOR::Spin(
            Array3::<c64>::new([n1, n2, n3]),
            Array3::<c64>::new([n1, n2, n3]),
        ),
        SpinScheme::Ncl => {
            return Err(
                "unsupported capability: spin_scheme='ncl' is not implemented in pw initialization"
                    .to_string(),
            )
        }
    };

    // Initialize density either from previous converged file (restart) or from atomic superposition.
    if dwmpi::is_root() {
        let mut loaded_from_restart = false;
        if input.geom_iter == 1 && input.control.get_restart() {
            match crate::restart::try_load_density_checkpoint(
                input.spin_scheme,
                &expected_checkpoint_meta,
                geom_ctx.blatt(),
                geom_ctx.rgtrans(),
                geom_ctx.gvec(),
                geom_ctx.pwden(),
                &mut rhog,
                &mut rho_3d,
            ) {
                Ok(message) => {
                    println!("   {}", message);
                    loaded_from_restart = true;
                }
                Err(err) => {
                    return Err(format!(
                        "failed to load restart density checkpoint: {}",
                        err
                    ));
                }
            }
        }

        if !loaded_from_restart {
            input.density_driver.from_atomic_super_position(
                input.pots,
                input.crystal,
                geom_ctx.rgtrans(),
                geom_ctx.gvec(),
                geom_ctx.pwden(),
                &mut rhog,
                &mut rho_3d,
            );

            if input.geom_iter == 1
                && !input.control.get_restart()
                && crate::restart::restart_density_files_exist(input.spin_scheme)
            {
                println!(
                    "   restart=false: ignore existing checkpoint files and build atomic initial density"
                );
            } else {
                println!();
                println!("   construct charge density from constituent atoms");
            }
        }
    }

    match input.spin_scheme {
        SpinScheme::Spin => {
            let (rhog_up, rhog_dn) = rhog.as_spin_mut().unwrap();

            dwmpi::bcast_slice(rhog_up.as_mut_slice(), dwmpi::comm_world());
            dwmpi::bcast_slice(rhog_dn.as_mut_slice(), dwmpi::comm_world());

            let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin_mut().unwrap();

            dwmpi::bcast_slice(rho_3d_up.as_mut_slice(), dwmpi::comm_world());
            dwmpi::bcast_slice(rho_3d_dn.as_mut_slice(), dwmpi::comm_world());
        }
        SpinScheme::NonSpin => {
            dwmpi::bcast_slice(rhog.as_non_spin_mut().unwrap().as_mut_slice(), dwmpi::comm_world());

            dwmpi::bcast_slice(rho_3d.as_non_spin_mut().unwrap().as_mut_slice(), dwmpi::comm_world());
        }
        SpinScheme::Ncl => {
            return Err(
                "unsupported capability: spin_scheme='ncl' is not implemented in pw broadcast"
                    .to_string(),
            )
        }
    }

    let mut total_rho = 0.0;

    if let RHOR::NonSpin(ref rho_3d) = &rho_3d {
        total_rho = rho_3d.sum().re * input.crystal.get_latt().volume() / geom_ctx.fftgrid().get_ntotf64();
    } else if let RHOR::Spin(ref rho_3d_up, ref rho_3d_dn) = rho_3d {
        let total_rho_up =
            rho_3d_up.sum().re * input.crystal.get_latt().volume() / geom_ctx.fftgrid().get_ntotf64();
        let total_rho_dn =
            rho_3d_dn.sum().re * input.crystal.get_latt().volume() / geom_ctx.fftgrid().get_ntotf64();

        total_rho = total_rho_up + total_rho_dn;
    }

    if dwmpi::is_root() {
        println!("   initial_charge = {}", total_rho);
    }

    // Core charge used by NLCC terms.
    input.orchestration_workspace.ensure_shape(npw_rho, [n1, n2, n3]);
    let (rhocoreg, rhocore_3d_workspace) = input.orchestration_workspace.core_charge_buffers_mut();

    nlcc::from_atomic_super_position(
        input.pots,
        input.crystal,
        geom_ctx.rgtrans(),
        geom_ctx.gvec(),
        geom_ctx.pwden(),
        rhocoreg,
        rhocore_3d_workspace,
    );

    // Symmetry helper used by force/stress post-processing.
    input.orchestration_workspace.update_atom_positions(input.crystal);

    let symdrv = symmetry::new(
        &input.crystal.get_latt().as_2d_array_row_major(),
        input.orchestration_workspace.atom_positions(),
        &input.crystal.get_atom_types(),
        EPS6,
    );

    if input.control.get_symmetry() && dwmpi::is_root() && verbosity >= VerbosityLevel::Verbose {
        println!();
        println!("   {:-^88}", " symmetry analysis ");
        symdrv.display();

        let n_sym = symdrv.get_n_sym_ops();
        let fft_comm_ops =
            symdrv.get_fft_commensurate_ops(geom_ctx.fftgrid().get_size(), input.kpts.get_k_mesh(), EPS6);
        println!(
            "   commensurate_ops (fft+kmesh) = {} / {}",
            fft_comm_ops.len(),
            n_sym
        );

        let kmesh = input.kpts.get_k_mesh();
        if kmesh[0] > 0 && kmesh[1] > 0 && kmesh[2] > 0 {
            let nk_full = kmesh[0] as usize * kmesh[1] as usize * kmesh[2] as usize;
            println!(
                "   ir_kpoints (symmetry-reduced) = {} / {}",
                input.kpts.get_n_kpts(),
                nk_full
            );
        }

        println!(
            "   sym_atom mapping dimensions   = {} atoms x {} ops",
            symdrv.get_sym_atom().len(),
            n_sym
        );

        crate::runtime_display::display_symmetry_equivalent_atoms(input.crystal, symdrv.as_ref());
    }

    let electronic_ctx = crate::ElectronicStepContext::build(&runtime_ctx, geom_ctx.blatt(), geom_ctx.gvec());

    Ok(GeometryPhaseArtifacts {
        geom_ctx,
        runtime_ctx,
        expected_checkpoint_meta,
        ewald,
        rhog,
        rho_3d,
        symdrv,
        electronic_ctx,
    })
}
