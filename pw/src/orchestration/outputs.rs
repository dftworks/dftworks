use control::Control;
use dfttypes::*;
use dwconsts::*;
use matrix::Matrix;
use vector3::Vector3f64;

pub(crate) struct PersistOutputsContext<'a> {
    pub control: &'a Control,
    pub crystal: &'a crystal::Crystal,
    pub blatt: &'a lattice::Lattice,
    pub expected_checkpoint_meta: &'a CheckpointMeta,
    pub rho_3d: &'a RHOR,
    pub electronic_ctx: &'a crate::ElectronicStepContext,
    pub vkevecs: &'a VKEigenVector,
}

pub(crate) fn persist_outputs(ctx: PersistOutputsContext<'_>) -> Result<(), String> {
    let repo = Hdf5FilePerKCheckpointRepository::default();

    if dwmpi::is_root() {
        if ctx.control.get_save_rho() {
            repo.save_density(ctx.rho_3d, ctx.blatt, Some(ctx.expected_checkpoint_meta))?;
        }

        ctx.crystal.output();
    }

    if ctx.control.get_save_wfc() || ctx.control.get_wannier90_export() {
        repo.save_wavefunctions(
            ctx.vkevecs,
            ctx.electronic_ctx.global_k_indices(),
            ctx.electronic_ctx.vpwwfc(),
            ctx.blatt,
            Some(ctx.expected_checkpoint_meta),
        )?;
    }

    Ok(())
}

pub(crate) struct ExitDecisionContext<'a, 'ks> {
    pub control: &'a Control,
    pub geom_iter: usize,
    pub stress_total: &'a Matrix<f64>,
    pub force_total: &'a [Vector3f64],
    pub vkevals: &'a VKEigenValue,
    pub vkevecs: &'a VKEigenVector,
    pub vkscf: &'a VKSCF<'ks>,
    pub electronic_ctx: &'a crate::ElectronicStepContext,
}

pub(crate) fn evaluate_exit_and_finalize(ctx: ExitDecisionContext<'_, '_>) -> bool {
    let force_max = force::get_max_force(ctx.force_total);
    let stress_max = stress::get_max_stress(ctx.stress_total);

    let mut should_exit = false;

    if ctx.control.get_geom_optim_cell() {
        if force_max < ctx.control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA
            && stress_max < ctx.control.get_geom_optim_stress_tolerance() * STRESS_KB_TO_HA
        {
            if dwmpi::is_root() {
                println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", ctx.geom_iter);
            }
            should_exit = true;
        }
    } else if force_max < ctx.control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
        if dwmpi::is_root() {
            println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", ctx.geom_iter);
        }
        should_exit = true;
    }

    if !should_exit && ctx.geom_iter >= ctx.control.get_geom_optim_max_steps() {
        if dwmpi::is_root() {
            println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", ctx.geom_iter);
        }
        should_exit = true;
    }

    if should_exit {
        if ctx.control.get_wannier90_export() {
            match wannier90::write_eig_inputs(
                ctx.control,
                ctx.vkevals,
                ctx.electronic_ctx.global_k_indices(),
            ) {
                Ok(summary) => {
                    if dwmpi::is_root() {
                        println!();
                        println!("   {:-^88}", " wannier90 eig export ");
                        for file in summary.written_files.iter() {
                            println!("   wrote {}", file);
                        }
                        println!(
                            "   run `w90-win` and `w90-amn` to generate remaining Wannier90 inputs"
                        );
                    }
                }
                Err(err) => {
                    if dwmpi::is_root() {
                        eprintln!("   wannier90 eig export failed: {}", err);
                    }
                }
            }
        }

        crate::post_processing(ctx.control, ctx.vkevals, ctx.vkevecs, ctx.vkscf);
    }

    should_exit
}
