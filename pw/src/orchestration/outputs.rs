use control::Control;
use dfttypes::*;
use dwconsts::*;
use gvector::GVector;
use pspot::PSPot;
use types::Matrix;
use types::Vector3f64;

pub(crate) fn persist_outputs(
    control: &Control,
    crystal: &crystal::Crystal,
    blatt: &lattice::Lattice,
    expected_checkpoint_meta: &CheckpointMeta,
    rho_3d: &RHOR,
    electronic_ctx: &crate::ElectronicStepContext,
    vkevecs: &VKEigenVector,
) -> Result<(), String> {
    let repo = Hdf5FilePerKCheckpointRepository::default();

    if dwmpi::is_root() {
        if control.get_save_rho() {
            repo.save_density(rho_3d, blatt, Some(expected_checkpoint_meta))?;
        }

        crystal.output();
    }

    if control.get_save_wfc() || control.get_wannier90_export() {
        repo.save_wavefunctions(
            vkevecs,
            electronic_ctx.global_k_indices(),
            electronic_ctx.vpwwfc(),
            blatt,
            Some(expected_checkpoint_meta),
        )?;
    }

    Ok(())
}

pub(crate) fn evaluate_exit_and_finalize(
    control: &Control,
    crystal: &crystal::Crystal,
    pots: &PSPot,
    kpts: &dyn kpts::KPTS,
    gvec: &GVector,
    geom_iter: usize,
    stress_total: &Matrix<f64>,
    force_total: &[Vector3f64],
    vkevals: &VKEigenValue,
    vkevecs: &VKEigenVector,
    vkscf: &VKSCF<'_>,
    electronic_ctx: &crate::ElectronicStepContext,
) -> Result<bool, String> {
    let force_max = force::get_max_force(force_total);
    let stress_max = stress::get_max_stress(stress_total);

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
    } else if force_max < control.get_geom_optim_force_tolerance() * FORCE_EV_TO_HA {
        if dwmpi::is_root() {
            println!("\n   {} : {:<5}", "geom_exit_tolerance_reached", geom_iter);
        }
        should_exit = true;
    }

    if !should_exit && geom_iter >= control.get_geom_optim_max_steps() {
        if dwmpi::is_root() {
            println!("\n   {} : {:<5}", "geom_exit_max_steps_reached", geom_iter);
        }
        should_exit = true;
    }

    if should_exit {
        if control.get_wannier90_export() {
            let summary =
                wannier90::write_eig_inputs(control, vkevals, electronic_ctx.global_k_indices())
                    .map_err(|err| format!("wannier90 eig export failed: {}", err))?;
            if dwmpi::is_root() {
                println!();
                println!("   {:-^88}", " wannier90 eig export ");
                for file in summary.written_files.iter() {
                    println!("   wrote {}", file);
                }
                println!("   run `w90-win` and `w90-amn` to generate remaining Wannier90 inputs");
            }

            let ppchi_dir = std::path::Path::new("properties").join("ppchi");
            let ppchi_summary = wannier90::write_ppchi_projected_outputs(
                control,
                crystal,
                pots,
                kpts,
                gvec,
                vkevals,
                electronic_ctx.global_k_indices(),
                vkevecs,
                electronic_ctx.vpwwfc(),
                &ppchi_dir,
            )
            .map_err(|err| format!("PP_CHI projected export failed: {}", err))?;
            if dwmpi::is_root() {
                println!();
                println!("   {:-^88}", " pp_chi projected export ");
                for file in ppchi_summary.written_files.iter() {
                    println!("   wrote {}", file);
                }
                println!(
                    "   rerun `w90-proj --seed {}.ppchi[.up|.dn] --input-dir {}` for alternate broadening",
                    control.get_wannier90_seedname().trim(),
                    ppchi_dir.display()
                );
            }
        }

        crate::post_processing(control, vkevals, vkevecs, vkscf);
    }

    Ok(should_exit)
}
