mod amn;
mod channel;
mod eig;
mod mesh;
mod mmn;
mod nnkp;
mod projected;
mod types;
mod win;

use amn::{build_trial_orbitals, merge_amn_part_files, write_amn_part_file};
use channel::{select_spin_channel, write_overlap_files_for_channel, write_win_for_channel};
use control::{Control, SpinScheme};
use crystal::Crystal;
use dfttypes::{VKEigenValue, VKEigenVector};
use dwconsts::HA_TO_EV;
use eig::{merge_rank_parts, write_local_eig_part_files};
use gvector::GVector;
use mesh::validate_k_mesh;
use projected::write_projmeta_file;
pub use projected::{run_projected_analysis, ProjectedConfig, ProjectedSummary};
use pspot::PSPot;
use pwbasis::PWBasis;
use std::io;
use std::path::Path;
pub use types::ExportSummary;

fn validate_seedname(control: &Control) -> io::Result<&str> {
    let seedname = control.get_wannier90_seedname().trim();
    if seedname.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90_seedname must not be empty",
        ));
    }
    Ok(seedname)
}

pub fn write_win_inputs(
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn kpts::KPTS,
) -> io::Result<ExportSummary> {
    let seedname = validate_seedname(control)?;
    validate_k_mesh(kpts.get_k_mesh())?;

    let mut summary = ExportSummary::default();

    dwmpi::barrier(dwmpi::comm_world());
    if dwmpi::is_root() {
        let pots = PSPot::new(control.get_pot_scheme_enum());
        match control.get_spin_scheme_enum() {
            SpinScheme::NonSpin => {
                summary.written_files.push(write_win_for_channel(
                    seedname, control, crystal, &pots, kpts, None,
                )?);
            }
            SpinScheme::Spin => {
                let up_seed = format!("{}.up", seedname);
                let dn_seed = format!("{}.dn", seedname);
                summary.written_files.push(write_win_for_channel(
                    &up_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    Some("up"),
                )?);
                summary.written_files.push(write_win_for_channel(
                    &dn_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    Some("down"),
                )?);
            }
            SpinScheme::Ncl => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "wannier90 export currently supports nonspin/spin only",
                ));
            }
        }
    }
    dwmpi::barrier(dwmpi::comm_world());

    Ok(summary)
}

pub fn write_overlap_inputs(
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn kpts::KPTS,
) -> io::Result<ExportSummary> {
    let seedname = validate_seedname(control)?;
    let mut summary = ExportSummary::default();

    // Ensure all ranks have reached post-processing after SCF outputs are on disk.
    dwmpi::barrier(dwmpi::comm_world());
    if dwmpi::is_root() {
        let pots = PSPot::new(control.get_pot_scheme_enum());
        match control.get_spin_scheme_enum() {
            SpinScheme::NonSpin => {
                let files = write_overlap_files_for_channel(
                    seedname, control, crystal, &pots, kpts, false, None,
                )?;
                summary.written_files.extend(files);
            }
            SpinScheme::Spin => {
                let up_seed = format!("{}.up", seedname);
                let dn_seed = format!("{}.dn", seedname);

                let up_files = write_overlap_files_for_channel(
                    &up_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    true,
                    Some("up"),
                )?;
                let dn_files = write_overlap_files_for_channel(
                    &dn_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    true,
                    Some("down"),
                )?;

                summary.written_files.extend(up_files);
                summary.written_files.extend(dn_files);
            }
            SpinScheme::Ncl => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "wannier90 export currently supports nonspin/spin only",
                ));
            }
        }
    }
    dwmpi::barrier(dwmpi::comm_world());

    Ok(summary)
}

pub fn write_eig_inputs(
    control: &Control,
    vkevals: &VKEigenValue,
    k_indices: &[usize],
) -> io::Result<ExportSummary> {
    let seedname = validate_seedname(control)?;
    write_eig_inputs_for_seedname(seedname, control.get_spin_scheme_enum(), vkevals, k_indices)
}

pub fn write_ppchi_projected_outputs(
    control: &Control,
    crystal: &Crystal,
    pots: &PSPot,
    kpts: &dyn kpts::KPTS,
    gvec: &GVector,
    vkevals: &VKEigenValue,
    k_indices: &[usize],
    vkevecs: &VKEigenVector,
    all_pwbasis: &[PWBasis],
    out_root: &Path,
) -> io::Result<ExportSummary> {
    let seedname = validate_seedname(control)?;
    let spin_scheme = control.get_spin_scheme_enum();
    let mut summary = ExportSummary::default();

    if dwmpi::is_root() {
        std::fs::create_dir_all(out_root)?;
    }
    dwmpi::barrier(dwmpi::comm_world());

    let ppchi_seed = format!("{}.ppchi", seedname);
    let ppchi_seed_prefix = out_root.join(&ppchi_seed);
    let ppchi_seed_prefix = ppchi_seed_prefix.display().to_string();
    let eig_summary =
        write_eig_inputs_for_seedname(&ppchi_seed_prefix, spin_scheme, vkevals, k_indices)?;
    summary.written_files.extend(eig_summary.written_files);

    let trial_orbitals = build_trial_orbitals(control.get_wannier90_num_wann(), crystal, pots)?;
    let num_proj = trial_orbitals.len();
    let rank = dwmpi::get_comm_world_rank();
    let nrank = dwmpi::get_comm_world_size();

    match spin_scheme {
        SpinScheme::NonSpin => {
            let eigvecs = select_spin_channel(vkevecs, None)?;
            let channel_seed = ppchi_seed.clone();
            let channel_prefix = out_root.join(&channel_seed);
            let channel_prefix = channel_prefix.display().to_string();
            let part_file = format!("{}.amn.part.rank{}", channel_prefix, rank);
            write_amn_part_file(
                &part_file,
                control.get_nband(),
                crystal,
                kpts,
                eigvecs,
                all_pwbasis,
                gvec,
                pots,
                trial_orbitals.as_slice(),
                k_indices,
            )?;

            dwmpi::barrier(dwmpi::comm_world());
            if dwmpi::is_root() {
                let amn_file = format!("{}.amn", channel_prefix);
                merge_amn_part_files(
                    &amn_file,
                    &|merge_rank| format!("{}.amn.part.rank{}", channel_prefix, merge_rank),
                    nrank,
                    control.get_nband(),
                    kpts.get_n_kpts(),
                    num_proj,
                )?;
                let projmeta_path = out_root.join(format!("{}.projmeta", channel_seed));
                write_projmeta_file(&projmeta_path, trial_orbitals.as_slice())?;
                summary.written_files.push(amn_file);
                summary
                    .written_files
                    .push(projmeta_path.display().to_string());

                let analysis_summary = run_projected_analysis(&ProjectedConfig {
                    seedname: channel_seed,
                    sigma_ev: control.get_dos_sigma() * HA_TO_EV,
                    ne: control.get_dos_ne(),
                    emin_ev: None,
                    emax_ev: None,
                    input_dir: out_root.to_path_buf(),
                    out_dir: out_root.join("nonspin"),
                    cache_path: None,
                    validation_tol: ProjectedConfig::default().validation_tol,
                })?;
                summary
                    .written_files
                    .extend(analysis_summary.generated_files.into_iter());
            }
        }
        SpinScheme::Spin => {
            let up_eigvecs = select_spin_channel(vkevecs, Some("up"))?;
            let dn_eigvecs = select_spin_channel(vkevecs, Some("down"))?;
            let up_seed = format!("{}.up", ppchi_seed);
            let dn_seed = format!("{}.dn", ppchi_seed);
            let up_prefix = out_root.join(&up_seed).display().to_string();
            let dn_prefix = out_root.join(&dn_seed).display().to_string();
            write_amn_part_file(
                &format!("{}.amn.part.rank{}", up_prefix, rank),
                control.get_nband(),
                crystal,
                kpts,
                up_eigvecs,
                all_pwbasis,
                gvec,
                pots,
                trial_orbitals.as_slice(),
                k_indices,
            )?;
            write_amn_part_file(
                &format!("{}.amn.part.rank{}", dn_prefix, rank),
                control.get_nband(),
                crystal,
                kpts,
                dn_eigvecs,
                all_pwbasis,
                gvec,
                pots,
                trial_orbitals.as_slice(),
                k_indices,
            )?;

            dwmpi::barrier(dwmpi::comm_world());
            if dwmpi::is_root() {
                let up_amn = format!("{}.amn", up_prefix);
                let dn_amn = format!("{}.amn", dn_prefix);
                merge_amn_part_files(
                    &up_amn,
                    &|merge_rank| format!("{}.amn.part.rank{}", up_prefix, merge_rank),
                    nrank,
                    control.get_nband(),
                    kpts.get_n_kpts(),
                    num_proj,
                )?;
                merge_amn_part_files(
                    &dn_amn,
                    &|merge_rank| format!("{}.amn.part.rank{}", dn_prefix, merge_rank),
                    nrank,
                    control.get_nband(),
                    kpts.get_n_kpts(),
                    num_proj,
                )?;

                let up_projmeta = out_root.join(format!("{}.projmeta", up_seed));
                let dn_projmeta = out_root.join(format!("{}.projmeta", dn_seed));
                write_projmeta_file(&up_projmeta, trial_orbitals.as_slice())?;
                write_projmeta_file(&dn_projmeta, trial_orbitals.as_slice())?;
                summary.written_files.push(up_amn);
                summary.written_files.push(dn_amn);
                summary
                    .written_files
                    .push(up_projmeta.display().to_string());
                summary
                    .written_files
                    .push(dn_projmeta.display().to_string());

                for (channel_seed, channel_dir) in [
                    (up_seed, out_root.join("up")),
                    (dn_seed, out_root.join("dn")),
                ] {
                    let analysis_summary = run_projected_analysis(&ProjectedConfig {
                        seedname: channel_seed,
                        sigma_ev: control.get_dos_sigma() * HA_TO_EV,
                        ne: control.get_dos_ne(),
                        emin_ev: None,
                        emax_ev: None,
                        input_dir: out_root.to_path_buf(),
                        out_dir: channel_dir,
                        cache_path: None,
                        validation_tol: ProjectedConfig::default().validation_tol,
                    })?;
                    summary
                        .written_files
                        .extend(analysis_summary.generated_files.into_iter());
                }
            }
        }
        SpinScheme::Ncl => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "PP_CHI projected export currently supports nonspin/spin only",
            ));
        }
    }

    dwmpi::barrier(dwmpi::comm_world());
    Ok(summary)
}

fn write_eig_inputs_for_seedname(
    seedname: &str,
    spin_scheme: SpinScheme,
    vkevals: &VKEigenValue,
    k_indices: &[usize],
) -> io::Result<ExportSummary> {
    let mut summary = ExportSummary::default();
    write_local_eig_part_files(seedname, vkevals, k_indices)?;
    dwmpi::barrier(dwmpi::comm_world());

    if dwmpi::is_root() {
        match spin_scheme {
            SpinScheme::NonSpin => {
                let eig_file = format!("{}.eig", seedname);
                merge_rank_parts(
                    &eig_file,
                    &|rank| format!("{}.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                summary.written_files.push(eig_file);
            }
            SpinScheme::Spin => {
                let up_eig_file = format!("{}.up.eig", seedname);
                let dn_eig_file = format!("{}.dn.eig", seedname);
                merge_rank_parts(
                    &up_eig_file,
                    &|rank| format!("{}.up.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                merge_rank_parts(
                    &dn_eig_file,
                    &|rank| format!("{}.dn.eig.part.rank{}", seedname, rank),
                    dwmpi::get_comm_world_size(),
                )?;
                summary.written_files.push(up_eig_file);
                summary.written_files.push(dn_eig_file);
            }
            SpinScheme::Ncl => {}
        }
    }

    dwmpi::barrier(dwmpi::comm_world());
    Ok(summary)
}

// End-to-end legacy wrapper:
// 1) write .win
// 2) write .nnkp/.amn/.mmn using saved SCF wavefunctions
// 3) write final .eig from distributed eigenvalue parts
pub fn export(
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn kpts::KPTS,
    vkevals: &VKEigenValue,
    k_indices: &[usize],
) -> io::Result<ExportSummary> {
    let mut summary = ExportSummary::default();

    let win_summary = write_win_inputs(control, crystal, kpts)?;
    summary.written_files.extend(win_summary.written_files);

    let overlap_summary = write_overlap_inputs(control, crystal, kpts)?;
    summary.written_files.extend(overlap_summary.written_files);

    let eig_summary = write_eig_inputs(control, vkevals, k_indices)?;
    summary.written_files.extend(eig_summary.written_files);

    Ok(summary)
}
