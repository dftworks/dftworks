mod amn;
mod channel;
mod eig;
mod mesh;
mod mmn;
mod nnkp;
mod projected;
mod types;
mod win;

use channel::{write_overlap_files_for_channel, write_win_for_channel};
use control::{Control, SpinScheme};
use crystal::Crystal;
use dfttypes::VKEigenValue;
use eig::{merge_rank_parts, write_local_eig_part_files};
use mesh::validate_k_mesh;
use mpi_sys::MPI_COMM_WORLD;
pub use projected::{run_projected_analysis, ProjectedConfig, ProjectedSummary};
use pspot::PSPot;
use std::io;
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

    dwmpi::barrier(MPI_COMM_WORLD);
    if dwmpi::is_root() {
        let pots = PSPot::new(control.get_pot_scheme());
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
    dwmpi::barrier(MPI_COMM_WORLD);

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
    dwmpi::barrier(MPI_COMM_WORLD);
    if dwmpi::is_root() {
        let pots = PSPot::new(control.get_pot_scheme());
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
    dwmpi::barrier(MPI_COMM_WORLD);

    Ok(summary)
}

pub fn write_eig_inputs(
    control: &Control,
    vkevals: &VKEigenValue,
    ik_first: usize,
) -> io::Result<ExportSummary> {
    let seedname = validate_seedname(control)?;
    let mut summary = ExportSummary::default();
    write_local_eig_part_files(seedname, vkevals, ik_first)?;
    dwmpi::barrier(MPI_COMM_WORLD);

    if dwmpi::is_root() {
        // Merge rank-local .eig parts into one file per exported channel.
        match control.get_spin_scheme_enum() {
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

    dwmpi::barrier(MPI_COMM_WORLD);
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
    ik_first: usize,
) -> io::Result<ExportSummary> {
    let mut summary = ExportSummary::default();

    let win_summary = write_win_inputs(control, crystal, kpts)?;
    summary.written_files.extend(win_summary.written_files);

    let overlap_summary = write_overlap_inputs(control, crystal, kpts)?;
    summary.written_files.extend(overlap_summary.written_files);

    let eig_summary = write_eig_inputs(control, vkevals, ik_first)?;
    summary.written_files.extend(eig_summary.written_files);

    Ok(summary)
}
