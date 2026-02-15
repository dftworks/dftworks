mod amn;
mod channel;
mod eig;
mod mesh;
mod mmn;
mod nnkp;
mod types;
mod win;

use channel::write_full_channel_data;
use control::{Control, SpinScheme};
use crystal::Crystal;
use dfttypes::VKEigenValue;
use eig::{merge_rank_parts, write_local_eig_part_files};
use mesh::{build_mesh_topology, read_k_shift, validate_k_mesh};
use mpi_sys::MPI_COMM_WORLD;
use pspot::PSPot;
use std::io;

pub use types::ExportSummary;

// End-to-end Wannier90 export pipeline.
//
// Responsibilities:
// 1) validate k-mesh metadata and neighbor topology
// 2) write per-channel Wannier text inputs (.win/.nnkp/.amn/.mmn)
// 3) collect distributed eigenvalue parts into final .eig files
//
// MPI behavior:
// - root rank writes text outputs that require global context
// - all ranks write local .eig parts
// - barriers enforce deterministic handoff points.
pub fn export(
    control: &Control,
    crystal: &Crystal,
    kpts: &dyn kpts::KPTS,
    vkevals: &VKEigenValue,
    ik_first: usize,
) -> io::Result<ExportSummary> {
    let seedname = control.get_wannier90_seedname().trim();
    if seedname.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wannier90_seedname must not be empty",
        ));
    }

    let k_mesh = kpts.get_k_mesh();
    validate_k_mesh(k_mesh)?;
    let k_shift = read_k_shift("in.kmesh")?;
    let fallback_topology = build_mesh_topology(kpts, k_mesh, k_shift)?;

    // Ensure all ranks have reached export after SCF outputs are on disk.
    dwmpi::barrier(MPI_COMM_WORLD);

    let mut summary = ExportSummary::default();

    if dwmpi::is_root() {
        let pots = PSPot::new(control.get_pot_scheme());

        match control.get_spin_scheme_enum() {
            SpinScheme::NonSpin => {
                // Single-channel export.
                let files = write_full_channel_data(
                    seedname,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    &fallback_topology,
                    control.is_spin(),
                    None,
                )?;
                summary.written_files.extend(files);
            }
            SpinScheme::Spin => {
                // Two independent channels with Wannier90's conventional
                // ".up"/".dn" seed naming.
                let up_seed = format!("{}.up", seedname);
                let dn_seed = format!("{}.dn", seedname);

                let up_files = write_full_channel_data(
                    &up_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    &fallback_topology,
                    true,
                    Some("up"),
                )?;
                let dn_files = write_full_channel_data(
                    &dn_seed,
                    control,
                    crystal,
                    &pots,
                    kpts,
                    &fallback_topology,
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
