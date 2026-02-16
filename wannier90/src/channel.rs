use crate::amn::{build_trial_orbitals, build_trial_orbitals_from_nnkp, write_amn_file};
use crate::mmn::write_mmn_file;
use crate::nnkp::build_topology_from_wannier90_pp;
use crate::win::write_win_file;
use control::Control;
use crystal::Crystal;
use dfttypes::VKEigenVector;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use pspot::PSPot;
use std::io;
use std::path::Path;
use types::c64;

pub(crate) fn write_win_for_channel(
    channel_seed: &str,
    control: &Control,
    crystal: &Crystal,
    pots: &PSPot,
    kpts: &dyn KPTS,
    spin_channel: Option<&str>,
) -> io::Result<String> {
    // The generated .win documents the default pseudo-atomic projector list.
    let trial_orbitals = build_trial_orbitals(control.get_wannier90_num_wann(), crystal, pots)?;
    let win_file = format!("{}.win", channel_seed);
    write_win_file(
        &win_file,
        control,
        crystal,
        kpts,
        spin_channel,
        &trial_orbitals,
    )?;
    Ok(win_file)
}

pub(crate) fn write_overlap_files_for_channel(
    channel_seed: &str,
    control: &Control,
    crystal: &Crystal,
    pots: &PSPot,
    kpts: &dyn KPTS,
    is_spin: bool,
    spin_channel: Option<&str>,
) -> io::Result<Vec<String>> {
    let nkpt = kpts.get_n_kpts();
    // AMN/MMN need saved wavefunction files from SCF output.
    ensure_wfc_files_present(is_spin, nkpt)?;

    // `w90-amn` requires `wannier90.x -pp` so that .nnkp is generated in the
    // canonical Wannier90 format (including projection semantics).
    let nnkp_file = format!("{}.nnkp", channel_seed);
    let topology = build_topology_from_wannier90_pp(channel_seed, kpts.get_n_kpts())?;

    // Drive projector semantics from nnkp (QE-style interface behavior).
    let trial_orbitals = build_trial_orbitals_from_nnkp(
        &nnkp_file,
        control.get_wannier90_num_wann(),
        crystal,
        pots,
    )?;

    let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
    let [n1, n2, n3] = fftgrid.get_size();
    let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);

    // Read wavefunctions/eigenvectors produced by SCF from HDF5 outputs.
    let (all_pwbasis, _blatt, all_eigvecs) = VKEigenVector::load_hdf5(is_spin, 0, nkpt - 1);
    let eigvecs = select_spin_channel(&all_eigvecs, spin_channel)?;

    let amn_file = format!("{}.amn", channel_seed);
    write_amn_file(
        &amn_file,
        control.get_nband(),
        crystal,
        kpts,
        eigvecs,
        &all_pwbasis,
        &gvec,
        pots,
        &trial_orbitals,
    )?;

    let mmn_file = format!("{}.mmn", channel_seed);
    write_mmn_file(
        &mmn_file,
        control.get_nband(),
        &topology,
        eigvecs,
        &all_pwbasis,
        &gvec,
    )?;

    Ok(vec![nnkp_file, amn_file, mmn_file])
}

fn select_spin_channel<'a>(
    all_eigvecs: &'a VKEigenVector,
    spin_channel: Option<&str>,
) -> io::Result<&'a [Matrix<c64>]> {
    // Enforce explicit channel selection so we never mix spin-resolved data.
    match (all_eigvecs, spin_channel) {
        (VKEigenVector::NonSpin(v), None) => Ok(v.as_slice()),
        (VKEigenVector::Spin(up, _), Some("up")) => Ok(up.as_slice()),
        (VKEigenVector::Spin(_, dn), Some("down")) => Ok(dn.as_slice()),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inconsistent spin channel selection for Wannier90 export",
        )),
    }
}

fn ensure_wfc_files_present(is_spin: bool, nkpt: usize) -> io::Result<()> {
    // Sanity check up front to fail fast with a clear message.
    for ik in 0..nkpt {
        let filename = if is_spin {
            format!("out.wfc.up.k.{}.hdf5", ik)
        } else {
            format!("out.wfc.k.{}.hdf5", ik)
        };

        if !Path::new(&filename).exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "missing required wavefunction file '{}'; run SCF with save_wfc=true before Wannier90 post-processing",
                    filename
                ),
            ));
        }
    }
    Ok(())
}
