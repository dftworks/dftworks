#![allow(warnings)]

use control::{Control, SpinScheme};
use dfttypes::*;
use gvector::GVector;
use kpts_distribution::KPointDomain;
use matrix::Matrix;
use ndarray::Array3;
use std::path::Path;
use types::c64;

const RESTART_LATTICE_TOL: f64 = 1.0e-8;
const RESTART_META_TOL: f64 = 1.0e-12;

#[inline]
fn spin_scheme_is_spin(spin_scheme: SpinScheme) -> Result<bool, String> {
    match spin_scheme {
        SpinScheme::NonSpin => Ok(false),
        SpinScheme::Spin => Ok(true),
        SpinScheme::Ncl => Err("restart requested for unsupported spin_scheme='ncl'".to_string()),
    }
}

pub(crate) fn restart_density_files_exist(spin_scheme: SpinScheme) -> bool {
    match spin_scheme_is_spin(spin_scheme) {
        Ok(is_spin) => Hdf5FilePerKCheckpointRepository::default()
            .density_filenames(is_spin)
            .iter()
            .all(|filename| Path::new(filename).exists()),
        Err(_) => false,
    }
}

pub(crate) fn try_load_density_checkpoint(
    spin_scheme: SpinScheme,
    expected_meta: &CheckpointMeta,
    expected_blatt: &lattice::Lattice,
    rgtrans: &rgtransform::RGTransform,
    gvec: &GVector,
    pwden: &pwdensity::PWDensity,
    rhog: &mut RHOG,
    rho_3d: &mut RHOR,
) -> Result<String, String> {
    let repo = Hdf5FilePerKCheckpointRepository::default();
    let is_spin = spin_scheme_is_spin(spin_scheme)?;

    if !restart_density_files_exist(spin_scheme) {
        return Err(match spin_scheme {
            SpinScheme::NonSpin => "restart requested but 'out.scf.rho.hdf5' is missing".to_string(),
            SpinScheme::Spin => {
                "restart requested but spin density checkpoints ('out.scf.rho.up.hdf5' and/or 'out.scf.rho.dn.hdf5') are missing".to_string()
            }
            SpinScheme::Ncl => "restart requested for unsupported spin_scheme='ncl'".to_string(),
        });
    }

    let (checkpoint_blatt, loaded_rho, checkpoint_meta_opt) = repo.load_density(is_spin)?;

    let checkpoint_meta = checkpoint_meta_opt.ok_or_else(|| {
        "checkpoint metadata is missing; regenerate checkpoints with the current schema".to_string()
    })?;

    checkpoint_meta.validate_against(expected_meta, RESTART_META_TOL)?;

    validate_checkpoint_lattice(
        expected_blatt,
        &checkpoint_blatt,
        RESTART_LATTICE_TOL,
        "density",
    )?;

    *rho_3d = loaded_rho;

    match (spin_scheme, rhog, rho_3d) {
        (SpinScheme::NonSpin, RHOG::NonSpin(rhog_ns), RHOR::NonSpin(rho_ns)) => {
            rgtrans.r3d_to_g1d(gvec, pwden, rho_ns.as_slice(), rhog_ns);
            Ok("loaded restart density from out.scf.rho.hdf5".to_string())
        }
        (SpinScheme::Spin, RHOG::Spin(rhog_up, rhog_dn), RHOR::Spin(rho_up, rho_dn)) => {
            rgtrans.r3d_to_g1d(gvec, pwden, rho_up.as_slice(), rhog_up);
            rgtrans.r3d_to_g1d(gvec, pwden, rho_dn.as_slice(), rhog_dn);
            Ok("loaded restart density from out.scf.rho.up.hdf5/out.scf.rho.dn.hdf5".to_string())
        }
        _ => Err("restart density spin-scheme mismatch while loading checkpoint".to_string()),
    }
}

pub(crate) fn try_load_wavefunction_checkpoint(
    spin_scheme: SpinScheme,
    k_domain: &KPointDomain,
    expected_blatt: &lattice::Lattice,
    expected_meta: &CheckpointMeta,
    vpwwfc: &[pwbasis::PWBasis],
    vkevecs: &mut VKEigenVector,
) -> Result<String, String> {
    if k_domain.is_empty() || vpwwfc.is_empty() {
        return Err("skip wavefunction restart: no local k-points on this rank".to_string());
    }
    let repo = Hdf5FilePerKCheckpointRepository::default();
    let is_spin = spin_scheme_is_spin(spin_scheme)?;

    let local_nk = k_domain.len();
    if local_nk != vpwwfc.len() {
        return Err(format!(
            "wavefunction restart mismatch: local_nk={} but local basis count={}",
            local_nk,
            vpwwfc.len()
        ));
    }

    for slot in k_domain.iter() {
        for filename in repo.wavefunction_filenames(is_spin, slot.global_index) {
            if !Path::new(&filename).exists() {
                return Err(format!(
                    "wavefunction restart files are incomplete for local slot {} (global k-index {}): missing '{}'",
                    slot.local_slot,
                    slot.global_index,
                    filename
                ));
            }
        }
    }

    validate_wavefunction_checkpoint_metadata(&repo, is_spin, k_domain, expected_meta)?;

    let (loaded_pwbasis, checkpoint_blatt, loaded_evecs) =
        repo.load_wavefunctions(is_spin, k_domain.global_indices())?;

    validate_checkpoint_lattice(
        expected_blatt,
        &checkpoint_blatt,
        RESTART_LATTICE_TOL,
        "wavefunction",
    )?;

    if loaded_pwbasis.len() != vpwwfc.len() {
        return Err(format!(
            "wavefunction restart mismatch: loaded basis count={} but expected {}",
            loaded_pwbasis.len(),
            vpwwfc.len()
        ));
    }

    for (slot, (loaded, expected)) in k_domain
        .iter()
        .zip(loaded_pwbasis.iter().zip(vpwwfc.iter()))
    {
        if loaded.get_k_index() != slot.global_index {
            return Err(format!(
                "wavefunction restart mismatch at local slot {}: loaded k_index={} expected={}",
                slot.local_slot,
                loaded.get_k_index(),
                slot.global_index
            ));
        }
        if expected.get_k_index() != slot.global_index {
            return Err(format!(
                "wavefunction setup mismatch at local slot {}: expected basis k_index={} but domain expects {}",
                slot.local_slot,
                expected.get_k_index(),
                slot.global_index
            ));
        }
        if loaded.get_n_plane_waves() != expected.get_n_plane_waves() {
            return Err(format!(
                "wavefunction restart mismatch at k_index {}: loaded npw={} expected={}",
                expected.get_k_index(),
                loaded.get_n_plane_waves(),
                expected.get_n_plane_waves()
            ));
        }
    }

    validate_loaded_wavefunction_shapes(&loaded_evecs, vkevecs, k_domain)?;

    *vkevecs = loaded_evecs;

    Ok(format!(
        "loaded restart wavefunctions for local_nk={}",
        local_nk
    ))
}

fn validate_wavefunction_checkpoint_metadata(
    repo: &dyn CheckpointRepository,
    is_spin: bool,
    k_domain: &KPointDomain,
    expected_meta: &CheckpointMeta,
) -> Result<(), String> {
    for slot in k_domain.iter() {
        for filename in repo.wavefunction_filenames(is_spin, slot.global_index) {
            let checkpoint_meta = read_checkpoint_meta_required(&filename)?;
            checkpoint_meta.validate_against(expected_meta, RESTART_META_TOL)?;
        }
    }

    Ok(())
}

fn read_checkpoint_meta_required(filename: &str) -> Result<CheckpointMeta, String> {
    let checkpoint_meta = CheckpointMeta::read_from_path_optional(filename)?;
    checkpoint_meta.ok_or_else(|| {
        format!(
            "checkpoint metadata is missing in '{}'; regenerate checkpoints with the current schema",
            filename
        )
    })
}

fn validate_loaded_wavefunction_shapes(
    loaded: &VKEigenVector,
    expected: &VKEigenVector,
    k_domain: &KPointDomain,
) -> Result<(), String> {
    match (loaded, expected) {
        (VKEigenVector::NonSpin(loaded_ns), VKEigenVector::NonSpin(expected_ns)) => {
            if loaded_ns.len() != expected_ns.len() {
                return Err(format!(
                    "wavefunction restart mismatch: loaded {} k-point blocks, expected {}",
                    loaded_ns.len(),
                    expected_ns.len()
                ));
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_ns.iter().zip(expected_ns.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow()
                    || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
        }
        (
            VKEigenVector::Spin(loaded_up, loaded_dn),
            VKEigenVector::Spin(expected_up, expected_dn),
        ) => {
            if loaded_up.len() != expected_up.len() || loaded_dn.len() != expected_dn.len() {
                return Err(format!(
                    "wavefunction restart mismatch: loaded blocks (up={}, dn={}), expected (up={}, dn={})",
                    loaded_up.len(),
                    loaded_dn.len(),
                    expected_up.len(),
                    expected_dn.len()
                ));
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_up.iter().zip(expected_up.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow()
                    || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch (up) at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
            for (local_slot, (loaded_mat, expected_mat)) in
                loaded_dn.iter().zip(expected_dn.iter()).enumerate()
            {
                if loaded_mat.nrow() != expected_mat.nrow()
                    || loaded_mat.ncol() != expected_mat.ncol()
                {
                    let k_index = k_domain
                        .global_index(local_slot)
                        .unwrap_or(k_domain.global_first_or_zero() + local_slot);
                    return Err(format!(
                        "wavefunction restart shape mismatch (dn) at k_index {}: loaded {}x{}, expected {}x{}",
                        k_index,
                        loaded_mat.nrow(),
                        loaded_mat.ncol(),
                        expected_mat.nrow(),
                        expected_mat.ncol()
                    ));
                }
            }
        }
        _ => {
            return Err(
                "wavefunction restart spin-scheme mismatch between loaded and expected storage"
                    .to_string(),
            )
        }
    }

    Ok(())
}

fn validate_checkpoint_lattice(
    expected: &lattice::Lattice,
    checkpoint: &lattice::Lattice,
    tol: f64,
    checkpoint_kind: &str,
) -> Result<(), String> {
    let max_diff = lattice_max_abs_diff(expected, checkpoint);
    if max_diff > tol {
        Err(format!(
            "{} checkpoint lattice mismatch: max |delta(b)|={:.3e} exceeds tolerance {:.3e}",
            checkpoint_kind, max_diff, tol
        ))
    } else {
        Ok(())
    }
}

fn lattice_max_abs_diff(lhs: &lattice::Lattice, rhs: &lattice::Lattice) -> f64 {
    let lhs_arr = lhs.as_2d_array_row_major();
    let rhs_arr = rhs.as_2d_array_row_major();
    let mut max_diff = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (lhs_arr[i][j] - rhs_arr[i][j]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    max_diff
}

pub(crate) fn checkpoint_meta_for_run(
    control: &Control,
    spin_scheme: SpinScheme,
    k_mesh: [i32; 3],
) -> CheckpointMeta {
    let spin_channels = match spin_scheme {
        SpinScheme::NonSpin => 1,
        SpinScheme::Spin => 2,
        SpinScheme::Ncl => 4,
    };

    let nband_usize = control.get_nband();
    if nband_usize > u32::MAX as usize {
        panic!(
            "nband={} exceeds checkpoint metadata limit {}",
            nband_usize,
            u32::MAX
        );
    }
    let nband = nband_usize as u32;

    CheckpointMeta::new(
        spin_channels,
        nband,
        control.get_ecut(),
        control.get_ecutrho(),
        k_mesh,
    )
}
