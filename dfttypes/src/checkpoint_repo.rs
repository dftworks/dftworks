#![allow(warnings)]

use crate::{CheckpointMeta, VKEigenVector, RHOR};
use lattice::Lattice;
use types::Matrix;
use ndarray::Array3;
use pwbasis::PWBasis;
use types::c64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointCodec {
    Hdf5FilePerK,
}

pub trait CheckpointRepository {
    fn codec(&self) -> CheckpointCodec;

    fn density_filenames(&self, is_spin: bool) -> Vec<String>;
    fn wavefunction_filenames(&self, is_spin: bool, ik_global: usize) -> Vec<String>;

    fn save_density(
        &self,
        rho: &RHOR,
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String>;

    fn load_density(
        &self,
        is_spin: bool,
    ) -> Result<(Lattice, RHOR, Option<CheckpointMeta>), String>;

    fn save_wavefunctions(
        &self,
        eigenvectors: &VKEigenVector,
        k_indices: &[usize],
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String>;

    fn load_wavefunctions(
        &self,
        is_spin: bool,
        k_indices: &[usize],
    ) -> Result<(Vec<PWBasis>, Lattice, VKEigenVector), String>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Hdf5FilePerKCheckpointRepository;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpinChannelFile {
    NonSpin,
    Up,
    Dn,
}

fn channels_for(is_spin: bool) -> &'static [SpinChannelFile] {
    if is_spin {
        &[SpinChannelFile::Up, SpinChannelFile::Dn]
    } else {
        &[SpinChannelFile::NonSpin]
    }
}

fn for_each_channel<F>(is_spin: bool, mut f: F) -> Result<(), String>
where
    F: FnMut(SpinChannelFile) -> Result<(), String>,
{
    for channel in channels_for(is_spin) {
        f(*channel)?;
    }
    Ok(())
}

fn first_channel(is_spin: bool) -> SpinChannelFile {
    if is_spin {
        SpinChannelFile::Up
    } else {
        SpinChannelFile::NonSpin
    }
}

fn density_filename(channel: SpinChannelFile) -> &'static str {
    match channel {
        SpinChannelFile::NonSpin => "out.scf.rho.hdf5",
        SpinChannelFile::Up => "out.scf.rho.up.hdf5",
        SpinChannelFile::Dn => "out.scf.rho.dn.hdf5",
    }
}

fn wavefunction_filename(channel: SpinChannelFile, ik_global: usize) -> String {
    match channel {
        SpinChannelFile::NonSpin => format!("out.wfc.k.{}.hdf5", ik_global),
        SpinChannelFile::Up => format!("out.wfc.up.k.{}.hdf5", ik_global),
        SpinChannelFile::Dn => format!("out.wfc.dn.k.{}.hdf5", ik_global),
    }
}

fn merge_spin_metadata(
    meta_up: Option<CheckpointMeta>,
    meta_dn: Option<CheckpointMeta>,
) -> Result<Option<CheckpointMeta>, String> {
    match (meta_up, meta_dn) {
        (Some(up), Some(dn)) => {
            if up != dn {
                Err("spin checkpoint metadata mismatch between up/down files".to_string())
            } else {
                Ok(Some(up))
            }
        }
        (None, None) => Ok(None),
        _ => Err("spin checkpoint metadata is present only in one channel file".to_string()),
    }
}

fn density_for_channel<'a>(
    rho: &'a RHOR,
    channel: SpinChannelFile,
) -> Result<&'a Array3<c64>, String> {
    match (rho, channel) {
        (RHOR::NonSpin(rho_ns), SpinChannelFile::NonSpin) => Ok(rho_ns),
        (RHOR::Spin(rho_up, _), SpinChannelFile::Up) => Ok(rho_up),
        (RHOR::Spin(_, rho_dn), SpinChannelFile::Dn) => Ok(rho_dn),
        (RHOR::NonSpin(_), SpinChannelFile::Up | SpinChannelFile::Dn) => Err(
            "density checkpoint write mismatch: nonspin storage requested for spin channel"
                .to_string(),
        ),
        (RHOR::Spin(_, _), SpinChannelFile::NonSpin) => Err(
            "density checkpoint write mismatch: spin storage requested for nonspin channel"
                .to_string(),
        ),
    }
}

fn wavefunctions_for_channel<'a>(
    eigenvectors: &'a VKEigenVector,
    channel: SpinChannelFile,
) -> Result<&'a [Matrix<c64>], String> {
    match (eigenvectors, channel) {
        (VKEigenVector::NonSpin(v), SpinChannelFile::NonSpin) => Ok(v.as_slice()),
        (VKEigenVector::Spin(up, _), SpinChannelFile::Up) => Ok(up.as_slice()),
        (VKEigenVector::Spin(_, dn), SpinChannelFile::Dn) => Ok(dn.as_slice()),
        (VKEigenVector::NonSpin(_), SpinChannelFile::Up | SpinChannelFile::Dn) => Err(
            "wavefunction checkpoint write mismatch: nonspin storage requested for spin channel"
                .to_string(),
        ),
        (VKEigenVector::Spin(_, _), SpinChannelFile::NonSpin) => Err(
            "wavefunction checkpoint write mismatch: spin storage requested for nonspin channel"
                .to_string(),
        ),
    }
}

fn channel_label(channel: SpinChannelFile) -> &'static str {
    match channel {
        SpinChannelFile::NonSpin => "nonspin",
        SpinChannelFile::Up => "up",
        SpinChannelFile::Dn => "dn",
    }
}

fn lattice_max_abs_diff(lhs: &Lattice, rhs: &Lattice) -> f64 {
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

impl Hdf5FilePerKCheckpointRepository {
    fn write_wavefunction_file(
        &self,
        filename: &str,
        eigen_vec: &Matrix<c64>,
        pwbasis: &PWBasis,
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let hdf5_file = hdf5::File::create(filename)
            .map_err(|e| format!("failed to create '{}': {}", filename, e))?;

        let mut eig_group = hdf5_file
            .create_group("EigenVector")
            .map_err(|e| format!("failed to create EigenVector in '{}': {}", filename, e))?;
        eigen_vec.save_hdf5(&mut eig_group);

        let mut pw_group = hdf5_file
            .create_group("PWBasis")
            .map_err(|e| format!("failed to create PWBasis in '{}': {}", filename, e))?;
        pwbasis.save_hdf5(&mut pw_group);

        let mut blatt_group = hdf5_file
            .create_group("BLattice")
            .map_err(|e| format!("failed to create BLattice in '{}': {}", filename, e))?;
        blatt.save_hdf5(&mut blatt_group);

        if let Some(meta) = checkpoint_meta {
            meta.write_to_file(&hdf5_file)
                .map_err(|e| format!("{} in '{}'", e, filename))?;
        }

        Ok(())
    }

    fn read_wavefunction_file(
        &self,
        filename: &str,
        need_basis_and_lattice: bool,
    ) -> Result<(Matrix<c64>, Option<PWBasis>, Option<Lattice>), String> {
        let hdf5_file = hdf5::File::open(filename)
            .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

        let eig_group = hdf5_file
            .group("EigenVector")
            .map_err(|e| format!("failed to open EigenVector in '{}': {}", filename, e))?;
        let eigen_vec = Matrix::<c64>::try_load_hdf5(&eig_group)
            .map_err(|e| format!("{} in '{}'", e, filename))?;

        if !need_basis_and_lattice {
            return Ok((eigen_vec, None, None));
        }

        let pw_group = hdf5_file
            .group("PWBasis")
            .map_err(|e| format!("failed to open PWBasis in '{}': {}", filename, e))?;
        let pwbasis =
            PWBasis::try_load_hdf5(&pw_group).map_err(|e| format!("{} in '{}'", e, filename))?;

        let blatt_group = hdf5_file
            .group("BLattice")
            .map_err(|e| format!("failed to open BLattice in '{}': {}", filename, e))?;
        let blatt =
            Lattice::try_load_hdf5(&blatt_group).map_err(|e| format!("{} in '{}'", e, filename))?;

        Ok((eigen_vec, Some(pwbasis), Some(blatt)))
    }

    fn write_density_file(
        &self,
        filename: &str,
        rho_3d: &Array3<c64>,
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let hdf5_file = hdf5::File::create(filename)
            .map_err(|e| format!("failed to create '{}': {}", filename, e))?;

        let mut rho_group = hdf5_file
            .create_group("RhoR")
            .map_err(|e| format!("failed to create RhoR in '{}': {}", filename, e))?;
        rho_3d.save_hdf5(&mut rho_group);

        let mut blatt_group = hdf5_file
            .create_group("BLattice")
            .map_err(|e| format!("failed to create BLattice in '{}': {}", filename, e))?;
        blatt.save_hdf5(&mut blatt_group);

        if let Some(meta) = checkpoint_meta {
            meta.write_to_file(&hdf5_file)
                .map_err(|e| format!("{} in '{}'", e, filename))?;
        }

        Ok(())
    }

    fn read_density_file(
        &self,
        filename: &str,
    ) -> Result<(Array3<c64>, Lattice, Option<CheckpointMeta>), String> {
        let hdf5_file = hdf5::File::open(filename)
            .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

        let rho_group = hdf5_file
            .group("RhoR")
            .map_err(|e| format!("failed to open RhoR in '{}': {}", filename, e))?;
        let rho_3d = Array3::<c64>::try_load_hdf5(&rho_group)
            .map_err(|e| format!("{} in '{}'", e, filename))?;

        let blatt_group = hdf5_file
            .group("BLattice")
            .map_err(|e| format!("failed to open BLattice in '{}': {}", filename, e))?;
        let blatt =
            Lattice::try_load_hdf5(&blatt_group).map_err(|e| format!("{} in '{}'", e, filename))?;

        let meta = CheckpointMeta::read_from_file_optional(&hdf5_file)
            .map_err(|e| format!("{} in '{}'", e, filename))?;

        Ok((rho_3d, blatt, meta))
    }
}

impl CheckpointRepository for Hdf5FilePerKCheckpointRepository {
    fn codec(&self) -> CheckpointCodec {
        CheckpointCodec::Hdf5FilePerK
    }

    fn density_filenames(&self, is_spin: bool) -> Vec<String> {
        channels_for(is_spin)
            .iter()
            .map(|channel| density_filename(*channel).to_string())
            .collect()
    }

    fn wavefunction_filenames(&self, is_spin: bool, ik_global: usize) -> Vec<String> {
        channels_for(is_spin)
            .iter()
            .map(|channel| wavefunction_filename(*channel, ik_global))
            .collect()
    }

    fn save_density(
        &self,
        rho: &RHOR,
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let is_spin = matches!(rho, RHOR::Spin(_, _));
        for_each_channel(is_spin, |channel| {
            let rho_3d = density_for_channel(rho, channel)?;
            self.write_density_file(density_filename(channel), rho_3d, blatt, checkpoint_meta)
        })
    }

    fn load_density(
        &self,
        is_spin: bool,
    ) -> Result<(Lattice, RHOR, Option<CheckpointMeta>), String> {
        const SPIN_LATTICE_MISMATCH_TOL: f64 = 1.0e-12;

        let mut blatt: Option<Lattice> = None;
        let mut rho_ns: Option<Array3<c64>> = None;
        let mut rho_up: Option<Array3<c64>> = None;
        let mut rho_dn: Option<Array3<c64>> = None;
        let mut meta_ns: Option<CheckpointMeta> = None;
        let mut meta_up: Option<CheckpointMeta> = None;
        let mut meta_dn: Option<CheckpointMeta> = None;

        for_each_channel(is_spin, |channel| {
            let filename = density_filename(channel);
            let (rho_channel, blatt_channel, meta_channel) = self.read_density_file(filename)?;

            if let Some(ref first_blatt) = blatt {
                let max_diff = lattice_max_abs_diff(first_blatt, &blatt_channel);
                if max_diff > SPIN_LATTICE_MISMATCH_TOL {
                    return Err(format!(
                        "density checkpoint lattice mismatch between channel files: '{}' differs from first channel by max |delta(b)|={:.3e} (tol={:.1e})",
                        filename, max_diff, SPIN_LATTICE_MISMATCH_TOL
                    ));
                }
            } else {
                blatt = Some(blatt_channel);
            }

            match channel {
                SpinChannelFile::NonSpin => {
                    rho_ns = Some(rho_channel);
                    meta_ns = meta_channel;
                }
                SpinChannelFile::Up => {
                    rho_up = Some(rho_channel);
                    meta_up = meta_channel;
                }
                SpinChannelFile::Dn => {
                    rho_dn = Some(rho_channel);
                    meta_dn = meta_channel;
                }
            }

            Ok(())
        })?;

        let blatt =
            blatt.ok_or_else(|| "density checkpoint load produced no channels".to_string())?;
        if is_spin {
            let up = rho_up
                .ok_or_else(|| "missing spin-up density channel while loading".to_string())?;
            let dn = rho_dn
                .ok_or_else(|| "missing spin-down density channel while loading".to_string())?;
            let meta = merge_spin_metadata(meta_up, meta_dn)?;
            Ok((blatt, RHOR::Spin(up, dn), meta))
        } else {
            let rho = rho_ns
                .ok_or_else(|| "missing non-spin density channel while loading".to_string())?;
            Ok((blatt, RHOR::NonSpin(rho), meta_ns))
        }
    }

    fn save_wavefunctions(
        &self,
        eigenvectors: &VKEigenVector,
        k_indices: &[usize],
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        if k_indices.len() != pwbasis.len() {
            return Err(format!(
                "wavefunction save mismatch: k_indices={} but pwbasis={}",
                k_indices.len(),
                pwbasis.len()
            ));
        }

        let is_spin = matches!(eigenvectors, VKEigenVector::Spin(_, _));
        for_each_channel(is_spin, |channel| {
            let vectors = wavefunctions_for_channel(eigenvectors, channel)?;
            if vectors.len() != pwbasis.len() {
                return Err(format!(
                    "wavefunction save mismatch: {} channel eigenvectors={} but pwbasis={}",
                    channel_label(channel),
                    vectors.len(),
                    pwbasis.len()
                ));
            }

            for (i, eigen_vec) in vectors.iter().enumerate() {
                let filename = wavefunction_filename(channel, k_indices[i]);
                self.write_wavefunction_file(
                    filename.as_str(),
                    eigen_vec,
                    &pwbasis[i],
                    blatt,
                    checkpoint_meta,
                )?;
            }

            Ok(())
        })
    }

    fn load_wavefunctions(
        &self,
        is_spin: bool,
        k_indices: &[usize],
    ) -> Result<(Vec<PWBasis>, Lattice, VKEigenVector), String> {
        if k_indices.is_empty() {
            return Err("cannot load VKEigenVector checkpoint for empty k-point list".to_string());
        }

        let first = first_channel(is_spin);
        let mut pwbasis_vec = Vec::<PWBasis>::with_capacity(k_indices.len());
        let mut blatt = Lattice::default();
        let mut vec_nonspin: Option<Vec<Matrix<c64>>> = None;
        let mut vec_up: Option<Vec<Matrix<c64>>> = None;
        let mut vec_dn: Option<Vec<Matrix<c64>>> = None;

        for_each_channel(is_spin, |channel| {
            let mut vecs = Vec::<Matrix<c64>>::with_capacity(k_indices.len());

            for &ik in k_indices.iter() {
                let filename = wavefunction_filename(channel, ik);
                let need_basis = channel == first;
                let (vec, basis_opt, blatt_opt) =
                    self.read_wavefunction_file(filename.as_str(), need_basis)?;
                vecs.push(vec);

                if let Some(basis) = basis_opt {
                    pwbasis_vec.push(basis);
                }
                if let Some(b) = blatt_opt {
                    blatt = b;
                }
            }

            match channel {
                SpinChannelFile::NonSpin => vec_nonspin = Some(vecs),
                SpinChannelFile::Up => vec_up = Some(vecs),
                SpinChannelFile::Dn => vec_dn = Some(vecs),
            }

            Ok(())
        })?;

        let eigenvectors = if is_spin {
            let up = vec_up
                .ok_or_else(|| "missing spin-up wavefunction channel while loading".to_string())?;
            let dn = vec_dn.ok_or_else(|| {
                "missing spin-down wavefunction channel while loading".to_string()
            })?;
            VKEigenVector::Spin(up, dn)
        } else {
            let ns = vec_nonspin
                .ok_or_else(|| "missing non-spin wavefunction channel while loading".to_string())?;
            VKEigenVector::NonSpin(ns)
        };

        Ok((pwbasis_vec, blatt, eigenvectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_filenames_follow_spin_channels() {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        assert_eq!(repo.density_filenames(false), vec!["out.scf.rho.hdf5"]);
        assert_eq!(
            repo.density_filenames(true),
            vec!["out.scf.rho.up.hdf5", "out.scf.rho.dn.hdf5"]
        );
    }

    #[test]
    fn test_wavefunction_filenames_follow_spin_channels() {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        assert_eq!(
            repo.wavefunction_filenames(false, 7),
            vec!["out.wfc.k.7.hdf5"]
        );
        assert_eq!(
            repo.wavefunction_filenames(true, 7),
            vec!["out.wfc.up.k.7.hdf5", "out.wfc.dn.k.7.hdf5"]
        );
    }

    #[test]
    fn test_merge_spin_metadata_requires_both_or_none() {
        let meta = CheckpointMeta::new(2, 16, 30.0, 120.0, [2, 2, 2]);
        assert_eq!(
            merge_spin_metadata(Some(meta), Some(meta)).expect("identical metadata should merge"),
            Some(meta)
        );
        assert!(merge_spin_metadata(Some(meta), None).is_err());
    }
}
