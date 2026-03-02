use kscf::*;
use lattice::*;
use types::*;
use ndarray::*;
use pwbasis::PWBasis;
use types::*;

use enum_as_inner::EnumAsInner;

mod checkpoint_repo;
pub use checkpoint_repo::{CheckpointCodec, CheckpointRepository, Hdf5FilePerKCheckpointRepository};

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;
const CHECKPOINT_META_GROUP: &str = "CheckpointMeta";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckpointMeta {
    pub schema_version: u32,
    pub spin_channels: u32,
    pub nband: u32,
    pub ecut_wfc: f64,
    pub ecut_rho: f64,
    pub k_mesh: [i32; 3],
}

impl CheckpointMeta {
    pub fn new(
        spin_channels: u32,
        nband: u32,
        ecut_wfc: f64,
        ecut_rho: f64,
        k_mesh: [i32; 3],
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            spin_channels,
            nband,
            ecut_wfc,
            ecut_rho,
            k_mesh,
        }
    }

    pub fn validate_against(&self, expected: &Self, tol: f64) -> Result<(), String> {
        let this = self.migrate_to_current()?;
        let expected = expected.migrate_to_current()?;

        if this.schema_version != expected.schema_version {
            return Err(format!(
                "checkpoint schema mismatch: found v{}, expected v{}",
                this.schema_version, expected.schema_version
            ));
        }
        if this.spin_channels != expected.spin_channels {
            return Err(format!(
                "checkpoint spin mismatch: found {} channel(s), expected {}",
                this.spin_channels, expected.spin_channels
            ));
        }
        if this.nband != expected.nband {
            return Err(format!(
                "checkpoint nband mismatch: found {}, expected {}",
                this.nband, expected.nband
            ));
        }
        if this.k_mesh != expected.k_mesh {
            return Err(format!(
                "checkpoint k-mesh mismatch: found {:?}, expected {:?}",
                this.k_mesh, expected.k_mesh
            ));
        }
        if (this.ecut_wfc - expected.ecut_wfc).abs() > tol {
            return Err(format!(
                "checkpoint ecut_wfc mismatch: found {:.12e}, expected {:.12e}",
                this.ecut_wfc, expected.ecut_wfc
            ));
        }
        if (this.ecut_rho - expected.ecut_rho).abs() > tol {
            return Err(format!(
                "checkpoint ecut_rho mismatch: found {:.12e}, expected {:.12e}",
                this.ecut_rho, expected.ecut_rho
            ));
        }
        Ok(())
    }

    pub fn migrate_to_current(&self) -> Result<Self, String> {
        let mut out = *self;
        match out.schema_version {
            CHECKPOINT_SCHEMA_VERSION => Ok(out),
            // Compatibility hook: allow legacy metadata marked as v0 while
            // preserving all parsed fields and normalizing to current schema.
            0 => {
                out.schema_version = CHECKPOINT_SCHEMA_VERSION;
                Ok(out)
            }
            other => Err(format!(
                "unsupported checkpoint schema version v{} (supported: v{} and migration hook v0)",
                other, CHECKPOINT_SCHEMA_VERSION
            )),
        }
    }

    pub fn write_to_file(&self, file: &hdf5::File) -> Result<(), String> {
        let group = file
            .create_group(CHECKPOINT_META_GROUP)
            .map_err(|e| format!("failed to create group '{}': {}", CHECKPOINT_META_GROUP, e))?;

        group
            .new_dataset_builder()
            .with_data(&[self.schema_version])
            .create("schema_version")
            .map_err(|e| format!("failed to write checkpoint schema_version: {}", e))?;
        group
            .new_dataset_builder()
            .with_data(&[self.spin_channels])
            .create("spin_channels")
            .map_err(|e| format!("failed to write checkpoint spin_channels: {}", e))?;
        group
            .new_dataset_builder()
            .with_data(&[self.nband])
            .create("nband")
            .map_err(|e| format!("failed to write checkpoint nband: {}", e))?;
        group
            .new_dataset_builder()
            .with_data(&[self.ecut_wfc])
            .create("ecut_wfc")
            .map_err(|e| format!("failed to write checkpoint ecut_wfc: {}", e))?;
        group
            .new_dataset_builder()
            .with_data(&[self.ecut_rho])
            .create("ecut_rho")
            .map_err(|e| format!("failed to write checkpoint ecut_rho: {}", e))?;
        group
            .new_dataset_builder()
            .with_data(&self.k_mesh)
            .create("k_mesh")
            .map_err(|e| format!("failed to write checkpoint k_mesh: {}", e))?;

        Ok(())
    }

    pub fn read_from_file_optional(file: &hdf5::File) -> Result<Option<Self>, String> {
        match file.group(CHECKPOINT_META_GROUP) {
            Ok(group) => Self::read_from_group(&group).map(Some),
            Err(_) => Ok(None),
        }
    }

    pub fn read_from_path_optional(path: &str) -> Result<Option<Self>, String> {
        let file =
            hdf5::File::open(path).map_err(|e| format!("failed to open '{}': {}", path, e))?;
        Self::read_from_file_optional(&file).map_err(|e| format!("{} in '{}'", e, path))
    }

    fn read_from_group(group: &hdf5::Group) -> Result<Self, String> {
        let schema_version = read_u32_scalar(group, "schema_version")?;
        let spin_channels = read_u32_scalar(group, "spin_channels")?;
        let nband = read_u32_scalar(group, "nband")?;
        let ecut_wfc = read_f64_scalar(group, "ecut_wfc")?;
        let ecut_rho = read_f64_scalar(group, "ecut_rho")?;
        let k_mesh = read_i32_vec3(group, "k_mesh")?;

        Self {
            schema_version,
            spin_channels,
            nband,
            ecut_wfc,
            ecut_rho,
            k_mesh,
        }
        .migrate_to_current()
    }
}

fn read_u32_scalar(group: &hdf5::Group, name: &str) -> Result<u32, String> {
    let data: Vec<u32> = group
        .dataset(name)
        .map_err(|e| format!("failed to open dataset '{}': {}", name, e))?
        .read()
        .map_err(|e| format!("failed to read dataset '{}': {}", name, e))?
        .to_vec();
    if data.len() != 1 {
        return Err(format!(
            "invalid scalar dataset '{}': expected len=1, got {}",
            name,
            data.len()
        ));
    }
    Ok(data[0])
}

fn read_f64_scalar(group: &hdf5::Group, name: &str) -> Result<f64, String> {
    let data: Vec<f64> = group
        .dataset(name)
        .map_err(|e| format!("failed to open dataset '{}': {}", name, e))?
        .read()
        .map_err(|e| format!("failed to read dataset '{}': {}", name, e))?
        .to_vec();
    if data.len() != 1 {
        return Err(format!(
            "invalid scalar dataset '{}': expected len=1, got {}",
            name,
            data.len()
        ));
    }
    Ok(data[0])
}

fn read_i32_vec3(group: &hdf5::Group, name: &str) -> Result<[i32; 3], String> {
    let data: Vec<i32> = group
        .dataset(name)
        .map_err(|e| format!("failed to open dataset '{}': {}", name, e))?
        .read()
        .map_err(|e| format!("failed to read dataset '{}': {}", name, e))?
        .to_vec();
    if data.len() != 3 {
        return Err(format!(
            "invalid vector dataset '{}': expected len=3, got {}",
            name,
            data.len()
        ));
    }
    Ok([data[0], data[1], data[2]])
}

#[derive(Debug, EnumAsInner)]
pub enum VKEigenValue {
    NonSpin(Vec<Vec<f64>>),
    Spin(Vec<Vec<f64>>, Vec<Vec<f64>>),
}

#[derive(Debug, EnumAsInner)]
pub enum VKEigenVector {
    NonSpin(Vec<Matrix<c64>>),
    Spin(Vec<Matrix<c64>>, Vec<Matrix<c64>>),
}

impl VKEigenVector {
    pub fn save_hdf5(&self, ik_first: usize, pwbasis: &[PWBasis], blatt: &Lattice) {
        self.save_hdf5_with_meta(ik_first, pwbasis, blatt, None);
    }

    pub fn save_hdf5_with_meta(
        &self,
        ik_first: usize,
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) {
        self.try_save_hdf5_with_meta(ik_first, pwbasis, blatt, checkpoint_meta)
            .expect("failed to save VKEigenVector checkpoint");
    }

    pub fn try_save_hdf5_with_meta(
        &self,
        ik_first: usize,
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let mut k_indices = Vec::with_capacity(pwbasis.len());
        for i in 0..pwbasis.len() {
            k_indices.push(ik_first + i);
        }
        self.try_save_hdf5_with_meta_for_kpoints(
            k_indices.as_slice(),
            pwbasis,
            blatt,
            checkpoint_meta,
        )
    }

    pub fn save_hdf5_with_meta_for_kpoints(
        &self,
        k_indices: &[usize],
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) {
        self.try_save_hdf5_with_meta_for_kpoints(k_indices, pwbasis, blatt, checkpoint_meta)
            .expect("failed to save VKEigenVector checkpoint");
    }

    pub fn try_save_hdf5_with_meta_for_kpoints(
        &self,
        k_indices: &[usize],
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        repo.save_wavefunctions(self, k_indices, pwbasis, blatt, checkpoint_meta)
    }

    pub fn load_hdf5(
        is_spin: bool,
        ik_first: usize,
        ik_last: usize,
    ) -> (Vec<PWBasis>, Lattice, VKEigenVector) {
        Self::try_load_hdf5(is_spin, ik_first, ik_last)
            .expect("failed to load VKEigenVector checkpoint")
    }

    pub fn try_load_hdf5(
        is_spin: bool,
        ik_first: usize,
        ik_last: usize,
    ) -> Result<(Vec<PWBasis>, Lattice, VKEigenVector), String> {
        let mut k_indices = Vec::new();
        for ik in ik_first..=ik_last {
            k_indices.push(ik);
        }
        Self::try_load_hdf5_for_kpoints(is_spin, k_indices.as_slice())
    }

    pub fn try_load_hdf5_for_kpoints(
        is_spin: bool,
        k_indices: &[usize],
    ) -> Result<(Vec<PWBasis>, Lattice, VKEigenVector), String> {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        repo.load_wavefunctions(is_spin, k_indices)
    }
}

#[derive(EnumAsInner)]
pub enum VKSCF<'a> {
    NonSpin(Vec<KSCF<'a>>),
    Spin(Vec<KSCF<'a>>, Vec<KSCF<'a>>),
}

#[derive(Debug, EnumAsInner)]
pub enum RHOG {
    NonSpin(Vec<c64>),
    Spin(Vec<c64>, Vec<c64>),
}

#[derive(Debug, EnumAsInner)]
pub enum RHOR {
    NonSpin(Array3<c64>),
    Spin(Array3<c64>, Array3<c64>),
}

impl RHOR {
    pub fn save_hdf5(&self, blatt: &Lattice) {
        self.save_hdf5_with_meta(blatt, None);
    }

    pub fn save_hdf5_with_meta(&self, blatt: &Lattice, checkpoint_meta: Option<&CheckpointMeta>) {
        self.try_save_hdf5_with_meta(blatt, checkpoint_meta)
            .expect("failed to save RHOR checkpoint");
    }

    pub fn try_save_hdf5_with_meta(
        &self,
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) -> Result<(), String> {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        repo.save_density(self, blatt, checkpoint_meta)
    }

    pub fn load_hdf5(is_spin: bool) -> (Lattice, RHOR) {
        let (blatt, rho, _meta) =
            Self::try_load_hdf5(is_spin).expect("failed to load RHOR checkpoint");
        (blatt, rho)
    }

    pub fn try_load_hdf5(
        is_spin: bool,
    ) -> Result<(Lattice, RHOR, Option<CheckpointMeta>), String> {
        let repo = Hdf5FilePerKCheckpointRepository::default();
        repo.load_density(is_spin)
    }
}

#[derive(Debug, EnumAsInner)]
pub enum DRHOR {
    NonSpin(Array3<c64>),
    Spin(Array3<c64>, Array3<c64>),
}

#[derive(Debug, EnumAsInner)]
pub enum VXCG {
    NonSpin(Vec<c64>),
    Spin(Vec<c64>, Vec<c64>),
}

#[derive(Debug, EnumAsInner)]
pub enum VXCR {
    NonSpin(Array3<c64>),
    Spin(Array3<c64>, Array3<c64>),
}

// cargo test  test_dfttypes --lib -- --nocapture
#[test]
fn test_vktypes() {
    let rhog = RHOG::NonSpin(vec![c64 { re: 0.0, im: 0.0 }; 3]);

    let rhog = rhog.as_non_spin().unwrap();

    println!("{:?}", rhog);

    let rhog = RHOG::Spin(
        vec![c64 { re: 1.0, im: 0.0 }; 3],
        vec![c64 { re: 2.0, im: 0.0 }; 3],
    );

    let (up, dn) = rhog.as_spin().unwrap();

    println!("up = {:?}", up);
    println!("dn = {:?}", dn);
}

#[test]
fn test_checkpoint_meta_migration_v0_to_current() {
    let legacy = CheckpointMeta {
        schema_version: 0,
        spin_channels: 2,
        nband: 32,
        ecut_wfc: 30.0,
        ecut_rho: 120.0,
        k_mesh: [4, 4, 1],
    };

    let migrated = legacy
        .migrate_to_current()
        .expect("legacy v0 checkpoint metadata should migrate");
    assert_eq!(migrated.schema_version, CHECKPOINT_SCHEMA_VERSION);
    assert_eq!(migrated.spin_channels, legacy.spin_channels);
    assert_eq!(migrated.nband, legacy.nband);
    assert_eq!(migrated.k_mesh, legacy.k_mesh);
}

#[test]
fn test_checkpoint_meta_validate_accepts_migrated_schema() {
    let expected = CheckpointMeta::new(2, 48, 40.0, 160.0, [6, 6, 2]);
    let legacy_same_payload = CheckpointMeta {
        schema_version: 0,
        spin_channels: 2,
        nband: 48,
        ecut_wfc: 40.0,
        ecut_rho: 160.0,
        k_mesh: [6, 6, 2],
    };

    legacy_same_payload
        .validate_against(&expected, 1.0e-12)
        .expect("v0 payload-compatible metadata should validate via migration hook");
}
