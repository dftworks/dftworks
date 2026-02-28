use kscf::*;
use lattice::*;
use matrix::*;
use ndarray::*;
use pwbasis::PWBasis;
use types::*;

use enum_as_inner::EnumAsInner;

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
        if self.schema_version != expected.schema_version {
            return Err(format!(
                "checkpoint schema mismatch: found v{}, expected v{}",
                self.schema_version, expected.schema_version
            ));
        }
        if self.spin_channels != expected.spin_channels {
            return Err(format!(
                "checkpoint spin mismatch: found {} channel(s), expected {}",
                self.spin_channels, expected.spin_channels
            ));
        }
        if self.nband != expected.nband {
            return Err(format!(
                "checkpoint nband mismatch: found {}, expected {}",
                self.nband, expected.nband
            ));
        }
        if self.k_mesh != expected.k_mesh {
            return Err(format!(
                "checkpoint k-mesh mismatch: found {:?}, expected {:?}",
                self.k_mesh, expected.k_mesh
            ));
        }
        if (self.ecut_wfc - expected.ecut_wfc).abs() > tol {
            return Err(format!(
                "checkpoint ecut_wfc mismatch: found {:.12e}, expected {:.12e}",
                self.ecut_wfc, expected.ecut_wfc
            ));
        }
        if (self.ecut_rho - expected.ecut_rho).abs() > tol {
            return Err(format!(
                "checkpoint ecut_rho mismatch: found {:.12e}, expected {:.12e}",
                self.ecut_rho, expected.ecut_rho
            ));
        }
        Ok(())
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

        Ok(Self {
            schema_version,
            spin_channels,
            nband,
            ecut_wfc,
            ecut_rho,
            k_mesh,
        })
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
        let mut k_indices = Vec::with_capacity(pwbasis.len());
        for i in 0..pwbasis.len() {
            k_indices.push(ik_first + i);
        }
        self.save_hdf5_with_meta_for_kpoints(
            k_indices.as_slice(),
            pwbasis,
            blatt,
            checkpoint_meta,
        );
    }

    pub fn save_hdf5_with_meta_for_kpoints(
        &self,
        k_indices: &[usize],
        pwbasis: &[PWBasis],
        blatt: &Lattice,
        checkpoint_meta: Option<&CheckpointMeta>,
    ) {
        assert_eq!(k_indices.len(), pwbasis.len());
        match self {
            VKEigenVector::NonSpin(v) => {
                assert_eq!(v.len(), pwbasis.len());
                for (i, eigen_vec) in v.iter().enumerate() {
                    let ik = k_indices[i];
                    let filename = format!("out.wfc.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::create(filename).unwrap();

                    let mut group_tmp = hdf5_file.create_group("EigenVector").unwrap();
                    eigen_vec.save_hdf5(&mut group_tmp);

                    // Write PWBasis information
                    let mut group_tmp = hdf5_file.create_group("PWBasis").unwrap();
                    pwbasis.get(i).unwrap().save_hdf5(&mut group_tmp);

                    // Write reciprocal lattice
                    let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                    blatt.save_hdf5(&mut group_tmp);

                    if let Some(meta) = checkpoint_meta {
                        meta.write_to_file(&hdf5_file)
                            .expect("failed to write checkpoint metadata");
                    }
                }
            }
            VKEigenVector::Spin(up, dn) => {
                assert_eq!(up.len(), pwbasis.len());
                assert_eq!(dn.len(), pwbasis.len());
                for (i, eigen_vec) in up.iter().enumerate() {
                    let ik = k_indices[i];
                    let filename = format!("out.wfc.up.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::create(filename).unwrap();

                    let mut group_tmp = hdf5_file.create_group("EigenVector").unwrap();
                    eigen_vec.save_hdf5(&mut group_tmp);

                    // Write PWBasis information
                    let mut group_tmp = hdf5_file.create_group("PWBasis").unwrap();
                    pwbasis.get(i).unwrap().save_hdf5(&mut group_tmp);

                    // Write reciprocal lattice
                    let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                    blatt.save_hdf5(&mut group_tmp);

                    if let Some(meta) = checkpoint_meta {
                        meta.write_to_file(&hdf5_file)
                            .expect("failed to write checkpoint metadata");
                    }
                }
                for (i, eigen_vec) in dn.iter().enumerate() {
                    let ik = k_indices[i];
                    let filename = format!("out.wfc.dn.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::create(filename).unwrap();

                    let mut group_tmp = hdf5_file.create_group("EigenVector").unwrap();
                    eigen_vec.save_hdf5(&mut group_tmp);

                    // Write PWBasis information
                    let mut group_tmp = hdf5_file.create_group("PWBasis").unwrap();
                    pwbasis.get(i).unwrap().save_hdf5(&mut group_tmp);

                    // Write reciprocal lattice
                    let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                    blatt.save_hdf5(&mut group_tmp);

                    if let Some(meta) = checkpoint_meta {
                        meta.write_to_file(&hdf5_file)
                            .expect("failed to write checkpoint metadata");
                    }
                }
            }
        }
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
        let nk = k_indices.len();
        if nk == 0 {
            return Err("cannot load VKEigenVector checkpoint for empty k-point list".to_string());
        }
        match is_spin {
            false => {
                let mut eigen_vecs = Vec::<Matrix<c64>>::with_capacity(nk);
                let mut pwbasis_vec = Vec::<PWBasis>::with_capacity(nk);
                let mut blatt = Lattice::default();

                for &ik in k_indices.iter() {
                    let filename = format!("out.wfc.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::open(&filename)
                        .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

                    let group_tmp = hdf5_file
                        .group("EigenVector")
                        .map_err(|e| format!("failed to open EigenVector in '{}': {}", filename, e))?;
                    let eigen_vec_tmp = Matrix::<c64>::try_load_hdf5(&group_tmp)
                        .map_err(|e| format!("{} in '{}'", e, filename))?;
                    eigen_vecs.push(eigen_vec_tmp);

                    // Load PWBasis information
                    let group_tmp = hdf5_file
                        .group("PWBasis")
                        .map_err(|e| format!("failed to open PWBasis in '{}': {}", filename, e))?;
                    let pwbasis_tmp = PWBasis::try_load_hdf5(&group_tmp)
                        .map_err(|e| format!("{} in '{}'", e, filename))?;
                    pwbasis_vec.push(pwbasis_tmp);

                    // Load reciprocal lattice
                    let group_tmp = hdf5_file
                        .group("BLattice")
                        .map_err(|e| format!("failed to open BLattice in '{}': {}", filename, e))?;
                    blatt = Lattice::try_load_hdf5(&group_tmp)
                        .map_err(|e| format!("{} in '{}'", e, filename))?;
                }
                Ok((pwbasis_vec, blatt, VKEigenVector::NonSpin(eigen_vecs)))
            }
            true => {
                let mut eigen_vecs_up = Vec::<Matrix<c64>>::with_capacity(nk);
                let mut eigen_vecs_dn = Vec::<Matrix<c64>>::with_capacity(nk);
                let mut pwbasis_vec = Vec::<PWBasis>::with_capacity(nk);
                let mut blatt = Lattice::default();

                for &ik in k_indices.iter() {
                    let filename = format!("out.wfc.up.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::open(&filename)
                        .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

                    let group_tmp = hdf5_file
                        .group("EigenVector")
                        .map_err(|e| format!("failed to open EigenVector in '{}': {}", filename, e))?;
                    eigen_vecs_up.push(
                        Matrix::<c64>::try_load_hdf5(&group_tmp)
                            .map_err(|e| format!("{} in '{}'", e, filename))?,
                    );

                    // Load PWBasis information
                    let group_tmp = hdf5_file
                        .group("PWBasis")
                        .map_err(|e| format!("failed to open PWBasis in '{}': {}", filename, e))?;
                    pwbasis_vec.push(
                        PWBasis::try_load_hdf5(&group_tmp)
                            .map_err(|e| format!("{} in '{}'", e, filename))?,
                    );

                    // Load reciprocal lattice
                    let group_tmp = hdf5_file
                        .group("BLattice")
                        .map_err(|e| format!("failed to open BLattice in '{}': {}", filename, e))?;
                    blatt = Lattice::try_load_hdf5(&group_tmp)
                        .map_err(|e| format!("{} in '{}'", e, filename))?;
                }
                for &ik in k_indices.iter() {
                    let filename = format!("out.wfc.dn.k.{}.hdf5", ik);
                    let hdf5_file = hdf5::File::open(&filename)
                        .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

                    let group_tmp = hdf5_file
                        .group("EigenVector")
                        .map_err(|e| format!("failed to open EigenVector in '{}': {}", filename, e))?;
                    eigen_vecs_dn.push(
                        Matrix::<c64>::try_load_hdf5(&group_tmp)
                            .map_err(|e| format!("{} in '{}'", e, filename))?,
                    );
                }
                Ok((
                    pwbasis_vec,
                    blatt,
                    VKEigenVector::Spin(eigen_vecs_up, eigen_vecs_dn),
                ))
            }
        }
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
        match self {
            RHOR::NonSpin(rho_3d) => {
                let hdf5_file = hdf5::File::create("out.scf.rho.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);

                if let Some(meta) = checkpoint_meta {
                    meta.write_to_file(&hdf5_file)
                        .expect("failed to write checkpoint metadata");
                }
            }
            RHOR::Spin(rho_3d_up, rho_3d_dn) => {
                // ---------------------- UP ----------------------
                let hdf5_file = hdf5::File::create("out.scf.rho.up.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d_up.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);

                if let Some(meta) = checkpoint_meta {
                    meta.write_to_file(&hdf5_file)
                        .expect("failed to write checkpoint metadata");
                }

                // ---------------------- DOWN ----------------------
                let hdf5_file = hdf5::File::create("out.scf.rho.dn.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d_dn.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);

                if let Some(meta) = checkpoint_meta {
                    meta.write_to_file(&hdf5_file)
                        .expect("failed to write checkpoint metadata");
                }
            }
        }
    }

    pub fn load_hdf5(is_spin: bool) -> (Lattice, RHOR) {
        let (blatt, rho, _meta) =
            Self::try_load_hdf5(is_spin).expect("failed to load RHOR checkpoint");
        (blatt, rho)
    }

    pub fn try_load_hdf5(
        is_spin: bool,
    ) -> Result<(Lattice, RHOR, Option<CheckpointMeta>), String> {
        match is_spin {
            false => {
                let filename = "out.scf.rho.hdf5";
                let hdf5_file = hdf5::File::open(filename)
                    .map_err(|e| format!("failed to open '{}': {}", filename, e))?;

                let group_tmp = hdf5_file
                    .group("RhoR")
                    .map_err(|e| format!("failed to open RhoR in '{}': {}", filename, e))?;
                let rho_3d = Array3::<c64>::try_load_hdf5(&group_tmp)
                    .map_err(|e| format!("{} in '{}'", e, filename))?;

                let group_tmp = hdf5_file
                    .group("BLattice")
                    .map_err(|e| format!("failed to open BLattice in '{}': {}", filename, e))?;
                let blatt = Lattice::try_load_hdf5(&group_tmp)
                    .map_err(|e| format!("{} in '{}'", e, filename))?;

                let meta = CheckpointMeta::read_from_file_optional(&hdf5_file)
                    .map_err(|e| format!("{} in '{}'", e, filename))?;

                Ok((blatt, RHOR::NonSpin(rho_3d), meta))
            }
            true => {
                // ---------------------- UP ----------------------
                let filename_up = "out.scf.rho.up.hdf5";
                let hdf5_file_up = hdf5::File::open(filename_up)
                    .map_err(|e| format!("failed to open '{}': {}", filename_up, e))?;

                let group_tmp = hdf5_file_up
                    .group("RhoR")
                    .map_err(|e| format!("failed to open RhoR in '{}': {}", filename_up, e))?;
                let rho_3d_up = Array3::<c64>::try_load_hdf5(&group_tmp)
                    .map_err(|e| format!("{} in '{}'", e, filename_up))?;

                // ---------------------- DOWN ----------------------
                let filename_dn = "out.scf.rho.dn.hdf5";
                let hdf5_file_dn = hdf5::File::open(filename_dn)
                    .map_err(|e| format!("failed to open '{}': {}", filename_dn, e))?;

                let group_tmp = hdf5_file_dn
                    .group("RhoR")
                    .map_err(|e| format!("failed to open RhoR in '{}': {}", filename_dn, e))?;
                let rho_3d_dn = Array3::<c64>::try_load_hdf5(&group_tmp)
                    .map_err(|e| format!("{} in '{}'", e, filename_dn))?;

                // Reciprocal lattice (identical for up/down)
                let group_tmp = hdf5_file_up
                    .group("BLattice")
                    .map_err(|e| format!("failed to open BLattice in '{}': {}", filename_up, e))?;
                let blatt = Lattice::try_load_hdf5(&group_tmp)
                    .map_err(|e| format!("{} in '{}'", e, filename_up))?;

                let meta_up = CheckpointMeta::read_from_file_optional(&hdf5_file_up)
                    .map_err(|e| format!("{} in '{}'", e, filename_up))?;
                let meta_dn = CheckpointMeta::read_from_file_optional(&hdf5_file_dn)
                    .map_err(|e| format!("{} in '{}'", e, filename_dn))?;

                let meta = match (meta_up, meta_dn) {
                    (Some(up), Some(dn)) => {
                        if up != dn {
                            return Err("spin checkpoint metadata mismatch between up/down files"
                                .to_string());
                        }
                        Some(up)
                    }
                    (None, None) => None,
                    _ => {
                        return Err(
                            "spin checkpoint metadata is present only in one channel file".to_string()
                        )
                    }
                };

                Ok((blatt, RHOR::Spin(rho_3d_up, rho_3d_dn), meta))
            }
        }
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
