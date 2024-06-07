use kscf::*;
use lattice::*;
use matrix::*;
use ndarray::*;
use pwbasis::PWBasis;
use types::*;

use enum_as_inner::EnumAsInner;

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
    pub fn save_hdf5(&self, ik_first: &usize, pwbasis: &[PWBasis], blatt: &Lattice) {
        match self {
            VKEigenVector::NonSpin(v) => {
                for (i, eigen_vec) in v.iter().enumerate() {
                    let ik = ik_first + i;
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
                }
            }
            VKEigenVector::Spin(up, dn) => {
                for (i, eigen_vec) in up.iter().enumerate() {
                    let ik = ik_first + i;
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
                }
                for (i, eigen_vec) in dn.iter().enumerate() {
                    let ik = ik_first + i;
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
                }
            }
        }
    }

    pub fn load_hdf5(&mut self) {
        todo!("Wavefunction cannot be loaded yet!");
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
        match self {
            RHOR::NonSpin(rho_3d) => {
                let hdf5_file = hdf5::File::create("out.scf.rho.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);
            }
            RHOR::Spin(rho_3d_up, rho_3d_dn) => {
                // ---------------------- UP ----------------------
                let hdf5_file = hdf5::File::create("out.scf.rho.up.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d_up.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);

                // ---------------------- DOWN ----------------------
                let hdf5_file = hdf5::File::create("out.scf.rho.dn.hdf5").unwrap();

                let mut group_tmp = hdf5_file.create_group("RhoR").unwrap();
                rho_3d_dn.save_hdf5(&mut group_tmp);

                // Write reciprocal lattice
                let mut group_tmp = hdf5_file.create_group("BLattice").unwrap();
                blatt.save_hdf5(&mut group_tmp);
            }
        }
    }

    pub fn load_hdf5(&mut self) -> Lattice {
        match self {
            RHOR::NonSpin(rho_3d) => {
                let hdf5_file = hdf5::File::open("out.scf.rho.hdf5").unwrap();

                let group_tmp = hdf5_file.group("RhoR").unwrap();
                *rho_3d = Array3::<c64>::load_hdf5(&group_tmp);

                let group_tmp = hdf5_file.group("BLattice").unwrap();
                Lattice::load_hdf5(&group_tmp)
            }
            RHOR::Spin(rho_3d_up, rho_3d_dn) => {
                let hdf5_file_up = hdf5::File::open("out.scf.rho.up.hdf5").unwrap();
                let hdf5_file_dn = hdf5::File::open("out.scf.rho.dn.hdf5").unwrap();

                let group_tmp = hdf5_file_up.group("RhoR").unwrap();
                *rho_3d_up = Array3::<c64>::load_hdf5(&group_tmp);

                let group_tmp = hdf5_file_dn.group("RhoR").unwrap();
                *rho_3d_dn = Array3::<c64>::load_hdf5(&group_tmp);

                let group_tmp = hdf5_file_up.group("BLattice").unwrap();
                Lattice::load_hdf5(&group_tmp)
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
