use matrix::*;
use ndarray::*;
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
