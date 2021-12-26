#![allow(warnings)]

pub mod spglib;
pub use spglib::*;

pub mod spglib_sys;
pub use spglib_sys::*;

pub trait SymmetryDriver {
    fn get_n_sym_ops(&self) -> usize;
    fn get_spacegroup_number(&self) -> i32;
    fn get_hall_number(&self) -> i32;
    fn get_rotation(&self, isym: usize) -> &[[i32; 3]; 3];
    fn get_translation(&self, isym: usize) -> &[f64];
    fn operation_on_vector(&self, isym: usize, v: &mut [f64; 3]);
    fn center_vector(&self, v: &mut [f64; 3]);
    fn display(&self);
    fn display_brief(&self);
    fn get_fft_commensurate_ops(
        &self,
        fftmesh: [usize; 3],
        kmesh: [i32; 3],
        symprec: f64,
    ) -> Vec<usize>;
    fn get_sym_atom(&self) -> &[Vec<usize>];
}

pub fn new(
    latt: &[[f64; 3]],
    position: &[[f64; 3]],
    types: &[i32],
    symprec: f64,
) -> Box<dyn SymmetryDriver> {
    let sym: Box<dyn SymmetryDriver>;

    sym = Box::new(SymmetryDriverSPGLIB::new(latt, position, types, symprec));

    sym
}

/// cargo test test_spglib --lib -- --nocapture         
#[test]
fn test_spglib() {
    let mesh: [i32; 3] = [2, 2, 2];
    let is_shift: [i32; 3] = [0, 0, 0];
    let mut lattice = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
    let mut position = [[-0.121, -0.122, -0.123], [0.128, 0.127, 0.126]];
    let types: [i32; 2] = [1, 2];

    let (kpts, mapping, k_unique, nk_unique) = get_ir_reciprocal_mesh(
        mesh.clone(),
        is_shift.clone(),
        &mut lattice,
        &mut position,
        &types,
        1.0E-6,
    );

    for (i, k) in kpts.iter().enumerate() {
        println!("{} {:?} -> {}", i, k, mapping[i]);
    }

    println!("k_unique    : {:?}", k_unique);

    println!("k_degeneracy: {:?}", nk_unique);

    println!("number of ir kpts = {}", k_unique.len());

    let dataset = new(&mut lattice, &mut position, &types, 1.0E-10);

    dataset.display();

    println!("sym_atom: {:?}", dataset.get_sym_atom());

    for i in 0..dataset.get_n_sym_ops() {
        let mut at_pos_0 = position[0].clone();

        dataset.operation_on_vector(i, &mut at_pos_0);
        dataset.center_vector(&mut at_pos_0);

        println!("ops {:2} {:.3?}", i, at_pos_0);
    }

    println!(" kpts = {:?}", kpts);

    for i in 0..dataset.get_n_sym_ops() {
        for ik in 0..kpts.len() {
            let mut k = kpts[ik].clone();

            //dataset.operation_on_vector(i, &mut k);

            //dataset.center_vector(&mut k);

            println!("ik {:2} k = {:.3?} ops {:2} {:.3?}", ik, kpts[ik], i, k);
        }
    }
}
