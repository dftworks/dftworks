#![allow(warnings)]

use itertools::Itertools;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};

use crate::spglib_sys as ffi;
use crate::spglib_sys::SpglibDataset;

use crate::SymmetryDriver;

pub struct SymmetryDriverSPGLIB {
    spacegroup_number: i32,
    hall_number: i32,
    international_symbol: String,
    hall_symbol: String,
    choice: String,
    transformation_matrix: [[f64; 3]; 3],
    origin_shift: [f64; 3],
    n_operations: usize,
    rotations: Vec<[[i32; 3]; 3]>,
    translations: Vec<[f64; 3]>,
    n_atoms: usize,
    wyckoffs: Vec<i32>,
    site_symmetry_symbols: Vec<String>,
    equivalent_atoms: Vec<i32>,
    crystallographic_orbits: Vec<i32>,
    primitive_lattice: [[f64; 3]; 3],
    mapping_to_primitive: Vec<i32>,
    n_std_atoms: usize,
    std_lattice: [[f64; 3]; 3],
    std_types: Vec<i32>,
    std_positions: Vec<[f64; 3]>,
    std_rotation_matrix: [[f64; 3]; 3],
    std_mapping_to_primitive: Vec<i32>,
    pointgroup_symbol: String,

    sym_atom: Vec<Vec<usize>>, // sym_atom[natoms][nsym]
}

impl SymmetryDriver for SymmetryDriverSPGLIB {
    fn get_sym_atom(&self) -> &[Vec<usize>] {
        &self.sym_atom
    }

    fn get_fft_commensurate_ops(
        &self,
        fftmesh: [usize; 3],
        kmesh: [i32; 3],
        symprec: f64,
    ) -> Vec<usize> {
        let mut viops = Vec::new();

        let n1 = fftmesh[0] as f64;
        let n2 = fftmesh[1] as f64;
        let n3 = fftmesh[2] as f64;

        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for i in 0..self.get_n_sym_ops() {
            let ft = self.get_translation(i);

            x = ft[0] * n1;
            y = ft[1] * n2;
            z = ft[2] * n3;

            //println!("ns = {}, {}, {}, {}", i, x, y, z);

            let b_fft = (x - x.round()).abs() < symprec
                && (y - y.round()).abs() < symprec
                && (z - z.round()).abs() < symprec;

            x = ft[0] * kmesh[0] as f64;
            y = ft[1] * kmesh[1] as f64;
            z = ft[2] * kmesh[2] as f64;

            let b_kmesh = (x - x.round()).abs() < symprec
                && (y - y.round()).abs() < symprec
                && (z - z.round()).abs() < symprec;

            if b_fft && b_kmesh {
                viops.push(i);
            }
        }

        viops
    }

    fn get_spacegroup_number(&self) -> i32 {
        self.spacegroup_number
    }

    fn get_hall_number(&self) -> i32 {
        self.hall_number
    }

    fn get_n_sym_ops(&self) -> usize {
        self.n_operations
    }

    fn get_rotation(&self, isym: usize) -> &[[i32; 3]; 3] {
        &self.rotations[isym]
    }

    fn get_translation(&self, isym: usize) -> &[f64] {
        &self.translations[isym]
    }

    fn operation_on_vector(&self, isym: usize, v: &mut [f64; 3]) {
        self.rotate_vector(isym, v);
        self.translate_vector(isym, v);
    }

    fn center_vector(&self, v: &mut [f64; 3]) {
        for x in v.iter_mut() {
            *x -= x.floor();
            if *x > 0.5 {
                *x -= 1.0;
            }
        }
    }

    fn display_brief(&self) {
        println!();
        println!("Space group type");
        println!("spacegroup_number    : {}", self.spacegroup_number);
        println!("hall_number          : {}", self.hall_number);
        println!("international_symbol : {}", self.international_symbol);
        println!("hall_symbol          : {}", self.hall_symbol);
        println!("choice               : {}", self.choice);

        println!();
        println!("Smmetry operations");
        println!("n_operations : {}", self.n_operations);

        println!();
        println!("Wyckoff positions and symmetrically equivalent atoms");
        println!("n_atoms                 : {}", self.n_atoms);
        println!("wyckoffs                : {:?}", self.wyckoffs);
        println!("site_symmetry_symbols   : {:?}", self.site_symmetry_symbols);
        println!("equivalent_atoms        : {:?}", self.equivalent_atoms);
        println!(
            "crystallographic_orbits : {:?}",
            self.crystallographic_orbits
        );
    }

    fn display(&self) {
        println!();
        println!("Space group type");
        println!("spacegroup_number    : {}", self.spacegroup_number);
        println!("hall_number          : {}", self.hall_number);
        println!("international_symbol : {}", self.international_symbol);
        println!("hall_symbol          : {}", self.hall_symbol);
        println!("choice               : {}", self.choice);

        println!();
        println!("Smmetry operations");
        println!("n_operations : {}", self.n_operations);
        for ns in 0..self.n_operations {
            println!("symmetry operation {}", ns);
            println!("rotations    : {:?}", self.rotations[ns]);
            println!("translations : {:?}", self.translations[ns]);
        }

        println!();
        println!("Wyckoff positions and symmetrically equivalent atoms");
        println!("n_atoms                 : {}", self.n_atoms);
        println!("wyckoffs                : {:?}", self.wyckoffs);
        println!("site_symmetry_symbols   : {:?}", self.site_symmetry_symbols);
        println!("equivalent_atoms        : {:?}", self.equivalent_atoms);
        println!(
            "crystallographic_orbits : {:?}",
            self.crystallographic_orbits
        );

        println!();
        println!("Transformation matrix and origin shift");
        println!("transformation_matrix : {:?}", self.transformation_matrix);
        println!("origin_shift          : {:?}", self.origin_shift);

        println!();
        println!("Standardized crystal structure after idealization");
        println!("n_std_atoms              : {}", self.n_std_atoms);
        println!("std_lattice              : {:?}", self.std_lattice);
        //println!("std_types                : {:?}", self.std_types);
        println!("std_positions            : {:?}", self.std_positions);
        println!("std_rotation_matrix      : {:?}", self.std_rotation_matrix);
        //println!("std_mapping_to_primitive : {:?}", self.std_mapping_to_primitive);

        println!();
        println!("Crystallographic point group");
        println!("pointgroup_symbol : {}", self.pointgroup_symbol);

        println!();
        println!("Intermediate data in symmetry search");
        println!("primitive_lattice    : {:?}", self.primitive_lattice);
        //println!("mapping_to_primitive : {:?}", self.mapping_to_primitive);
    }
}

impl SymmetryDriverSPGLIB {
    pub fn new(
        latt: &[[f64; 3]],
        position: &[[f64; 3]],
        types: &[i32],
        symprec: f64,
    ) -> SymmetryDriverSPGLIB {
        let ptr = unsafe {
            &*ffi::spg_get_dataset(
                latt.as_ptr(),
                position.as_ptr(),
                types.as_ptr(),
                position.len() as i32,
                symprec,
            )
        };

        let spacegroup_number = ptr.spacegroup_number as i32;

        let hall_number = ptr.hall_number as i32;
        let international_symbol = String::from(
            CString::from(unsafe { CStr::from_ptr(ptr.international_symbol.as_ptr()) })
                .to_str()
                .unwrap(),
        );

        let hall_symbol = String::from(
            CString::from(unsafe { CStr::from_ptr(ptr.hall_symbol.as_ptr()) })
                .to_str()
                .unwrap(),
        );

        let choice = String::from(
            CString::from(unsafe { CStr::from_ptr(ptr.choice.as_ptr()) })
                .to_str()
                .unwrap(),
        );

        let transformation_matrix = ptr.transformation_matrix;
        let origin_shift = ptr.origin_shift;
        let n_operations = ptr.n_operations as usize;
        let rotations = unsafe { std::slice::from_raw_parts(ptr.rotations, n_operations).to_vec() };

        let translations =
            unsafe { std::slice::from_raw_parts(ptr.translations, n_operations).to_vec() };

        let n_atoms = ptr.n_atoms as usize;
        let wyckoffs = unsafe { std::slice::from_raw_parts(ptr.wyckoffs, n_atoms).to_vec() };
        // TODO
        let site_symmetry_symbols = Vec::new();

        let equivalent_atoms = unsafe {
            std::slice::from_raw_parts(ptr.equivalent_atoms as *const i32, n_atoms).to_vec()
        };
        let crystallographic_orbits =
            unsafe { std::slice::from_raw_parts(ptr.crystallographic_orbits, n_atoms).to_vec() };

        let primitive_lattice = ptr.primitive_lattice;

        let mapping_to_primitive =
            unsafe { std::slice::from_raw_parts(ptr.mapping_to_primitive, n_atoms).to_vec() };

        let n_std_atoms = ptr.n_std_atoms as usize;

        let std_lattice = ptr.std_lattice;

        let std_types = unsafe { std::slice::from_raw_parts(ptr.std_types, n_std_atoms).to_vec() };
        let std_positions =
            unsafe { std::slice::from_raw_parts(ptr.std_positions, n_std_atoms).to_vec() };
        let std_rotation_matrix = ptr.std_rotation_matrix;
        let std_mapping_to_primitive = unsafe {
            std::slice::from_raw_parts(ptr.std_mapping_to_primitive, n_std_atoms).to_vec()
        };

        let pointgroup_symbol = String::from(
            CString::from(unsafe { CStr::from_ptr(ptr.pointgroup_symbol.as_ptr()) })
                .to_str()
                .unwrap(),
        );
        //
        //

        let natoms = position.len();

        let mut sym_atom = Vec::new();

        for iat in 0..natoms {
            let mut ns_atom = Vec::new();

            for ns in 0..n_operations {
                ns_atom.push(search_for_matching_atom(
                    &rotations[ns],
                    &translations[ns],
                    position,
                    ns,
                    iat,
                ));
            }

            sym_atom.push(ns_atom);
        }
        //

        SymmetryDriverSPGLIB {
            spacegroup_number,
            hall_number,
            international_symbol,
            hall_symbol,
            choice,
            transformation_matrix,
            origin_shift,
            n_operations,
            rotations,
            translations,
            n_atoms,
            wyckoffs,
            site_symmetry_symbols,
            equivalent_atoms,
            crystallographic_orbits,
            primitive_lattice,
            mapping_to_primitive,
            n_std_atoms,
            std_lattice,
            std_types,
            std_positions,
            std_rotation_matrix,
            std_mapping_to_primitive,
            pointgroup_symbol,
            sym_atom,
        }
    }

    pub fn rotate_vector(&self, i: usize, v: &mut [f64; 3]) {
        let s = v.clone();
        let m = self.rotations[i];

        v[0] = m[0][0] as f64 * s[0] + m[0][1] as f64 * s[1] + m[0][2] as f64 * s[2];
        v[1] = m[1][0] as f64 * s[0] + m[1][1] as f64 * s[1] + m[1][2] as f64 * s[2];
        v[2] = m[2][0] as f64 * s[0] + m[2][1] as f64 * s[1] + m[2][2] as f64 * s[2];
    }

    pub fn translate_vector(&self, i: usize, v: &mut [f64; 3]) {
        let s = self.translations[i];

        v[0] += s[0];
        v[1] += s[1];
        v[2] += s[2];
    }
}

fn center_vector(v: &mut [f64; 3]) {
    for x in v.iter_mut() {
        *x -= x.floor();
        if *x > 0.5 {
            *x -= 1.0;
        }
    }
}

fn search_for_matching_atom(
    rt: &[[i32; 3]],
    ft: &[f64],
    position: &[[f64; 3]],
    isym: usize,
    iat: usize,
) -> usize {
    let atom = position[iat];

    let mut rat = [0.0; 3];

    rat[0] =
        rt[0][0] as f64 * atom[0] + rt[0][1] as f64 * atom[1] + rt[0][2] as f64 * atom[2] + ft[0];
    rat[1] =
        rt[1][0] as f64 * atom[0] + rt[1][1] as f64 * atom[1] + rt[1][2] as f64 * atom[2] + ft[1];
    rat[2] =
        rt[2][0] as f64 * atom[0] + rt[2][1] as f64 * atom[1] + rt[2][2] as f64 * atom[2] + ft[2];

    center_vector(&mut rat);

    let mut nmatch = position.len();

    for (iat, at) in position.iter().enumerate() {
        let mut tat = *at;

        center_vector(&mut tat);

        if (tat[0] - rat[0]).abs() < 1.0E-5
            && (tat[1] - rat[1]).abs() < 1.0E-5
            && (tat[2] - rat[2]).abs() < 1.0E-5
        {
            nmatch = iat;
            break;
        }
    }

    nmatch
}

pub fn get_ir_reciprocal_mesh(
    mesh: [i32; 3],
    is_shift: [i32; 3],
    lattice: &mut [[f64; 3]],
    position: &mut [[f64; 3]],
    types: &[i32],
    symprec: f64,
) -> (Vec<[f64; 3]>, Vec<i32>, Vec<i32>, Vec<usize>) {
    let nkpt = (mesh[0] * mesh[1] * mesh[2]) as usize;

    let mut grid_address = vec![[0i32; 3]; nkpt];
    let mut ir_mapping_table = vec![0i32; nkpt];

    unsafe {
        ffi::spg_get_ir_reciprocal_mesh(
            grid_address.as_mut_ptr(),
            ir_mapping_table.as_mut_ptr(),
            mesh.as_ptr(),
            is_shift.as_ptr(),
            1,
            lattice.as_mut_ptr(),
            position.as_mut_ptr(),
            types.as_ptr(),
            position.len() as i32,
            symprec,
        );
    }

    let mut kpts = vec![[0f64; 3]; nkpt];

    for (i, k) in grid_address.iter().enumerate() {
        kpts[i][0] = k[0] as f64 / mesh[0] as f64;
        kpts[i][1] = k[1] as f64 / mesh[1] as f64;
        kpts[i][2] = k[2] as f64 / mesh[2] as f64;
    }

    let ir_ikpt: Vec<i32> = ir_mapping_table.clone().into_iter().unique().collect();

    let mut ir_ikpt_degeneracy: Vec<usize> = Vec::new();

    for k in ir_ikpt.iter() {
        ir_ikpt_degeneracy.push(ir_mapping_table.iter().filter(|&v| *v == *k).count());
    }

    (kpts, ir_mapping_table, ir_ikpt, ir_ikpt_degeneracy)
}
