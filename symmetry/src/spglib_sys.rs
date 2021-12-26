use std::os::raw::*;

#[repr(C)]
pub struct SpglibDataset {
    pub spacegroup_number: c_int,
    pub hall_number: c_int,
    pub international_symbol: [c_char; 11],
    pub hall_symbol: [c_char; 17],
    pub choice: [c_char; 6],
    pub transformation_matrix: [[f64; 3]; 3],
    pub origin_shift: [f64; 3],
    pub n_operations: c_int,
    pub rotations: *mut [[c_int; 3]; 3],
    pub translations: *mut [f64; 3],
    pub n_atoms: c_int,
    pub wyckoffs: *mut c_int,
    pub site_symmetry_symbols: *mut [c_char; 7],
    pub equivalent_atoms: *mut c_int,
    pub crystallographic_orbits: *mut c_int,
    pub primitive_lattice: [[f64; 3]; 3],
    pub mapping_to_primitive: *mut c_int,
    pub n_std_atoms: c_int,
    pub std_lattice: [[f64; 3]; 3],
    pub std_types: *mut c_int,
    pub std_positions: *mut [f64; 3],
    pub std_rotation_matrix: [[f64; 3]; 3],
    pub std_mapping_to_primitive: *mut c_int,
    pub pointgroup_symbol: [c_char; 6],
}

extern "C" {
    pub fn spg_get_symmetry(
        rotation: *mut [[c_int; 3]; 3],
        translation: *mut [f64; 3],
        max_size: c_int,
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        symprec: f64,
    ) -> c_int;

    pub fn spg_get_ir_reciprocal_mesh(
        grid_address: *mut [c_int; 3],
        ir_mapping_table: *mut c_int,
        mesh: *const c_int,
        is_shift: *const c_int,
        is_time_reversal: c_int,
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        symprec: f64,
    ) -> c_int;

    pub fn spg_get_dataset(
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        symprec: f64,
    ) -> *mut SpglibDataset;

    pub fn spgat_get_dataset(
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        symprec: f64,
        angle_tolerance: f64,
    ) -> *mut SpglibDataset;

    pub fn spgat_get_dataset_with_hall_number(
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        hall_number: c_int,
        symprec: f64,
        angle_tolerance: f64,
    ) -> *mut SpglibDataset;

}
