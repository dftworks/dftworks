#![allow(warnings)]

use symops::{classify_symmetry, detect_symmetry, DetectOptions, Structure, SymOp};

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

#[derive(Clone, Debug)]
pub struct SymmetryDriverInternal {
    spacegroup_number: i32,
    hall_number: i32,
    point_group_hint: String,
    rotations: Vec<[[i32; 3]; 3]>,
    translations: Vec<[f64; 3]>,
    sym_atom: Vec<Vec<usize>>,
}

impl SymmetryDriverInternal {
    pub fn new(
        latt: &[[f64; 3]],
        position: &[[f64; 3]],
        types: &[i32],
        symprec: f64,
    ) -> SymmetryDriverInternal {
        let symprec = if symprec > 0.0 { symprec } else { 1.0e-6 };
        let options = DetectOptions {
            symprec,
            metric_tol: symprec * 10.0,
            validate_group: true,
        };

        let lattice = normalize_lattice_rows(latt);
        let structure = Structure {
            lattice,
            positions: position.to_vec(),
            atom_types: types.to_vec(),
        };

        let operations = detect_symmetry(&structure, options)
            .ok()
            .map(|detected| detected.operations)
            .filter(|ops| !ops.is_empty())
            .unwrap_or_else(|| vec![SymOp::identity()]);

        let classification = classify_symmetry(&operations).ok();
        let spacegroup_number = classification
            .as_ref()
            .and_then(|c| c.space_group_number)
            .map(i32::from)
            .unwrap_or(0);
        let point_group_hint = classification
            .as_ref()
            .map(|c| c.point_group_hint.to_string())
            .unwrap_or_else(|| "1".to_string());

        let rotations: Vec<[[i32; 3]; 3]> = operations.iter().map(|op| *op.rotation()).collect();
        let translations: Vec<[f64; 3]> = operations.iter().map(|op| op.translation()).collect();
        let sym_atom = build_sym_atom_map(&rotations, &translations, position, types, symprec);

        SymmetryDriverInternal {
            spacegroup_number,
            hall_number: 0,
            point_group_hint,
            rotations,
            translations,
            sym_atom,
        }
    }

    fn rotate_vector(&self, isym: usize, v: &mut [f64; 3]) {
        let src = *v;
        let m = self.rotations[isym];
        v[0] = m[0][0] as f64 * src[0] + m[0][1] as f64 * src[1] + m[0][2] as f64 * src[2];
        v[1] = m[1][0] as f64 * src[0] + m[1][1] as f64 * src[1] + m[1][2] as f64 * src[2];
        v[2] = m[2][0] as f64 * src[0] + m[2][1] as f64 * src[1] + m[2][2] as f64 * src[2];
    }

    fn translate_vector(&self, isym: usize, v: &mut [f64; 3]) {
        let t = self.translations[isym];
        v[0] += t[0];
        v[1] += t[1];
        v[2] += t[2];
    }
}

impl SymmetryDriver for SymmetryDriverInternal {
    fn get_n_sym_ops(&self) -> usize {
        self.rotations.len()
    }

    fn get_spacegroup_number(&self) -> i32 {
        self.spacegroup_number
    }

    fn get_hall_number(&self) -> i32 {
        self.hall_number
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
        center_vector(v);
    }

    fn display(&self) {
        println!();
        println!("   Symmetry (self-contained)");
        println!("   spacegroup_number : {}", self.spacegroup_number);
        println!("   hall_number       : {}", self.hall_number);
        println!("   point_group_hint  : {}", self.point_group_hint);
        println!("   n_operations      : {}", self.rotations.len());
        for i in 0..self.rotations.len() {
            println!("   symmetry operation {}", i);
            println!("   rotations    : {:?}", self.rotations[i]);
            println!("   translations : {:?}", self.translations[i]);
        }
    }

    fn display_brief(&self) {
        println!();
        println!("   Symmetry (self-contained)");
        println!("   spacegroup_number : {}", self.spacegroup_number);
        println!("   hall_number       : {}", self.hall_number);
        println!("   point_group_hint  : {}", self.point_group_hint);
        println!("   n_operations      : {}", self.rotations.len());
    }

    fn get_fft_commensurate_ops(
        &self,
        fftmesh: [usize; 3],
        kmesh: [i32; 3],
        symprec: f64,
    ) -> Vec<usize> {
        let symprec = if symprec > 0.0 { symprec } else { 1.0e-6 };
        let n1 = fftmesh[0] as f64;
        let n2 = fftmesh[1] as f64;
        let n3 = fftmesh[2] as f64;

        let mut out = Vec::new();
        for i in 0..self.get_n_sym_ops() {
            let ft = self.get_translation(i);

            let b_fft = ((ft[0] * n1) - (ft[0] * n1).round()).abs() < symprec
                && ((ft[1] * n2) - (ft[1] * n2).round()).abs() < symprec
                && ((ft[2] * n3) - (ft[2] * n3).round()).abs() < symprec;

            let b_kmesh = ((ft[0] * kmesh[0] as f64) - (ft[0] * kmesh[0] as f64).round()).abs()
                < symprec
                && ((ft[1] * kmesh[1] as f64) - (ft[1] * kmesh[1] as f64).round()).abs() < symprec
                && ((ft[2] * kmesh[2] as f64) - (ft[2] * kmesh[2] as f64).round()).abs() < symprec;

            if b_fft && b_kmesh {
                out.push(i);
            }
        }

        out
    }

    fn get_sym_atom(&self) -> &[Vec<usize>] {
        &self.sym_atom
    }
}

pub fn new(
    latt: &[[f64; 3]],
    position: &[[f64; 3]],
    types: &[i32],
    symprec: f64,
) -> Box<dyn SymmetryDriver> {
    Box::new(SymmetryDriverInternal::new(latt, position, types, symprec))
}

fn normalize_lattice_rows(latt: &[[f64; 3]]) -> [[f64; 3]; 3] {
    if latt.len() >= 3 {
        [latt[0], latt[1], latt[2]]
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

fn build_sym_atom_map(
    rotations: &[[[i32; 3]; 3]],
    translations: &[[f64; 3]],
    position: &[[f64; 3]],
    types: &[i32],
    symprec: f64,
) -> Vec<Vec<usize>> {
    let natoms = position.len();
    let mut sym_atom = Vec::with_capacity(natoms);
    for iat in 0..natoms {
        let mut mapped_by_op = Vec::with_capacity(rotations.len());
        for isym in 0..rotations.len() {
            mapped_by_op.push(search_for_matching_atom(
                &rotations[isym],
                &translations[isym],
                position,
                types,
                iat,
                symprec,
            ));
        }
        sym_atom.push(mapped_by_op);
    }
    sym_atom
}

fn search_for_matching_atom(
    rotation: &[[i32; 3]],
    translation: &[f64; 3],
    position: &[[f64; 3]],
    types: &[i32],
    iat: usize,
    symprec: f64,
) -> usize {
    let atom = position[iat];
    let mapped = apply_operation(rotation, translation, atom);
    let mapped_centered = [
        wrap_centered(mapped[0]),
        wrap_centered(mapped[1]),
        wrap_centered(mapped[2]),
    ];
    let ref_type = types.get(iat).copied();

    for (jat, target) in position.iter().enumerate() {
        if ref_type.is_some() && types.get(jat).copied() != ref_type {
            continue;
        }
        let target_centered = [
            wrap_centered(target[0]),
            wrap_centered(target[1]),
            wrap_centered(target[2]),
        ];
        if (mapped_centered[0] - target_centered[0]).abs() < symprec
            && (mapped_centered[1] - target_centered[1]).abs() < symprec
            && (mapped_centered[2] - target_centered[2]).abs() < symprec
        {
            return jat;
        }
    }

    // Keep index in-bounds even if no perfect match is found.
    iat
}

fn apply_operation(rotation: &[[i32; 3]], translation: &[f64; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] as f64 * vector[0]
            + rotation[0][1] as f64 * vector[1]
            + rotation[0][2] as f64 * vector[2]
            + translation[0],
        rotation[1][0] as f64 * vector[0]
            + rotation[1][1] as f64 * vector[1]
            + rotation[1][2] as f64 * vector[2]
            + translation[1],
        rotation[2][0] as f64 * vector[0]
            + rotation[2][1] as f64 * vector[1]
            + rotation[2][2] as f64 * vector[2]
            + translation[2],
    ]
}

fn center_vector(v: &mut [f64; 3]) {
    for x in v.iter_mut() {
        *x -= x.floor();
        if *x > 0.5 {
            *x -= 1.0;
        }
    }
}

fn normalize_fractional(x: f64) -> f64 {
    let mut wrapped = x - x.floor();
    if wrapped >= 1.0 {
        wrapped -= 1.0;
    }
    if wrapped < 0.0 {
        wrapped += 1.0;
    }
    if wrapped.abs() < 1.0e-12 || (1.0 - wrapped).abs() < 1.0e-12 {
        0.0
    } else {
        wrapped
    }
}

fn wrap_centered(x: f64) -> f64 {
    let mut wrapped = x - x.round();
    if wrapped >= 0.5 {
        wrapped -= 1.0;
    }
    if wrapped < -0.5 {
        wrapped += 1.0;
    }
    if wrapped.abs() < 1.0e-12 {
        0.0
    } else {
        wrapped
    }
}

fn flatten_index(i: usize, j: usize, k: usize, n2: usize, n3: usize) -> usize {
    (i * n2 + j) * n3 + k
}

fn map_k_to_grid_index(
    k: [f64; 3],
    mesh: [i32; 3],
    is_shift: [i32; 3],
    tol: f64,
) -> Option<[usize; 3]> {
    let tol = tol.max(1.0e-8);
    let mut idx = [0usize; 3];
    for d in 0..3 {
        let n = mesh[d];
        if n <= 0 {
            return None;
        }

        let n_f = n as f64;
        let shift = is_shift[d] as f64;
        let k_norm = normalize_fractional(k[d]);
        let raw = 2.0 * n_f * k_norm - shift;
        let i = (0.5 * raw).round() as i64;
        let n_i64 = n as i64;
        let i_mod = ((i % n_i64) + n_i64) % n_i64;

        let reconstructed = (2.0 * i_mod as f64 + shift) / (2.0 * n_f);
        if wrap_centered(reconstructed - k_norm).abs() > tol {
            return None;
        }
        idx[d] = i_mod as usize;
    }
    Some(idx)
}

pub fn get_ir_reciprocal_mesh(
    mesh: [i32; 3],
    is_shift: [i32; 3],
    lattice: &mut [[f64; 3]],
    position: &mut [[f64; 3]],
    types: &[i32],
    symprec: f64,
) -> (Vec<[f64; 3]>, Vec<i32>, Vec<i32>, Vec<usize>) {
    if mesh[0] <= 0 || mesh[1] <= 0 || mesh[2] <= 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let n1 = mesh[0] as usize;
    let n2 = mesh[1] as usize;
    let n3 = mesh[2] as usize;
    let nkpt = n1 * n2 * n3;

    let mut grid_address = vec![[0i32; 3]; nkpt];
    let mut kpts = vec![[0.0f64; 3]; nkpt];

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let idx = flatten_index(i, j, k, n2, n3);
                grid_address[idx] = [i as i32, j as i32, k as i32];
                kpts[idx] = [
                    (2 * i as i32 + is_shift[0]) as f64 / (2 * mesh[0]) as f64,
                    (2 * j as i32 + is_shift[1]) as f64 / (2 * mesh[1]) as f64,
                    (2 * k as i32 + is_shift[2]) as f64 / (2 * mesh[2]) as f64,
                ];
            }
        }
    }

    let driver = SymmetryDriverInternal::new(lattice, position, types, symprec);
    let mut ir_mapping_table = vec![-1i32; nkpt];

    for idx in 0..nkpt {
        if ir_mapping_table[idx] >= 0 {
            continue;
        }
        let representative = idx as i32;
        ir_mapping_table[idx] = representative;
        let k0 = kpts[idx];
        for rotation in driver.rotations.iter() {
            let mapped = [
                rotation[0][0] as f64 * k0[0]
                    + rotation[0][1] as f64 * k0[1]
                    + rotation[0][2] as f64 * k0[2],
                rotation[1][0] as f64 * k0[0]
                    + rotation[1][1] as f64 * k0[1]
                    + rotation[1][2] as f64 * k0[2],
                rotation[2][0] as f64 * k0[0]
                    + rotation[2][1] as f64 * k0[1]
                    + rotation[2][2] as f64 * k0[2],
            ];
            if let Some([mi, mj, mk]) = map_k_to_grid_index(mapped, mesh, is_shift, symprec) {
                let midx = flatten_index(mi, mj, mk, n2, n3);
                if ir_mapping_table[midx] < 0 {
                    ir_mapping_table[midx] = representative;
                }
            }
        }
    }

    for i in 0..nkpt {
        if ir_mapping_table[i] < 0 {
            ir_mapping_table[i] = i as i32;
        }
    }

    let mut ir_ikpt = Vec::<i32>::new();
    for &mapped in ir_mapping_table.iter() {
        if !ir_ikpt.iter().any(|v| *v == mapped) {
            ir_ikpt.push(mapped);
        }
    }

    let mut ir_ikpt_degeneracy: Vec<usize> = Vec::with_capacity(ir_ikpt.len());
    for k in ir_ikpt.iter() {
        ir_ikpt_degeneracy.push(ir_mapping_table.iter().filter(|&v| *v == *k).count());
    }

    (kpts, ir_mapping_table, ir_ikpt, ir_ikpt_degeneracy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn silicon_diamond_primitive_cell() -> ([[f64; 3]; 3], [[f64; 3]; 2], [i32; 2]) {
        // Primitive FCC cell with two-atom Si basis.
        let lattice = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
        let position = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]];
        let types = [1, 1];
        (lattice, position, types)
    }

    #[test]
    fn test_internal_driver_builds_ops() {
        let lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let position = [[0.0, 0.0, 0.0]];
        let types = [1];
        let dataset = SymmetryDriverInternal::new(&lattice, &position, &types, 1.0e-6);
        assert!(dataset.get_n_sym_ops() >= 1);
        assert!(dataset.get_sym_atom().len() == 1);
    }

    #[test]
    fn test_silicon_driver_has_nontrivial_symmetry() {
        let (lattice, position, types) = silicon_diamond_primitive_cell();
        let dataset = SymmetryDriverInternal::new(&lattice, &position, &types, 1.0e-6);

        println!("Si symmetry op count: {}", dataset.get_n_sym_ops());
        for isym in 0..dataset.get_n_sym_ops().min(8) {
            println!(
                "Si op {:2}: R={:?}, t={:?}",
                isym,
                dataset.get_rotation(isym),
                dataset.get_translation(isym)
            );
        }
        println!("Si sym_atom map: {:?}", dataset.get_sym_atom());

        assert!(dataset.get_n_sym_ops() > 1);
        assert_eq!(dataset.get_sym_atom().len(), position.len());
        for atom_map in dataset.get_sym_atom().iter() {
            assert_eq!(atom_map.len(), dataset.get_n_sym_ops());
            assert!(atom_map.iter().all(|&idx| idx < position.len()));
        }

        // Identity operation should always be present.
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let has_identity = (0..dataset.get_n_sym_ops()).any(|isym| {
            dataset.get_rotation(isym) == &identity
                && dataset
                    .get_translation(isym)
                    .iter()
                    .all(|x| x.abs() < 1.0e-10)
        });
        assert!(has_identity);
    }

    #[test]
    fn test_silicon_ir_reciprocal_mesh_mapping_consistency() {
        let mesh: [i32; 3] = [4, 4, 4];
        let is_shift: [i32; 3] = [0, 0, 0];
        let (mut lattice, mut position, types) = silicon_diamond_primitive_cell();

        let (kpts, mapping, k_unique, nk_unique) =
            get_ir_reciprocal_mesh(mesh, is_shift, &mut lattice, &mut position, &types, 1.0E-6);

        println!(
            "Si IR mesh summary: total_kpts={}, ir_kpts={}",
            kpts.len(),
            k_unique.len()
        );
        println!("Si IR representatives: {:?}", k_unique);
        println!("Si IR degeneracies: {:?}", nk_unique);
        println!(
            "Si mapping sample (first 16): {:?}",
            &mapping[..mapping.len().min(16)]
        );

        assert_eq!(kpts.len(), 64);
        assert_eq!(mapping.len(), 64);
        assert!(!k_unique.is_empty());
        assert_eq!(k_unique.len(), nk_unique.len());

        let mut counts = BTreeMap::<i32, usize>::new();
        for &mapped in mapping.iter() {
            assert!(mapped >= 0);
            assert!((mapped as usize) < kpts.len());
            *counts.entry(mapped).or_insert(0) += 1;
        }

        assert_eq!(counts.len(), k_unique.len());
        assert_eq!(counts.values().sum::<usize>(), kpts.len());

        for (ir_idx, expected_deg) in k_unique.iter().zip(nk_unique.iter()) {
            assert_eq!(counts.get(ir_idx).copied().unwrap_or(0), *expected_deg);
        }
    }
}
