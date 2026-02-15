use dwconsts::*;
use itertools::Itertools;
use lattice::Lattice;
use matrix::Matrix;
use pspot::PSPot;
use vector3::*;

use std::io::Write;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

// Crystal structure container.
//
// Coordinates:
// - lattice vectors stored in Bohr
// - atomic positions stored in fractional coordinates
// - helper conversions provide Cartesian views when needed
#[derive(Debug, Default)]
pub struct Crystal {
    scale_a: f64,
    scale_b: f64,
    scale_c: f64,
    latt: Lattice,
    cell_mask: Matrix<f64>,
    atom_positions: Vec<Vector3f64>,
    atom_species: Vec<String>,
    atom_indices_by_specie: Vec<Vec<usize>>,
}

impl Crystal {
    pub fn get_n_total_electrons(&self, pots: &PSPot) -> f64 {
        // Sum valence electrons by species multiplicity.
        let mut sum = 0.0;

        for (isp, sp) in self.get_unique_species().iter().enumerate() {
            let natom_for_this_specie = self.atom_indices_by_specie[isp].len();

            let zion = pots.get_psp(sp).get_zion();

            sum += zion * natom_for_this_specie as f64;
        }

        sum
    }

    pub fn get_zions(&self, pots: &PSPot) -> Vec<f64> {
        // Per-atom ionic charges aligned with atom ordering.
        let natoms = self.get_n_atoms();

        let mut zions = vec![0.0; natoms];

        for (isp, sp) in self.get_unique_species().iter().enumerate() {
            let zion = pots.get_psp(sp).get_zion();

            for idx in self.atom_indices_by_specie[isp].iter() {
                zions[*idx] = zion;
            }
        }

        zions
    }

    pub fn get_atom_indices_of_specie(&self, isp: usize) -> &[usize] {
        &self.atom_indices_by_specie[isp]
    }

    pub fn get_latt(&self) -> &Lattice {
        &self.latt
    }

    pub fn get_cell_mask(&self) -> &Matrix<f64> {
        &self.cell_mask
    }

    pub fn get_unique_species(&self) -> Vec<String> {
        // Preserve first-occurrence order while removing duplicates.
        self.atom_species.clone().into_iter().unique().collect()
    }

    pub fn get_n_unique_species(&self) -> usize {
        self.atom_species
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<String>>()
            .len()
    }

    pub fn get_atom_positions_of_specie(&self, isp: usize) -> Vec<Vector3f64> {
        // Gather atoms by specie index map.
        let atom_indices = self.get_atom_indices_of_specie(isp);

        let mut atom_positions_for_this_specie = vec![Vector3f64::zeros(); atom_indices.len()];

        for (i, idx) in atom_indices.iter().enumerate() {
            atom_positions_for_this_specie[i] = self.atom_positions[*idx];
        }

        atom_positions_for_this_specie
    }

    pub fn get_n_atoms(&self) -> usize {
        self.atom_positions.len()
    }

    pub fn get_atom_positions(&self) -> &[Vector3f64] {
        &self.atom_positions
    }

    pub fn get_atom_positions_cart(&self) -> Vec<Vector3f64> {
        // Convert all fractional atomic positions to Cartesian coordinates.
        let natoms = self.atom_positions.len();

        let mut atoms_cart = vec![Vector3f64::zeros(); natoms];

        for iat in 0..natoms {
            self.latt.frac_to_cart(
                self.atom_positions[iat].as_slice(),
                atoms_cart[iat].as_mut_slice(),
            );
        }

        atoms_cart
    }

    pub fn set_atom_positions_from_cart(&mut self, atoms_cart: &[Vector3f64]) {
        // Update internal fractional coordinates from Cartesian inputs.
        let natoms = self.atom_positions.len();

        let mut tpos = Vector3f64::zeros();

        for iat in 0..natoms {
            self.latt
                .cart_to_frac(atoms_cart[iat].as_slice(), tpos.as_mut_slice());

            self.atom_positions[iat] = tpos;
        }
    }

    pub fn set_atom_positions_from_frac(&mut self, atoms_frac: &[Vector3f64]) {
        let natoms = self.atom_positions.len();

        for iat in 0..natoms {
            self.atom_positions[iat] = atoms_frac[iat];
        }
    }

    pub fn set_lattice_vectors(&mut self, latt: &Lattice) {
        // Replace full lattice object (used by geometry optimizers).
        self.latt = latt.clone();
    }

    pub fn get_atom_types(&self) -> Vec<i32> {
        let mut types = vec![0; self.get_n_atoms()];

        for (isp, _sp) in self.get_unique_species().iter().enumerate() {
            for idx in self.atom_indices_by_specie[isp].iter() {
                types[*idx] = isp as i32 + 1;
            }
        }

        types
    }

    pub fn get_atom_species(&self) -> &[String] {
        &self.atom_species
    }

    pub fn new() -> Crystal {
        Crystal::default()
    }

    pub fn read_file(&mut self, inpfile: &str) {
        // Parse in.crystal with format:
        // line 1: scale_a scale_b scale_c
        // line 2-4: lattice vectors + T/F mask flags per Cartesian component
        // remaining lines: species x y z (fractional atomic positions)
        let file = File::open(inpfile).unwrap();
        let lines = BufReader::new(file).lines();

        self.atom_positions = Vec::new();
        self.atom_species = Vec::new();

        let mut vec_a = [0.0; 3];
        let mut vec_b = [0.0; 3];
        let mut vec_c = [0.0; 3];

        self.cell_mask = Matrix::new(3, 3);

        for (i, line) in lines.enumerate() {
            let s: Vec<&str> = line.as_ref().unwrap().split_whitespace().collect();

            match i {
                0 => {
                    // Independent scale factors for three lattice vectors.
                    self.scale_a = s[0].parse().unwrap();
                    self.scale_b = s[1].parse().unwrap();
                    self.scale_c = s[2].parse().unwrap();
                }

                1 => {
                    // Lattice vector a and optimization mask flags.
                    vec_a[0] = s[0].parse().unwrap();
                    vec_a[1] = s[1].parse().unwrap();
                    vec_a[2] = s[2].parse().unwrap();

                    for iv in 0..3 {
                        vec_a[iv] *= self.scale_a * ANG_TO_BOHR;
                    }

                    if s[3].to_lowercase() == "t" {
                        self.cell_mask[[0, 0]] = 1.0;
                    } else {
                        self.cell_mask[[0, 0]] = 0.0;
                    }

                    if s[4].to_lowercase() == "t" {
                        self.cell_mask[[1, 0]] = 1.0;
                    } else {
                        self.cell_mask[[1, 0]] = 0.0;
                    }

                    if s[5].to_lowercase() == "t" {
                        self.cell_mask[[2, 0]] = 1.0;
                    } else {
                        self.cell_mask[[2, 0]] = 0.0;
                    }
                }

                2 => {
                    // Lattice vector b and optimization mask flags.
                    vec_b[0] = s[0].parse().unwrap();
                    vec_b[1] = s[1].parse().unwrap();
                    vec_b[2] = s[2].parse().unwrap();

                    for iv in 0..3 {
                        vec_b[iv] *= self.scale_b * ANG_TO_BOHR;
                    }

                    if s[3].to_lowercase() == "t" {
                        self.cell_mask[[0, 1]] = 1.0;
                    } else {
                        self.cell_mask[[0, 1]] = 0.0;
                    }

                    if s[4].to_lowercase() == "t" {
                        self.cell_mask[[1, 1]] = 1.0;
                    } else {
                        self.cell_mask[[1, 1]] = 0.0;
                    }

                    if s[5].to_lowercase() == "t" {
                        self.cell_mask[[2, 1]] = 1.0;
                    } else {
                        self.cell_mask[[2, 1]] = 0.0;
                    }
                }

                3 => {
                    // Lattice vector c and optimization mask flags.
                    vec_c[0] = s[0].parse().unwrap();
                    vec_c[1] = s[1].parse().unwrap();
                    vec_c[2] = s[2].parse().unwrap();

                    for iv in 0..3 {
                        vec_c[iv] *= self.scale_c * ANG_TO_BOHR;
                    }

                    if s[3].to_lowercase() == "t" {
                        self.cell_mask[[0, 2]] = 1.0;
                    } else {
                        self.cell_mask[[0, 2]] = 0.0;
                    }

                    if s[4].to_lowercase() == "t" {
                        self.cell_mask[[1, 2]] = 1.0;
                    } else {
                        self.cell_mask[[1, 2]] = 0.0;
                    }

                    if s[5].to_lowercase() == "t" {
                        self.cell_mask[[2, 2]] = 1.0;
                    } else {
                        self.cell_mask[[2, 2]] = 0.0;
                    }
                }

                // atoms
                _ => {
                    if s.len() == 0 {
                        continue;
                    };

                    let symbol = s[0].to_string();
                    let x: f64 = s[1].parse().unwrap();
                    let y: f64 = s[2].parse().unwrap();
                    let z: f64 = s[3].parse().unwrap();

                    // Atomic position remains fractional.
                    self.atom_species.push(symbol);
                    self.atom_positions.push(Vector3f64 { x, y, z });
                }
            }

            // Keep lattice object synchronized while parsing.
            self.latt = Lattice::new(&vec_a, &vec_b, &vec_c);
        }

        // Build specie -> atom-index lookup for fast grouped operations.

        let unique_species: Vec<String> = self.get_unique_species();

        let nsp = unique_species.len();

        self.atom_indices_by_specie = vec![Vec::new(); nsp];

        for (at_index, at_symbol) in self.atom_species.iter().enumerate() {
            for (isp, sp) in unique_species.iter().enumerate() {
                if *sp == *at_symbol {
                    self.atom_indices_by_specie[isp].push(at_index);
                }
            }
        }
    }

    pub fn output(&self) {
        // Write current structure in the same schema as in.crystal.
        let mut f = File::create("out.crystal").unwrap();

        writeln!(f, "{} {} {}", self.scale_a, self.scale_b, self.scale_c).unwrap();

        let a = self.latt.get_vector_a();

        writeln!(
            f,
            "{:.12?} {:.12?} {:.12?} {} {} {}",
            a.x / self.scale_a * BOHR_TO_ANG,
            a.y / self.scale_a * BOHR_TO_ANG,
            a.z / self.scale_a * BOHR_TO_ANG,
            float_to_char(self.cell_mask[[0, 0]]),
            float_to_char(self.cell_mask[[1, 0]]),
            float_to_char(self.cell_mask[[2, 0]])
        )
        .unwrap();

        let b = self.latt.get_vector_b();

        writeln!(
            f,
            "{:.12?} {:.12?} {:.12?} {} {} {}",
            b.x / self.scale_b * BOHR_TO_ANG,
            b.y / self.scale_b * BOHR_TO_ANG,
            b.z / self.scale_b * BOHR_TO_ANG,
            float_to_char(self.cell_mask[[0, 1]]),
            float_to_char(self.cell_mask[[1, 1]]),
            float_to_char(self.cell_mask[[2, 1]])
        )
        .unwrap();

        let c = self.latt.get_vector_c();
        writeln!(
            f,
            "{:.12?} {:.12?} {:.12?} {} {} {}",
            c.x / self.scale_c * BOHR_TO_ANG,
            c.y / self.scale_c * BOHR_TO_ANG,
            c.z / self.scale_c * BOHR_TO_ANG,
            float_to_char(self.cell_mask[[0, 2]]),
            float_to_char(self.cell_mask[[1, 2]]),
            float_to_char(self.cell_mask[[2, 2]])
        )
        .unwrap();

        for (i, at) in self.atom_positions.iter().enumerate() {
            writeln!(f, "{} {} {} {}", self.atom_species[i], at.x, at.y, at.z).unwrap();
        }
    }

    pub fn display(&self) {
        println!("   {:-^88}", " crystal structure ");
        println!();

        println!("   lattice_vectors");
        println!();

        let vec_a = self.latt.get_vector_a();
        println!(
            "   a = {:20.12}  {:20.12}  {:20.12}",
            vec_a.x * BOHR_TO_ANG,
            vec_a.y * BOHR_TO_ANG,
            vec_a.z * BOHR_TO_ANG
        );

        let vec_b = self.latt.get_vector_b();
        println!(
            "   b = {:20.12}  {:20.12}  {:20.12}",
            vec_b.x * BOHR_TO_ANG,
            vec_b.y * BOHR_TO_ANG,
            vec_b.z * BOHR_TO_ANG
        );

        let vec_c = self.latt.get_vector_c();
        println!(
            "   c = {:20.12}  {:20.12}  {:20.12}",
            vec_c.x * BOHR_TO_ANG,
            vec_c.y * BOHR_TO_ANG,
            vec_c.z * BOHR_TO_ANG
        );

        println!();
        println!("   natoms = {}", self.get_atom_positions().len());
        println!("   atom_positions\n");
        println!("                fractional                                                cartesian (A)");
        println!();

        for (i, atom) in self.get_atom_positions().iter().enumerate() {
            let mut pos_c = Vector3f64::zeros();

            self.latt
                .frac_to_cart(atom.as_slice(), pos_c.as_mut_slice());

            println!(
                "   {:<3} {:>4} : {:16.12}  {:16.12}  {:16.12}  {:20.12}  {:20.12}  {:20.12}",
                i + 1,
                self.atom_species[i],
                atom.x,
                atom.y,
                atom.z,
                pos_c.x * BOHR_TO_ANG,
                pos_c.y * BOHR_TO_ANG,
                pos_c.z * BOHR_TO_ANG
            );
        }

        println!();

        for (isp, sp) in self.get_unique_species().iter().enumerate() {
            println!(
                "   {} : {:?}",
                sp,
                self.get_atom_indices_of_specie(isp)
                    .iter()
                    .map(|x| x + 1)
                    .collect::<Vec<usize>>()
            );
        }
    }
}

fn float_to_char(fval: f64) -> String {
    if fval > 0.0 {
        "T".to_string()
    } else {
        "F".to_string()
    }
}

#[test]
fn test_crystal() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    println!("{:?}", d);
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    crystal.display();
    crystal.output();
}
