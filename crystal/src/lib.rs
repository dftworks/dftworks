//! Crystal structure module for DFT calculations.
//!
//! This module provides functionality for reading, storing, and manipulating
//! crystal structures including lattice vectors, atomic positions, and species.
//! Positions are stored in fractional coordinates relative to the lattice vectors.

use dwconsts::*;
use itertools::Itertools;
use lattice::Lattice;
use matrix::Matrix;
use pspot::PSPot;
use vector3::*;

use std::io::Write;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

/// Crystal structure representation.
///
/// Stores the crystal structure including lattice vectors, atomic positions
/// (in fractional coordinates), and species information. The structure can be
/// read from a file format or constructed programmatically.
#[derive(Debug, Default)]
pub struct Crystal {
    /// Scale factor for lattice vector a (in Angstroms)
    scale_a: f64,
    /// Scale factor for lattice vector b (in Angstroms)
    scale_b: f64,
    /// Scale factor for lattice vector c (in Angstroms)
    scale_c: f64,
    /// Lattice vectors defining the unit cell (stored in Bohr)
    latt: Lattice,
    /// Cell mask matrix (3x3) indicating which lattice vector components
    /// are allowed to vary during optimization (1.0 = T, 0.0 = F)
    cell_mask: Matrix<f64>,
    /// Atomic positions in fractional coordinates (relative to lattice vectors)
    atom_positions: Vec<Vector3f64>,
    /// Atomic species labels for each atom (same order as atom_positions)
    atom_species: Vec<String>,
    /// Indices of atoms grouped by species for efficient lookup
    /// `atom_indices_by_specie[isp]` contains all atom indices for species `isp`
    atom_indices_by_specie: Vec<Vec<usize>>,
}

impl Crystal {
    /// Calculate the total number of valence electrons in the crystal.
    ///
    /// Sums up the valence charge (zion) for all atoms based on their
    /// pseudopotential definitions.
    ///
    /// # Arguments
    /// * `pots` - Pseudopotential container providing zion values for each species
    ///
    /// # Returns
    /// Total number of valence electrons as a floating point number
    pub fn get_n_total_electrons(&self, pots: &PSPot) -> f64 {
        let mut sum = 0.0;

        for (isp, sp) in self.get_unique_species().iter().enumerate() {
            let natom_for_this_specie = self.atom_indices_by_specie[isp].len();

            let zion = pots.get_psp(sp).get_zion();

            sum += zion * natom_for_this_specie as f64;
        }

        sum
    }

    /// Get the valence charge (zion) for each atom.
    ///
    /// Returns a vector where each element corresponds to the valence charge
    /// of the atom at that index, based on its species' pseudopotential.
    ///
    /// # Arguments
    /// * `pots` - Pseudopotential container providing zion values for each species
    ///
    /// # Returns
    /// Vector of valence charges, one per atom
    pub fn get_zions(&self, pots: &PSPot) -> Vec<f64> {
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

    /// Get atom indices for a specific species.
    ///
    /// # Arguments
    /// * `isp` - Species index (0-based)
    ///
    /// # Returns
    /// Slice of atom indices belonging to this species
    pub fn get_atom_indices_of_specie(&self, isp: usize) -> &[usize] {
        &self.atom_indices_by_specie[isp]
    }

    /// Get a reference to the lattice structure.
    ///
    /// # Returns
    /// Reference to the Lattice object containing the unit cell vectors
    pub fn get_latt(&self) -> &Lattice {
        &self.latt
    }

    /// Get the cell mask matrix.
    ///
    /// The cell mask indicates which lattice vector components can vary
    /// during structural optimization (1.0 = allowed, 0.0 = fixed).
    ///
    /// # Returns
    /// Reference to the 3x3 cell mask matrix
    pub fn get_cell_mask(&self) -> &Matrix<f64> {
        &self.cell_mask
    }

    /// Get a list of unique atomic species in the crystal.
    ///
    /// # Returns
    /// Vector of unique species strings, in order of first appearance
    pub fn get_unique_species(&self) -> Vec<String> {
        self.atom_species.iter().unique().cloned().collect()
    }

    /// Get the number of unique atomic species.
    ///
    /// # Returns
    /// Number of distinct species in the crystal
    pub fn get_n_unique_species(&self) -> usize {
        self.atom_species.iter().unique().count()
    }

    /// Get atomic positions for a specific species.
    ///
    /// Returns positions in fractional coordinates for all atoms of the given species.
    ///
    /// # Arguments
    /// * `isp` - Species index (0-based)
    ///
    /// # Returns
    /// Vector of fractional positions for atoms of this species
    pub fn get_atom_positions_of_specie(&self, isp: usize) -> Vec<Vector3f64> {
        let atom_indices = self.get_atom_indices_of_specie(isp);

        let mut atom_positions_for_this_specie = vec![Vector3f64::zeros(); atom_indices.len()];

        for (i, idx) in atom_indices.iter().enumerate() {
            atom_positions_for_this_specie[i] = self.atom_positions[*idx];
        }

        atom_positions_for_this_specie
    }

    /// Get the total number of atoms in the crystal.
    ///
    /// # Returns
    /// Total number of atoms
    pub fn get_n_atoms(&self) -> usize {
        self.atom_positions.len()
    }

    /// Get atomic positions in fractional coordinates.
    ///
    /// # Returns
    /// Slice of fractional positions for all atoms
    pub fn get_atom_positions(&self) -> &[Vector3f64] {
        &self.atom_positions
    }

    /// Convert atomic positions from fractional to Cartesian coordinates.
    ///
    /// Returns positions in Cartesian coordinates (in Bohr) by applying
    /// the lattice transformation matrix.
    ///
    /// # Returns
    /// Vector of Cartesian positions for all atoms
    pub fn get_atom_positions_cart(&self) -> Vec<Vector3f64> {
        let mut atoms_cart = vec![Vector3f64::zeros(); self.atom_positions.len()];
        // Store reference to avoid borrow checker issues in closure
        let latt = &self.latt;

        // Use iterator chain for better vectorization
        self.atom_positions
            .iter()
            .zip(atoms_cart.iter_mut())
            .for_each(|(pos_frac, pos_cart)| {
                latt.frac_to_cart(pos_frac.as_slice(), pos_cart.as_mut_slice());
            });

        atoms_cart
    }

    /// Set atomic positions from Cartesian coordinates.
    ///
    /// Converts Cartesian coordinates (in Bohr) to fractional coordinates
    /// and updates the internal positions.
    ///
    /// # Arguments
    /// * `atoms_cart` - Slice of Cartesian positions (must match number of atoms)
    pub fn set_atom_positions_from_cart(&mut self, atoms_cart: &[Vector3f64]) {
        let mut tpos = Vector3f64::zeros();
        // Store reference to avoid borrow checker issues in closure
        let latt = &self.latt;

        // Use iterator chain for better vectorization
        self.atom_positions
            .iter_mut()
            .zip(atoms_cart.iter())
            .for_each(|(pos_frac, pos_cart)| {
                latt.cart_to_frac(pos_cart.as_slice(), tpos.as_mut_slice());
                *pos_frac = tpos;
            });
    }

    /// Set atomic positions directly from fractional coordinates.
    ///
    /// # Arguments
    /// * `atoms_frac` - Slice of fractional positions (must match number of atoms)
    pub fn set_atom_positions_from_frac(&mut self, atoms_frac: &[Vector3f64]) {
        // Use iterator chain for better vectorization
        self.atom_positions
            .iter_mut()
            .zip(atoms_frac.iter())
            .for_each(|(pos, frac)| *pos = *frac);
    }

    /// Update the lattice vectors.
    ///
    /// # Arguments
    /// * `latt` - New lattice structure to use
    pub fn set_lattice_vectors(&mut self, latt: &Lattice) {
        self.latt = latt.clone();
    }

    /// Get atom type indices (1-based species index for each atom).
    ///
    /// Each atom is assigned a type number corresponding to its species index + 1.
    /// Atoms of the same species have the same type number.
    ///
    /// # Returns
    /// Vector of type indices (1-based) for each atom
    pub fn get_atom_types(&self) -> Vec<i32> {
        let mut types = vec![0; self.get_n_atoms()];

        // Assign type numbers: first species = 1, second = 2, etc.
        for (isp, _sp) in self.get_unique_species().iter().enumerate() {
            for idx in self.atom_indices_by_specie[isp].iter() {
                types[*idx] = isp as i32 + 1;
            }
        }

        types
    }

    /// Get atomic species labels.
    ///
    /// # Returns
    /// Slice of species strings, one per atom
    pub fn get_atom_species(&self) -> &[String] {
        &self.atom_species
    }

    /// Create a new empty crystal structure.
    ///
    /// # Returns
    /// A new Crystal instance with default (empty) values
    pub fn new() -> Crystal {
        Crystal::default()
    }

    /// Read crystal structure from a file.
    ///
    /// File format:
    /// - Line 0: scale_a scale_b scale_c (scale factors in Angstroms)
    /// - Line 1: a_x a_y a_z mask_a_x mask_a_y mask_a_z (vector a components and cell mask)
    /// - Line 2: b_x b_y b_z mask_b_x mask_b_y mask_b_z (vector b components and cell mask)
    /// - Line 3: c_x c_y c_z mask_c_x mask_c_y mask_c_z (vector c components and cell mask)
    /// - Lines 4+: species x y z (atom species and fractional coordinates)
    ///
    /// Mask values: "T" or "t" = 1.0 (allowed to vary), "F" or "f" = 0.0 (fixed)
    /// Lattice vectors are converted from Angstroms to Bohr internally.
    ///
    /// # Arguments
    /// * `inpfile` - Path to the input crystal structure file
    pub fn read_file(&mut self, inpfile: &str) {
        let file = File::open(inpfile).unwrap();
        let lines = BufReader::new(file).lines();

        // Initialize storage
        self.atom_positions = Vec::new();
        self.atom_species = Vec::new();

        // Temporary storage for lattice vectors (in Angstroms before conversion)
        let mut vec_a = [0.0; 3];
        let mut vec_b = [0.0; 3];
        let mut vec_c = [0.0; 3];

        // Initialize 3x3 cell mask matrix
        self.cell_mask = Matrix::new(3, 3);

        // Parse file line by line
        for (i, line) in lines.enumerate() {
            let s: Vec<&str> = line.as_ref().unwrap().split_whitespace().collect();

            match i {
                // Line 0: Scale factors
                0 => {
                    self.scale_a = s[0].parse().unwrap();
                    self.scale_b = s[1].parse().unwrap();
                    self.scale_c = s[2].parse().unwrap();
                }

                // Line 1: Lattice vector a and its cell mask
                1 => {
                    // Parse vector a components
                    vec_a[0] = s[0].parse().unwrap();
                    vec_a[1] = s[1].parse().unwrap();
                    vec_a[2] = s[2].parse().unwrap();

                    // Apply scale factor and convert Angstroms to Bohr
                    for iv in 0..3 {
                        vec_a[iv] *= self.scale_a * ANG_TO_BOHR;
                    }

                    // Parse cell mask for vector a (columns 0, 1, 2 of row 0)
                    // "T" or "t" means the component can vary (1.0), otherwise fixed (0.0)
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

                // Line 2: Lattice vector b and its cell mask
                2 => {
                    // Parse vector b components
                    vec_b[0] = s[0].parse().unwrap();
                    vec_b[1] = s[1].parse().unwrap();
                    vec_b[2] = s[2].parse().unwrap();

                    // Apply scale factor and convert Angstroms to Bohr
                    for iv in 0..3 {
                        vec_b[iv] *= self.scale_b * ANG_TO_BOHR;
                    }

                    // Parse cell mask for vector b (columns 0, 1, 2 of row 1)
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

                // Line 3: Lattice vector c and its cell mask
                3 => {
                    // Parse vector c components
                    vec_c[0] = s[0].parse().unwrap();
                    vec_c[1] = s[1].parse().unwrap();
                    vec_c[2] = s[2].parse().unwrap();

                    // Apply scale factor and convert Angstroms to Bohr
                    for iv in 0..3 {
                        vec_c[iv] *= self.scale_c * ANG_TO_BOHR;
                    }

                    // Parse cell mask for vector c (columns 0, 1, 2 of row 2)
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

                // Lines 4+: Atomic positions (species and fractional coordinates)
                _ => {
                    // Skip empty lines
                    if s.is_empty() {
                        continue;
                    }

                    // Parse atom species and fractional coordinates
                    let symbol = s[0].to_string();
                    let x: f64 = s[1].parse().unwrap();
                    let y: f64 = s[2].parse().unwrap();
                    let z: f64 = s[3].parse().unwrap();

                    self.atom_species.push(symbol);
                    self.atom_positions.push(Vector3f64 { x, y, z });
                }
            }

        }

        // Set lattice after all vectors are read (now in Bohr)
        self.latt = Lattice::new(&vec_a, &vec_b, &vec_c);

        // Group atoms by species for efficient lookup
        let unique_species: Vec<String> = self.get_unique_species();
        let nsp = unique_species.len();

        // Initialize storage: one vector per species
        self.atom_indices_by_specie = vec![Vec::new(); nsp];

        // Create a map from species string to species index for O(1) lookup
        // This avoids O(n*m) nested loop, making grouping O(n) instead
        let species_to_index: HashMap<_, _> = unique_species
            .iter()
            .enumerate()
            .map(|(i, sp)| (sp.as_str(), i))
            .collect();

        // Assign each atom to its species group
        for (at_index, at_symbol) in self.atom_species.iter().enumerate() {
            if let Some(&isp) = species_to_index.get(at_symbol.as_str()) {
                self.atom_indices_by_specie[isp].push(at_index);
            }
        }
    }

    /// Write crystal structure to output file "out.crystal".
    ///
    /// Output format matches the input format, with lattice vectors
    /// converted back from Bohr to Angstroms.
    pub fn output(&self) {
        let mut f = File::create("out.crystal").unwrap();

        // Write scale factors
        writeln!(f, "{} {} {}", self.scale_a, self.scale_b, self.scale_c).unwrap();

        // Write lattice vector a (convert from Bohr to Angstroms)
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

        // Write lattice vector b (convert from Bohr to Angstroms)
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

        // Write lattice vector c (convert from Bohr to Angstroms)
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

        // Write atomic positions (species and fractional coordinates)
        for (i, at) in self.atom_positions.iter().enumerate() {
            writeln!(f, "{} {} {} {}", self.atom_species[i], at.x, at.y, at.z).unwrap();
        }
    }

    /// Display crystal structure information to stdout.
    ///
    /// Prints a formatted table showing:
    /// - Lattice vectors (in Angstroms)
    /// - Number of atoms
    /// - Atomic positions in both fractional and Cartesian coordinates
    /// - Species grouping information
    pub fn display(&self) {
        println!("   {:-^88}", " crystal structure ");
        println!();

        // Print lattice vectors
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

        // Print atomic information
        println!();
        println!("   natoms = {}", self.get_atom_positions().len());
        println!("   atom_positions\n");
        println!("                fractional                                                cartesian (A)");
        println!();

        // Print each atom with both fractional and Cartesian coordinates
        for (i, atom) in self.get_atom_positions().iter().enumerate() {
            let mut pos_c = Vector3f64::zeros();

            // Convert fractional to Cartesian for display
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

        // Print species grouping (atom indices for each species, 1-based)
        for (isp, sp) in self.get_unique_species().iter().enumerate() {
            println!(
                "   {} : {:?}",
                sp,
                self.get_atom_indices_of_specie(isp)
                    .iter()
                    .map(|x| x + 1) // Convert to 1-based indexing for display
                    .collect::<Vec<usize>>()
            );
        }
    }
}

/// Convert a floating point value to a cell mask character.
///
/// Used for output formatting: positive values become "T" (True, allowed to vary),
/// zero or negative values become "F" (False, fixed).
///
/// # Arguments
/// * `fval` - Floating point value (typically 1.0 or 0.0)
///
/// # Returns
/// "T" if value > 0.0, "F" otherwise
fn float_to_char(fval: f64) -> String {
    if fval > 0.0 {
        "T".to_string()
    } else {
        "F".to_string()
    }
}

#[cfg(test)]
mod tests;

