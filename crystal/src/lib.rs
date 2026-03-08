use dwconsts::*;
use itertools::Itertools;
use lattice::Lattice;
use types::Matrix;
use pspot::PSPot;
use types::*;

use std::io::Write;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or(line)
}

fn parse_bool_mask(token: &str, path: &str, line_no: usize, axis: &str) -> Result<f64, String> {
    match token.trim().to_ascii_lowercase().as_str() {
        "t" => Ok(1.0),
        "f" => Ok(0.0),
        other => Err(format!(
            "{}:{}: invalid cell mask '{}' for {}; expected T or F",
            path, line_no, other, axis
        )),
    }
}

fn parse_f64_token(
    tokens: &[&str],
    index: usize,
    path: &str,
    line_no: usize,
    field: &str,
) -> Result<f64, String> {
    let raw = tokens.get(index).ok_or_else(|| {
        format!(
            "{}:{}: missing {} (expected enough columns on this line)",
            path, line_no, field
        )
    })?;
    raw.parse::<f64>().map_err(|err| {
        format!(
            "{}:{}: failed to parse {} from '{}': {}",
            path, line_no, field, raw, err
        )
    })
}

// Crystal structure container.
//
// Coordinates:
// - lattice vectors stored in Bohr
// - atomic positions stored in fractional coordinates
// - helper conversions provide Cartesian views when needed
#[derive(Debug)]
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

impl Default for Crystal {
    fn default() -> Self {
        Self {
            scale_a: 0.0,
            scale_b: 0.0,
            scale_c: 0.0,
            latt: Lattice::default(),
            cell_mask: Matrix::<f64>::new(0, 0),
            atom_positions: Vec::new(),
            atom_species: Vec::new(),
            atom_indices_by_specie: Vec::new(),
        }
    }
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
        if let Err(err) = self.try_read_file(inpfile) {
            panic!("{}", err);
        }
    }

    pub fn try_read_file(&mut self, inpfile: &str) -> Result<(), String> {
        // Parse in.crystal with format:
        // line 1: scale_a scale_b scale_c
        // line 2-4: lattice vectors + T/F mask flags per Cartesian component
        // remaining lines: species x y z (fractional atomic positions)
        let file = File::open(inpfile)
            .map_err(|err| format!("failed to read '{}': {}", inpfile, err))?;
        let lines = BufReader::new(file).lines();

        let mut atom_positions = Vec::new();
        let mut atom_species = Vec::new();

        let mut vec_a = [0.0; 3];
        let mut vec_b = [0.0; 3];
        let mut vec_c = [0.0; 3];

        let mut cell_mask = Matrix::new(3, 3);
        let mut scale_a = 0.0;
        let mut scale_b = 0.0;
        let mut scale_c = 0.0;
        let mut content_line_index = 0usize;

        for (line_idx, line_res) in lines.enumerate() {
            let line_no = line_idx + 1;
            let line = line_res
                .map_err(|err| format!("{}:{}: failed to read line: {}", inpfile, line_no, err))?;
            let stripped = strip_comment(&line);
            let s: Vec<&str> = stripped.split_whitespace().collect();
            if s.is_empty() {
                continue;
            }

            match content_line_index {
                0 => {
                    if s.len() < 3 {
                        return Err(format!(
                            "{}:{}: expected three scale factors on the first content line",
                            inpfile, line_no
                        ));
                    }
                    // Independent scale factors for three lattice vectors.
                    scale_a = parse_f64_token(&s, 0, inpfile, line_no, "scale_a")?;
                    scale_b = parse_f64_token(&s, 1, inpfile, line_no, "scale_b")?;
                    scale_c = parse_f64_token(&s, 2, inpfile, line_no, "scale_c")?;
                }

                1 => {
                    if s.len() < 6 {
                        return Err(format!(
                            "{}:{}: expected lattice vector a followed by three T/F mask flags",
                            inpfile, line_no
                        ));
                    }
                    // Lattice vector a and optimization mask flags.
                    vec_a[0] = parse_f64_token(&s, 0, inpfile, line_no, "a_x")?;
                    vec_a[1] = parse_f64_token(&s, 1, inpfile, line_no, "a_y")?;
                    vec_a[2] = parse_f64_token(&s, 2, inpfile, line_no, "a_z")?;

                    for iv in 0..3 {
                        vec_a[iv] *= scale_a * ANG_TO_BOHR;
                    }

                    cell_mask[(0, 0)] = parse_bool_mask(s[3], inpfile, line_no, "a_x")?;
                    cell_mask[(1, 0)] = parse_bool_mask(s[4], inpfile, line_no, "a_y")?;
                    cell_mask[(2, 0)] = parse_bool_mask(s[5], inpfile, line_no, "a_z")?;
                }

                2 => {
                    if s.len() < 6 {
                        return Err(format!(
                            "{}:{}: expected lattice vector b followed by three T/F mask flags",
                            inpfile, line_no
                        ));
                    }
                    // Lattice vector b and optimization mask flags.
                    vec_b[0] = parse_f64_token(&s, 0, inpfile, line_no, "b_x")?;
                    vec_b[1] = parse_f64_token(&s, 1, inpfile, line_no, "b_y")?;
                    vec_b[2] = parse_f64_token(&s, 2, inpfile, line_no, "b_z")?;

                    for iv in 0..3 {
                        vec_b[iv] *= scale_b * ANG_TO_BOHR;
                    }

                    cell_mask[(0, 1)] = parse_bool_mask(s[3], inpfile, line_no, "b_x")?;
                    cell_mask[(1, 1)] = parse_bool_mask(s[4], inpfile, line_no, "b_y")?;
                    cell_mask[(2, 1)] = parse_bool_mask(s[5], inpfile, line_no, "b_z")?;
                }

                3 => {
                    if s.len() < 6 {
                        return Err(format!(
                            "{}:{}: expected lattice vector c followed by three T/F mask flags",
                            inpfile, line_no
                        ));
                    }
                    // Lattice vector c and optimization mask flags.
                    vec_c[0] = parse_f64_token(&s, 0, inpfile, line_no, "c_x")?;
                    vec_c[1] = parse_f64_token(&s, 1, inpfile, line_no, "c_y")?;
                    vec_c[2] = parse_f64_token(&s, 2, inpfile, line_no, "c_z")?;

                    for iv in 0..3 {
                        vec_c[iv] *= scale_c * ANG_TO_BOHR;
                    }

                    cell_mask[(0, 2)] = parse_bool_mask(s[3], inpfile, line_no, "c_x")?;
                    cell_mask[(1, 2)] = parse_bool_mask(s[4], inpfile, line_no, "c_y")?;
                    cell_mask[(2, 2)] = parse_bool_mask(s[5], inpfile, line_no, "c_z")?;
                }

                // atoms
                _ => {
                    if s.len() < 4 {
                        return Err(format!(
                            "{}:{}: expected atomic line '<species> x y z'",
                            inpfile, line_no
                        ));
                    }

                    let symbol = s[0].to_string();
                    let x = parse_f64_token(&s, 1, inpfile, line_no, "atom_x")?;
                    let y = parse_f64_token(&s, 2, inpfile, line_no, "atom_y")?;
                    let z = parse_f64_token(&s, 3, inpfile, line_no, "atom_z")?;

                    // Atomic position remains fractional.
                    atom_species.push(symbol);
                    atom_positions.push(Vector3f64::new(x, y, z));
                }
            }
            content_line_index += 1;
        }

        if content_line_index < 4 {
            return Err(format!(
                "{}: expected 4 non-empty header lines before atomic positions",
                inpfile
            ));
        }

        let latt = Lattice::new(&vec_a, &vec_b, &vec_c);
        // Build specie -> atom-index lookup for fast grouped operations.
        let unique_species: Vec<String> = atom_species.clone().into_iter().unique().collect();

        let nsp = unique_species.len();

        let mut atom_indices_by_specie = vec![Vec::new(); nsp];

        for (at_index, at_symbol) in atom_species.iter().enumerate() {
            for (isp, sp) in unique_species.iter().enumerate() {
                if *sp == *at_symbol {
                    atom_indices_by_specie[isp].push(at_index);
                }
            }
        }

        self.scale_a = scale_a;
        self.scale_b = scale_b;
        self.scale_c = scale_c;
        self.latt = latt;
        self.cell_mask = cell_mask;
        self.atom_positions = atom_positions;
        self.atom_species = atom_species;
        self.atom_indices_by_specie = atom_indices_by_specie;

        Ok(())
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
            float_to_char(self.cell_mask[(0, 0)]),
            float_to_char(self.cell_mask[(1, 0)]),
            float_to_char(self.cell_mask[(2, 0)])
        )
        .unwrap();

        let b = self.latt.get_vector_b();

        writeln!(
            f,
            "{:.12?} {:.12?} {:.12?} {} {} {}",
            b.x / self.scale_b * BOHR_TO_ANG,
            b.y / self.scale_b * BOHR_TO_ANG,
            b.z / self.scale_b * BOHR_TO_ANG,
            float_to_char(self.cell_mask[(0, 1)]),
            float_to_char(self.cell_mask[(1, 1)]),
            float_to_char(self.cell_mask[(2, 1)])
        )
        .unwrap();

        let c = self.latt.get_vector_c();
        writeln!(
            f,
            "{:.12?} {:.12?} {:.12?} {} {} {}",
            c.x / self.scale_c * BOHR_TO_ANG,
            c.y / self.scale_c * BOHR_TO_ANG,
            c.z / self.scale_c * BOHR_TO_ANG,
            float_to_char(self.cell_mask[(0, 2)]),
            float_to_char(self.cell_mask[(1, 2)]),
            float_to_char(self.cell_mask[(2, 2)])
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

#[test]
fn test_try_read_file_reports_invalid_mask() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("dftworks-crystal-invalid-mask-{}", nanos));
    let content = "\
1.0 1.0 1.0
1.0 0.0 0.0 X T T
0.0 1.0 0.0 T T T
0.0 0.0 1.0 T T T
Si 0.0 0.0 0.0
";
    fs::write(&path, content).expect("write temp in.crystal");

    let mut crystal = Crystal::new();
    let err = crystal
        .try_read_file(path.to_str().expect("temp path should be utf8"))
        .expect_err("invalid in.crystal should fail");
    assert!(err.contains("expected T or F"));

    fs::remove_file(path).expect("remove temp in.crystal");
}
