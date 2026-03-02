use crystal::Crystal;
use lattice::Lattice;
use symmetry::*;
use types::*;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use crate::{KPTS, KptsError};

pub struct KptsMesh {
    k_frac: Vec<Vector3f64>,
    k_degeneracy: Vec<usize>,
    k_weight: Vec<f64>,
    k_mesh: [i32; 3],
    nk_total: usize,
    is_symmetry_reduced: bool,
}

impl KptsMesh {
    pub fn new(crystal: &Crystal, use_symmetry: bool) -> KptsMesh {
        Self::try_new(crystal, use_symmetry)
            .unwrap_or_else(|err| panic!("failed to build mesh k-points: {}", err))
    }

    pub fn try_new(crystal: &Crystal, use_symmetry: bool) -> Result<KptsMesh, KptsError> {
        // Read Monkhorst-Pack mesh + shift from in.kmesh.
        let (k_mesh, is_shift) = read_k_mesh()?;

        // Convert crystal data to the format expected by the symmetry helper.
        let mut lattice = crystal.get_latt().as_2d_array_row_major();
        let natoms = crystal.get_n_atoms();

        let mut position = vec![[0.0; 3]; natoms];
        for (i, at) in crystal.get_atom_positions().iter().enumerate() {
            position[i][0] = at.x;
            position[i][1] = at.y;
            position[i][2] = at.z;
        }

        let mut types = vec![0i32; natoms];

        for (i, at) in crystal.get_atom_species().iter().enumerate() {
            for (itype, sp) in crystal.get_unique_species().iter().enumerate() {
                if sp == at {
                    types[i] = itype as i32;
                }
            }
        }

        // Build irreducible reciprocal mesh using symmetry reduction.
        let (kpts, _mapping, k_unique, nk_unique) = get_ir_reciprocal_mesh(
            k_mesh,
            is_shift,
            &mut lattice,
            &mut position,
            &types,
            1.0E-05,
        );

        let nk_total = kpts.len();
        let use_reduced = use_symmetry && !k_unique.is_empty() && k_unique.len() == nk_unique.len();
        let nk = if use_reduced {
            k_unique.len()
        } else {
            nk_total
        };

        let mut k_frac = vec![Vector3f64::zeros(); nk];
        let mut k_weight = vec![0.0; nk];
        let mut k_degeneracy = vec![0; nk];

        if use_reduced {
            for ik in 0..nk {
                let idx = k_unique[ik] as usize;
                let k = kpts[idx];
                k_frac[ik] = Vector3f64::new(k[0], k[1], k[2]);
                k_degeneracy[ik] = nk_unique[ik];
                k_weight[ik] = nk_unique[ik] as f64 / nk_total as f64;
            }
        } else {
            for (ik, k) in kpts.iter().enumerate() {
                k_frac[ik] = Vector3f64::new(k[0], k[1], k[2]);
                k_weight[ik] = 1.0 / nk_total as f64;
                k_degeneracy[ik] = 1;
            }
        }

        Ok(KptsMesh {
            k_frac,
            k_degeneracy,
            k_weight,
            k_mesh,
            nk_total,
            is_symmetry_reduced: use_reduced,
        })
    }
}

impl KPTS for KptsMesh {
    fn get_k_mesh(&self) -> [i32; 3] {
        self.k_mesh
    }

    fn get_k_frac(&self, k_index: usize) -> Vector3f64 {
        self.k_frac[k_index]
    }

    fn get_k_weight(&self, k_index: usize) -> f64 {
        self.k_weight[k_index]
    }

    fn get_k_degeneracy(&self, k_index: usize) -> usize {
        self.k_degeneracy[k_index]
    }

    fn get_n_kpts(&self) -> usize {
        self.k_frac.len()
    }

    fn frac_to_cart(&self, k_frac: &Vector3f64, blatt: &Lattice) -> Vector3f64 {
        // k_cart = k1*b1 + k2*b2 + k3*b3
        let a = blatt.get_vector_a();
        let b = blatt.get_vector_b();
        let c = blatt.get_vector_c();

        let mut k_cart = Vector3f64::zeros();

        k_cart.x = k_frac.x * a.x + k_frac.y * b.x + k_frac.z * c.x;
        k_cart.y = k_frac.x * a.y + k_frac.y * b.y + k_frac.z * c.y;
        k_cart.z = k_frac.x * a.z + k_frac.y * b.z + k_frac.z * c.z;

        k_cart
    }

    fn display(&self) {
        println!();
        if self.is_symmetry_reduced {
            println!("   {:-^88}", " IR k-points (symmetry-reduced) ");
            println!(
                "   {:12} {} / {}",
                "nkpt (IR/full) =",
                self.get_n_kpts(),
                self.nk_total
            );
        } else {
            println!("   {:-^88}", " k-points (fractional) ");
            println!("{:12} {:^6} {}", "", "nkpt =", self.get_n_kpts());
        }
        println!();

        println!(
            "{:12} {:^6} {:^16} {:^16} {:^16} {:^12} {:^12}",
            "", "index", "k1", "k2", "k3", "degeneracy", "weight"
        );

        for ik in 0..self.get_n_kpts() {
            let xk_frac = self.get_k_frac(ik);
            let xk_degeneracy = self.get_k_degeneracy(ik);
            let xk_weight = self.get_k_weight(ik);

            println!(
                "{:12} {:^6} {:16.12} {:16.12} {:16.12} {:^12} {:12.8}",
                "",
                ik + 1,
                xk_frac.x,
                xk_frac.y,
                xk_frac.z,
                xk_degeneracy,
                xk_weight
            );
        }
    }
}

fn parse_i32_token(tokens: &[&str], idx: usize, line_no: usize, label: &str) -> Result<i32, KptsError> {
    let value = tokens.get(idx).ok_or_else(|| {
        KptsError::new(format!(
            "in.kmesh:{}: missing '{}' token at column {}",
            line_no, label, idx
        ))
    })?;
    value.parse::<i32>().map_err(|e| {
        KptsError::new(format!(
            "in.kmesh:{}: invalid {} '{}': {}",
            line_no, label, value, e
        ))
    })
}

fn read_k_mesh() -> Result<([i32; 3], [i32; 3]), KptsError> {
    // in.kmesh format:
    // line 1: nk1 nk2 nk3
    // line 2: shift1 shift2 shift3 (0/1)
    let lines = read_file_data_to_vec("in.kmesh")?;
    if lines.len() < 2 {
        return Err(KptsError::new(
            "in.kmesh: expected at least 2 lines: mesh and shift",
        ));
    }

    let s: Vec<&str> = lines[0].split_whitespace().collect();
    if s.len() != 3 {
        return Err(KptsError::new(format!(
            "in.kmesh:1: expected 3 mesh integers, got {}",
            s.len()
        )));
    }
    let nk1 = parse_i32_token(s.as_slice(), 0, 1, "nk1")?;
    let nk2 = parse_i32_token(s.as_slice(), 1, 1, "nk2")?;
    let nk3 = parse_i32_token(s.as_slice(), 2, 1, "nk3")?;
    if nk1 <= 0 || nk2 <= 0 || nk3 <= 0 {
        return Err(KptsError::new(format!(
            "in.kmesh:1: mesh sizes must be > 0, got [{}, {}, {}]",
            nk1, nk2, nk3
        )));
    }

    let s: Vec<&str> = lines[1].split_whitespace().collect();
    if s.len() != 3 {
        return Err(KptsError::new(format!(
            "in.kmesh:2: expected 3 shift integers, got {}",
            s.len()
        )));
    }
    let k1_shift = parse_i32_token(s.as_slice(), 0, 2, "k1_shift")?;
    let k2_shift = parse_i32_token(s.as_slice(), 1, 2, "k2_shift")?;
    let k3_shift = parse_i32_token(s.as_slice(), 2, 2, "k3_shift")?;

    for (axis, shift) in [("k1_shift", k1_shift), ("k2_shift", k2_shift), ("k3_shift", k3_shift)] {
        if shift != 0 && shift != 1 {
            return Err(KptsError::new(format!(
                "in.kmesh:2: {} must be 0 or 1, got {}",
                axis, shift
            )));
        }
    }

    Ok(([nk1, nk2, nk3], [k1_shift, k2_shift, k3_shift]))
}

fn read_file_data_to_vec(kfile: &str) -> Result<Vec<String>, KptsError> {
    // Lightweight line reader used by k-point input parsers.
    let file = File::open(kfile)
        .map_err(|e| KptsError::new(format!("failed to open '{}': {}", kfile, e)))?;
    let mut lines = Vec::new();
    for (line_idx, line_res) in BufReader::new(file).lines().enumerate() {
        let line = line_res.map_err(|e| {
            KptsError::new(format!(
                "failed to read '{}', line {}: {}",
                kfile,
                line_idx + 1,
                e
            ))
        })?;
        lines.push(line);
    }
    Ok(lines)
}
