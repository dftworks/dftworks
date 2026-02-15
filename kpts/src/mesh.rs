use crystal::Crystal;
use lattice::Lattice;
use symmetry::*;
use vector3::*;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use crate::KPTS;

pub struct KptsMesh {
    k_frac: Vec<Vector3f64>,
    k_degeneracy: Vec<usize>,
    k_weight: Vec<f64>,
    k_mesh: [i32; 3],
}

impl KptsMesh {
    pub fn new(crystal: &Crystal) -> KptsMesh {
        // Read Monkhorst-Pack mesh + shift from in.kmesh.
        let (k_mesh, is_shift) = read_k_mesh();

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
        let (kpts, _mapping, _k_unique, _nk_unique) = get_ir_reciprocal_mesh(
            k_mesh,
            is_shift,
            &mut lattice,
            &mut position,
            &types,
            1.0E-05,
        );

        let nk = kpts.len();

        let mut k_frac = vec![Vector3f64::zeros(); nk];

        let mut k_weight = vec![0.0; nk];

        let mut k_degeneracy = vec![0; nk];

        let nk_total = kpts.len();

        for (ik, k) in kpts.iter().enumerate() {
            k_frac[ik] = Vector3f64 {
                x: k[0],
                y: k[1],
                z: k[2],
            };

            // Current path assigns uniform weights on the reduced list.
            k_weight[ik] = 1.0 / nk_total as f64;
            k_degeneracy[ik] = 1;
        }

        KptsMesh {
            k_frac,
            k_degeneracy,
            k_weight,
            k_mesh,
        }
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
        println!("   {:-^88}", " k-points (fractional) ");
        println!();

        println!("{:12} {:^6} {}", "", "nkpt =", self.get_n_kpts());
        println!();

        println!(
            "{:12} {:^6} {:^16} {:^16} {:^16} {:^12}",
            "", "index", "k1", "k2", "k3", "degeneracy"
        );

        for ik in 0..self.get_n_kpts() {
            let xk_frac = self.get_k_frac(ik);
            let xk_degeneracy = self.get_k_degeneracy(ik);

            println!(
                "{:12} {:^6} {:16.12} {:16.12} {:16.12} {:^12}",
                "",
                ik + 1,
                xk_frac.x,
                xk_frac.y,
                xk_frac.z,
                xk_degeneracy
            );
        }
    }
}

fn read_k_mesh() -> ([i32; 3], [i32; 3]) {
    // in.kmesh format:
    // line 1: nk1 nk2 nk3
    // line 2: shift1 shift2 shift3 (0/1)
    let lines = read_file_data_to_vec("in.kmesh");

    let s: Vec<&str> = lines[0].split_whitespace().collect();
    let nk1 = s[0].parse().unwrap();
    let nk2 = s[1].parse().unwrap();
    let nk3 = s[2].parse().unwrap();

    let s: Vec<&str> = lines[1].split_whitespace().collect();
    let k1_shift = s[0].parse().unwrap();
    let k2_shift = s[1].parse().unwrap();
    let k3_shift = s[2].parse().unwrap();

    ([nk1, nk2, nk3], [k1_shift, k2_shift, k3_shift])
}

fn read_file_data_to_vec(kfile: &str) -> Vec<String> {
    // Lightweight line reader used by k-point input parsers.
    let file = File::open(kfile).unwrap();
    let lines = BufReader::new(file).lines();
    let lines: Vec<String> = lines.map_while(std::io::Result::ok).collect();

    lines
}
