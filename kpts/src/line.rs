use lattice::Lattice;
use vector3::*;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use crate::KPTS;

pub struct KptsLine {
    k_frac: Vec<Vector3f64>,
    k_degeneracy: Vec<usize>,
    k_weight: Vec<f64>,
}

impl KptsLine {
    pub fn new() -> KptsLine {
        let k_frac = read_k_line();

        let nk = k_frac.len();

        let k_weight = vec![1.0 / nk as f64; nk];

        let k_degeneracy = vec![1; nk];

        KptsLine {
            k_frac,
            k_degeneracy,
            k_weight,
        }
    }
}

impl KPTS for KptsLine {
    fn get_k_mesh(&self) -> [i32; 3] {
        println!("get_k_mesh not implemented");
        //std::process::exit(1);

        [0, 0, 0]
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

fn read_k_line() -> Vec<Vector3f64> {
    let lines = read_file_data_to_vec("in.kline");

    let s: Vec<&str> = lines[0].split_whitespace().collect();
    let npts: usize = s[1].parse().unwrap();

    let mut kpts = Vec::new();

    // L 0.5 0.5 0.5 G 0.0 0.0 0.0
    for line in lines[1..].iter() {
        let s: Vec<&str> = line.split_whitespace().collect();

        let x1: f64 = s[1].parse().unwrap();
        let y1: f64 = s[2].parse().unwrap();
        let z1: f64 = s[3].parse().unwrap();

        let x2: f64 = s[5].parse().unwrap();
        let y2: f64 = s[6].parse().unwrap();
        let z2: f64 = s[7].parse().unwrap();

        let xspace = (x2 - x1) / ((npts - 1) as f64);
        let yspace = (y2 - y1) / ((npts - 1) as f64);
        let zspace = (z2 - z1) / ((npts - 1) as f64);

        for i in 0..npts {
            let if64 = i as f64;

            let x = x1 + if64 * xspace;
            let y = y1 + if64 * yspace;
            let z = z1 + if64 * zspace;

            kpts.push(Vector3f64 { x, y, z });
        }
    }

    kpts
}

fn read_file_data_to_vec(kfile: &str) -> Vec<String> {
    let file = File::open(kfile).unwrap();
    let lines = BufReader::new(file).lines();
    let lines: Vec<String> = lines.map_while(std::io::Result::ok).collect();

    lines
}
