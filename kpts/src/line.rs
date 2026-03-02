use lattice::Lattice;
use vector3::*;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use crate::{KPTS, KptsError};

pub struct KptsLine {
    k_frac: Vec<Vector3f64>,
    k_degeneracy: Vec<usize>,
    k_weight: Vec<f64>,
}

impl KptsLine {
    pub fn new() -> KptsLine {
        Self::try_new().unwrap_or_else(|err| panic!("failed to build line k-points: {}", err))
    }

    pub fn try_new() -> Result<KptsLine, KptsError> {
        // Read explicit high-symmetry path from in.kline.
        let k_frac = read_k_line()?;
        if k_frac.is_empty() {
            return Err(KptsError::new("in.kline did not produce any k-points"));
        }

        let nk = k_frac.len();

        let k_weight = vec![1.0 / nk as f64; nk];

        let k_degeneracy = vec![1; nk];

        Ok(KptsLine {
            k_frac,
            k_degeneracy,
            k_weight,
        })
    }
}

impl KPTS for KptsLine {
    fn get_k_mesh(&self) -> [i32; 3] {
        // Line mode does not represent a regular 3D mesh.
        println!("get_k_mesh not implemented");

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

fn parse_f64_token(tokens: &[&str], idx: usize, line_no: usize, label: &str) -> Result<f64, KptsError> {
    let value = tokens.get(idx).ok_or_else(|| {
        KptsError::new(format!(
            "in.kline:{}: missing '{}' token at column {}",
            line_no, label, idx
        ))
    })?;
    value.parse::<f64>().map_err(|e| {
        KptsError::new(format!(
            "in.kline:{}: invalid {} '{}': {}",
            line_no, label, value, e
        ))
    })
}

fn read_k_line() -> Result<Vec<Vector3f64>, KptsError> {
    // in.kline format:
    // line 1: "npts <N>"
    // next lines: label1 k1x k1y k1z label2 k2x k2y k2z
    // Each segment is linearly interpolated with N points.
    let lines = read_file_data_to_vec("in.kline")?;
    if lines.is_empty() {
        return Err(KptsError::new("in.kline is empty"));
    }

    let header: Vec<&str> = lines[0].split_whitespace().collect();
    if header.len() != 2 || header[0] != "npts" {
        return Err(KptsError::new(
            "in.kline:1: expected header format 'npts <N>'",
        ));
    }
    let npts: usize = header[1].parse().map_err(|e| {
        KptsError::new(format!(
            "in.kline:1: invalid npts '{}': {}",
            header[1], e
        ))
    })?;
    if npts < 2 {
        return Err(KptsError::new(format!(
            "in.kline:1: npts must be >= 2, got {}",
            npts
        )));
    }

    let mut kpts = Vec::new();

    // Example segment:
    // L 0.5 0.5 0.5 G 0.0 0.0 0.0
    for (line_idx, line) in lines[1..].iter().enumerate() {
        let line_no = line_idx + 2;
        let s: Vec<&str> = line.split_whitespace().collect();
        if s.is_empty() {
            continue;
        }
        if s.len() != 8 {
            return Err(KptsError::new(format!(
                "in.kline:{}: expected 8 tokens '<label1> k1x k1y k1z <label2> k2x k2y k2z', got {}",
                line_no,
                s.len()
            )));
        }

        let x1 = parse_f64_token(s.as_slice(), 1, line_no, "k1x")?;
        let y1 = parse_f64_token(s.as_slice(), 2, line_no, "k1y")?;
        let z1 = parse_f64_token(s.as_slice(), 3, line_no, "k1z")?;

        let x2 = parse_f64_token(s.as_slice(), 5, line_no, "k2x")?;
        let y2 = parse_f64_token(s.as_slice(), 6, line_no, "k2y")?;
        let z2 = parse_f64_token(s.as_slice(), 7, line_no, "k2z")?;

        let xspace = (x2 - x1) / ((npts - 1) as f64);
        let yspace = (y2 - y1) / ((npts - 1) as f64);
        let zspace = (z2 - z1) / ((npts - 1) as f64);

        for i in 0..npts {
            let if64 = i as f64;

            let x = x1 + if64 * xspace;
            let y = y1 + if64 * yspace;
            let z = z1 + if64 * zspace;

            kpts.push(Vector3f64::new(x, y, z));
        }
    }

    if kpts.is_empty() {
        return Err(KptsError::new("in.kline did not define any path segments"));
    }

    Ok(kpts)
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
