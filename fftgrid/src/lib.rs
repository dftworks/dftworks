//#![allow(warnings)]

use lattice::Lattice;
use std::{f64::consts, fmt};
use utility;

#[derive(Debug, Clone)]
pub struct FFTGrid {
    n1: usize,
    n2: usize,
    n3: usize,
}

impl FFTGrid {
    pub fn get_ntotf64(&self) -> f64 {
        (self.n1 * self.n2 * self.n3) as f64
    }

    pub fn get_ntot(&self) -> usize {
        self.n1 * self.n2 * self.n3
    }

    pub fn get_n1(&self) -> usize {
        self.n1
    }

    pub fn get_n2(&self) -> usize {
        self.n2
    }

    pub fn get_n3(&self) -> usize {
        self.n3
    }

    pub fn get_size(&self) -> [usize; 3] {
        [self.n1, self.n2, self.n3]
    }

    pub fn get_n1_left(&self) -> i32 {
        utility::fft_left_end(self.n1)
    }

    pub fn get_n1_right(&self) -> i32 {
        utility::fft_right_end(self.n1)
    }

    pub fn get_n2_left(&self) -> i32 {
        utility::fft_left_end(self.n2)
    }

    pub fn get_n2_right(&self) -> i32 {
        utility::fft_right_end(self.n2)
    }

    pub fn get_n3_left(&self) -> i32 {
        utility::fft_left_end(self.n3)
    }

    pub fn get_n3_right(&self) -> i32 {
        utility::fft_right_end(self.n3)
    }

    pub fn new(latt: &Lattice, ecutrho: f64) -> FFTGrid {
        let gmax = (2.0 * ecutrho).sqrt();

        let twopi = 2.0 * consts::PI;

        let mut n1 = (2.0 * gmax * latt.get_vector_a().norm2() / twopi).ceil() as usize;
        let mut n2 = (2.0 * gmax * latt.get_vector_b().norm2() / twopi).ceil() as usize;
        let mut n3 = (2.0 * gmax * latt.get_vector_c().norm2() / twopi).ceil() as usize;

        n1 = get_fftwn(n1);
        n2 = get_fftwn(n2);
        n3 = get_fftwn(n3);

        FFTGrid { n1, n2, n3 }
    }
}

impl fmt::Display for FFTGrid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let outstr = format!("{} x {} x {}", self.n1, self.n2, self.n3);

        write!(f, "{}", outstr)
    }
}

fn get_fftwn(n: usize) -> usize {
    let mut bres: bool = false;

    let mut tn = n;

    while !bres {
        bres = is_fftw_ok(tn);
        if !bres {
            tn += 1;
        }
    }

    tn
}

fn is_fftw_ok(n_to_check: usize) -> bool {
    const FACTORS: [usize; 6] = [2, 3, 5, 7, 11, 13];

    let mut pcnt = vec![0usize; 6];

    let mut tn = n_to_check;

    for (i, fi) in FACTORS.iter().enumerate() {
        while tn % fi == 0 && tn != 1 {
            tn = tn / fi;
            pcnt[i] += 1;
        }
    }

    if tn == 1 {
        true
    } else {
        false
    }
}
