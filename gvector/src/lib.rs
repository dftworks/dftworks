//#![allow(warnings)]

use itertools::multizip;
use lattice::Lattice;
use utility;
use vector3::{Vector3f64, Vector3i32};

#[derive(Debug)]
pub struct GVector {
    miller: Vec<Vector3i32>,
    cart: Vec<Vector3f64>,
}

impl GVector {
    pub fn new(latt: &Lattice, n1: usize, n2: usize, n3: usize) -> GVector {
        let blatt = latt.reciprocal();

        let nsize = n1 * n2 * n3;

        // generate miller index

        let mut t_miller = vec![Vector3i32::zeros(); nsize];

        set_miller(t_miller.as_mut_slice(), n1, n2, n3);

        // calculate cartesian coordinates of miller index

        let mut t_cart = vec![Vector3f64::zeros(); nsize];

        miller_to_cart(t_cart.as_mut_slice(), t_miller.as_slice(), &blatt);

        // calculate the length of each G vector

        let mut t_g: Vec<f64> = vec![0.0; nsize];

        set_g_norm(t_g.as_mut_slice(), t_cart.as_slice());

        // calculate the index of ordered G vector according to length

        let ordered_index = utility::argsort(&t_g);

        // sort cart

        let mut cart = vec![Vector3f64::zeros(); nsize];

        for (i, &j) in ordered_index.iter().enumerate() {
            cart[i] = t_cart[j];
        }

        // sort miller

        let mut miller = vec![Vector3i32::zeros(); nsize];

        for (i, &j) in ordered_index.iter().enumerate() {
            miller[i] = t_miller[j];
        }

        // Construct a new GVector struct

        GVector { miller, cart }
    }
    pub fn get_miller(&self) -> &[Vector3i32] {
        self.miller.as_slice()
    }

    pub fn get_cart(&self) -> &[Vector3f64] {
        self.cart.as_slice()
    }

    pub fn set_g_vector_index(&self, ecut: f64, xk: Vector3f64, gindex: &mut [usize]) {
        let mut npw = 0;

        let two_ecut = 2.0 * ecut;

        for (i, g) in self.cart.iter().enumerate() {
            let kg = xk + *g;

            let kq2 = kg.x * kg.x + kg.y * kg.y + kg.z * kg.z;

            if kq2 <= two_ecut {
                gindex[npw] = i;

                npw += 1;
            }
        }
    }

    // |k+G|^2 < 2*Ecut
    pub fn get_n_plane_waves(&self, ecut: f64, xk: Vector3f64) -> usize {
        let mut npw = 0;

        let two_ecut = 2.0 * ecut;

        for g in self.cart.iter() {
            let kg = xk + *g;

            let kq2 = kg.x * kg.x + kg.y * kg.y + kg.z * kg.z;

            if kq2 <= two_ecut {
                npw += 1;
            }
        }

        npw
    }
}

fn set_g_norm(g: &mut [f64], cart: &[Vector3f64]) {
    for (x, y) in multizip((g.iter_mut(), cart.iter())) {
        *x = y.norm2();
    }
}

// x = i * a.x + j * b.x + k * c.x
// y = i * a.y + j * b.y + k * c.y
// z = i * a.z + j * b.z + k * c.z
fn miller_to_cart(cart: &mut [Vector3f64], miller: &[Vector3i32], blatt: &Lattice) {
    let a = blatt.get_vector_a();
    let b = blatt.get_vector_b();
    let c = blatt.get_vector_c();

    for (ct, mi) in multizip((cart.iter_mut(), miller.iter())) {
        let i = mi.x as f64;
        let j = mi.y as f64;
        let k = mi.z as f64;

        ct.x = i * a.x + j * b.x + k * c.x;
        ct.y = i * a.y + j * b.y + k * c.y;
        ct.z = i * a.z + j * b.z + k * c.z;
    }
}

fn set_miller(miller: &mut [Vector3i32], n1: usize, n2: usize, n3: usize) {
    let i1 = utility::fft_left_end(n1);
    let i2 = utility::fft_left_end(n2);
    let i3 = utility::fft_left_end(n3);

    let j1 = utility::fft_right_end(n1);
    let j2 = utility::fft_right_end(n2);
    let j3 = utility::fft_right_end(n3);

    let mut ig = 0;
    for i in i1..j1 + 1 {
        for j in i2..j2 + 1 {
            for k in i3..j3 + 1 {
                miller[ig].x = i;
                miller[ig].y = j;
                miller[ig].z = k;

                ig += 1;
            }
        }
    }
}

#[test]
fn test_gvector() {
    use dwconsts::*;
    
    // length in angstrom
    let mut latt = Lattice::new(&[3.0, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 3.0]);

    latt.scaled_by(ANG_TO_BOHR);

    let ecutrho = 120.0; // hartree
    let ecut = 20.0; // hartree
    let xk = Vector3f64::zeros();

    let blatt = latt.reciprocal();

    let gvec = GVector::new(&blatt, 20, 20, 20);

    let mut npw = gvec.get_n_plane_waves(ecut, xk);

    println!("Total plane waves for wave functioin = {}", npw);

    npw = gvec.get_n_plane_waves(ecutrho, xk);

    println!("Total plane waves for charge density = {}", npw);
}
