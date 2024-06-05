use matrix::*;
use vector3::*;

use std::{f64::consts, fmt};

#[derive(Debug, Default, Clone)]
pub struct Lattice {
    data: Matrix<f64>,
}

impl Lattice {
    pub fn new(a: &[f64], b: &[f64], c: &[f64]) -> Lattice {
        let mut data = Matrix::<f64>::new(3, 3);

        data.set_col(0, a);
        data.set_col(1, b);
        data.set_col(2, c);

        Lattice { data }
    }

    pub fn get_metric_tensor(&self) -> Matrix<f64> {
        let mut g = Matrix::<f64>::new(3, 3);

        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        g[[0, 0]] = a.dot_product(&a);
        g[[0, 1]] = a.dot_product(&b);
        g[[0, 2]] = a.dot_product(&c);

        g[[1, 0]] = b.dot_product(&a);
        g[[1, 1]] = b.dot_product(&b);
        g[[1, 2]] = b.dot_product(&c);

        g[[2, 0]] = c.dot_product(&a);
        g[[2, 1]] = c.dot_product(&b);
        g[[2, 2]] = c.dot_product(&c);

        g
    }

    pub fn as_matrix(&self) -> &Matrix<f64> {
        &self.data
    }

    pub fn as_mut_matrix(&mut self) -> &mut Matrix<f64> {
        &mut self.data
    }

    pub fn as_2d_array_col_major(&self) -> [[f64; 3]; 3] {
        let mut latt = [[0.0f64; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                latt[j][i] = self.data[[j, i]];
            }
        }

        latt
    }

    pub fn as_2d_array_row_major(&self) -> [[f64; 3]; 3] {
        let mut latt = [[0.0f64; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                latt[i][j] = self.data[[j, i]];
            }
        }

        latt
    }

    // ( a x b ) . c
    pub fn volume(&self) -> f64 {
        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        a.cross_product(&b).dot_product(&c)
    }

    // ra = 2 x PI x (b x c) / volume
    // rb = 2 x PI x (c x a) / volume
    // rc = 2 x PI x (a x b) / volume
    pub fn reciprocal(&self) -> Lattice {
        let factor = 2.0 * consts::PI / self.volume();

        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        let blatt_a = b.cross_product(&c) * factor;
        let blatt_b = c.cross_product(&a) * factor;
        let blatt_c = a.cross_product(&b) * factor;

        Lattice::new(&blatt_a.to_vec(), &blatt_b.to_vec(), &blatt_c.to_vec())
    }

    pub fn get_vector_a(&self) -> Vector3f64 {
        let v = self.data.get_col(0);

        Vector3f64 {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    pub fn get_vector_b(&self) -> Vector3f64 {
        let v = self.data.get_col(1);

        Vector3f64 {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    pub fn get_vector_c(&self) -> Vector3f64 {
        let v = self.data.get_col(2);

        Vector3f64 {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    pub fn scaled_by(&mut self, f: f64) {
        // for v in self.data.as_mut_slice().iter_mut() {
        //     *v *= f;
        // }

        self.data.as_mut_slice().iter_mut().for_each(|v| *v *= f);
    }

    pub fn frac_to_cart(&self, pos_f: &[f64], pos_c: &mut [f64]) {
        for i in 0..3 {
            pos_c[i] = 0.0;

            for j in 0..3 {
                pos_c[i] += self.data[[i, j]] * pos_f[j];
            }
        }
    }

    pub fn cart_to_frac(&self, pos_c: &[f64], pos_f: &mut [f64]) {
        let mut mat = self.data.clone();

        mat.inv();

        for i in 0..3 {
            pos_f[i] = 0.0;

            for j in 0..3 {
                pos_f[i] += mat[[i, j]] * pos_c[j];
            }
        }
    }

    /// Save the PWBasis to a HDF5 file.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        self.data.save_hdf5(group)
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        write!(f,
               "{}\n{:25.16}\t{:25.16}\t{:25.16}\n{:25.16}\t{:25.16}\t{:25.16}\n{:25.16}\t{:25.16}\t{:25.16}", "Lattice",
               a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
    }
}

#[test]
fn test_lattice() {
    let latt = Lattice::new(&[1.0, 0.1, 0.0], &[0.0, 1.0, 0.2], &[0.0, 0.3, 1.0]);

    println!("{}", latt);

    println!("volume = {}", latt.volume());

    let fact = 0.1;

    let mut latt_1 = latt.clone();
    latt_1.scaled_by(fact);

    println!("scaled by {}\n{}", fact, latt_1);
    println!("volume = {}", latt_1.volume());

    let blatt = latt.reciprocal();

    println!("blatt = \n{}", blatt);

    println!(
        "latt^T * blatt = \n{}",
        latt.as_matrix().transpose().dot(blatt.as_matrix())
    );

    let pos_f = Vector3f64 {
        x: 0.2,
        y: 0.3,
        z: 0.4,
    };

    let mut pos_c = Vector3f64::zeros();

    latt.frac_to_cart(pos_f.as_slice(), pos_c.as_mut_slice());

    println!("pos_f = {}", pos_f);
    println!("pos_c = {}", pos_c);

    let mut pos_f_2 = Vector3f64::zeros();

    latt.cart_to_frac(pos_c.as_slice(), pos_f_2.as_mut_slice());

    println!("pos_f_2 = {}", pos_f_2);
}
