use types::*;
use nalgebra::Matrix3;
use types::*;

use std::{f64::consts, fmt};

// 3x3 lattice container (column vectors a,b,c).
//
// Conventions:
// - internal matrix stores lattice vectors by columns
// - direct and reciprocal transforms follow physics convention:
//   b_i · a_j = 2*pi*delta_ij
#[derive(Debug, Default, Clone)]
pub struct Lattice {
    data: Matrix<f64>,
}

impl Lattice {
    pub fn new(a: &[f64], b: &[f64], c: &[f64]) -> Lattice {
        // Build lattice with columns (a,b,c).
        let mut data = Matrix::<f64>::new(3, 3);

        data.set_col(0, a);
        data.set_col(1, b);
        data.set_col(2, c);

        Lattice { data }
    }

    pub fn get_metric_tensor(&self) -> Matrix<f64> {
        // Metric tensor G_ij = a_i · a_j.
        let mut g = Matrix::<f64>::new(3, 3);

        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        g[[0, 0]] = a.dot(&a);
        g[[0, 1]] = a.dot(&b);
        g[[0, 2]] = a.dot(&c);

        g[[1, 0]] = b.dot(&a);
        g[[1, 1]] = b.dot(&b);
        g[[1, 2]] = b.dot(&c);

        g[[2, 0]] = c.dot(&a);
        g[[2, 1]] = c.dot(&b);
        g[[2, 2]] = c.dot(&c);

        g
    }

    pub fn as_matrix(&self) -> &Matrix<f64> {
        &self.data
    }

    pub fn as_mut_matrix(&mut self) -> &mut Matrix<f64> {
        &mut self.data
    }

    pub fn as_2d_array_col_major(&self) -> [[f64; 3]; 3] {
        // Returns [vector_index][component].
        let mut latt = [[0.0f64; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                latt[j][i] = self.data[[j, i]];
            }
        }

        latt
    }

    pub fn as_2d_array_row_major(&self) -> [[f64; 3]; 3] {
        // Returns [component][vector_index].
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
        // Signed cell volume.
        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        a.cross(&b).dot(&c)
    }

    // ra = 2 x PI x (b x c) / volume
    // rb = 2 x PI x (c x a) / volume
    // rc = 2 x PI x (a x b) / volume
    pub fn reciprocal(&self) -> Lattice {
        // Reciprocal lattice vectors satisfying b_i · a_j = 2*pi*delta_ij.
        let factor = 2.0 * consts::PI / self.volume();

        let a = self.get_vector_a();
        let b = self.get_vector_b();
        let c = self.get_vector_c();

        let blatt_a = b.cross(&c) * factor;
        let blatt_b = c.cross(&a) * factor;
        let blatt_c = a.cross(&b) * factor;

        Lattice::new(
            &blatt_a.as_slice().to_vec(),
            &blatt_b.as_slice().to_vec(),
            &blatt_c.as_slice().to_vec(),
        )
    }

    pub fn get_vector_a(&self) -> Vector3f64 {
        let v = self.data.get_col(0);

        Vector3f64::new(v[0], v[1], v[2])
    }

    pub fn get_vector_b(&self) -> Vector3f64 {
        let v = self.data.get_col(1);

        Vector3f64::new(v[0], v[1], v[2])
    }

    pub fn get_vector_c(&self) -> Vector3f64 {
        let v = self.data.get_col(2);

        Vector3f64::new(v[0], v[1], v[2])
    }

    pub fn scaled_by(&mut self, f: f64) {
        // Uniformly scales all lattice vectors by factor f.
        self.data.as_mut_slice().iter_mut().for_each(|v| *v *= f);
    }

    pub fn frac_to_cart(&self, pos_f: &[f64], pos_c: &mut [f64]) {
        // Cartesian position = lattice_matrix * fractional_position.
        let latt = Matrix3::<f64>::from_column_slice(self.data.as_slice());
        let pos_frac = Vector3f64::new(pos_f[0], pos_f[1], pos_f[2]);
        let pos_cart = latt * pos_frac;
        pos_c[..3].copy_from_slice(pos_cart.as_slice());
    }

    pub fn cart_to_frac(&self, pos_c: &[f64], pos_f: &mut [f64]) {
        // Fractional position = inverse(lattice_matrix) * Cartesian position.
        let latt = Matrix3::<f64>::from_column_slice(self.data.as_slice());
        let latt_inv = latt
            .try_inverse()
            .expect("lattice matrix is singular in cart_to_frac");
        let pos_cart = Vector3f64::new(pos_c[0], pos_c[1], pos_c[2]);
        let pos_frac = latt_inv * pos_cart;
        pos_f[..3].copy_from_slice(pos_frac.as_slice());
    }

    /// Save the lattice to a HDF5 file.
    pub fn save_hdf5(&self, group: &mut hdf5::Group) {
        self.data.save_hdf5(group);
    }

    /// Load the lattice from a HDF5 file.
    pub fn load_hdf5(group: &hdf5::Group) -> Self {
        Self::try_load_hdf5(group).expect("failed to load Lattice from HDF5")
    }

    /// Fallible HDF5 loader used by restart/checkpoint paths.
    pub fn try_load_hdf5(group: &hdf5::Group) -> Result<Self, String> {
        let data = Matrix::<f64>::try_load_hdf5(group)?;
        if data.nrow() != 3 || data.ncol() != 3 {
            return Err(format!(
                "invalid lattice matrix shape: expected 3x3, got {}x{}",
                data.nrow(),
                data.ncol()
            ));
        }
        Ok(Lattice { data })
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

    let pos_f = Vector3f64::new(0.2, 0.3, 0.4);

    let mut pos_c = Vector3f64::zeros();

    latt.frac_to_cart(pos_f.as_slice(), pos_c.as_mut_slice());

    println!("pos_f = {}", pos_f);
    println!("pos_c = {}", pos_c);

    let mut pos_f_2 = Vector3f64::zeros();

    latt.cart_to_frac(pos_c.as_slice(), pos_f_2.as_mut_slice());

    println!("pos_f_2 = {}", pos_f_2);
}
