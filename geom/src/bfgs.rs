#![allow(warnings)]
use crystal::Crystal;
use lattice::Lattice;
use matrix::*;
use types::*;

use optimization::OptimizationDriver;

pub struct BFGS {
    optim: Box<dyn OptimizationDriver>,

    alpha: f64,
    nstep: usize,

    iter: usize,
}

use crate::GeomOptimizationDriver;

impl BFGS {
    pub fn new(alpha: f64, nstep: usize) -> BFGS {
        BFGS {
            optim: optimization::new("bfgs", alpha, nstep),
            alpha,
            nstep,
            iter: 0,
        }
    }
}

impl GeomOptimizationDriver for BFGS {
    fn compute_next_configuration(
        &mut self,
        crystal: &mut Crystal,
        force: &[Vector3f64],
        stress: &Matrix<f64>,
        force_mask: &[Vector3f64],
        stress_mask: &Matrix<f64>,
        latt0: &Lattice,
        bcell_move: bool,
    ) {
        if self.iter == self.nstep {
            self.optim = optimization::new("bfgs", self.alpha, self.nstep);

            self.iter = 0;
        }

        if bcell_move {
            move_cell_and_ions(self, crystal, force, stress, force_mask, stress_mask, latt0);
        } else {
            move_ions(self, crystal, force, force_mask);
        }

        self.iter += 1;
    }
}

fn flatten_vec3(v: &[Vector3f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for item in v {
        out.extend_from_slice(item.as_slice());
    }
    out
}

fn scatter_vec3(flat: &[f64], out: &mut [Vector3f64]) {
    debug_assert_eq!(flat.len(), out.len() * 3);
    for (dst, chunk) in out.iter_mut().zip(flat.chunks_exact(3)) {
        dst.x = chunk[0];
        dst.y = chunk[1];
        dst.z = chunk[2];
    }
}

fn move_cell_and_ions(
    driver: &mut BFGS,
    crystal: &mut Crystal,
    force: &[Vector3f64],
    stress: &Matrix<f64>,
    force_mask: &[Vector3f64],
    stress_mask: &Matrix<f64>,
    latt0: &Lattice,
) {
    let volume = crystal.get_latt().volume();
    let precon = volume.powf(1.0 / 3.0);

    let mut gcoord = compute_generalized_coordinates(crystal, latt0);

    let gforce = compute_generalized_forces(crystal, latt0, force, stress);

    // mask

    let natoms = crystal.get_n_atoms();

    let mut gmask = vec![
        Vector3f64::new(1.0, 1.0, 1.0);
        3 + natoms
    ];

    for i in 0..3 {
        let t = stress_mask.get_col(i);

        gmask[i].x = t[0];
        gmask[i].y = t[1];
        gmask[i].z = t[2];
    }

    for i in 0..natoms {
        gmask[3 + i] = force_mask[i];
    }

    let mut vin = flatten_vec3(&gcoord);
    let vout = flatten_vec3(&gforce);

    println!("vin  = {:?}", vin);
    println!("vout = {:?}", vout);

    let vmask = flatten_vec3(&gmask);

    driver.optim.compute_next_input(vin.as_mut_slice(), vout.as_slice(), vmask.as_slice());
    scatter_vec3(&vin, &mut gcoord);

    // set new lattice

    let mut epsilon = Matrix::<f64>::new(3, 3);

    for i in 0..3 {
        epsilon[[0, i]] = gcoord[i].x;
        epsilon[[1, i]] = gcoord[i].y;
        epsilon[[2, i]] = gcoord[i].z;
    }

    println!("epsilon before symmetrization = \n {}", epsilon);

    for i in 0..3 {
        for j in i + 1..3 {
            epsilon[[i, j]] = (epsilon[[i, j]] + epsilon[[j, i]]) / 2.0;
            epsilon[[j, i]] = epsilon[[i, j]];
        }
    }

    println!("epsilon after symmetrization = \n {}", epsilon);

    set_lattice_vectors_with_strain(crystal, latt0, &epsilon);

    // set new atom coordinates

    crystal.set_atom_positions_from_frac(&gcoord[3..]);
}

fn move_ions(
    driver: &mut BFGS,
    crystal: &mut Crystal,
    force: &[Vector3f64],
    force_mask: &[Vector3f64],
) {
    let natoms = crystal.get_n_atoms();

    let mut gforce = vec![Vector3f64::zeros(); natoms];

    force::cart_to_frac(crystal.get_latt(), force, &mut gforce);

    for f in gforce.iter_mut() {
        f.x *= -1.0;
        f.y *= -1.0;
        f.z *= -1.0;
    }

    println!("force = {:?}", gforce);

    let mut atoms_frac = crystal.get_atom_positions().to_vec();

    let mut vin = flatten_vec3(&atoms_frac);
    let vout = flatten_vec3(&gforce);
    let mask = flatten_vec3(force_mask);

    driver.optim.compute_next_input(vin.as_mut_slice(), vout.as_slice(), mask.as_slice());
    scatter_vec3(&vin, &mut atoms_frac);

    crystal.set_atom_positions_from_frac(&atoms_frac);
}

fn set_lattice_vectors_with_strain(crystal: &mut Crystal, latt0: &Lattice, strain: &Matrix<f64>) {
    let mut factor = strain.clone();

    for i in 0..3 {
        factor[[i, i]] += 1.0;
    }

    let mlatt = factor.dot(latt0.as_matrix());

    let new_latt = Lattice::new(mlatt.get_col(0), mlatt.get_col(1), mlatt.get_col(2));

    crystal.set_lattice_vectors(&new_latt);
}

fn compute_generalized_coordinates(crystal: &Crystal, latt0: &Lattice) -> Vec<Vector3f64> {
    let natoms = crystal.get_n_atoms();

    let mut gcoord = vec![Vector3f64::zeros(); 3 + natoms];

    // cell : strain tensor

    let mut latt0_inv = latt0.as_matrix().clone();

    latt0_inv.inv();

    let mut epsilon = crystal.get_latt().as_matrix().dot(&latt0_inv);

    for i in 0..3 {
        epsilon[[i, i]] -= 1.0;
    }

    for i in 0..3 {
        gcoord[i].x = epsilon[[0, i]];
        gcoord[i].y = epsilon[[1, i]];
        gcoord[i].z = epsilon[[2, i]];
    }

    // atoms : fractional coordinates

    let atoms_frac = crystal.get_atom_positions().to_vec();

    for i in 0..natoms {
        gcoord[3 + i] = atoms_frac[i];
    }

    gcoord
}

fn compute_generalized_forces(
    crystal: &Crystal,
    latt0: &Lattice,
    force: &[Vector3f64],
    stress: &Matrix<f64>,
) -> Vec<Vector3f64> {
    let natoms = crystal.get_n_atoms();

    let mut gforce = vec![Vector3f64::zeros(); 3 + natoms];

    let volume = crystal.get_latt().volume();

    // cell : stress

    let mut latt0_inv = latt0.as_matrix().clone();

    latt0_inv.inv();

    let mut epsilon_plus_inv = crystal.get_latt().as_matrix().dot(&latt0_inv);
    epsilon_plus_inv.inv();

    let gstress = stress.dot(&epsilon_plus_inv);

    let factor = -1.0;

    // use stress directly

    for i in 0..3 {
        let t = stress.get_col(i);

        gforce[i].x = t[0] * factor;
        gforce[i].y = t[1] * factor;
        gforce[i].z = t[2] * factor;
    }

    // atoms : force

    force::cart_to_frac(crystal.get_latt(), force, &mut gforce[3..]);

    let g = crystal.get_latt().get_metric_tensor();

    let volume = crystal.get_latt().volume();

    let factor = -1.0; // / volume.powf(2.0 / 3.0);

    for i in 0..natoms {
        //let v = g.dot(&gforce[3 + i].to_vec());

        let v = gforce[3 + i].as_slice().to_vec();

        gforce[3 + i].x = v[0] * factor;
        gforce[3 + i].y = v[1] * factor;
        gforce[3 + i].z = v[2] * factor;
    }

    gforce
}
