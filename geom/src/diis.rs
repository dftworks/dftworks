#![allow(warnings)]
use crystal::Crystal;
use lattice::Lattice;
use nalgebra::Matrix3;
use types::*;

use optimization::OptimizationDriver;

pub struct DIIS {
    optim: Box<dyn OptimizationDriver>,

    alpha: f64,
    nstep: usize,

    iter: usize,
}

use crate::GeomOptimizationDriver;

impl DIIS {
    pub fn new(alpha: f64, nstep: usize) -> DIIS {
        DIIS {
            optim: optimization::new("diis", alpha, nstep),
            alpha,
            nstep,
            iter: 0,
        }
    }
}

impl GeomOptimizationDriver for DIIS {
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
        // if self.iter == self.nstep {
        //     self.optim = optimization::new("diis", self.alpha, self.nstep);

        //     self.iter = 0;
        // }

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

fn matrix_to_matrix3(m: &Matrix<f64>) -> Matrix3<f64> {
    Matrix3::<f64>::from_column_slice(m.as_slice())
}

fn matrix3_to_lattice(m: &Matrix3<f64>) -> Lattice {
    let a = [m[(0, 0)], m[(1, 0)], m[(2, 0)]];
    let b = [m[(0, 1)], m[(1, 1)], m[(2, 1)]];
    let c = [m[(0, 2)], m[(1, 2)], m[(2, 2)]];
    Lattice::new(&a, &b, &c)
}

fn lattice_to_matrix3(latt: &Lattice) -> Matrix3<f64> {
    Matrix3::<f64>::from_column_slice(latt.as_matrix().as_slice())
}

fn move_cell_and_ions(
    driver: &mut DIIS,
    crystal: &mut Crystal,
    force: &[Vector3f64],
    stress: &Matrix<f64>,
    force_mask: &[Vector3f64],
    stress_mask: &Matrix<f64>,
    latt0: &Lattice,
) {
    let mut gcoord = compute_generalized_coordinates(crystal, latt0);

    let gforce = compute_generalized_forces(crystal, latt0, force, stress);

    // mask

    let natoms = crystal.get_n_atoms();

    let mut gmask = vec![
        Vector3f64::new(1.0, 1.0, 1.0);
        3 + natoms
    ];

    for i in 0..3 {
        gmask[i].x = stress_mask[(0, i)];
        gmask[i].y = stress_mask[(1, i)];
        gmask[i].z = stress_mask[(2, i)];
    }

    for i in 0..natoms {
        gmask[3 + i] = force_mask[i];
    }

    let mut vin = flatten_vec3(&gcoord);
    let vout = flatten_vec3(&gforce);
    let vmask = flatten_vec3(&gmask);

    //println!("vin  = {:?}", vin);
    //println!("vout = {:?}", vout);
    //println!("mask = {:?}", vmask);

    driver.optim.compute_next_input(vin.as_mut_slice(), vout.as_slice(), vmask.as_slice());
    scatter_vec3(&vin, &mut gcoord);

    // set new lattice

    let latt = Lattice::new(&vin[0..3], &vin[3..6], &vin[6..9]);

    crystal.set_lattice_vectors(&latt);

    // set new atom coordinates

    crystal.set_atom_positions_from_frac(&gcoord[3..]);
}

fn move_ions(
    driver: &mut DIIS,
    crystal: &mut Crystal,
    force: &[Vector3f64],
    force_mask: &[Vector3f64],
) {
    let natoms = crystal.get_n_atoms();

    let mut gforce = vec![Vector3f64::zeros(); natoms];

    force::cart_to_frac(crystal.get_latt(), force, &mut gforce);

    gforce.iter_mut().for_each(|f| {
        f.x *= -1.0;
        f.y *= -1.0;
        f.z *= -1.0;
    });

    //println!("force = {:?}", gforce);

    let mut atoms_frac = crystal.get_atom_positions().to_vec();

    let mut vin = flatten_vec3(&atoms_frac);
    let vout = flatten_vec3(&gforce);
    let mask = flatten_vec3(force_mask);

    driver.optim.compute_next_input(vin.as_mut_slice(), vout.as_slice(), mask.as_slice());
    scatter_vec3(&vin, &mut atoms_frac);

    crystal.set_atom_positions_from_frac(&atoms_frac);
}

fn set_lattice_vectors_with_strain(crystal: &mut Crystal, latt0: &Lattice, strain: &Matrix3<f64>) {
    let mut factor = strain.clone();
    for i in 0..3 {
        factor[(i, i)] += 1.0;
    }
    let mlatt = factor * lattice_to_matrix3(latt0);
    let new_latt = matrix3_to_lattice(&mlatt);
    crystal.set_lattice_vectors(&new_latt);
}

fn compute_generalized_coordinates(crystal: &Crystal, _latt0: &Lattice) -> Vec<Vector3f64> {
    let natoms = crystal.get_n_atoms();

    let mut gcoord = vec![Vector3f64::zeros(); 3 + natoms];

    // cell : lattice vectors
    let latt = crystal.get_latt().as_matrix();
    for i in 0..3 {
        gcoord[i].x = latt[(0, i)];
        gcoord[i].y = latt[(1, i)];
        gcoord[i].z = latt[(2, i)];
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
    _latt0: &Lattice,
    force: &[Vector3f64],
    stress: &Matrix<f64>,
) -> Vec<Vector3f64> {
    let natoms = crystal.get_n_atoms();

    let mut gforce = vec![Vector3f64::zeros(); 3 + natoms];

    // cell : stress
    let volume = crystal.get_latt().volume();
    let latt_inv = lattice_to_matrix3(crystal.get_latt())
        .try_inverse()
        .expect("lattice matrix is singular in DIIS generalized forces");
    let stress_mat = matrix_to_matrix3(stress);
    let cforce = (stress_mat * latt_inv.transpose()) * volume;

    let factor = -1.0;

    // transform stress to forces on lattice vectors

    for i in 0..3 {
        gforce[i].x = cforce[(0, i)] * factor;
        gforce[i].y = cforce[(1, i)] * factor;
        gforce[i].z = cforce[(2, i)] * factor;
    }

    // atoms : force

    force::cart_to_frac(crystal.get_latt(), force, &mut gforce[3..]);

    let factor = -1.0; // / volume.powf(2.0 / 3.0);

    for i in 0..natoms {
        gforce[3 + i].x *= factor;
        gforce[3 + i].y *= factor;
        gforce[3 + i].z *= factor;
    }

    gforce
}
