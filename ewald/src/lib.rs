//#![allow(warnings)]

use crystal::*;
use dwconsts::*;
use gvector::*;
use matrix::*;
use num_traits::identities::Zero;
use pwdensity::*;
use special;
use types::*;
use utility;
use vector3::*;

// Ewald ion-ion interaction decomposition for periodic crystals.
//
// Splits long-range Coulomb sum into:
// - short-range real-space part (erfc-screened)
// - reciprocal-space part (Gaussian-damped)
// - analytic G=0/self/background correction
//
// The same split is applied consistently for energy, forces, and stress.
pub struct Ewald {
    energy: f64,
    force: Vec<Vector3f64>,
    stress: Matrix<f64>,
}

impl Ewald {
    pub fn new(crystal: &Crystal, zions: &[f64], gvec: &GVector, pwden: &PWDensity) -> Ewald {
        // Choose splitting parameter from reciprocal cutoff target.
        let eta = get_eta_based_on_gcut(pwden, 1E-30);

        // Real-space cutoff implied by eta and tolerance.
        let rmax = get_rmax_based_on_eta(eta, 1E-30);

        let nn_cells = make_near_cells(crystal, rmax);

        let natoms = crystal.get_n_atoms();

        // Energy decomposition.

        let energy_r = compute_energy_real_space_part(crystal, zions, eta, &nn_cells);
        let energy_g = compute_energy_g_space_part(crystal, zions, gvec, pwden, eta);
        let energy_g0 = compute_energy_g0_part(crystal, zions, eta);

        let energy = energy_r + energy_g + energy_g0;

        // Force decomposition (real + reciprocal).

        let mut force = vec![Vector3f64::zeros(); natoms];

        let force_r = compute_force_real_space_part(crystal, zions, eta, &nn_cells);
        let force_g = compute_force_g_space_part(crystal, zions, gvec, pwden, eta);

        for i in 0..natoms {
            force[i].x = force_r[i].x + force_g[i].x;
            force[i].y = force_r[i].y + force_g[i].y;
            force[i].z = force_r[i].z + force_g[i].z;
        }

        // Stress decomposition (real + reciprocal).

        let mut stress = Matrix::<f64>::new(3, 3);
        let stress_r = compute_stress_real_space_part(crystal, zions, eta, &nn_cells);
        let stress_g = compute_stress_g_space_part(crystal, zions, gvec, pwden, eta);

        let volume = crystal.get_latt().volume();
        for i in 0..3 {
            for j in 0..3 {
                stress[[i, j]] = -1.0 * (stress_r[[i, j]] + stress_g[[i, j]]) / volume;
            }
        }

        Ewald {
            energy,
            force,
            stress,
        }
    }

    pub fn get_energy(&self) -> f64 {
        self.energy
    }

    pub fn get_force(&self) -> &[Vector3f64] {
        &self.force
    }

    pub fn get_stress(&self) -> &Matrix<f64> {
        &self.stress
    }
}

fn make_near_cells(crystal: &Crystal, rmax: f64) -> Vec<Vector3i32> {
    let latt = crystal.get_latt();

    let a = latt.get_vector_a();
    let b = latt.get_vector_b();
    let c = latt.get_vector_c();

    let na = (rmax / a.norm2()).ceil() as i32 + 2;
    let nb = (rmax / b.norm2()).ceil() as i32 + 2;
    let nc = (rmax / c.norm2()).ceil() as i32 + 2;

    // Candidate translation vectors and their squared lengths.
    let mut t_rs: Vec<Vector3i32> = Vec::new();
    let mut t_r2: Vec<f64> = Vec::new();

    for ia in -na..na {
        for ib in -nb..nb {
            for ic in -nc..nc {
                let x = a.x * ia as f64 + b.x * ib as f64 + c.x * ic as f64;
                let y = a.y * ia as f64 + b.y * ib as f64 + c.y * ic as f64;
                let z = a.z * ia as f64 + b.z * ib as f64 + c.z * ic as f64;

                let r2 = x * x + y * y + z * z;

                if r2 < rmax * rmax {
                    t_r2.push(r2);
                    t_rs.push(Vector3i32 {
                        x: ia,
                        y: ib,
                        z: ic,
                    });
                }
            }
        }
    }

    // Sort neighbors by distance to improve convergence behavior in loops.
    let ordered_index = utility::argsort(&t_r2);

    let mut rs: Vec<Vector3i32> = vec![Vector3i32 { x: 0, y: 0, z: 0 }; t_rs.len()];

    for (i, &j) in ordered_index.iter().enumerate() {
        rs[i] = t_rs[j];
    }

    rs
}

// 4pi/G^2*exp(-G^2/4/eta) = eps

fn get_eta_based_on_gcut(pwden: &PWDensity, eps: f64) -> f64 {
    // Solve damping relation at gmax to pick eta.
    let gmax = pwden.get_gmax();

    let g2 = gmax * gmax;

    -0.25 * g2 / (eps * g2 / FOURPI).ln()
}

fn get_rmax_based_on_eta(eta: f64, eps: f64) -> f64 {
    // Increase rmax until erfc(sqrt(eta) r) tail is below tolerance.
    let mut rmax = 0.0;

    while special::erfc(rmax * eta.sqrt()) > rmax * eps {
        rmax += 0.1;
    }

    rmax
}

fn compute_energy_real_space_part(
    crystal: &Crystal,
    zions: &[f64],
    eta: f64,
    nn_cells: &[Vector3i32],
) -> f64 {
    let latt = crystal.get_latt();

    let a = latt.get_vector_a();
    let b = latt.get_vector_b();
    let c = latt.get_vector_c();

    let natoms = crystal.get_n_atoms();

    let atoms = crystal.get_atom_positions();

    let eta_sqrt = eta.sqrt();

    // Real-space screened pair sum.
    let mut sum = 0.0;

    for cell in nn_cells.iter() {
        for i in 0..natoms {
            let ati = &atoms[i];

            for j in 0..natoms {
                if j == i {
                    continue;
                }

                let atj = &atoms[j];

                // d_i - d_j - R, crystal coordinates

                let fa = ati.x - atj.x - cell.x as f64;
                let fb = ati.y - atj.y - cell.y as f64;
                let fc = ati.z - atj.z - cell.z as f64;

                // get cartesion coordinates

                let dx = a.x * fa + b.x * fb + c.x * fc;
                let dy = a.y * fa + b.y * fb + c.y * fc;
                let dz = a.z * fa + b.z * fb + c.z * fc;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                sum += 0.5 * zions[i] * zions[j] / r * special::erfc(eta_sqrt * r);
            }
        }
    }

    sum
}

fn compute_energy_g_space_part(
    crystal: &Crystal,
    zions: &[f64],
    gvec: &GVector,
    pwden: &PWDensity,
    eta: f64,
) -> f64 {
    let npw = pwden.get_n_plane_waves();

    let g = pwden.get_g();

    let gidx = pwden.get_gindex();

    let miller = gvec.get_miller();

    let atoms = crystal.get_atom_positions();

    // Reciprocal-space structure-factor sum (G != 0).
    let mut sum = 0.0;

    for i in 1..npw {
        let mill = miller[gidx[i]];
        let g2 = g[i] * g[i];

        let mut s = c64::zero();

        for (iat, atom) in atoms.iter().enumerate() {
            let gd =
                TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);
            s += zions[iat]
                * c64 {
                    re: gd.cos(),
                    im: gd.sin(),
                };
        }

        sum += s.norm_sqr() * (-g2 / 4.0 / eta).exp() / g2;
    }

    let volume = crystal.get_latt().volume();

    sum * FOURPI / 2.0 / volume
}

fn compute_energy_g0_part(crystal: &Crystal, zions: &[f64], eta: f64) -> f64 {
    // Includes self term and compensating background/G=0 contribution.
    let volume = crystal.get_latt().volume();

    let s: f64 = zions.iter().sum();
    let s2: f64 = zions.iter().map(|x| x * x).sum();

    -(eta / PI).sqrt() * s2 - 0.5 * s * s * FOURPI / volume / 4.0 / eta
}

fn compute_force_g_space_part(
    crystal: &Crystal,
    zions: &[f64],
    gvec: &GVector,
    pwden: &PWDensity,
    eta: f64,
) -> Vec<Vector3f64> {
    let npw = pwden.get_n_plane_waves();

    let g = pwden.get_g();

    let gidx = pwden.get_gindex();

    let miller = gvec.get_miller();

    let atoms = crystal.get_atom_positions();

    let cart = gvec.get_cart();

    let natoms = crystal.get_n_atoms();

    // Reciprocal-space ionic forces.
    let mut force = vec![Vector3f64::zeros(); natoms];

    for iat in 0..natoms {
        let mut v = Vector3c64::zeros();

        for i in 1..npw {
            let mill = miller[gidx[i]];

            let g2 = g[i] * g[i];

            // Structure factor s = sum_beta Z_beta exp(i G·d_beta).

            let mut s = c64::zero();

            for (jat, atom) in atoms.iter().enumerate() {
                let gd = TWOPI
                    * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);
                s += zions[jat]
                    * c64 {
                        re: gd.cos(),
                        im: gd.sin(),
                    };
            }

            //

            let atom = &atoms[iat];

            let ngd =
                -TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let mut comm = s
                * zions[iat]
                * c64 {
                    re: ngd.cos(),
                    im: ngd.sin(),
                };

            comm *= (-g2 / 4.0 / eta).exp() / g2;

            //

            let gcoord = cart[gidx[i]];

            v.x += I_C64 * gcoord.x * comm;
            v.y += I_C64 * gcoord.y * comm;
            v.z += I_C64 * gcoord.z * comm;
        }

        let volume = crystal.get_latt().volume();

        v.x *= FOURPI / 2.0 / volume;
        v.y *= FOURPI / 2.0 / volume;
        v.z *= FOURPI / 2.0 / volume;

        force[iat].x += 2.0 * v.x.re;
        force[iat].y += 2.0 * v.y.re;
        force[iat].z += 2.0 * v.z.re;
    }

    force
}

fn compute_force_real_space_part(
    crystal: &Crystal,
    zions: &[f64],
    eta: f64,
    nn_cells: &[Vector3i32],
) -> Vec<Vector3f64> {
    let latt = crystal.get_latt();

    let a = latt.get_vector_a();
    let b = latt.get_vector_b();
    let c = latt.get_vector_c();

    let natoms = crystal.get_n_atoms();

    let atoms = crystal.get_atom_positions();

    let eta_sqrt = eta.sqrt();

    // Real-space screened pair-force sum.
    let mut force = vec![Vector3f64::zeros(); natoms];

    for i in 0..natoms {
        let ati = &atoms[i];

        let mut v = Vector3f64::zeros();

        for cell in nn_cells.iter() {
            for j in 0..natoms {
                if j == i {
                    continue;
                }

                let atj = &atoms[j];

                // d_i - d_j - R, crystal coordinates

                let fa = ati.x - atj.x - cell.x as f64;
                let fb = ati.y - atj.y - cell.y as f64;
                let fc = ati.z - atj.z - cell.z as f64;

                // get cartesion coordinates

                let dx = a.x * fa + b.x * fb + c.x * fc;
                let dy = a.y * fa + b.y * fb + c.y * fc;
                let dz = a.z * fa + b.z * fb + c.z * fc;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let mut comm = 0.5 * zions[i] * zions[j] / r / r;
                comm *= 2.0 / PI.sqrt() * (-eta * r * r).exp() * eta_sqrt
                    + special::erfc(eta_sqrt * r) / r;

                v.x += comm * dx;
                v.y += comm * dy;
                v.z += comm * dz;
            } // beta
        } // R
        force[i].x += v.x;
        force[i].y += v.y;
        force[i].z += v.z;
    } // alpha

    force
}

pub fn compute_stress_real_space_part(
    crystal: &Crystal,
    zions: &[f64],
    eta: f64,
    nn_cells: &[Vector3i32],
) -> Matrix<f64> {
    let latt = crystal.get_latt();

    let a = latt.get_vector_a();
    let b = latt.get_vector_b();
    let c = latt.get_vector_c();

    let natoms = crystal.get_n_atoms();

    let atoms = crystal.get_atom_positions();

    let eta_sqrt = eta.sqrt();

    // Real-space screened stress tensor contribution.
    let mut stress = Matrix::<f64>::new(3, 3);

    for cell in nn_cells.iter() {
        for i in 0..natoms {
            let ati = &atoms[i];

            for j in 0..natoms {
                if j == i {
                    continue;
                }

                let atj = &atoms[j];

                // d_i - d_j - R, crystal coordinates

                let fa = ati.x - atj.x - cell.x as f64;
                let fb = ati.y - atj.y - cell.y as f64;
                let fc = ati.z - atj.z - cell.z as f64;

                // get cartesion coordinates

                let dx = a.x * fa + b.x * fb + c.x * fc;
                let dy = a.y * fa + b.y * fb + c.y * fc;
                let dz = a.z * fa + b.z * fb + c.z * fc;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let vecv = [dx, dy, dz];

                for ii in 0..3 {
                    for jj in 0..3 {
                        stress[[jj, ii]] += 0.5
                            * eta_sqrt
                            * zions[i]
                            * zions[j]
                            * (-2.0 / PI.sqrt() * (-eta * r * r).exp()
                                - special::erfc(eta_sqrt * r) / r)
                            * vecv[ii]
                            * vecv[jj]
                            / r
                            / r;
                    }
                }
            }
        }
    }

    stress
}

pub fn compute_stress_g_space_part(
    crystal: &Crystal,
    zions: &[f64],
    gvec: &GVector,
    pwden: &PWDensity,
    eta: f64,
) -> Matrix<f64> {
    let npw = pwden.get_n_plane_waves();

    let volume = crystal.get_latt().volume();

    let g = pwden.get_g();

    let gidx = pwden.get_gindex();

    let cart = gvec.get_cart();

    let miller = gvec.get_miller();

    let atoms = crystal.get_atom_positions();

    let unit_mat = Matrix::<f64>::unit(3);

    // Reciprocal-space stress tensor contribution.
    let mut stress = Matrix::<f64>::new(3, 3);

    for ipw in 1..npw {
        let mill = miller[gidx[ipw]];
        let g2 = g[ipw] * g[ipw];

        let mut s = c64::zero();

        for (iat, atom) in atoms.iter().enumerate() {
            let gd =
                TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);
            s += zions[iat]
                * c64 {
                    re: gd.cos(),
                    im: gd.sin(),
                };
        }

        let comm = s.norm_sqr() * (-g2 / 4.0 / eta).exp() / g2 * FOURPI / volume / 2.0;

        let gcoord = cart[gidx[ipw]].as_slice();

        for i in 0..3 {
            for j in 0..3 {
                stress[[j, i]] += comm
                    * (2.0 / g2 * gcoord[i] * gcoord[j] * (g2 / 4.0 / eta + 1.0)
                        - unit_mat[[j, i]]);
            }
        }
    }

    let mut s = f64::zero();

    for (iat, _atom) in atoms.iter().enumerate() {
        s += zions[iat];
    }

    for i in 0..3 {
        stress[[i, i]] += s * s * PI / 2.0 / volume / eta;
    }

    stress
}
