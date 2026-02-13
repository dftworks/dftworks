//#![allow(warnings)]

use crate::EigenSolver;
use control::*;
use dwconsts::*;
use linalg;
use matrix::*;
use num_traits::identities::Zero;
use types::c64;
use utility;

pub struct EigenSolverPCG {
    npw: usize,
    nband: usize,

    // work space
    x0: Vec<c64>,
    h_x0: Vec<c64>,
    d0: Vec<c64>,
    h_d0: Vec<c64>,
    g0: Vec<c64>,
    g1: Vec<c64>,
    pg0: Vec<c64>,
    pg1: Vec<c64>,
    precond: Vec<c64>,
}

impl EigenSolverPCG {
    pub fn new(npw: usize, nband: usize) -> EigenSolverPCG {
        let mut x0 = Vec::with_capacity(npw);
        let mut h_x0 = Vec::with_capacity(npw);
        let mut d0 = Vec::with_capacity(npw);
        let mut h_d0 = Vec::with_capacity(npw);
        let mut g0 = Vec::with_capacity(npw);
        let mut g1 = Vec::with_capacity(npw);
        let mut pg0 = Vec::with_capacity(npw);
        let mut pg1 = Vec::with_capacity(npw);
        let mut precond = Vec::with_capacity(npw);

        x0.resize(npw, c64::zero());
        h_x0.resize(npw, c64::zero());
        d0.resize(npw, c64::zero());
        h_d0.resize(npw, c64::zero());
        g0.resize(npw, c64::zero());
        g1.resize(npw, c64::zero());
        pg0.resize(npw, c64::zero());
        pg1.resize(npw, c64::zero());
        precond.resize(npw, c64::zero());

        EigenSolverPCG { npw, nband, x0, h_x0, d0, h_d0, g0, g1, pg0, pg1, precond }
    }
}

impl EigenSolver for EigenSolverPCG {
    ///evecs: initial guess as input; store new eigenvectors

    fn compute(
        &mut self, ham_on_psi: &mut dyn FnMut(&[c64], &mut [c64]), ham_diag: &[f64], evecs: &mut Matrix<c64>, evals: &mut [f64], occ: &[f64], tol_eigval: f64, max_cg_loop: usize,
        max_scf_iter: usize,
    ) -> (usize, usize) {
        let mut n_hpsi = 0;
        let mut n_band_converged = 0;

        let mut omega0: f64 = 0.0;
        let mut omega: f64 = 0.0;

        for iband in 0..self.nband {
            // orthogonalize to lower bands and normalize

            if iband > 0 {
                gram_schmidt(evecs, iband);
            }

            // nscf or scf
            let tol2_eigval = if max_scf_iter <= 1 {
                tol_eigval
            } else {
                if occ[iband] > EPS3 {
                    tol_eigval
                } else {
                    (tol_eigval * 10.0).max(EPS5)
                }
            };

            //println!("tol2_eigval = {}", tol2_eigval);

            self.x0.copy_from_slice(evecs.get_col(iband));

            utility::normalize_vector_c64(&mut self.x0);

            //

            ham_on_psi(&self.x0, &mut self.h_x0);
            n_hpsi += 1;

            omega0 = utility::zdot_product(&self.x0, &self.h_x0).re;

            // calculate one state

            let mut bconverged = false;

            for cg_iter in 0..max_cg_loop {
                compute_preconditioner(&self.x0, ham_diag, &mut self.precond);

                // gradient

                for i in 0..self.npw {
                    self.g1[i] = self.h_x0[i] - omega0 * self.x0[i];
                }

                // orthogonalize g1 to all lower bands

                orthogonalize_to_lower_bands(evecs, iband, &mut self.g1);

                for i in 0..self.npw {
                    self.pg1[i] = self.g1[i] * self.precond[i];
                }

                // orthogonalize pg1 to all lower bands

                orthogonalize_to_lower_bands(evecs, iband, &mut self.pg1);

                utility::normalize_vector_c64(&mut self.pg1);

                let mut beta: c64;
                if cg_iter == 1 {
                    beta = c64::zero();
                } else {
                    let mut xx = c64::zero();

                    for i in 0..self.npw {
                        //Polak-Ribiere
                        xx += self.g1[i] * (self.pg1[i] - self.pg0[i]).conj();
                        //Fletcher-Reeves
                        //xx += self.g1[i].conj() * self.pg1[i];
                    }

                    beta = xx / utility::zdot_product(&self.pg0, &self.g0);

                    beta = c64 { re: beta.re.max(0.0), im: 0.0 };
                }

                for i in 0..self.npw {
                    self.d0[i] = -self.pg1[i] + beta * self.d0[i];
                }

                let proj = utility::zdot_product(&self.x0, &self.d0);

                utility::add_and_zscale(&self.x0, &mut self.d0, -proj); // d0 = d0 - proj*x0

                utility::normalize_vector_c64(&mut self.d0);

                ham_on_psi(&self.d0, &mut self.h_d0);
                n_hpsi += 1;

                let alpha = get_alpha(&self.x0, &self.d0, &self.h_x0, &self.h_d0);

                let t = (1.0 + alpha.norm_sqr()).sqrt();
                let cs = 1.0 / t;
                let sn = alpha / t;

                for i in 0..self.npw {
                    self.x0[i] = cs * self.x0[i] + sn * self.d0[i];
                }

                for i in 0..self.npw {
                    self.h_x0[i] = cs * self.h_x0[i] + sn * self.h_d0[i];
                }

                omega = (utility::zdot_product(&self.x0, &self.h_x0)).re;

                if (omega - omega0).abs() < tol2_eigval {
                    bconverged = true;

                    n_band_converged += 1;

                    break;
                }

                self.pg0.copy_from_slice(&self.pg1);
                self.g0.copy_from_slice(&self.g1);

                omega0 = omega;
            }

            evals[iband] = omega;
            evecs.set_col(iband, &self.x0);
        }

        (n_band_converged, n_hpsi)
    }
}

pub fn get_alpha(x0: &[c64], d0: &[c64], h_x0: &[c64], h_d0: &[c64]) -> c64 {
    let mut mh: Matrix<c64> = Matrix::new(2, 2);
    let mut mo: Matrix<c64> = Matrix::new(2, 2);

    mh[[0, 0]] = utility::zdot_product(x0, h_x0);
    mh[[0, 1]] = utility::zdot_product(x0, h_d0);
    mh[[1, 0]] = utility::zdot_product(d0, h_x0);
    mh[[1, 1]] = utility::zdot_product(d0, h_d0);

    mo[[0, 0]] = utility::zdot_product(x0, x0);
    mo[[0, 1]] = utility::zdot_product(x0, d0);
    mo[[1, 0]] = utility::zdot_product(d0, x0);
    mo[[1, 1]] = utility::zdot_product(d0, d0);

    //println!("mo = {}", mo);
    //println!("mh = {}", mh);
    mo.pinv(); // in situ inverse

    let oinvh = mo.dot(&mh);

    //println!("oinvh = {}", oinvh);

    let (_es, ev) = linalg::eigh(&oinvh);
    // println!("es = {}", es[0]);
    let ev0 = ev.get_col(0);

    let alpha = ev0[1] / ev0[0];

    //println!("alpha = {}", alpha);

    alpha
}

fn compute_preconditioner(psi: &[c64], kin: &[f64], kgg: &mut [c64]) {
    let mut ek = c64::zero();

    for (&psig, &ekg) in psi.iter().zip(kin.iter()) {
        ek += psig.norm_sqr() * ekg;
    }

    for (&ekg, k) in kin.iter().zip(kgg.iter_mut()) {
        let x = ekg / (ek * 1.5);

        let x2 = x * x;
        let x3 = x * x2;
        let x4 = x * x3;

        let y = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x3;

        *k = 1.0 * y / (y + 16.0 * x4);

        *k *= 2.0 / (1.5 * ek);
    }
}

fn gram_schmidt(evecs: &mut Matrix<c64>, iband: usize) {
    let npw = evecs.nrow();
    let mut v = vec![c64::zero(); npw];
    v.copy_from_slice(evecs.get_col(iband));

    // Pre-allocate projection coefficients
    let mut proj = Vec::with_capacity(iband);

    // Compute all projections first (Classical Gram-Schmidt)
    for j in 0..iband {
        let psi = evecs.get_col(j);
        proj.push(utility::zdot_product(psi, evecs.get_col(iband)));
    }

    // Apply all projections using SIMD-friendly operations
    for (j, &proj_coeff) in proj.iter().enumerate() {
        let psi = evecs.get_col(j);
        // Use iterator for better vectorization
        v.iter_mut().zip(psi.iter()).for_each(|(vi, &psi_i)| {
            *vi -= proj_coeff * psi_i;
        });
    }

    utility::normalize_vector_c64(&mut v);
    evecs.set_col(iband, &v);
}

fn orthogonalize_to_lower_bands(evecs: &Matrix<c64>, ibnd: usize, y: &mut [c64]) {
    // Pre-allocate projection coefficients with capacity
    let mut proj = Vec::with_capacity(ibnd);

    // Compute all projections first using iterator for better vectorization
    for i in 0..ibnd {
        let psi = evecs.get_col(i);
        proj.push(utility::zdot_product(psi, y));
    }

    // Apply all projections using SIMD-friendly operations
    for (i, &proj_coeff) in proj.iter().enumerate() {
        let psi = evecs.get_col(i);
        // Use iterator for better vectorization
        y.iter_mut().zip(psi.iter()).for_each(|(yi, &psi_i)| {
            *yi -= proj_coeff * psi_i;
        });
    }
}

#[test]
fn test_sparse_solver_pcg() {
    let n: usize = 100;
    let nev: usize = 20;

    let m = utility::make_matrix(n);

    let mut m_dot_v = |v: &[c64], vp: &mut [c64]| {
        let x = m.dot(&v.to_vec());
        for i in 0..x.len() {
            vp[i] = x[i];
        }
    };
    let mut evals = vec![0.0; nev];
    let mut evecs = Matrix::new(n, nev);

    let mut h_diag = vec![0.0; n];
    for i in 0..n {
        h_diag[i] = m[[i, i]].re;
    }

    let mut occ = vec![0.0; n];

    for i in 0..nev {
        occ[i] = 1.0;
    }

    for i in 0..nev {
        utility::make_normalized_rand_vector(evecs.get_mut_col(i));
    }

    let max_cg_loop = 100;
    let max_scf_iter = 1;

    let mut sparse = EigenSolverPCG::new(n, nev);
    let (nconv, niter) = sparse.compute(&mut m_dot_v, &h_diag, &mut evecs, &mut evals, &occ, EPS5, max_cg_loop, max_scf_iter);

    println!("eigenvalues converged = {}, niter = {}", nconv, niter);

    let (nconv, niter) = sparse.compute(&mut m_dot_v, &h_diag, &mut evecs, &mut evals, &occ, EPS10, max_cg_loop, max_scf_iter);

    println!("eigenvalues converged = {}, niter = {}", nconv, niter);

    let (es, ev) = linalg::eigh(&m);

    println!(" pcg \t\t eigh");

    for i in 0..nev {
        println!("{:20.16} {:20.16}", evals[i], es[i]);
    }
}
