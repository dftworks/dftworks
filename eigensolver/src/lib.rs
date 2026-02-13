#![allow(warnings)]

mod pcg;
use pcg::*;

use control::*;
use matrix::Matrix;

use types::c64;

pub trait EigenSolver {
    fn compute(
        &mut self, ham_on_psi: &mut dyn FnMut(&[c64], &mut [c64]), ham_diag: &[f64], evecs: &mut Matrix<c64>, evals: &mut [f64], occ: &[f64], tol_eigval: f64, max_cg_loop: usize,
        max_scf_iter: usize,
    ) -> (usize, usize);
}

pub fn new(solver_scheme: &str, n: usize, nev: usize) -> Box<dyn EigenSolver> {
    let sparse: Box<dyn EigenSolver>;

    match solver_scheme {
        "pcg" => {
            sparse = Box::new(EigenSolverPCG::new(n, nev));
        }

        &_ => {
            sparse = Box::new(EigenSolverPCG::new(n, nev));
        }
    }

    sparse
}
