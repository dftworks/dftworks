#![allow(warnings)]

mod pcg;
use pcg::*;

mod lobpcg;
use lobpcg::*;

use control::*;
use types::Matrix;

use types::c64;

pub trait EigenSolver {
    /// Improves the eigenpairs in `evecs`/`evals` (initial guess on input).
    /// A band counts as converged when its eigenvalue changes by less than
    /// `tol_occupied` (bands with occupation > EPS3) or `tol_empty` (other bands).
    /// Returns (number of converged bands, number of H|psi> applications).
    fn compute(
        &mut self, ham_on_psi: &mut dyn FnMut(&[c64], &mut [c64]), ham_diag: &[f64], evecs: &mut Matrix<c64>, evals: &mut [f64], occ: &[f64], tol_occupied: f64, tol_empty: f64,
        max_iter: usize,
    ) -> (usize, usize);
}

pub fn new(solver_scheme: EigenSolverScheme, n: usize, nev: usize) -> Box<dyn EigenSolver> {
    let sparse: Box<dyn EigenSolver>;

    match solver_scheme {
        EigenSolverScheme::Pcg => {
            sparse = Box::new(EigenSolverPCG::new(n, nev));
        }

        EigenSolverScheme::Lobpcg => {
            sparse = Box::new(EigenSolverLOBPCG::new(n, nev));
        }
    }

    sparse
}
