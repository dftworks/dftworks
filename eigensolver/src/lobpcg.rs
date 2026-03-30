// Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) eigensolver.
// "B" stands for Block — all bands are updated simultaneously via a
// Rayleigh–Ritz projection on the trial subspace {X, W, P}.
//
// Reference: A. V. Knyazev, "Toward the Optimal Preconditioned Eigensolver:
// Locally Optimal Block Preconditioned Conjugate Gradient Method",
// SIAM J. Sci. Comput. 23(2), 517-541 (2001).
//
// The algorithm solves for the nband smallest eigenpairs of H by iterating
// on a trial subspace {X, W, P} where X are the current Ritz vectors,
// W the preconditioned residuals, and P the conjugate directions from the
// previous step.

use crate::pcg::compute_preconditioner;
use crate::EigenSolver;
use dwconsts::*;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;
use num_traits::identities::Zero;
use types::c64;
use types::*;
use utility;

pub struct EigenSolverLOBPCG {
    npw: usize,
    nband: usize,
    hx: Vec<c64>,
    w: Vec<c64>,
    hw: Vec<c64>,
    p: Vec<c64>,
    hp: Vec<c64>,
    precond: Vec<c64>,
}

impl EigenSolverLOBPCG {
    pub fn new(npw: usize, nband: usize) -> Self {
        Self {
            npw,
            nband,
            hx: vec![c64::zero(); npw * nband],
            w: vec![c64::zero(); npw * nband],
            hw: vec![c64::zero(); npw * nband],
            p: vec![c64::zero(); npw * nband],
            hp: vec![c64::zero(); npw * nband],
            precond: vec![c64::zero(); npw],
        }
    }
}

impl EigenSolver for EigenSolverLOBPCG {
    fn compute(
        &mut self, ham_on_psi: &mut dyn FnMut(&[c64], &mut [c64]), ham_diag: &[f64], evecs: &mut Matrix<c64>, evals: &mut [f64], occ: &[f64], tol_eigval: f64, max_cg_loop: usize,
        max_scf_iter: usize,
    ) -> (usize, usize) {
        let npw = self.npw;
        let nb = self.nband;
        let mut n_hpsi = 0usize;

        // Local copy of X (eigenvector block)
        let mut x = vec![c64::zero(); npw * nb];
        let mut lambda = vec![0.0f64; nb];
        let mut lambda_old = vec![0.0f64; nb];

        for i in 0..nb {
            bcol_mut(&mut x, i, npw).copy_from_slice(evecs.get_col(i));
        }

        // Orthonormalize initial guess
        mgs_block(&mut x, npw, nb);

        // Compute HX = H * X
        for i in 0..nb {
            ham_on_psi(bcol(&x, i, npw), bcol_mut(&mut self.hx, i, npw));
            n_hpsi += 1;
        }

        // Initial Rayleigh quotients
        for i in 0..nb {
            lambda[i] = utility::zdot_product(bcol(&x, i, npw), bcol(&self.hx, i, npw)).re;
        }

        let mut n_band_converged = 0;
        let mut has_p = false;

        for _iter in 0..max_cg_loop {
            lambda_old.copy_from_slice(&lambda);

            // --- Compute preconditioned residuals W_i = T^{-1} (H X_i - lambda_i X_i) ---
            for i in 0..nb {
                let lam = lambda[i];
                for k in 0..npw {
                    self.w[i * npw + k] = self.hx[i * npw + k] - c64::from(lam) * x[i * npw + k];
                }
                compute_preconditioner(bcol(&x, i, npw), ham_diag, &mut self.precond);
                for k in 0..npw {
                    self.w[i * npw + k] *= self.precond[k];
                }
            }

            // Orthogonalize W against X, then orthonormalize W internally
            orthogonalize_block_against(&mut self.w, &x, npw, nb, nb);
            let w_ok = mgs_block_checked(&mut self.w, npw, nb);
            let n_active_w = w_ok.iter().filter(|&&v| v).count();

            // If no active W columns, all bands are converged — exit early
            if n_active_w == 0 {
                n_band_converged = nb;
                break;
            }

            // Compute HW for surviving columns
            for i in 0..nb {
                if w_ok[i] {
                    ham_on_psi(bcol(&self.w, i, npw), bcol_mut(&mut self.hw, i, npw));
                    n_hpsi += 1;
                } else {
                    bcol_mut(&mut self.hw, i, npw).iter_mut().for_each(|v| *v = c64::zero());
                }
            }

            // Orthogonalize P against X and W, then orthonormalize
            let mut use_p = has_p;
            if use_p {
                orthogonalize_block_against(&mut self.p, &x, npw, nb, nb);
                orthogonalize_block_against(&mut self.p, &self.w, npw, nb, nb);
                let p_ok = mgs_block_checked(&mut self.p, npw, nb);
                if p_ok.iter().any(|&ok| !ok) {
                    use_p = false;
                } else {
                    // Recompute HP = H * P (avoids accumulating reconstruction error)
                    for i in 0..nb {
                        ham_on_psi(bcol(&self.p, i, npw), bcol_mut(&mut self.hp, i, npw));
                        n_hpsi += 1;
                    }
                }
            }

            // --- Build subspace Hamiltonian H_sub = basis^H * H*basis ---
            let sdim = if use_p { 3 * nb } else { 2 * nb };
            let mut h_sub = DMatrix::<c64>::zeros(sdim, sdim);

            fill_h_block(&mut h_sub, &x, &self.hx, 0, 0, npw, nb);
            fill_h_block(&mut h_sub, &self.w, &self.hw, nb, nb, npw, nb);
            fill_h_cross(&mut h_sub, &x, &self.hw, 0, nb, npw, nb);
            if use_p {
                fill_h_block(&mut h_sub, &self.p, &self.hp, 2 * nb, 2 * nb, npw, nb);
                fill_h_cross(&mut h_sub, &x, &self.hp, 0, 2 * nb, npw, nb);
                fill_h_cross(&mut h_sub, &self.w, &self.hp, nb, 2 * nb, npw, nb);
            }

            // Penalise dead W columns so they never appear among the nb
            // smallest Ritz values (zero rows/columns would produce spurious
            // eigenvalue = 0 that would be incorrectly selected).
            for i in 0..nb {
                if !w_ok[i] {
                    let idx = nb + i;
                    for j in 0..sdim {
                        h_sub[(idx, j)] = c64::zero();
                        h_sub[(j, idx)] = c64::zero();
                    }
                    h_sub[(idx, idx)] = c64::from(1.0e30);
                }
            }

            // Hermitianize to absorb rounding
            let h_herm = (&h_sub + h_sub.adjoint()) * c64::from(0.5);

            // --- Solve standard eigenvalue problem ---
            let se = SymmetricEigen::new(h_herm);

            let mut idx: Vec<usize> = (0..sdim).collect();
            idx.sort_by(|&a, &b| se.eigenvalues[a].partial_cmp(&se.eigenvalues[b]).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..nb {
                lambda[i] = se.eigenvalues[idx[i]];
            }

            // --- Reconstruct X and P from Ritz vectors ---
            let mut x_new = vec![c64::zero(); npw * nb];
            let mut p_new = vec![c64::zero(); npw * nb];

            for i in 0..nb {
                let ci = idx[i];

                // X-block contribution
                for j in 0..nb {
                    let c = se.eigenvectors[(j, ci)];
                    axpy_col(&mut x_new, i, &x, j, c, npw);
                }

                // W-block contribution (goes into both X_new and P_new)
                for j in 0..nb {
                    let c = se.eigenvectors[(nb + j, ci)];
                    axpy_col(&mut x_new, i, &self.w, j, c, npw);
                    axpy_col(&mut p_new, i, &self.w, j, c, npw);
                }

                // P-block contribution
                if use_p {
                    for j in 0..nb {
                        let c = se.eigenvectors[(2 * nb + j, ci)];
                        axpy_col(&mut x_new, i, &self.p, j, c, npw);
                        axpy_col(&mut p_new, i, &self.p, j, c, npw);
                    }
                }
            }

            x.copy_from_slice(&x_new);
            self.p.copy_from_slice(&p_new);
            has_p = true;

            // Re-orthonormalize X then recompute HX = H*X from scratch
            // (avoids error amplification from the Ritz reconstruction)
            mgs_block(&mut x, npw, nb);
            for i in 0..nb {
                ham_on_psi(bcol(&x, i, npw), bcol_mut(&mut self.hx, i, npw));
                n_hpsi += 1;
            }

            // Recompute lambda from the fresh HX
            for i in 0..nb {
                lambda[i] = utility::zdot_product(bcol(&x, i, npw), bcol(&self.hx, i, npw)).re;
            }

            // --- Check convergence ---
            n_band_converged = 0;
            for i in 0..nb {
                let tol2 = if max_scf_iter <= 1 {
                    tol_eigval
                } else if occ[i] > EPS3 {
                    tol_eigval
                } else {
                    (tol_eigval * 10.0).max(EPS5)
                };
                if (lambda[i] - lambda_old[i]).abs() < tol2 {
                    n_band_converged += 1;
                }
            }

            if n_band_converged == nb {
                break;
            }
        }

        // Copy results back
        evals[..nb].copy_from_slice(&lambda);
        for i in 0..nb {
            evecs.set_col(i, bcol(&x, i, npw));
        }

        (n_band_converged, n_hpsi)
    }
}

// ---------------------------------------------------------------------------
// Column-access helpers for flat npw*nband buffers (column-major layout)
// ---------------------------------------------------------------------------

#[inline]
fn bcol(buf: &[c64], i: usize, npw: usize) -> &[c64] {
    &buf[i * npw..(i + 1) * npw]
}

#[inline]
fn bcol_mut(buf: &mut [c64], i: usize, npw: usize) -> &mut [c64] {
    &mut buf[i * npw..(i + 1) * npw]
}

/// out[:,io] += alpha * src[:,is]
#[inline]
fn axpy_col(out: &mut [c64], io: usize, src: &[c64], is: usize, alpha: c64, npw: usize) {
    let o = io * npw;
    let s = is * npw;
    for k in 0..npw {
        out[o + k] += alpha * src[s + k];
    }
}

// ---------------------------------------------------------------------------
// Modified Gram-Schmidt routines
// ---------------------------------------------------------------------------

/// Orthonormalize columns of `buf` via modified Gram-Schmidt.
fn mgs_block(buf: &mut [c64], npw: usize, ncol: usize) {
    for i in 0..ncol {
        for j in 0..i {
            let proj = utility::zdot_product(bcol(buf, j, npw), bcol(buf, i, npw));
            let (left, right) = buf.split_at_mut(i * npw);
            let cj = &left[j * npw..(j + 1) * npw];
            let ci = &mut right[..npw];
            for k in 0..npw {
                ci[k] -= proj * cj[k];
            }
        }
        let norm = utility::zdot_product(bcol(buf, i, npw), bcol(buf, i, npw)).re.sqrt();
        if norm > EPS20 {
            let inv = 1.0 / norm;
            bcol_mut(buf, i, npw).iter_mut().for_each(|v| *v *= inv);
        }
    }
}

/// Orthonormalize columns; returns per-column survival flag.
fn mgs_block_checked(buf: &mut [c64], npw: usize, ncol: usize) -> Vec<bool> {
    let mut ok = vec![true; ncol];
    for i in 0..ncol {
        for j in 0..i {
            if !ok[j] {
                continue;
            }
            let proj = utility::zdot_product(bcol(buf, j, npw), bcol(buf, i, npw));
            let (left, right) = buf.split_at_mut(i * npw);
            let cj = &left[j * npw..(j + 1) * npw];
            let ci = &mut right[..npw];
            for k in 0..npw {
                ci[k] -= proj * cj[k];
            }
        }
        let norm = utility::zdot_product(bcol(buf, i, npw), bcol(buf, i, npw)).re.sqrt();
        if norm > EPS10 {
            let inv = 1.0 / norm;
            bcol_mut(buf, i, npw).iter_mut().for_each(|v| *v *= inv);
        } else {
            bcol_mut(buf, i, npw).iter_mut().for_each(|v| *v = c64::zero());
            ok[i] = false;
        }
    }
    ok
}

// ---------------------------------------------------------------------------
// Block orthogonalization
// ---------------------------------------------------------------------------

/// Orthogonalize columns of `a` against columns of `b`:  A -= B (B^H A)
fn orthogonalize_block_against(a: &mut [c64], b: &[c64], npw: usize, na: usize, nb_cols: usize) {
    for i in 0..na {
        for j in 0..nb_cols {
            let proj = utility::zdot_product(&b[j * npw..(j + 1) * npw], &a[i * npw..(i + 1) * npw]);
            for k in 0..npw {
                a[i * npw + k] -= proj * b[j * npw + k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subspace Hamiltonian assembly
// ---------------------------------------------------------------------------

/// Diagonal block:  h_sub[off+i, off+j] = <a_i | ha_j>
fn fill_h_block(h_sub: &mut DMatrix<c64>, a: &[c64], ha: &[c64], r_off: usize, c_off: usize, npw: usize, nb: usize) {
    for i in 0..nb {
        for j in 0..nb {
            h_sub[(r_off + i, c_off + j)] = utility::zdot_product(bcol(a, i, npw), bcol(ha, j, npw));
        }
    }
}

/// Off-diagonal cross block:  h_sub[r_off+i, c_off+j] = <a_i | hb_j>
/// and its Hermitian conjugate  h_sub[c_off+j, r_off+i] = conj(...)
fn fill_h_cross(h_sub: &mut DMatrix<c64>, a: &[c64], hb: &[c64], r_off: usize, c_off: usize, npw: usize, nb: usize) {
    for i in 0..nb {
        for j in 0..nb {
            let val = utility::zdot_product(bcol(a, i, npw), bcol(hb, j, npw));
            h_sub[(r_off + i, c_off + j)] = val;
            h_sub[(c_off + j, r_off + i)] = val.conj();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_solver_lobpcg() {
    use nalgebra::DVector;

    let n: usize = 100;
    let nev: usize = 20;

    let m = utility::make_matrix(n);
    let m_na = m.as_dmatrix().clone();

    let mut m_dot_v = |v: &[c64], vp: &mut [c64]| {
        let x = &m_na * DVector::from_column_slice(v);
        vp.copy_from_slice(x.as_slice());
    };
    let mut evals = vec![0.0; nev];
    let mut evecs = Matrix::new(n, nev);

    let mut h_diag = vec![0.0; n];
    for i in 0..n {
        h_diag[i] = m[(i, i)].re;
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

    let mut solver = EigenSolverLOBPCG::new(n, nev);
    let (nconv, niter) = solver.compute(&mut m_dot_v, &h_diag, &mut evecs, &mut evals, &occ, EPS5, max_cg_loop, max_scf_iter);

    println!("eigenvalues converged = {}, niter = {}", nconv, niter);

    let (nconv, niter) = solver.compute(&mut m_dot_v, &h_diag, &mut evecs, &mut evals, &occ, EPS10, max_cg_loop, max_scf_iter);

    println!("eigenvalues converged = {}, niter = {}", nconv, niter);

    let se = SymmetricEigen::new(m.as_dmatrix().clone());
    let mut es = se.eigenvalues.as_slice().to_vec();
    es.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!(" lobpcg \t\t eigh");

    for i in 0..nev {
        println!("{:20.16} {:20.16}", evals[i], es[i]);
    }
}
