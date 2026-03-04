use dwconsts::*;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;
use types::Matrix;
use types::MatrixExt;
use num_traits::identities::Zero;
use types::c64;

pub fn rotate_wfc(
    op_on_v: &mut dyn FnMut(&[c64], &mut [c64]),
    evc_in: &Matrix<c64>,
    evc_out: &mut Matrix<c64>,
    eval_out: &mut [f64],
) {
    let nbnd = evc_in.ncol();
    let mut hxi = vec![c64::zero(); evc_in.nrow()];

    let mut sbh = Matrix::<c64>::new(nbnd, nbnd);
    for i in 0..nbnd {
        let xi = evc_in.get_col(i);

        op_on_v(&xi, &mut hxi);

        for j in 0..nbnd {
            let xj = evc_in.get_col(j);

            sbh[(j, i)] = utility::zdot_product(&xj, &hxi);
        }
    }

    let se = SymmetricEigen::new(sbh.as_dmatrix().clone());
    let mut evals = se.eigenvalues.as_slice().to_vec();
    let mut evs = se.eigenvectors;
    let mut idx: Vec<usize> = (0..evals.len()).collect();
    idx.sort_by(|&i, &j| {
        evals[i]
            .partial_cmp(&evals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if !idx.iter().enumerate().all(|(i, &j)| i == j) {
        let evals_old = evals.clone();
        let evs_old = evs.clone();
        for (dst, &src) in idx.iter().enumerate() {
            evals[dst] = evals_old[src];
            let col = evs_old.column(src).into_owned();
            evs.set_column(dst, &col);
        }
    }

    eval_out[..nbnd].copy_from_slice(&evals[..nbnd]);
    let rotated: DMatrix<c64> = evc_in.as_dmatrix() * &evs;
    evc_out.as_mut_slice().copy_from_slice(rotated.as_slice());
}
