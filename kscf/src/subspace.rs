use dwconsts::*;
use matrix::Matrix;
use nalgebra::DMatrix;
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

            sbh[[j, i]] = utility::zdot_product(&xj, &hxi);
        }
    }

    let (evals, evs) = linalg::eigh(&sbh);

    eval_out[..nbnd].copy_from_slice(&evals[..nbnd]);
    let rotated: DMatrix<c64> = evc_in.as_dmatrix() * evs.as_dmatrix();
    evc_out.as_mut_slice().copy_from_slice(rotated.as_slice());
}
