use dwconsts::*;
use matrix::Matrix;
use num_traits::identities::Zero;
use types::c64;

pub fn rotate_wfc(
    op_on_v: &mut dyn FnMut(&[c64], &mut [c64]),
    evc_in: &Matrix<c64>,
    evc_out: &mut Matrix<c64>,
    eval_out: &mut [f64],
) {
    let nbnd = evc_in.ncol();
    let npw = evc_in.nrow();

    let mut hxi = vec![c64::zero(); npw];

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

    for ib in 0..nbnd {
        let xi = evc_out.get_mut_col(ib);

        xi.iter_mut().for_each(|x| *x = c64::zero());

        for jb in 0..nbnd {
            let xj = evc_in.get_col(jb);

            let f = evs[[jb, ib]];

            for i in 0..npw {
                xi[i] += xj[i] * f;
            }
        }
    }
}
