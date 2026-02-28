#![allow(warnings)]

use dfttypes::*;
use gvector::GVector;
use mixing::Mixing;
use pwdensity::PWDensity;
use rayon::prelude::*;
use rgtransform::RGTransform;
use types::c64;

const PARALLEL_MIN_LEN: usize = 8192;

#[inline]
fn use_parallel_for_len(len: usize) -> bool {
    len >= PARALLEL_MIN_LEN && rayon::current_num_threads() > 1
}

pub fn compute_rho_of_g(
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    rho_3d: &mut RHOR,
    rhog_out: &mut [c64],
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        rgtrans.r3d_to_g1d(gvec, pwden, rho_3d.as_slice(), rhog_out);
    }
}

pub fn compute_next_density(
    pwden: &PWDensity,
    mixing: &mut dyn Mixing,
    rhog_out: &[c64],
    rhog_diff: &mut [c64],
    rhog: &mut RHOG,
) {
    if let RHOG::NonSpin(rhog) = rhog {
        debug_assert_eq!(rhog_out.len(), rhog_diff.len());
        debug_assert_eq!(rhog.len(), rhog_diff.len());

        // mix old and new densities to get the density for the next iteration
        if use_parallel_for_len(rhog_diff.len()) {
            rhog_diff
                .par_iter_mut()
                .zip(rhog_out.par_iter())
                .zip(rhog.par_iter())
                .for_each(|((d, out), old)| {
                    *d = *out - *old;
                });
        } else {
            for ipw in 0..rhog_diff.len() {
                rhog_diff[ipw] = rhog_out[ipw] - rhog[ipw];
            }
        }

        mixing.compute_next_density(pwden.get_g(), rhog, rhog_diff);
    }
}
