//! Reference scalar implementation of XC kernels.
//! This implementation uses simple loops and is intended for correctness verification and fallback.

use crate::traits::{XCKernel, XCFamily};
use crate::correlation::pz::pz_unpolarized;
use crate::exchange::slater::slater_unpolarized;

/// Scalar CPU implementation of LDA-PZ
pub struct LdaPzScalar;

impl XCKernel for LdaPzScalar {
    fn family(&self) -> XCFamily {
        XCFamily::LDA
    }

    fn compute_lda(&self, rho: &[f64], vxc: &mut [f64], exc: &mut [f64]) {
        let n = rho.len();
        debug_assert_eq!(vxc.len(), n);
        debug_assert_eq!(exc.len(), n);

        for i in 0..n {
            // Use same logic as original: abs() handling might be needed if density is noisy
            // matching original pz implementation which expects positive density
            let t = rho[i].abs(); 

            // Exchange
            let (vx, ex) = slater_unpolarized(t);
            
            // Correlation
            let (vc, ec) = pz_unpolarized(t);

            vxc[i] += vx + vc;
            exc[i] = ex + ec;
        }
    }
}
