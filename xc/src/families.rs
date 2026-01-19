//! Type-safe wrappers for functional families.

use crate::traits::{XCKernel, XCFamily};
use vector3::Vector3;

/// Type-safe trait for LDA kernels
pub trait LdaKernel: Send + Sync {
    fn compute(&self, rho: &[f64], vxc: &mut [f64], exc: &mut [f64]);
}

/// Type-safe trait for GGA kernels
pub trait GgaKernel: Send + Sync {
    fn compute(&self, rho: &[f64], grad: &[Vector3<f64>], vxc: &mut [f64], exc: &mut [f64]);
}

// Blanket implementation for any XCKernel that claims to be LDA
impl<T: XCKernel> LdaKernel for T {
    fn compute(&self, rho: &[f64], vxc: &mut [f64], exc: &mut [f64]) {
        debug_assert_eq!(self.family(), XCFamily::LDA);
        self.compute_lda(rho, vxc, exc);
    }
}
