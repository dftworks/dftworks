//! Core traits for the XC module redesign.

use vector3::Vector3;
use types::c64;

/// Enum representing the family of an XC functional.
/// This determines which compute method is valid to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XCFamily {
    /// Local Density Approximation
    LDA,
    /// Generalized Gradient Approximation
    GGA,
    /// Meta-Generalized Gradient Approximation (Future proofing)
    MetaGGA,
}

/// The core interface for any XC functional kernel.
/// Implementation should be stateless and thread-safe.
pub trait XCKernel: Send + Sync {
    /// Returns the family of this functional
    fn family(&self) -> XCFamily;

    /// Compute XC for a batch of points (LDA).
    /// Used when `family() == LDA`.
    /// 
    /// # Arguments
    /// * `rho` - Input density batch [N]
    /// * `vxc` - Output potential batch [N] (added to existing values)
    /// * `exc` - Output energy density batch [N] (overwrites existing values)
    fn compute_lda(&self, rho: &[f64], vxc: &mut [f64], exc: &mut [f64]);

    /// Compute XC for a batch with gradients (GGA).
    /// Used when `family() == GGA`.
    fn compute_gga(&self, _rho: &[f64], _drho: &[Vector3<f64>], _vxc: &mut [f64], _exc: &mut [f64]) {
        unimplemented!("GGA not implemented for this kernel");
    }
}
