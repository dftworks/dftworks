//! Exchange-correlation functionals for Density Functional Theory calculations
//!
//! This module provides implementations of various exchange-correlation (XC)
//! functionals used in DFT, including:
//! - LDA (Local Density Approximation) functionals
//! - LSDA (Local Spin Density Approximation) functionals
//! - GGA (Generalized Gradient Approximation) functionals (stub)
//!
//! # Usage
//!
//! ## Type-safe enum-based API (Recommended)
//!
//! ```rust,no_run
//! use xc::{XCFunctional, XC};
//! use dfttypes::*;
//! use ndarray::Array3;
//!
//! // Create functional from string (with error handling)
//! let func = XCFunctional::from_str("lda-pz")?;
//!
//! // Or use the enum directly
//! let func = XCFunctional::LdaPz;
//!
//! // Use it
//! func.potential_and_energy(&rho, None, &mut vxc, &mut exc);
//! ```
//!
//! ## Legacy trait object API (for backward compatibility)
//!
//! ```rust,no_run
//! use xc::{new, XC};
//!
//! let xc = new("lda-pz")?;
//! xc.potential_and_energy(&rho, None, &mut vxc, &mut exc);
//! ```

#![allow(warnings)]

mod error;
mod exchange;
mod correlation;
mod functional;
mod lda;
mod pbe;

pub use error::XCError;
pub use functional::XCFunctional;
pub use exchange::slater;
pub use correlation::pz;

use dfttypes::*;
use ndarray::*;
use types::*;

/// Trait for exchange-correlation functionals
///
/// This trait defines the interface for computing exchange-correlation
/// potentials and energy densities from electron densities.
pub trait XC {
    /// Computes the exchange-correlation potential and energy density
    ///
    /// # Arguments
    /// * `rho` - Electron density (spin-polarized or non-spin-polarized)
    /// * `drho` - Density gradient (required for GGA functionals, None for LDA)
    /// * `vxc` - Output: exchange-correlation potential (mutated in place)
    /// * `exc` - Output: exchange-correlation energy density (mutated in place)
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    );
}

// Legacy implementations for backward compatibility
mod legacy {
    use super::*;
    use crate::lda::{ldapz, lsdapz};
    use crate::pbe;

    /// Legacy LDA-PZ implementation (for backward compatibility)
    pub struct XCLDAPZ {}

    impl XCLDAPZ {
        pub fn new() -> XCLDAPZ {
            XCLDAPZ {}
        }
    }

    impl XC for XCLDAPZ {
        fn potential_and_energy(
            &self,
            rho: &RHOR,
            drho: Option<&DRHOR>,
            vxc: &mut VXCR,
            exc: &mut Array3<c64>,
        ) {
            ldapz::compute(rho, drho, vxc, exc);
        }
    }

    /// Legacy LSDA-PZ implementation (for backward compatibility)
    pub struct XCLSDAPZ {}

    impl XCLSDAPZ {
        pub fn new() -> XCLSDAPZ {
            XCLSDAPZ {}
        }
    }

    impl XC for XCLSDAPZ {
        fn potential_and_energy(
            &self,
            rho: &RHOR,
            drho: Option<&DRHOR>,
            vxc: &mut VXCR,
            exc: &mut Array3<c64>,
        ) {
            lsdapz::compute(rho, drho, vxc, exc);
        }
    }

    /// Legacy PBE implementation (for backward compatibility)
    pub struct XCPBE {}

    impl XCPBE {
        pub fn new() -> XCPBE {
            XCPBE {}
        }
    }

    impl XC for XCPBE {
        fn potential_and_energy(
            &self,
            rho: &RHOR,
            drho: Option<&DRHOR>,
            vxc: &mut VXCR,
            exc: &mut Array3<c64>,
        ) {
            pbe::compute(rho, drho, vxc, exc);
        }
    }
}

// Re-export legacy types for backward compatibility
pub use legacy::{XCLDAPZ, XCLSDAPZ, XCPBE};

/// Creates a new exchange-correlation functional from a string identifier
///
/// # Arguments
/// * `xc_scheme` - String identifier (e.g., "lda-pz", "lsda-pz", "pbe")
///
/// # Returns
/// * `Box<dyn XC>` if the scheme is recognized
/// * Panics if the scheme is not supported
///
/// # Example
/// ```
/// use xc::new;
/// let xc = new("lda-pz");
/// ```
///
/// # Note
/// This function panics on error for backward compatibility.
/// For error handling, use `try_new()` instead.
/// For better performance and type safety, consider using `XCFunctional` enum instead.
pub fn new(xc_scheme: &str) -> Box<dyn XC> {
    try_new(xc_scheme).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        panic!("Failed to create XC functional: {}", e);
    })
}

/// Creates a new exchange-correlation functional from a string identifier (with error handling)
///
/// # Arguments
/// * `xc_scheme` - String identifier (e.g., "lda-pz", "lsda-pz", "pbe")
///
/// # Returns
/// * `Ok(Box<dyn XC>)` if the scheme is recognized
/// * `Err(XCError::UnknownScheme)` if the scheme is not supported
///
/// # Example
/// ```
/// use xc::try_new;
/// let xc = try_new("lda-pz")?;
/// ```
///
/// # Note
/// This function returns a trait object for backward compatibility.
/// For better performance and type safety, consider using `XCFunctional` enum instead.
pub fn try_new(xc_scheme: &str) -> Result<Box<dyn XC>, XCError> {
    match xc_scheme {
        "lda-pz" => Ok(Box::new(XCLDAPZ::new())),
        "lsda-pz" => Ok(Box::new(XCLSDAPZ::new())),
        "pbe" => Ok(Box::new(XCPBE::new())),
        _ => Err(XCError::UnknownScheme(xc_scheme.to_string())),
    }
}
