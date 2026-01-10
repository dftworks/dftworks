//! Exchange-correlation functional types and dispatch

use crate::error::XCError;
use crate::XC;
use dfttypes::*;
use ndarray::Array3;
use types::c64;

/// Type-safe enumeration of available exchange-correlation functionals
///
/// This enum provides zero-cost abstraction - no heap allocation or
/// dynamic dispatch overhead compared to trait objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XCFunctional {
    /// Local Density Approximation with Perdew-Zunger correlation
    /// Uses Slater exchange and PZ correlation (unpolarized)
    LdaPz,
    /// Local Spin Density Approximation with Perdew-Zunger correlation
    /// Uses Slater exchange and PZ correlation (spin-polarized)
    LsdaPz,
    /// Perdew-Burke-Ernzerhof generalized gradient approximation
    /// (Currently a stub - not yet implemented)
    Pbe,
}

impl XCFunctional {
    /// Creates an XCFunctional from a string identifier
    ///
    /// # Arguments
    /// * `scheme` - String identifier (e.g., "lda-pz", "lsda-pz", "pbe")
    ///
    /// # Returns
    /// * `Ok(XCFunctional)` if the scheme is recognized
    /// * `Err(XCError::UnknownScheme)` if the scheme is not supported
    ///
    /// # Example
    /// ```
    /// use xc::XCFunctional;
    /// let func = XCFunctional::from_str("lda-pz")?;
    /// ```
    pub fn from_str(scheme: &str) -> Result<Self, XCError> {
        match scheme {
            "lda-pz" => Ok(XCFunctional::LdaPz),
            "lsda-pz" => Ok(XCFunctional::LsdaPz),
            "pbe" => Ok(XCFunctional::Pbe),
            _ => Err(XCError::UnknownScheme(scheme.to_string())),
        }
    }

    /// Returns the string identifier for this functional
    ///
    /// # Example
    /// ```
    /// use xc::XCFunctional;
    /// let func = XCFunctional::LdaPz;
    /// assert_eq!(func.as_str(), "lda-pz");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            XCFunctional::LdaPz => "lda-pz",
            XCFunctional::LsdaPz => "lsda-pz",
            XCFunctional::Pbe => "pbe",
        }
    }

    /// Returns whether this functional requires density gradients
    ///
    /// LDA functionals only need the density, while GGA functionals
    /// (like PBE) also need the density gradient.
    pub fn needs_gradient(&self) -> bool {
        match self {
            XCFunctional::LdaPz | XCFunctional::LsdaPz => false,
            XCFunctional::Pbe => true,
        }
    }

    /// Returns whether this functional supports spin-polarized calculations
    pub fn supports_spin(&self) -> bool {
        match self {
            XCFunctional::LdaPz => false,
            XCFunctional::LsdaPz | XCFunctional::Pbe => true,
        }
    }
}

impl XC for XCFunctional {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
        match self {
            XCFunctional::LdaPz => {
                crate::lda::ldapz::compute(rho, drho, vxc, exc);
            }
            XCFunctional::LsdaPz => {
                crate::lda::lsdapz::compute(rho, drho, vxc, exc);
            }
            XCFunctional::Pbe => {
                crate::pbe::compute(rho, drho, vxc, exc);
            }
        }
    }
}
