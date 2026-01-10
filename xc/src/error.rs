//! Error types for the XC module

use std::fmt;

/// Errors that can occur in XC functional calculations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XCError {
    /// Unknown or unsupported XC scheme
    UnknownScheme(String),
    /// Invalid density input
    InvalidDensity(String),
    /// Unsupported spin configuration for this functional
    UnsupportedSpin,
}

impl fmt::Display for XCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XCError::UnknownScheme(scheme) => {
                write!(f, "Unknown XC scheme: '{}'. Supported schemes: lda-pz, lsda-pz, pbe", scheme)
            }
            XCError::InvalidDensity(msg) => {
                write!(f, "Invalid density: {}", msg)
            }
            XCError::UnsupportedSpin => {
                write!(f, "This functional does not support the requested spin configuration")
            }
        }
    }
}

impl std::error::Error for XCError {}
