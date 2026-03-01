//! Van der Waals corrections for DFT calculations
//!
//! This module implements dispersion corrections to add long-range
//! correlation effects missing in standard DFT functionals.
//!
//! ## Supported Methods
//!
//! - **DFT-D3**: Grimme's D3 dispersion correction with zero-damping or BJ-damping
//! - **DFT-D3(BJ)**: D3 with Becke-Johnson damping (recommended)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use vdw::{VdwCorrection, VdwMethod, VdwDamping};
//! use crystal::Crystal;
//!
//! // Create D3(BJ) correction for PBE functional
//! let vdw = VdwCorrection::new(VdwMethod::D3, VdwDamping::BJ, "pbe");
//!
//! // Compute dispersion energy
//! let e_disp = vdw.energy(&crystal)?;
//!
//! // Compute dispersion forces
//! let forces_disp = vdw.forces(&crystal)?;
//! ```

use crystal::Crystal;
use vector3::Vector3f64;

mod d3_params;
use d3_params::{get_atomic_number, get_c6_coefficient, get_r0};

/// Van der Waals correction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdwMethod {
    /// DFT-D3 (Grimme)
    D3,
}

/// Damping function for van der Waals correction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdwDamping {
    /// Zero-damping (original D3)
    Zero,
    /// Becke-Johnson damping (recommended, more accurate)
    BJ,
}

/// Van der Waals correction calculator
pub struct VdwCorrection {
    method: VdwMethod,
    damping: VdwDamping,
    xc_scheme: String,
    s6: f64,  // Global scaling factor for C6
    s8: f64,  // Global scaling factor for C8
    a1: f64,  // Damping parameter 1
    a2: f64,  // Damping parameter 2
}

impl VdwCorrection {
    /// Create new van der Waals correction calculator
    ///
    /// # Arguments
    ///
    /// * `method` - Correction method (D3, etc.)
    /// * `damping` - Damping function (Zero, BJ)
    /// * `xc_scheme` - XC functional name for parameter selection
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vdw = VdwCorrection::new(VdwMethod::D3, VdwDamping::BJ, "pbe");
    /// ```
    pub fn new(method: VdwMethod, damping: VdwDamping, xc_scheme: &str) -> Self {
        let (s6, s8, a1, a2) = Self::get_damping_parameters(xc_scheme, damping);

        VdwCorrection {
            method,
            damping,
            xc_scheme: xc_scheme.to_lowercase(),
            s6,
            s8,
            a1,
            a2,
        }
    }

    /// Get damping parameters for functional and damping type
    fn get_damping_parameters(xc_scheme: &str, damping: VdwDamping) -> (f64, f64, f64, f64) {
        let xc = xc_scheme.to_lowercase();

        match damping {
            VdwDamping::Zero => {
                // Zero-damping parameters
                match xc.as_str() {
                    "pbe" => (1.0, 0.722, 1.217, 14.0),
                    "lda" | "lda-pz" => (1.0, 1.0, 1.0, 14.0),
                    "blyp" => (1.0, 1.682, 1.094, 14.0),
                    _ => {
                        eprintln!("Warning: No D3 parameters for '{}', using PBE defaults", xc);
                        (1.0, 0.722, 1.217, 14.0)
                    }
                }
            }
            VdwDamping::BJ => {
                // BJ-damping parameters (recommended)
                match xc.as_str() {
                    "pbe" => (1.0, 0.7875, 0.4289, 4.4407),
                    "lda" | "lda-pz" => (1.0, 1.0, 0.3946, 4.5718),
                    "blyp" => (1.0, 2.0674, 0.4298, 4.2359),
                    _ => {
                        eprintln!("Warning: No D3(BJ) parameters for '{}', using PBE defaults", xc);
                        (1.0, 0.7875, 0.4289, 4.4407)
                    }
                }
            }
        }
    }

    /// Compute dispersion energy (in Ry)
    ///
    /// Returns E_disp = sum_{i<j} E_disp(i,j)
    pub fn energy(&self, crystal: &Crystal) -> Result<f64, String> {
        match self.method {
            VdwMethod::D3 => self.energy_d3(crystal),
        }
    }

    /// Compute dispersion forces (in Ry/bohr)
    ///
    /// Returns force on each atom: F_i = -dE_disp/dR_i
    pub fn forces(&self, crystal: &Crystal) -> Result<Vec<Vector3f64>, String> {
        match self.method {
            VdwMethod::D3 => self.forces_d3(crystal),
        }
    }

    /// Compute dispersion contribution to stress tensor (in Ry/bohr^3)
    ///
    /// Returns stress tensor (3x3 symmetric matrix)
    pub fn stress(&self, crystal: &Crystal) -> Result<[[f64; 3]; 3], String> {
        match self.method {
            VdwMethod::D3 => self.stress_d3(crystal),
        }
    }

    /// DFT-D3 energy calculation
    fn energy_d3(&self, crystal: &Crystal) -> Result<f64, String> {
        let nat = crystal.get_n_atoms();
        let symbols = crystal.get_atom_species();
        let positions = crystal.get_atom_positions_cart();

        let mut e_disp = 0.0;

        // Sum over all atom pairs i < j
        for i in 0..nat {
            let zi = get_atomic_number(&symbols[i])?;
            let c6_i = get_c6_coefficient(zi);
            let r0_i = get_r0(zi);

            for j in (i + 1)..nat {
                let zj = get_atomic_number(&symbols[j])?;
                let c6_j = get_c6_coefficient(zj);
                let r0_j = get_r0(zj);

                // Distance vector and magnitude
                let rij_vec = Vector3f64::new(
                    positions[j].x - positions[i].x,
                    positions[j].y - positions[i].y,
                    positions[j].z - positions[i].z,
                );
                let rij = rij_vec.norm2();

                if rij < 1e-6 {
                    continue; // Skip if atoms too close (shouldn't happen)
                }

                // C6 coefficient for pair (geometric mean)
                let c6_ij = (c6_i * c6_j).sqrt();

                // van der Waals radius for pair
                let r0_ij = r0_i + r0_j;

                // Damping function
                let f_damp = self.damping_function(rij, r0_ij);

                // Dispersion energy for this pair
                // E_ij = -s6 * C6_ij / r_ij^6 * f_damp(r_ij)
                let e_ij = -self.s6 * c6_ij / rij.powi(6) * f_damp;

                e_disp += e_ij;
            }
        }

        // Convert from Hartree to Ry (E_Ry = 2 * E_Ha)
        Ok(e_disp * 2.0)
    }

    /// DFT-D3 forces calculation
    fn forces_d3(&self, crystal: &Crystal) -> Result<Vec<Vector3f64>, String> {
        let nat = crystal.get_n_atoms();
        let symbols = crystal.get_atom_species();
        let positions = crystal.get_atom_positions_cart();

        let mut forces = vec![Vector3f64::zeros(); nat];

        // Sum over all atom pairs i < j
        for i in 0..nat {
            let zi = get_atomic_number(&symbols[i])?;
            let c6_i = get_c6_coefficient(zi);
            let r0_i = get_r0(zi);

            for j in (i + 1)..nat {
                let zj = get_atomic_number(&symbols[j])?;
                let c6_j = get_c6_coefficient(zj);
                let r0_j = get_r0(zj);

                // Distance vector and magnitude
                let rij_vec = Vector3f64::new(
                    positions[j].x - positions[i].x,
                    positions[j].y - positions[i].y,
                    positions[j].z - positions[i].z,
                );
                let rij = rij_vec.norm2();

                if rij < 1e-6 {
                    continue;
                }

                let c6_ij = (c6_i * c6_j).sqrt();
                let r0_ij = r0_i + r0_j;

                // Damping function and its derivative
                let f_damp = self.damping_function(rij, r0_ij);
                let df_damp = self.damping_derivative(rij, r0_ij);

                // Energy derivative: dE_ij/dr_ij
                // E_ij = -s6 * C6_ij / r^6 * f_damp(r)
                // dE/dr = -s6 * C6_ij * [df/dr / r^6 - 6 * f / r^7]
                let de_dr = -self.s6 * c6_ij * (df_damp / rij.powi(6) - 6.0 * f_damp / rij.powi(7));

                // Force vector: F = -dE/dR = -(dE/dr) * (r_vec / r)
                let f_vec = rij_vec * (-de_dr / rij);

                // Newton's third law: F_i = -F_ij, F_j = F_ij
                forces[i] = forces[i] + (f_vec * -1.0);
                forces[j] = forces[j] + f_vec;
            }
        }

        // Convert from Hartree/bohr to Ry/bohr
        for f in &mut forces {
            *f = *f * 2.0;
        }

        Ok(forces)
    }

    /// DFT-D3 stress calculation
    fn stress_d3(&self, crystal: &Crystal) -> Result<[[f64; 3]; 3], String> {
        let nat = crystal.get_n_atoms();
        let symbols = crystal.get_atom_species();
        let positions = crystal.get_atom_positions_cart();
        let volume = crystal.get_latt().volume();

        let mut stress = [[0.0; 3]; 3];

        // Sum over all atom pairs i < j
        for i in 0..nat {
            let zi = get_atomic_number(&symbols[i])?;
            let c6_i = get_c6_coefficient(zi);
            let r0_i = get_r0(zi);

            for j in (i + 1)..nat {
                let zj = get_atomic_number(&symbols[j])?;
                let c6_j = get_c6_coefficient(zj);
                let r0_j = get_r0(zj);

                let rij_vec = Vector3f64::new(
                    positions[j].x - positions[i].x,
                    positions[j].y - positions[i].y,
                    positions[j].z - positions[i].z,
                );
                let rij = rij_vec.norm2();

                if rij < 1e-6 {
                    continue;
                }

                let c6_ij = (c6_i * c6_j).sqrt();
                let r0_ij = r0_i + r0_j;

                let f_damp = self.damping_function(rij, r0_ij);
                let df_damp = self.damping_derivative(rij, r0_ij);

                let de_dr = -self.s6 * c6_ij * (df_damp / rij.powi(6) - 6.0 * f_damp / rij.powi(7));

                // Stress contribution: σ_αβ = (1/V) * sum_ij r_α * F_β
                // where F = (dE/dr) * (r_vec / r)
                let r_components = [rij_vec.x, rij_vec.y, rij_vec.z];
                for alpha in 0..3 {
                    for beta in 0..3 {
                        stress[alpha][beta] += (de_dr / rij) * r_components[alpha] * r_components[beta];
                    }
                }
            }
        }

        // Normalize by volume and convert to Ry/bohr^3
        for alpha in 0..3 {
            for beta in 0..3 {
                stress[alpha][beta] *= 2.0 / volume; // 2.0 for Ha->Ry, divide by volume
            }
        }

        Ok(stress)
    }

    /// Damping function
    fn damping_function(&self, r: f64, r0: f64) -> f64 {
        match self.damping {
            VdwDamping::Zero => {
                // Zero-damping: f(r) = 1 / (1 + exp(-a1*(r/r0 - 1)))
                let sr = r / (self.a2 * r0);
                1.0 / (1.0 + 6.0 * sr.powi(-self.a1 as i32))
            }
            VdwDamping::BJ => {
                // BJ-damping: f(r) = r^n / (r^n + (a1*r0 + a2)^n)
                // For n=6:
                let r0_scaled = self.a1 * r0 + self.a2;
                r.powi(6) / (r.powi(6) + r0_scaled.powi(6))
            }
        }
    }

    /// Derivative of damping function
    fn damping_derivative(&self, r: f64, r0: f64) -> f64 {
        match self.damping {
            VdwDamping::Zero => {
                let sr = r / (self.a2 * r0);
                let exp_term = (-self.a1 * (sr - 1.0)).exp();
                let denom = (1.0 + exp_term).powi(2);
                self.a1 * exp_term / (self.a2 * r0 * denom)
            }
            VdwDamping::BJ => {
                let r0_scaled = self.a1 * r0 + self.a2;
                let r6 = r.powi(6);
                let r0_6 = r0_scaled.powi(6);
                6.0 * r.powi(5) * r0_6 / (r6 + r0_6).powi(2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdw_correction_creation() {
        let vdw = VdwCorrection::new(VdwMethod::D3, VdwDamping::BJ, "pbe");
        assert_eq!(vdw.method, VdwMethod::D3);
        assert_eq!(vdw.damping, VdwDamping::BJ);
        assert_eq!(vdw.s6, 1.0);
    }

    #[test]
    fn test_damping_function() {
        let vdw = VdwCorrection::new(VdwMethod::D3, VdwDamping::BJ, "pbe");

        // At r >> r0, damping should approach 1
        let f_large = vdw.damping_function(10.0, 1.0);
        assert!(f_large > 0.99);

        // At r << r0, damping should approach 0
        let f_small = vdw.damping_function(0.1, 1.0);
        assert!(f_small < 0.01);
    }
}
