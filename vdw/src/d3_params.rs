//! DFT-D3 atomic parameters database
//!
//! Contains C6 dispersion coefficients and van der Waals radii
//! for elements commonly used in DFT calculations.
//!
//! Reference: S. Grimme et al., J. Chem. Phys. 132, 154104 (2010)

/// Get atomic number from element symbol
pub fn get_atomic_number(symbol: &str) -> Result<u32, String> {
    let symbol_upper = symbol.to_uppercase();

    match symbol_upper.as_str() {
        "H" => Ok(1),
        "HE" => Ok(2),
        "LI" => Ok(3),
        "BE" => Ok(4),
        "B" => Ok(5),
        "C" => Ok(6),
        "N" => Ok(7),
        "O" => Ok(8),
        "F" => Ok(9),
        "NE" => Ok(10),
        "NA" => Ok(11),
        "MG" => Ok(12),
        "AL" => Ok(13),
        "SI" => Ok(14),
        "P" => Ok(15),
        "S" => Ok(16),
        "CL" => Ok(17),
        "AR" => Ok(18),
        "K" => Ok(19),
        "CA" => Ok(20),
        "SC" => Ok(21),
        "TI" => Ok(22),
        "V" => Ok(23),
        "CR" => Ok(24),
        "MN" => Ok(25),
        "FE" => Ok(26),
        "CO" => Ok(27),
        "NI" => Ok(28),
        "CU" => Ok(29),
        "ZN" => Ok(30),
        "GA" => Ok(31),
        "GE" => Ok(32),
        "AS" => Ok(33),
        "SE" => Ok(34),
        "BR" => Ok(35),
        "KR" => Ok(36),
        _ => Err(format!("Unknown element symbol: {}", symbol)),
    }
}

/// Get C6 dispersion coefficient for element (in Hartree·bohr^6)
///
/// These are reference C6 values for free atoms from DFT-D3 parametrization.
/// The actual C6 for atom pairs is computed as geometric mean: C6_ij = sqrt(C6_i * C6_j)
pub fn get_c6_coefficient(z: u32) -> f64 {
    match z {
        1 => 0.14,      // H
        2 => 0.08,      // He
        3 => 1387.0,    // Li
        4 => 214.0,     // Be
        5 => 99.5,      // B
        6 => 46.6,      // C
        7 => 24.2,      // N
        8 => 15.6,      // O
        9 => 9.52,      // F
        10 => 6.38,     // Ne
        11 => 1556.0,   // Na
        12 => 627.0,    // Mg
        13 => 528.0,    // Al
        14 => 305.0,    // Si
        15 => 185.0,    // P
        16 => 134.0,    // S
        17 => 94.6,     // Cl
        18 => 64.3,     // Ar
        19 => 3897.0,   // K
        20 => 2221.0,   // Ca
        21 => 1383.0,   // Sc
        22 => 1044.0,   // Ti
        23 => 832.0,    // V
        24 => 602.0,    // Cr
        25 => 552.0,    // Mn
        26 => 482.0,    // Fe
        27 => 408.0,    // Co
        28 => 373.0,    // Ni
        29 => 253.0,    // Cu
        30 => 284.0,    // Zn
        31 => 498.0,    // Ga
        32 => 354.0,    // Ge
        33 => 246.0,    // As
        34 => 210.0,    // Se
        35 => 162.0,    // Br
        36 => 129.6,    // Kr
        _ => {
            eprintln!("Warning: No C6 coefficient for Z={}, using default 100.0", z);
            100.0
        }
    }
}

/// Get van der Waals radius R0 for element (in bohr)
///
/// These are covalent radii used in the DFT-D3 damping function.
/// The cutoff radius for pair ij is: R0_ij = R0_i + R0_j
pub fn get_r0(z: u32) -> f64 {
    match z {
        1 => 1.001,     // H
        2 => 1.012,     // He
        3 => 0.825,     // Li
        4 => 1.408,     // Be
        5 => 1.485,     // B
        6 => 1.452,     // C
        7 => 1.397,     // N
        8 => 1.342,     // O
        9 => 1.287,     // F
        10 => 1.243,    // Ne
        11 => 1.144,    // Na
        12 => 1.364,    // Mg
        13 => 1.639,    // Al
        14 => 1.716,    // Si
        15 => 1.705,    // P
        16 => 1.683,    // S
        17 => 1.639,    // Cl
        18 => 1.595,    // Ar
        19 => 1.485,    // K
        20 => 1.474,    // Ca
        21 => 1.562,    // Sc
        22 => 1.562,    // Ti
        23 => 1.562,    // V
        24 => 1.562,    // Cr
        25 => 1.562,    // Mn
        26 => 1.562,    // Fe
        27 => 1.562,    // Co
        28 => 1.562,    // Ni
        29 => 1.562,    // Cu
        30 => 1.562,    // Zn
        31 => 1.650,    // Ga
        32 => 1.727,    // Ge
        33 => 1.760,    // As
        34 => 1.771,    // Se
        35 => 1.749,    // Br
        36 => 1.727,    // Kr
        _ => {
            eprintln!("Warning: No R0 for Z={}, using default 1.5", z);
            1.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_number_lookup() {
        assert_eq!(get_atomic_number("H").unwrap(), 1);
        assert_eq!(get_atomic_number("C").unwrap(), 6);
        assert_eq!(get_atomic_number("Si").unwrap(), 14);
        assert_eq!(get_atomic_number("si").unwrap(), 14); // Case-insensitive

        assert!(get_atomic_number("Xx").is_err());
    }

    #[test]
    fn test_c6_coefficients() {
        // Check common elements
        assert!((get_c6_coefficient(1) - 0.14).abs() < 1e-6);  // H
        assert!((get_c6_coefficient(6) - 46.6).abs() < 1e-6);  // C
        assert!((get_c6_coefficient(14) - 305.0).abs() < 1e-6); // Si
    }

    #[test]
    fn test_r0_values() {
        // Check common elements
        assert!((get_r0(1) - 1.001).abs() < 1e-6);   // H
        assert!((get_r0(6) - 1.452).abs() < 1e-6);   // C
        assert!((get_r0(14) - 1.716).abs() < 1e-6);  // Si
    }
}
