use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=GFORTRAN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MATRIX_LINK_GFORTRAN");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=LAPACK_DIR");

    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=blas");

    // Add LAPACK_DIR to search path if specified
    if let Ok(lapack_dir) = env::var("LAPACK_DIR") {
        println!("cargo:rustc-link-search=native={}", lapack_dir);
    }

    // macOS + MacPorts/Homebrew LAPACK static linkage needs explicit GNU Fortran runtime libs
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let link_gfortran = env::var("MATRIX_LINK_GFORTRAN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        if link_gfortran {
            // Try environment variable first
            let gfortran_lib_dir = env::var("GFORTRAN_LIB_DIR").ok().or_else(|| {
                // Try common locations in order of preference
                let mut candidates = vec![
                    // Homebrew (Apple Silicon) - gcc@13, gcc@12, gcc@11, etc
                    "/opt/homebrew/lib/gcc/13".to_string(),
                    "/opt/homebrew/lib/gcc/12".to_string(),
                    "/opt/homebrew/lib/gcc/11".to_string(),
                    "/opt/homebrew/opt/gcc/lib/gcc/current".to_string(),
                    // Homebrew (Intel)
                    "/usr/local/lib/gcc/13".to_string(),
                    "/usr/local/lib/gcc/12".to_string(),
                    "/usr/local/lib/gcc/11".to_string(),
                    "/usr/local/opt/gcc/lib/gcc/current".to_string(),
                    // MacPorts
                    "/opt/local/lib/libgcc".to_string(),
                    "/opt/local/lib/gcc12".to_string(),
                    "/opt/local/lib/gcc11".to_string(),
                ];

                // Add Conda prefix if available
                if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
                    candidates.push(format!("{}/lib", conda_prefix));
                }

                candidates
                    .into_iter()
                    .find(|dir| Path::new(dir).exists())
            });

            match gfortran_lib_dir {
                Some(dir) => {
                    println!("cargo:rustc-link-search=native={}", dir);
                }
                None => {
                    eprintln!("WARNING: gfortran library directory not found.");
                    eprintln!("If build fails with missing gfortran symbols, try:");
                    eprintln!("  export GFORTRAN_LIB_DIR=/path/to/gcc/lib");
                    eprintln!("Common package manager commands:");
                    eprintln!("  Homebrew: brew install gcc");
                    eprintln!("  MacPorts: sudo port install gcc13");
                }
            }

            println!("cargo:rustc-link-lib=gfortran");
            println!("cargo:rustc-link-lib=quadmath");
            println!("cargo:rustc-link-lib=gcc_s");
        }
    }
}
