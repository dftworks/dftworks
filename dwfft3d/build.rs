use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=FFTW_DIR");
    println!("cargo:rerun-if-env-changed=FFTW_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    // Try to find FFTW library path
    let fftw_lib_dir = env::var("FFTW_LIB_DIR").ok().or_else(|| {
        env::var("FFTW_DIR").ok().map(|d| format!("{}/lib", d))
    });

    if let Some(dir) = fftw_lib_dir {
        println!("cargo:rustc-link-search=native={}", dir);
    } else {
        // Try pkg-config first (most reliable cross-platform method)
        if let Ok(lib_dir) = pkg_config_fftw() {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        } else {
            // Try common package manager locations
            let mut candidates = vec![
                // Homebrew (Apple Silicon)
                "/opt/homebrew/lib".to_string(),
                // Homebrew (Intel)
                "/usr/local/lib".to_string(),
                // MacPorts
                "/opt/local/lib".to_string(),
                // System paths (Linux)
                "/usr/lib".to_string(),
                "/usr/lib64".to_string(),
                "/usr/lib/x86_64-linux-gnu".to_string(),
            ];

            // Add Conda prefix if available
            if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
                candidates.push(format!("{}/lib", conda_prefix));
            }

            let mut found = false;
            for candidate in &candidates {
                let path = Path::new(candidate);
                // Check if libfftw3 actually exists at this location
                if path.join("libfftw3.a").exists()
                    || path.join("libfftw3.so").exists()
                    || path.join("libfftw3.dylib").exists() {
                    println!("cargo:rustc-link-search=native={}", candidate);
                    found = true;
                    break;
                }
            }

            if !found {
                eprintln!("WARNING: FFTW3 library not found.");
                eprintln!("Build may fail. Please install FFTW3 or set FFTW_DIR or FFTW_LIB_DIR.");
                eprintln!("Common package manager commands:");
                eprintln!("  Homebrew (macOS): brew install fftw");
                eprintln!("  MacPorts (macOS): sudo port install fftw-3");
                eprintln!("  Conda:            conda install -c conda-forge fftw");
                eprintln!("  Ubuntu/Debian:    sudo apt install libfftw3-dev");
                eprintln!("  RHEL/CentOS:      sudo yum install fftw-devel");
                eprintln!("");
                eprintln!("Or set environment variables:");
                eprintln!("  export FFTW_DIR=/path/to/fftw");
                eprintln!("  export FFTW_LIB_DIR=/path/to/fftw/lib");
            }
        }
    }

    println!("cargo:rustc-link-lib=fftw3");
    println!("cargo:rustc-link-lib=fftw3_threads");
}

fn pkg_config_fftw() -> Result<String, ()> {
    let output = Command::new("pkg-config")
        .args(["--libs-only-L", "fftw3"])
        .output()
        .map_err(|_| ())?;

    if !output.status.success() {
        return Err(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lib_dir = stdout
        .trim()
        .strip_prefix("-L")
        .ok_or(())?
        .to_string();

    Ok(lib_dir)
}
