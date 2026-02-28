use std::env;
use std::path::PathBuf;

fn main() {
    // Environment-driven MPI library discovery
    // Users can set MPI_LIB_DIR to specify MPI library location
    // Common locations are checked as fallbacks

    let mpi_lib_dir = env::var("MPI_LIB_DIR").ok();

    if let Some(dir) = mpi_lib_dir {
        println!("cargo:rustc-link-search=native={}", dir);
        println!("cargo:rerun-if-env-changed=MPI_LIB_DIR");
        return;
    }

    // Try common package manager locations in order of preference
    let mut candidates: Vec<String> = vec![
        // Homebrew (Apple Silicon)
        "/opt/homebrew/lib".to_string(),
        // Homebrew (Intel)
        "/usr/local/lib".to_string(),
        // MacPorts MPICH
        "/opt/local/lib/mpich-mp".to_string(),
        // MacPorts OpenMPI
        "/opt/local/lib/openmpi-mp".to_string(),
        // MacPorts base
        "/opt/local/lib".to_string(),
        // System paths (Linux)
        "/usr/lib".to_string(),
        "/usr/lib64".to_string(),
    ];

    // Add Conda prefix if available
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        candidates.push(format!("{}/lib", conda_prefix));
    }

    let mut found = false;
    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", candidate);
            found = true;
        }
    }

    if !found {
        eprintln!("WARNING: No MPI library directory found.");
        eprintln!("Please set MPI_LIB_DIR environment variable to your MPI library path.");
        eprintln!("Common package manager commands:");
        eprintln!("  Homebrew (macOS): brew install open-mpi");
        eprintln!("  MacPorts (macOS): sudo port install mpich-default");
        eprintln!("  Conda:            conda install -c conda-forge openmpi");
        eprintln!("  Ubuntu/Debian:    sudo apt install libopenmpi-dev");
        eprintln!("  RHEL/CentOS:      sudo yum install openmpi-devel");
    }

    println!("cargo:rerun-if-env-changed=MPI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
}
