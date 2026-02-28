use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=DFTWORKS_RPATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    // Add runtime library paths (rpath) for dynamic linking
    // Users can set DFTWORKS_RPATH to a colon-separated list of paths
    // If not set, common package manager locations are used

    if let Ok(custom_rpath) = env::var("DFTWORKS_RPATH") {
        // User-specified rpath (colon-separated)
        for path in custom_rpath.split(':').filter(|p| !p.is_empty()) {
            add_rpath(path);
        }
    } else {
        // Auto-detect common library locations
        let mut candidates = vec![
            // Homebrew (Apple Silicon)
            "/opt/homebrew/lib".to_string(),
            "/opt/homebrew/lib/gcc/current".to_string(),
            "/opt/homebrew/opt/gcc/lib/gcc/current".to_string(),
            // Homebrew (Intel)
            "/usr/local/lib".to_string(),
            "/usr/local/opt/gcc/lib/gcc/current".to_string(),
            // MacPorts
            "/opt/local/lib".to_string(),
            "/opt/local/lib/libgcc".to_string(),
            "/opt/local/lib/mpich-mp".to_string(),
            "/opt/local/lib/openmpi-mp".to_string(),
        ];

        // Add Conda prefix if available
        if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
            candidates.push(format!("{}/lib", conda_prefix));
        }

        // Add custom LAPACK dir if set
        if let Ok(lapack_dir) = env::var("LAPACK_DIR") {
            candidates.push(lapack_dir);
        }

        for candidate in &candidates {
            if Path::new(candidate).exists() {
                add_rpath(candidate);
            }
        }
    }

    // Git commit hash for provenance tracking
    let git_commit = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=DWWORKS_GIT_COMMIT={}", git_commit);

    // Build features for provenance tracking
    let mut features: Vec<String> = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_string))
        .map(|name| name.to_ascii_lowercase().replace('_', "-"))
        .collect();
    features.sort();
    features.dedup();
    println!(
        "cargo:rustc-env=DWWORKS_BUILD_FEATURES={}",
        features.join(",")
    );
}

fn add_rpath(path: &str) {
    // Use platform-specific rpath syntax
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);

    // Note: Windows uses different DLL search mechanisms
}
