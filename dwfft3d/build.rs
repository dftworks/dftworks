use std::collections::BTreeSet;
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=FFTW_DIR");
    println!("cargo:rerun-if-env-changed=FFTW_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=DFTWORKS_ALLOW_PATH_FALLBACK");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let mut search_dirs: BTreeSet<String> = BTreeSet::new();

    if let Ok(dir) = env::var("FFTW_LIB_DIR") {
        add_if_exists(&mut search_dirs, dir);
    }

    if let Ok(prefix) = env::var("FFTW_DIR") {
        add_if_exists(&mut search_dirs, format!("{}/lib", prefix));
    }

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        add_if_exists(&mut search_dirs, format!("{}/lib", conda_prefix));
    }

    for dir in pkg_config_lib_dirs(&["fftw3", "fftw3_threads"]) {
        add_if_exists(&mut search_dirs, dir);
    }

    for dir in standard_system_dirs() {
        if has_fftw3(&dir) {
            add_if_exists(&mut search_dirs, dir);
        }
    }

    // Optional fallback for package-manager specific locations.
    if allow_path_fallback() {
        for dir in platform_fallback_dirs() {
            if has_fftw3(&dir) {
                add_if_exists(&mut search_dirs, dir);
            }
        }
    }

    if search_dirs.is_empty() {
        panic!(
            "Failed to discover FFTW3 libraries.\n\
             Tried: FFTW_LIB_DIR/FFTW_DIR, pkg-config(fftw3), standard system paths.\n\
             Optional fallback paths are disabled by default; enable with DFTWORKS_ALLOW_PATH_FALLBACK=1.\n\
             Install hints:\n\
               - Ubuntu/Debian: sudo apt install libfftw3-dev\n\
               - RHEL/CentOS:   sudo yum install fftw-devel\n\
               - Homebrew:      brew install fftw\n\
               - MacPorts:      sudo port install fftw-3\n\
             Or set:\n\
               export FFTW_DIR=/path/to/fftw\n\
               export FFTW_LIB_DIR=/path/to/fftw/lib"
        );
    }

    for dir in search_dirs {
        println!("cargo:rustc-link-search=native={}", dir);
    }

    println!("cargo:rustc-link-lib=fftw3");
    println!("cargo:rustc-link-lib=fftw3_threads");
}

fn allow_path_fallback() -> bool {
    env::var("DFTWORKS_ALLOW_PATH_FALLBACK")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn add_if_exists(out: &mut BTreeSet<String>, dir: String) {
    if Path::new(&dir).exists() {
        out.insert(dir);
    }
}

fn standard_system_dirs() -> Vec<String> {
    vec![
        "/usr/lib".to_string(),
        "/usr/lib64".to_string(),
        "/usr/lib/x86_64-linux-gnu".to_string(),
        "/usr/lib/aarch64-linux-gnu".to_string(),
    ]
}

fn platform_fallback_dirs() -> Vec<String> {
    vec![
        "/opt/homebrew/lib".to_string(),
        "/usr/local/lib".to_string(),
        "/opt/local/lib".to_string(),
    ]
}

fn has_fftw3(dir: &str) -> bool {
    let path = Path::new(dir);
    path.join("libfftw3.a").exists()
        || path.join("libfftw3.so").exists()
        || path.join("libfftw3.dylib").exists()
}

fn pkg_config_lib_dirs(pkgs: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for pkg in pkgs {
        let output = Command::new("pkg-config")
            .args(["--libs-only-L", pkg])
            .output();

        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for token in stdout.split_whitespace() {
            if let Some(dir) = token.strip_prefix("-L") {
                out.push(dir.to_string());
            }
        }
    }
    out
}
