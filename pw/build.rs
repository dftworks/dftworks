use std::process::Command;

fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib/libgcc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib/mpich-mp");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/lapack/lib");

    let git_commit = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=DWWORKS_GIT_COMMIT={}", git_commit);

    let mut features: Vec<String> = std::env::vars()
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
