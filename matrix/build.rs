fn main() {
    println!(
        "cargo:rustc-link-search={}",
        "/opt/intel/oneapi/mkl/2021.1.1/lib"
    );
    println!("cargo:rustc-link-lib={}", "mkl_intel_lp64");
    println!("cargo:rustc-link-lib={}", "mkl_core");
    println!("cargo:rustc-link-lib={}", "mkl_sequential");
    println!("cargo:rustc-link-lib={}", "pthread");
    println!("cargo:rustc-link-lib={}", "m");
    println!("cargo:rustc-link-lib={}", "dl");
}
