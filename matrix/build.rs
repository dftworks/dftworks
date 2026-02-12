fn main() {
    println!("cargo:rustc-link-lib={}", "lapack");
    println!("cargo:rustc-link-lib={}", "blas");

    // macOS + MacPorts LAPACK static linkage needs explicit GNU Fortran runtime libs.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-search=native=/opt/local/lib/libgcc");
        println!("cargo:rustc-link-lib={}", "gfortran");
        println!("cargo:rustc-link-lib={}", "quadmath");
        println!("cargo:rustc-link-lib={}", "gcc_s");
    }
}
