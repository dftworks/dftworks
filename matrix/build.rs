fn main() {
    println!("cargo:rustc-link-lib={}", "lapack");
    println!("cargo:rustc-link-lib={}", "blas");
}
