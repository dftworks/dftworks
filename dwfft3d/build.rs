fn main() {
    println!("cargo:rustc-link-lib={}", "fftw3");
    println!("cargo:rustc-link-lib={}", "fftw3_threads");
}
