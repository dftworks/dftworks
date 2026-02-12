fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib/libgcc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/local/lib/mpich-mp");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/spglib/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/lapack/lib");
}
