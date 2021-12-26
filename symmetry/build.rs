fn main() {
    println!("cargo:rustc-link-search={}", "/opt/spglib/lib");
    println!("cargo:rustc-link-lib={}", "symspg");
}
