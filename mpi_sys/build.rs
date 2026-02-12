fn main() {
    println!("cargo:rustc-link-search=native=/opt/local/lib");
    println!("cargo:rustc-link-search=native=/opt/local/lib/mpich-mp");
}
