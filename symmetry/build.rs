fn main() {
    println!("cargo:rerun-if-env-changed=SPGLIB_LIB_DIR");

    let spglib_lib_dir =
        std::env::var("SPGLIB_LIB_DIR").unwrap_or_else(|_| "/opt/spglib/lib".to_string());

    println!("cargo:rustc-link-search=native={}", spglib_lib_dir);
    println!("cargo:rustc-link-lib=symspg");
}
