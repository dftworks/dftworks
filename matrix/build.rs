fn main() {
    println!("cargo:rerun-if-env-changed=GFORTRAN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MATRIX_LINK_GFORTRAN");

    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=blas");

    // macOS + MacPorts LAPACK static linkage needs explicit GNU Fortran runtime libs.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let link_gfortran = std::env::var("MATRIX_LINK_GFORTRAN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        if link_gfortran {
            let gfortran_lib_dir = std::env::var("GFORTRAN_LIB_DIR")
                .unwrap_or_else(|_| "/opt/local/lib/libgcc".to_string());

            if std::path::Path::new(&gfortran_lib_dir).exists() {
                println!("cargo:rustc-link-search=native={}", gfortran_lib_dir);
            }

            println!("cargo:rustc-link-lib=gfortran");
            println!("cargo:rustc-link-lib=quadmath");
            println!("cargo:rustc-link-lib=gcc_s");
        }
    }
}
