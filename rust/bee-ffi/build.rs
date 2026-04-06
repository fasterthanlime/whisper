fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::generate(&crate_dir)
        .expect("cbindgen failed")
        .write_to_file(format!("{crate_dir}/include/bee_ffi.h"));
}
