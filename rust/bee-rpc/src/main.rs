use std::path::PathBuf;

fn main() {
    let descriptor = bee_rpc::bee_ipc_service_descriptor();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("generated");
    std::fs::create_dir_all(&out_dir).unwrap();

    // Generate both client and server Swift bindings
    let code = vox_codegen::targets::swift::generate_service(descriptor);
    let out_path = out_dir.join("BeeIPC.swift");
    std::fs::write(&out_path, &code).unwrap();
    println!("Wrote {}", out_path.display());
}
