use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    std::fs::create_dir_all(&out_dir).unwrap();

    let services = [
        ("ImeIPC", bee_rpc::ime_service_descriptor()),
        ("AppIPC", bee_rpc::app_service_descriptor()),
    ];

    for (name, descriptor) in services {
        let code = vox_codegen::targets::swift::generate_service(descriptor);
        let out_path = out_dir.join(format!("{name}.swift"));
        std::fs::write(&out_path, &code).unwrap();
        println!("Wrote {}", out_path.display());
    }
}
