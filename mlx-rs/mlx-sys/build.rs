extern crate cmake;

use cmake::Config;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

fn link_common() {
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol.
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

fn build_and_link_mlx_c() -> PathBuf {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");
    link_common();
    PathBuf::from("src/mlx-c")
}

fn link_from_prefix(prefix: &Path) -> PathBuf {
    let lib_dir = prefix.join("lib");
    let alt_lib_dir = prefix.join("build/lib");
    let include_dir = prefix.join("include");

    if !include_dir.exists() {
        panic!(
            "MLX_SYS_PREFIX={} is missing include directory",
            prefix.display()
        );
    }

    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if alt_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", alt_lib_dir.display());
    }

    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");
    link_common();
    include_dir
}

fn parse_prefix_from_env() -> Option<PathBuf> {
    let value = env::var("MLX_SYS_PREFIX").ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

fn header_paths(include_root: &Path) -> Vec<PathBuf> {
    vec![
        include_root.join("mlx/c/mlx.h"),
        include_root.join("mlx/c/linalg.h"),
        include_root.join("mlx/c/error.h"),
        include_root.join("mlx/c/transforms_impl.h"),
    ]
}

fn emit_rerun_directives() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREFIX");
}

fn main() {
    emit_rerun_directives();

    let include_root = if let Some(prefix) = parse_prefix_from_env() {
        eprintln!(
            "mlx-sys: using prebuilt libraries from MLX_SYS_PREFIX={}",
            prefix.display()
        );
        link_from_prefix(&prefix)
    } else {
        build_and_link_mlx_c()
    };

    let headers = header_paths(&include_root);
    for header in &headers {
        if !header.exists() {
            panic!("missing header for bindgen: {}", header.display());
        }
        println!("cargo:rerun-if-changed={}", header.display());
    }

    // generate bindings
    let mut builder = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .clang_arg(format!("-I{}", include_root.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for header in &headers {
        builder = builder.header(header.to_string_lossy());
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
