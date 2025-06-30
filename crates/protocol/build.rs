//! Build script for protocol_basilca crate
//!
//! This script compiles .proto files into Rust code using tonic-build.
//! Generated code is placed in src/gen/ directory.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Configure tonic-build with proper module structure
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/gen")
        // Ensure all protobuf types are generated in a flat structure
        .file_descriptor_set_path(out_dir.join("descriptor.bin"))
        .include_file("mod.rs")
        .compile(
            &[
                "proto/common.proto",
                "proto/executor_control.proto",
                "proto/miner_discovery.proto",
                "proto/validator_api.proto",
                "proto/executor_registration.proto",
                "proto/executor_management.proto",
            ],
            &["proto"],
        )?;

    // Tell cargo to recompile if any proto files change
    println!("cargo:rerun-if-changed=proto/");

    Ok(())
}
