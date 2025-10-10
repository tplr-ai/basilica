//! Build script for protocol_basilca crate
//!
//! This script compiles .proto files into Rust code using tonic-build.
//! Generated code is placed in src/gen/ directory.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Create the gen directory if it doesn't exist
    std::fs::create_dir_all("src/gen")?;

    // Configure tonic-build with proper module structure
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/gen")
        // Ensure all protobuf types are generated in a flat structure
        .file_descriptor_set_path(out_dir.join("descriptor.bin"))
        .include_file("mod.rs")
        // Enable serde serialization for messages that don't contain timestamps
        .type_attribute(
            "ChallengeParameters",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(
            "ChallengeResult",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(
            "GpuPerformanceBaseline",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(
            "MachineInfo",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute("GpuSpec", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute("CpuSpec", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(
            "MemorySpec",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute("OsInfo", "#[derive(serde::Serialize, serde::Deserialize)]")
        // Add serde support for authentication messages
        .type_attribute(
            "MinerAuthentication",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        // Add serde support for ResourceLimits
        .type_attribute(
            "ResourceLimits",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .compile(
            &[
                "proto/common.proto",
                "proto/miner_discovery.proto",
                "proto/validator_api.proto",
                "proto/gpu_pow.proto",
                "proto/billing.proto",
                "proto/payments.proto",
                "proto/rental.proto",
            ],
            &["proto"],
        )?;

    // Tell cargo to recompile if any proto files change
    println!("cargo:rerun-if-changed=proto/");

    Ok(())
}
