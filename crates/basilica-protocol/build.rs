//! Build script for basilica-protocol crate.
//!
//! Compiles `.proto` files into Rust code with tonic-build. Generated code
//! is emitted to `OUT_DIR` and included via `tonic::include_proto!` — the
//! files are never written into the source tree.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("descriptor.bin"))
        // Enable serde serialization for select messages
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
        // Add serde support for ComputeType enum
        .type_attribute(
            "ComputeType",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .compile(
            &[
                "proto/common.proto",
                "proto/miner_discovery.proto",
                "proto/validator_api.proto",
                "proto/billing.proto",
                "proto/rental.proto",
                "proto/miner_payouts.proto",
            ],
            &["proto"],
        )?;

    println!("cargo:rerun-if-changed=proto/");

    Ok(())
}
