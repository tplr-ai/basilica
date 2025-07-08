//! Build script for bittensor crate to generate correct metadata based on network

use std::env;
use std::fs;
use std::path::Path;

/// Configuration data extracted from TOML files
#[derive(Default)]
struct ConfigData {
    network: Option<String>,
    chain_endpoint: Option<String>,
}

/// Try to detect network and endpoint from configuration files
fn detect_config_from_files() -> ConfigData {
    // Try to find config files in common locations
    let config_paths = vec![
        "miner.toml",
        "validator.toml",
        "config/miner.toml",
        "config/validator.toml",
        "../config/miner.toml",
        "../config/validator.toml",
        "../../miner.toml",     // From crates/bittensor to root
        "../../validator.toml", // From crates/bittensor to root
        "../../config/miner.toml",
        "../../config/validator.toml",
    ];

    let mut config = ConfigData::default();

    for path in config_paths {
        if let Ok(contents) = fs::read_to_string(path) {
            println!("cargo:warning=Found and reading configuration from: {path}");

            // Parse TOML to find network and chain_endpoint
            let mut in_bittensor_section = false;

            for line in contents.lines() {
                let line = line.trim();

                // Check for [bittensor] section
                if line == "[bittensor]" {
                    in_bittensor_section = true;
                    continue;
                } else if line.starts_with('[') {
                    in_bittensor_section = false;
                }

                // Look for network in any section or bittensor section
                if line.starts_with("network") && line.contains('=') {
                    if let Some(value) = line.split('=').nth(1) {
                        let network = value
                            .trim()
                            .trim_matches('"')
                            .trim_matches('\'')
                            .to_string();
                        if !network.is_empty() && config.network.is_none() {
                            config.network = Some(network);
                        }
                    }
                }

                // Look for chain_endpoint in bittensor section
                if in_bittensor_section && line.starts_with("chain_endpoint") && line.contains('=')
                {
                    if let Some(value) = line.split('=').nth(1) {
                        // Remove comments and trim
                        let value = if let Some(comment_pos) = value.find('#') {
                            &value[..comment_pos]
                        } else {
                            value
                        };

                        let endpoint = value
                            .trim()
                            .trim_matches('"')
                            .trim_matches('\'')
                            .to_string();
                        if !endpoint.is_empty() && config.chain_endpoint.is_none() {
                            config.chain_endpoint = Some(endpoint);
                        }
                    }
                }
            }

            // If we found what we need, stop searching
            if config.network.is_some() {
                break;
            }
        }
    }

    config
}

fn main() {
    println!("cargo:rerun-if-env-changed=BITTENSOR_NETWORK");
    println!("cargo:rerun-if-changed=metadata/finney.rs");
    println!("cargo:rerun-if-changed=metadata/test.rs");
    println!("cargo:rerun-if-changed=metadata/local.rs");

    // Detect configuration from files
    let config = detect_config_from_files();

    // Determine which network we're building for
    // Priority: 1. Environment variable, 2. Config file, 3. Default to finney
    let network = env::var("BITTENSOR_NETWORK")
        .ok()
        .or(config.network)
        .unwrap_or_else(|| {
            println!("cargo:warning=No network configuration found, defaulting to 'finney'");
            "finney".to_string()
        });

    println!("cargo:warning=Building bittensor metadata for network: {network}");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let metadata_path = Path::new(&out_dir).join("metadata.rs");

    // Copy the appropriate pre-generated metadata file
    let metadata_source = match network.as_str() {
        "test" => "metadata/test.rs",
        "local" => "metadata/local.rs",
        "finney" => "metadata/finney.rs",
        _ => {
            println!("cargo:warning=Unknown network '{network}', using finney metadata");
            "metadata/finney.rs"
        }
    };

    // Copy the metadata file to the output directory
    if let Err(e) = fs::copy(metadata_source, &metadata_path) {
        panic!(
            "Failed to copy metadata file from {metadata_source} to {metadata_path:?}: {e}. \
            Make sure to generate metadata files first using `cargo run --bin generate-metadata`"
        );
    }

    println!("cargo:warning=Using pre-generated metadata from {metadata_source}");
}
