//! Build script for bittensor crate to generate correct metadata based on network

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use parity_scale_codec::Decode;
use subxt_codegen::syn::parse_quote;
use subxt_codegen::CodegenBuilder;
use subxt_metadata::Metadata;
use subxt_utils_fetchmetadata::{self as fetch_metadata, MetadataVersion};

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

#[tokio::main]
async fn main() {
    println!("cargo:rerun-if-env-changed=BITTENSOR_NETWORK");
    println!("cargo:rerun-if-env-changed=METADATA_CHAIN_ENDPOINT");
    println!("cargo:rerun-if-changed=miner.toml");
    println!("cargo:rerun-if-changed=validator.toml");
    println!("cargo:rerun-if-changed=config/miner.toml");
    println!("cargo:rerun-if-changed=config/validator.toml");

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

    // Get the appropriate endpoint
    // Priority: 1. Env var, 2. Config file chain_endpoint, 3. Auto-detect from network
    let endpoint = if let Ok(custom_endpoint) = env::var("METADATA_CHAIN_ENDPOINT") {
        custom_endpoint
    } else if let Some(config_endpoint) = config.chain_endpoint {
        println!("cargo:warning=Using chain_endpoint from config: {config_endpoint}");
        config_endpoint
    } else {
        // Auto-detect endpoint based on network (same logic as BittensorConfig::get_chain_endpoint)
        match network.as_str() {
            "test" => "wss://test.finney.opentensor.ai:443".to_string(),
            "local" => "ws://subtensor:9944".to_string(),
            "finney" | _ => "wss://entrypoint-finney.opentensor.ai:443".to_string(),
        }
    };

    println!(
        "cargo:warning=Building bittensor metadata for network: {network} with endpoint: {endpoint}"
    );

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let metadata_path = Path::new(&out_dir).join("metadata.rs");

    // Fetch metadata from the chain
    let metadata_bytes = fetch_metadata::from_url(
        endpoint.as_str().try_into().unwrap(),
        MetadataVersion::Latest,
    )
    .await
    .expect("Failed to fetch metadata from chain");

    let mut metadata_bytes: &[u8] = &metadata_bytes;
    let metadata = Metadata::decode(&mut metadata_bytes).expect("Failed to decode metadata");

    // Generate code with same configuration as crabtensor
    let mut codegen = CodegenBuilder::new();
    codegen.set_additional_global_derives(vec![parse_quote!(Clone)]);
    codegen.add_derives_for_type(
        parse_quote!(pallet_subtensor::rpc_info::neuron_info::NeuronInfoLite),
        vec![
            parse_quote!(serde::Deserialize),
            parse_quote!(serde::Serialize),
        ],
        true,
    );

    let code = codegen
        .generate(metadata)
        .expect("Failed to generate code from metadata");
    let file_output = File::create(&metadata_path).expect("Failed to create metadata file");

    // Format the generated code
    let mut process = Command::new("rustfmt")
        .stdin(Stdio::piped())
        .stdout(file_output)
        .spawn()
        .unwrap_or_else(|_| {
            // If rustfmt is not available, write unformatted
            let mut file = File::create(&metadata_path).expect("Failed to create metadata file");
            write!(file, "{code}").expect("Failed to write metadata");
            std::process::exit(0);
        });

    if let Some(stdin) = process.stdin.as_mut() {
        write!(stdin, "{code}").expect("Failed to write to rustfmt");
    }

    process.wait().expect("Failed to wait for rustfmt");

    // Create a module file that re-exports the generated metadata
    let mod_path = Path::new(&out_dir).join("mod.rs");
    let mut mod_file = File::create(mod_path).expect("Failed to create mod.rs");
    writeln!(mod_file, "pub mod metadata;").expect("Failed to write mod.rs");
}
