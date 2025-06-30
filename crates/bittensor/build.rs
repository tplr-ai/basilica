//! Build script for bittensor crate to generate correct metadata based on network

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use parity_scale_codec::Decode;
use subxt_codegen::syn::parse_quote;
use subxt_codegen::CodegenBuilder;
use subxt_metadata::Metadata;
use subxt_utils_fetchmetadata::{self as fetch_metadata, MetadataVersion};

#[tokio::main]
async fn main() {
    println!("cargo:rerun-if-env-changed=BITTENSOR_NETWORK");
    println!("cargo:rerun-if-env-changed=METADATA_CHAIN_ENDPOINT");

    // Determine which network we're building for
    let network = env::var("BITTENSOR_NETWORK").unwrap_or_else(|_| "finney".to_string());

    // Get the appropriate endpoint
    let endpoint = if let Ok(custom_endpoint) = env::var("METADATA_CHAIN_ENDPOINT") {
        custom_endpoint
    } else {
        match network.as_str() {
            "test" | "testnet" => "wss://test.finney.opentensor.ai:443".to_string(),
            "local" => "ws://127.0.0.1:9944".to_string(),
            _ => "wss://entrypoint-finney.opentensor.ai:443".to_string(), // Default to mainnet
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
