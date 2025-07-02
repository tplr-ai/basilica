//! Tool to generate metadata files for different Bittensor networks
//! 
//! Usage: cargo run --bin generate-metadata -- [network]
//! 
//! Networks: finney, test, local

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::process::{Command, Stdio};

use parity_scale_codec::Decode;
use subxt_codegen::syn::parse_quote;
use subxt_codegen::CodegenBuilder;
use subxt_metadata::Metadata;
use subxt_utils_fetchmetadata::{self as fetch_metadata, MetadataVersion};

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let network = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    // Create metadata directory if it doesn't exist
    fs::create_dir_all("metadata").expect("Failed to create metadata directory");

    match network {
        "all" => {
            generate_metadata("finney", "wss://entrypoint-finney.opentensor.ai:443").await;
            generate_metadata("test", "wss://test.finney.opentensor.ai:443").await;
            println!("Skipping local network - requires running local node");
        }
        "finney" => {
            generate_metadata("finney", "wss://entrypoint-finney.opentensor.ai:443").await;
        }
        "test" => {
            generate_metadata("test", "wss://test.finney.opentensor.ai:443").await;
        }
        "local" => {
            let endpoint = env::var("LOCAL_ENDPOINT").unwrap_or_else(|_| "ws://localhost:9944".to_string());
            generate_metadata("local", &endpoint).await;
        }
        _ => {
            eprintln!("Unknown network: {}", network);
            eprintln!("Usage: cargo run --bin generate-metadata -- [network]");
            eprintln!("Networks: finney, test, local, all");
            std::process::exit(1);
        }
    }
}

async fn generate_metadata(network: &str, endpoint: &str) {
    println!("Generating metadata for {} network from {}", network, endpoint);

    // Fetch metadata from the chain
    let metadata_bytes: Vec<u8> = match fetch_metadata::from_url(
        endpoint.try_into().unwrap(),
        MetadataVersion::Latest,
    )
    .await
    {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to fetch metadata for {}: {}", network, e);
            return;
        }
    };

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

    let output_path = format!("metadata/{}.rs", network);
    
    // Try to format the code
    let formatted_code = match Command::new("rustfmt")
        .arg("--edition=2021")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(mut process) => {
            if let Some(stdin) = process.stdin.as_mut() {
                write!(stdin, "{}", code).expect("Failed to write to rustfmt");
            }
            match process.wait_with_output() {
                Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
                Err(_) => code.to_string(),
            }
        }
        Err(_) => code.to_string(),
    };

    // Write the metadata file
    let mut file = File::create(&output_path).expect("Failed to create metadata file");
    write!(file, "{}", formatted_code).expect("Failed to write metadata");

    println!("Generated metadata for {} network at {}", network, output_path);
}