use subxt::{OnlineClient, PolkadotConfig};

const NETUID: u16 = 387;
const ENDPOINT: &str = "wss://test.finney.opentensor.ai:443/";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing axon discovery methods on netuid {NETUID} (TESTNET)");
    println!("Connecting to: {ENDPOINT}");

    // Initialize the client
    let client = OnlineClient::<PolkadotConfig>::from_url(ENDPOINT).await?;

    // Method 1: Get axon info via metagraph (current implementation)
    println!("\n=== Method 1: Metagraph-based Discovery ===");
    test_metagraph_method(&client).await?;

    // Method 2: Direct get_axon_info extrinsic calls
    println!("\n=== Method 2: Direct get_axon_info Extrinsic ===");
    test_direct_extrinsic(&client).await?;

    Ok(())
}

async fn test_metagraph_method(
    client: &OnlineClient<PolkadotConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use runtime API to get metagraph
    let runtime_api = client.runtime_api().at_latest().await?;

    // Import the generated API types
    use bittensor::api::api;

    let metagraph = runtime_api
        .call(
            api::runtime_apis::subnet_info_runtime_api::SubnetInfoRuntimeApi.get_metagraph(NETUID),
        )
        .await?
        .ok_or("Subnet not found")?;

    println!("Total neurons in metagraph: {}", metagraph.hotkeys.len());
    println!("Total axons in metagraph: {}", metagraph.axons.len());

    // Display first 5 neurons with valid axons
    let mut count = 0;
    for (uid, hotkey) in metagraph.hotkeys.iter().enumerate() {
        if let Some(axon) = metagraph.axons.get(uid) {
            if axon.ip != 0 && axon.port != 0 {
                println!("\nNeuron UID: {uid}");
                println!("  Hotkey: 0x{}", hex::encode(hotkey.0));
                println!("  Raw IP (u128): {}", axon.ip);
                println!("  IP Type: {}", axon.ip_type);
                println!("  Port: {}", axon.port);

                // Try to decode IP - test both methods
                let decoded_ip = if axon.ip_type == 4 {
                    // Method 1: Lower 32 bits (current encoding)
                    let lower_bits = axon.ip as u32;
                    let lower_bytes = lower_bits.to_be_bytes();
                    let lower_ip = format!(
                        "{}.{}.{}.{}",
                        lower_bytes[0], lower_bytes[1], lower_bytes[2], lower_bytes[3]
                    );

                    // Method 2: Upper 32 bits (discovery.rs decoding expectation)
                    let upper_bytes = [
                        (axon.ip >> 120) as u8,
                        (axon.ip >> 112) as u8,
                        (axon.ip >> 104) as u8,
                        (axon.ip >> 96) as u8,
                    ];
                    let upper_ip = format!(
                        "{}.{}.{}.{}",
                        upper_bytes[0], upper_bytes[1], upper_bytes[2], upper_bytes[3]
                    );

                    println!("  Decoded IP (lower 32 bits): {lower_ip}");
                    println!("  Decoded IP (upper 32 bits): {upper_ip}");

                    // Return whichever looks valid
                    if lower_ip != "0.0.0.0" {
                        lower_ip
                    } else {
                        upper_ip
                    }
                } else {
                    format!("IPv6: {:x}", axon.ip)
                };

                println!("  Final decoded IP: {decoded_ip}");

                count += 1;
                if count >= 5 {
                    break;
                }
            }
        }
    }

    if count == 0 {
        println!("No neurons with valid axon info found!");
    }

    Ok(())
}

async fn test_direct_extrinsic(
    client: &OnlineClient<PolkadotConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    // For direct storage access, we'll query the Axons storage map
    // The Axons storage map is: Axons<T>: map hasher(blake2_128_concat) (u16, T::AccountId) => AxonInfoOf

    println!("Fetching axon info directly from storage...");

    // First get a few hotkeys from the metagraph to query directly
    let runtime_api = client.runtime_api().at_latest().await?;
    use bittensor::api::api;

    let metagraph = runtime_api
        .call(
            api::runtime_apis::subnet_info_runtime_api::SubnetInfoRuntimeApi.get_metagraph(NETUID),
        )
        .await?
        .ok_or("Subnet not found")?;

    // Query storage for specific hotkeys
    let mut count = 0;
    for (uid, hotkey) in metagraph.hotkeys.iter().enumerate().take(10) {
        // Skip if no axon in metagraph
        if let Some(axon) = metagraph.axons.get(uid) {
            if axon.ip == 0 || axon.port == 0 {
                continue;
            }

            // Build storage query for Axons(netuid, hotkey)
            let storage_query = api::storage().subtensor_module().axons(NETUID, hotkey);

            // Query the storage
            match client
                .storage()
                .at_latest()
                .await?
                .fetch(&storage_query)
                .await?
            {
                Some(axon_info) => {
                    println!("\nDirect Storage Query for UID {uid}:");
                    println!("  Hotkey: 0x{}", hex::encode(hotkey.0));
                    println!("  Block: {}", axon_info.block);
                    println!("  Version: {}", axon_info.version);
                    println!("  Raw IP (u128): {}", axon_info.ip);
                    println!("  Port: {}", axon_info.port);
                    println!("  IP Type: {}", axon_info.ip_type);
                    println!("  Protocol: {}", axon_info.protocol);

                    // Compare with metagraph data
                    println!("  Metagraph IP: {}", axon.ip);
                    println!("  Match: {}", axon.ip == axon_info.ip);

                    // Try to decode IP address using both methods
                    if axon_info.ip_type == 4 {
                        let lower_bits = axon_info.ip as u32;
                        let lower_bytes = lower_bits.to_be_bytes();
                        let lower_ip = format!(
                            "{}.{}.{}.{}",
                            lower_bytes[0], lower_bytes[1], lower_bytes[2], lower_bytes[3]
                        );

                        let upper_bytes = [
                            (axon_info.ip >> 120) as u8,
                            (axon_info.ip >> 112) as u8,
                            (axon_info.ip >> 104) as u8,
                            (axon_info.ip >> 96) as u8,
                        ];
                        let upper_ip = format!(
                            "{}.{}.{}.{}",
                            upper_bytes[0], upper_bytes[1], upper_bytes[2], upper_bytes[3]
                        );

                        println!("  Decoded IP (lower 32 bits): {lower_ip}");
                        println!("  Decoded IP (upper 32 bits): {upper_ip}");
                    }

                    count += 1;
                    if count >= 5 {
                        break;
                    }
                }
                None => {
                    println!(
                        "\nNo storage entry for UID {} (hotkey: 0x{})",
                        uid,
                        hex::encode(hotkey.0)
                    );
                }
            }
        }
    }

    if count == 0 {
        println!("No axon entries found!");
    }

    Ok(())
}
