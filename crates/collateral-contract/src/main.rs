use alloy_primitives::U256;
use anyhow::Result;
use clap::{Parser, Subcommand};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use collateral_contract::{
    config::{CollateralNetworkConfig, Network},
    CollateralEvent, CollateralEventWithMeta,
};
use hex::FromHex;
use std::collections::HashMap;
use std::str::FromStr;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "collateral-cli")]
#[command(about = "A CLI for interacting with the Collateral contract")]
#[command(version = "1.0")]
struct Cli {
    /// Network to connect to
    #[arg(long, env = "NETWORK", value_enum, default_value = "mainnet")]
    network: Network,

    /// Contract address to use
    #[arg(long, env = "CONTRACT_ADDRESS")]
    contract_address: Option<String>,

    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transaction commands
    #[command(subcommand)]
    Tx(TxCommands),
    /// Query commands
    #[command(subcommand)]
    Query(QueryCommands),
    /// Event scanning commands
    #[command(subcommand)]
    Events(EventCommands),
}

#[derive(Subcommand)]
enum TxCommands {
    /// Deposit collateral for an node (alpha-only tx path; TAO msg.value is intentionally not exposed)
    Deposit {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Node ID as string
        #[arg(long)]
        node_id: String,
        /// Alpha hotkey as hex string (32 bytes). Required when claiming ownership on a new node.
        #[arg(long)]
        alpha_hotkey: String,
        /// Alpha amount to deposit in wei
        #[arg(long)]
        alpha_amount: String,
    },
    /// Reclaim collateral for an node (alpha destination is owner-derived on-chain)
    ReclaimCollateral {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Node ID as string
        #[arg(long)]
        node_id: String,
        /// URL for proof of reclaim
        #[arg(long)]
        url: String,
        /// SHA-256 checksum of URL content as hex string (32 bytes)
        #[arg(long)]
        url_content_sha256: String,
    },
    /// Finalize a reclaim request
    FinalizeReclaim {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Reclaim request ID
        #[arg(long)]
        reclaim_request_id: String,
    },
    /// Deny a reclaim request
    DenyReclaim {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Reclaim request ID
        #[arg(long)]
        reclaim_request_id: String,
        /// URL for proof of denial
        #[arg(long)]
        url: String,
        /// SHA-256 checksum of URL content as hex string (32 bytes)
        #[arg(long)]
        url_content_sha256: String,
    },
    /// Slash collateral for an node (alpha-only tx path; TAO slash amount is intentionally fixed to zero)
    SlashCollateral {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Node ID as string
        #[arg(long)]
        node_id: String,
        /// Alpha amount to slash in wei
        #[arg(long)]
        slash_alpha_amount: String,
        /// URL for proof of slashing
        #[arg(long)]
        url: String,
        /// SHA-256 checksum of URL content as hex string (32 bytes)
        #[arg(long)]
        url_content_sha256: String,
    },
    /// Burn register for the contract hotkey
    BurnRegister {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
    },
    /// Enable or disable TAO deposits
    UpdateTaoDepositsEnabled {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Whether TAO deposits should be enabled
        #[arg(long)]
        enabled: bool,
    },
    /// Enable or disable alpha deposits
    UpdateAlphaDepositsEnabled {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Whether alpha deposits should be enabled
        #[arg(long)]
        enabled: bool,
    },
}

#[derive(Subcommand)]
enum QueryCommands {
    /// Get the contract version
    Version,
    /// Get the network UID
    Netuid,
    /// Get the trustee address
    Trustee,
    /// Get the decision timeout
    DecisionTimeout,
    /// Get the contract coldkey
    ContractColdkey,
    /// Get the contract hotkey
    ContractHotkey,
    /// Get the minimum collateral increase
    MinCollateralIncrease,
    /// Get the minimum alpha collateral increase
    MinAlphaCollateralIncrease,
    /// Get the miner address for an node
    NodeToMiner {
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Node ID as string
        #[arg(long)]
        node_id: String,
    },
    /// Get both TAO and alpha collateral amounts for a node
    Collaterals {
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Node ID as string
        #[arg(long)]
        node_id: String,
    },
    /// Check if TAO deposits are enabled
    TaoDepositsEnabled,
    /// Check if alpha deposits are enabled
    AlphaDepositsEnabled,
    /// Get reclaim details by request ID
    Reclaims {
        /// Reclaim request ID
        #[arg(long)]
        reclaim_request_id: String,
    },
}

#[derive(Subcommand)]
enum EventCommands {
    /// Scan for contract events
    Scan {
        /// Starting block number (defaults to 0)
        #[arg(long, default_value_t = 0)]
        from_block: u64,
        /// Ending block number (defaults to current chain head)
        #[arg(long)]
        to_block: Option<u64>,
        /// Output format: json or pretty
        #[arg(long, default_value = "pretty")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging using the unified system
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let base_filter = format!("basilica_protocol=info,{}", binary_name);
    let default_filter = format!("basilica_protocol=info,{}=info", binary_name);
    basilica_common::logging::init_logging(&cli.verbosity, &base_filter, &default_filter)?;
    let network_config =
        CollateralNetworkConfig::from_network(&cli.network, cli.contract_address, None)?;

    println!("Using network: {:?}", cli.network);
    println!("Contract address: {}", network_config.contract_address);
    println!("RPC URL: {}", network_config.rpc_url);

    match cli.command {
        Commands::Tx(tx_cmd) => handle_tx_command(tx_cmd, &network_config).await,
        Commands::Query(query_cmd) => handle_query_command(query_cmd, &network_config).await,
        Commands::Events(event_cmd) => handle_event_command(event_cmd, &network_config).await,
    }
}

async fn handle_tx_command(
    cmd: TxCommands,
    network_config: &CollateralNetworkConfig,
) -> Result<()> {
    match cmd {
        TxCommands::Deposit {
            private_key,
            hotkey,
            node_id,
            alpha_hotkey,
            alpha_amount,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let node_uuid = Uuid::parse_str(&node_id)?;
            let alpha_hotkey_bytes = parse_hotkey(&alpha_hotkey)?;
            let alpha_amount_u256 = parse_u256(&alpha_amount)?;

            println!(
                "Depositing {} alpha (wei) for node {} with hotkey {} (TAO msg.value is fixed to 0 in this CLI path)",
                alpha_amount, node_id, hotkey
            );
            collateral_contract::deposit(
                &private_key,
                hotkey_bytes,
                node_uuid.into_bytes(),
                alpha_hotkey_bytes,
                alpha_amount_u256,
                network_config,
            )
            .await?;
            println!("Deposit transaction completed successfully!");
        }
        TxCommands::ReclaimCollateral {
            private_key,
            hotkey,
            node_id,
            url,
            url_content_sha256,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let checksum = parse_sha256_checksum(&url_content_sha256)?;
            let node_uuid = Uuid::parse_str(&node_id)?;

            println!(
                "Reclaiming collateral for node {} with hotkey {}",
                node_id, hotkey
            );
            collateral_contract::reclaim_collateral(
                &private_key,
                hotkey_bytes,
                node_uuid.into_bytes(),
                &url,
                checksum,
                network_config,
            )
            .await?;
            println!("Reclaim collateral transaction completed successfully!");
        }
        TxCommands::FinalizeReclaim {
            private_key,
            reclaim_request_id,
        } => {
            let request_id = parse_u256(&reclaim_request_id)?;

            println!("Finalizing reclaim request {}", reclaim_request_id);
            collateral_contract::finalize_reclaim(&private_key, request_id, network_config).await?;
            println!("Finalize reclaim transaction completed successfully!");
        }
        TxCommands::DenyReclaim {
            private_key,
            reclaim_request_id,
            url,
            url_content_sha256,
        } => {
            let request_id = parse_u256(&reclaim_request_id)?;
            let checksum = parse_sha256_checksum(&url_content_sha256)?;

            println!("Denying reclaim request {}", reclaim_request_id);
            collateral_contract::deny_reclaim(
                &private_key,
                request_id,
                &url,
                checksum,
                network_config,
            )
            .await?;
            println!("Deny reclaim transaction completed successfully!");
        }
        TxCommands::SlashCollateral {
            private_key,
            hotkey,
            node_id,
            slash_alpha_amount,
            url,
            url_content_sha256,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let checksum = parse_sha256_checksum(&url_content_sha256)?;
            let node_uuid = Uuid::parse_str(&node_id)?;
            let alpha_amount = parse_u256(&slash_alpha_amount)?;

            println!(
                "Slashing {} alpha (wei) for node {} with hotkey {} (TAO slash amount is fixed to 0 in this CLI path)",
                slash_alpha_amount, node_id, hotkey
            );
            collateral_contract::slash_collateral(
                &private_key,
                hotkey_bytes,
                node_uuid.into_bytes(),
                alpha_amount,
                &url,
                checksum,
                network_config,
            )
            .await?;
            println!("Slash collateral transaction completed successfully!");
        }
        TxCommands::BurnRegister { private_key } => {
            println!("Burning register for contract hotkey");
            collateral_contract::burn_register(&private_key, network_config).await?;
            println!("Burn register completed successfully!");
        }
        TxCommands::UpdateTaoDepositsEnabled {
            private_key,
            enabled,
        } => {
            println!("Setting TAO deposits enabled: {}", enabled);
            collateral_contract::update_tao_deposits_enabled(&private_key, enabled, network_config)
                .await?;
            println!("TAO deposits enabled updated successfully!");
        }
        TxCommands::UpdateAlphaDepositsEnabled {
            private_key,
            enabled,
        } => {
            println!("Setting alpha deposits enabled: {}", enabled);
            collateral_contract::update_alpha_deposits_enabled(
                &private_key,
                enabled,
                network_config,
            )
            .await?;
            println!("Alpha deposits enabled updated successfully!");
        }
    }
    Ok(())
}

async fn handle_query_command(
    cmd: QueryCommands,
    network_config: &CollateralNetworkConfig,
) -> Result<()> {
    match cmd {
        QueryCommands::Version => {
            let result = collateral_contract::get_version(network_config).await?;
            println!("Contract version: {}", result);
        }
        QueryCommands::Netuid => {
            let result = collateral_contract::netuid(network_config).await?;
            println!("Network UID: {}", result);
        }
        QueryCommands::Trustee => {
            let result = collateral_contract::trustee(network_config).await?;
            println!("Trustee address: {}", result);
        }
        QueryCommands::DecisionTimeout => {
            let result = collateral_contract::decision_timeout(network_config).await?;
            println!("Decision timeout: {} seconds", result);
        }
        QueryCommands::ContractColdkey => {
            let result = collateral_contract::contract_coldkey(network_config).await?;
            println!("Contract coldkey: 0x{}", hex::encode(result));
        }
        QueryCommands::ContractHotkey => {
            let result = collateral_contract::validator_hotkey(network_config).await?;
            println!("Validator hotkey: 0x{}", hex::encode(result));
        }
        QueryCommands::MinCollateralIncrease => {
            let result = collateral_contract::min_collateral_increase(network_config).await?;
            println!("Minimum collateral increase: {} wei", result);
        }
        QueryCommands::MinAlphaCollateralIncrease => {
            let result = collateral_contract::min_alpha_collateral_increase(network_config).await?;
            println!("Minimum alpha collateral increase: {}", result);
        }
        QueryCommands::TaoDepositsEnabled => {
            let result = collateral_contract::tao_deposits_enabled(network_config).await?;
            println!("TAO deposits enabled: {}", result);
        }
        QueryCommands::AlphaDepositsEnabled => {
            let result = collateral_contract::alpha_deposits_enabled(network_config).await?;
            println!("Alpha deposits enabled: {}", result);
        }
        QueryCommands::NodeToMiner { hotkey, node_id } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let node_id_clone = node_id.clone();
            let node_uuid = Uuid::parse_str(&node_id)?;
            let result = collateral_contract::node_to_miner(
                hotkey_bytes,
                node_uuid.into_bytes(),
                network_config,
            )
            .await?;
            println!("Miner address for node {}: {}", node_id_clone, result);
        }
        QueryCommands::Collaterals { hotkey, node_id } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let node_id_clone = node_id.clone();
            let node_uuid = Uuid::parse_str(&node_id)?;
            let (tao, alpha) = collateral_contract::collaterals(
                hotkey_bytes,
                node_uuid.into_bytes(),
                network_config,
            )
            .await?;
            println!("Collaterals for node {}:", node_id_clone);
            println!("  TAO:   {} wei", tao);
            println!("  Alpha: {} wei", alpha);
        }
        QueryCommands::Reclaims { reclaim_request_id } => {
            let request_id = parse_u256(&reclaim_request_id)?;
            let result = collateral_contract::reclaims(request_id, network_config).await?;
            println!("Reclaim details for request {}:", reclaim_request_id);
            println!("  Hotkey: {}", hex::encode(result.hotkey));
            println!("  Node ID: {}", Uuid::from_bytes(result.node_id));
            println!("  Miner: {}", result.miner);
            println!("  Amount: {} wei", result.amount);
            println!("  Alpha coldkey: {}", hex::encode(result.alpha_coldkey));
            println!("  Alpha amount: {} wei", result.alpha_amount);
            println!("  Deny timeout: {}", result.deny_timeout);
        }
    }
    Ok(())
}

async fn handle_event_command(
    cmd: EventCommands,
    network_config: &CollateralNetworkConfig,
) -> Result<()> {
    match cmd {
        EventCommands::Scan {
            from_block,
            to_block,
            format,
        } => {
            println!("Scanning events from block {}", from_block);
            let (to_block, events) = match to_block {
                Some(to_block) => {
                    collateral_contract::scan_events_with_scope(
                        from_block,
                        to_block,
                        network_config,
                    )
                    .await?
                }
                None => collateral_contract::scan_events(from_block, network_config).await?,
            };

            println!("Scanned blocks {} to {}", from_block, to_block);

            if format == "json" {
                print_events_json(&events)?;
            } else {
                print_events_pretty(&events);
            }
        }
    }
    Ok(())
}

// Helper functions for parsing inputs

fn parse_hotkey(hotkey: &str) -> Result<[u8; 32]> {
    let hotkey = hotkey.strip_prefix("0x").unwrap_or(hotkey);
    if hotkey.len() != 64 {
        return Err(anyhow::anyhow!(
            "Hotkey must be 32 bytes (64 hex characters)"
        ));
    }
    let bytes = Vec::from_hex(hotkey)?;
    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    Ok(array)
}

fn parse_u256(value: &str) -> Result<U256> {
    Ok(U256::from_str(value)?)
}

fn parse_sha256_checksum(checksum: &str) -> Result<[u8; 32]> {
    let checksum = checksum.strip_prefix("0x").unwrap_or(checksum);
    if checksum.len() != 64 {
        return Err(anyhow::anyhow!(
            "SHA-256 checksum must be 32 bytes (64 hex characters)"
        ));
    }
    let bytes = Vec::from_hex(checksum)?;
    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    Ok(array)
}

fn print_events_pretty(events: &HashMap<u64, Vec<CollateralEventWithMeta>>) {
    if events.is_empty() {
        println!("No events found");
        return;
    }

    for (block_number, block_events) in events {
        println!("\nBlock {}: {} events", block_number, block_events.len());
        for (i, event_with_meta) in block_events.iter().enumerate() {
            println!("  Event {} (tx: {}, log_index: {}):", i + 1, event_with_meta.tx_hash, event_with_meta.log_index);
            let event = &event_with_meta.event;
            match event {
                CollateralEvent::Deposit(deposit) => {
                    println!("    Type: Deposit");
                    println!("    Hotkey: {}", hex::encode(deposit.hotkey.as_slice()));
                    println!("    Node ID: {}", hex::encode(deposit.nodeId.as_slice()));
                    println!("    Miner: {}", deposit.miner);
                    println!(
                        "    Alpha Hotkey: {}",
                        hex::encode(deposit.alphaHotkey.as_slice())
                    );
                    println!("    Alpha Amount: {} wei", deposit.alphaAmount);
                }
                CollateralEvent::ReclaimProcessStarted(reclaim_started) => {
                    println!("    Type: ReclaimProcessStarted");
                    println!("    Request ID: {}", reclaim_started.reclaimRequestId);
                    println!(
                        "    Hotkey: {}",
                        hex::encode(reclaim_started.hotkey.as_slice())
                    );
                    println!(
                        "    Node ID: {}",
                        hex::encode(reclaim_started.nodeId.as_slice())
                    );
                    println!("    Miner: {}", reclaim_started.miner);
                    println!("    TAO Amount: {} wei", reclaim_started.amount);
                    println!(
                        "    Alpha Coldkey: {}",
                        hex::encode(reclaim_started.alphaColdkey.as_slice())
                    );
                    println!("    Alpha Amount: {} wei", reclaim_started.alphaAmount);
                    println!("    Expiration: {}", reclaim_started.expirationTime);
                    println!("    URL: {}", reclaim_started.url);
                    println!(
                        "    URL Content SHA-256: {}",
                        hex::encode(reclaim_started.urlContentSha256.as_slice())
                    );
                }
                CollateralEvent::Denied(denied) => {
                    println!("    Type: Denied");
                    println!("    Request ID: {}", denied.reclaimRequestId);
                    println!("    URL: {}", denied.url);
                    println!(
                        "    URL Content SHA-256: {}",
                        hex::encode(denied.urlContentSha256.as_slice())
                    );
                }
                CollateralEvent::Reclaimed(reclaimed) => {
                    println!("    Type: Reclaimed");
                    println!("    Request ID: {}", reclaimed.reclaimRequestId);
                    println!("    Hotkey: {}", hex::encode(reclaimed.hotkey.as_slice()));
                    println!("    Node ID: {}", hex::encode(reclaimed.nodeId.as_slice()));
                    println!("    Miner: {}", reclaimed.miner);
                    println!(
                        "    Alpha Coldkey: {}",
                        hex::encode(reclaimed.alphaColdkey.as_slice())
                    );
                    println!("    Alpha Amount: {} wei", reclaimed.alphaAmount);
                }
                CollateralEvent::Slashed(slashed) => {
                    println!("    Type: Slashed");
                    println!("    Hotkey: {}", hex::encode(slashed.hotkey.as_slice()));
                    println!("    Node ID: {}", hex::encode(slashed.nodeId.as_slice()));
                    println!("    Miner: {}", slashed.miner);
                    println!("    Alpha Amount: {} wei", slashed.slashAlphaAmount);
                    println!("    URL: {}", slashed.url);
                    println!(
                        "    URL Content SHA-256: {}",
                        hex::encode(slashed.urlContentSha256.as_slice())
                    );
                }
            }
        }
    }
}

fn print_events_json(events: &HashMap<u64, Vec<CollateralEventWithMeta>>) -> Result<()> {
    let mut json_events = serde_json::Map::new();

    for (block_number, block_events) in events {
        let mut json_block_events = Vec::new();

        for event_with_meta in block_events {
            let mut json_event = event_with_meta.event.to_json();
            if let serde_json::Value::Object(ref mut map) = json_event {
                map.insert(
                    "type".to_string(),
                    serde_json::Value::String(event_with_meta.event.event_type().to_string()),
                );
                map.insert(
                    "txHash".to_string(),
                    serde_json::Value::String(event_with_meta.tx_hash.clone()),
                );
                map.insert(
                    "logIndex".to_string(),
                    serde_json::json!(event_with_meta.log_index),
                );
            }
            json_block_events.push(json_event);
        }

        json_events.insert(
            block_number.to_string(),
            serde_json::Value::Array(json_block_events),
        );
    }

    let output = serde_json::Value::Object(json_events);
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
