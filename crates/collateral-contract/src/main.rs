use alloy_primitives::U256;
use anyhow::Result;
use clap::{Parser, Subcommand};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use collateral_contract::{
    config::{CollateralNetworkConfig, Network},
    CollateralEvent,
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
    #[arg(long, value_enum, default_value = "mainnet")]
    network: Network,

    /// Contract address to use
    #[arg(long)]
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
    /// Deposit collateral for an executor
    Deposit {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Executor ID as string
        #[arg(long)]
        executor_id: String,
        /// Amount to deposit in wei
        #[arg(long)]
        amount: String,
    },
    /// Reclaim collateral for an executor
    ReclaimCollateral {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Executor ID as string
        #[arg(long)]
        executor_id: String,
        /// URL for proof of reclaim
        #[arg(long)]
        url: String,
        /// MD5 checksum of URL content as hex string (16 bytes)
        #[arg(long)]
        url_content_md5_checksum: String,
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
        /// MD5 checksum of URL content as hex string (16 bytes)
        #[arg(long)]
        url_content_md5_checksum: String,
    },
    /// Slash collateral for an executor
    SlashCollateral {
        /// Private key for signing the transaction (hex string)
        #[arg(long, env = "PRIVATE_KEY")]
        private_key: String,
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Executor ID as string
        #[arg(long)]
        executor_id: String,
        /// URL for proof of slashing
        #[arg(long)]
        url: String,
        /// MD5 checksum of URL content as hex string (16 bytes)
        #[arg(long)]
        url_content_md5_checksum: String,
    },
}

#[derive(Subcommand)]
enum QueryCommands {
    /// Get the network UID
    Netuid,
    /// Get the trustee address
    Trustee,
    /// Get the decision timeout
    DecisionTimeout,
    /// Get the minimum collateral increase
    MinCollateralIncrease,
    /// Get the miner address for an executor
    ExecutorToMiner {
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Executor ID as string
        #[arg(long)]
        executor_id: String,
    },
    /// Get the collateral amount for an executor
    Collaterals {
        /// Hotkey as hex string (32 bytes)
        #[arg(long)]
        hotkey: String,
        /// Executor ID as string
        #[arg(long)]
        executor_id: String,
    },
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
        /// Starting block number
        #[arg(long)]
        from_block: u64,
        /// Ending block number
        #[arg(long)]
        to_block: u64,
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
    let network_config = CollateralNetworkConfig::from_network(&cli.network, cli.contract_address)?;

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
            executor_id,
            amount,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let amount_u256 = parse_u256(&amount)?;
            let executor_uuid = Uuid::parse_str(&executor_id)?;

            println!(
                "Depositing {} in wei for executor {} with hotkey {}",
                amount, executor_id, hotkey
            );
            collateral_contract::deposit(
                &private_key,
                hotkey_bytes,
                executor_uuid.into_bytes(),
                amount_u256,
                network_config,
            )
            .await?;
            println!("Deposit transaction completed successfully!");
        }
        TxCommands::ReclaimCollateral {
            private_key,
            hotkey,
            executor_id,
            url,
            url_content_md5_checksum,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let checksum = parse_md5_checksum(&url_content_md5_checksum)?;
            let executor_uuid = Uuid::parse_str(&executor_id)?;

            println!(
                "Reclaiming collateral for executor {} with hotkey {}",
                executor_id, hotkey
            );
            collateral_contract::reclaim_collateral(
                &private_key,
                hotkey_bytes,
                executor_uuid.into_bytes(),
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
            url_content_md5_checksum,
        } => {
            let request_id = parse_u256(&reclaim_request_id)?;
            let checksum = parse_md5_checksum(&url_content_md5_checksum)?;

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
            executor_id,
            url,
            url_content_md5_checksum,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let checksum = parse_md5_checksum(&url_content_md5_checksum)?;
            let executor_uuid = Uuid::parse_str(&executor_id)?;

            println!(
                "Slashing all collateral for executor {} with hotkey {}",
                executor_id, hotkey
            );
            collateral_contract::slash_collateral(
                &private_key,
                hotkey_bytes,
                executor_uuid.into_bytes(),
                &url,
                checksum,
                network_config,
            )
            .await?;
            println!("Slash collateral transaction completed successfully!");
        }
    }
    Ok(())
}

async fn handle_query_command(
    cmd: QueryCommands,
    network_config: &CollateralNetworkConfig,
) -> Result<()> {
    match cmd {
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
        QueryCommands::MinCollateralIncrease => {
            let result = collateral_contract::min_collateral_increase(network_config).await?;
            println!("Minimum collateral increase: {} in wei", result);
        }
        QueryCommands::ExecutorToMiner {
            hotkey,
            executor_id,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let executor_id_clone = executor_id.clone();
            let executor_uuid = Uuid::parse_str(&executor_id)?;
            let result = collateral_contract::executor_to_miner(
                hotkey_bytes,
                executor_uuid.into_bytes(),
                network_config,
            )
            .await?;
            println!(
                "Miner address for executor {}: {}",
                executor_id_clone, result
            );
        }
        QueryCommands::Collaterals {
            hotkey,
            executor_id,
        } => {
            let hotkey_bytes = parse_hotkey(&hotkey)?;
            let executor_id_clone = executor_id.clone();
            let executor_uuid = Uuid::parse_str(&executor_id)?;
            let result = collateral_contract::collaterals(
                hotkey_bytes,
                executor_uuid.into_bytes(),
                network_config,
            )
            .await?;
            println!(
                "Collateral for executor {}: {} in wei",
                executor_id_clone, result
            );
        }
        QueryCommands::Reclaims { reclaim_request_id } => {
            let request_id = parse_u256(&reclaim_request_id)?;
            let result = collateral_contract::reclaims(request_id, network_config).await?;
            println!("Reclaim details for request {}:", reclaim_request_id);
            println!("  Hotkey: {}", hex::encode(result.hotkey));
            println!("  Executor ID: {}", Uuid::from_bytes(result.executor_id));
            println!("  Miner: {}", result.miner);
            println!("  Amount: {} in wei", result.amount);
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
            let (to_block, events) =
                collateral_contract::scan_events_with_scope(from_block, to_block, network_config)
                    .await?;

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

fn parse_md5_checksum(checksum: &str) -> Result<u128> {
    let checksum = checksum.strip_prefix("0x").unwrap_or(checksum);
    if checksum.len() != 32 {
        return Err(anyhow::anyhow!(
            "MD5 checksum must be 16 bytes (32 hex characters)"
        ));
    }
    let bytes = Vec::from_hex(checksum)?;
    let mut array = [0u8; 16];
    array.copy_from_slice(&bytes);
    Ok(u128::from_be_bytes(array))
}

fn print_events_pretty(events: &HashMap<u64, Vec<CollateralEvent>>) {
    if events.is_empty() {
        println!("No events found");
        return;
    }

    for (block_number, block_events) in events {
        println!("\nBlock {}: {} events", block_number, block_events.len());
        for (i, event) in block_events.iter().enumerate() {
            println!("  Event {}:", i + 1);
            match event {
                CollateralEvent::Deposit(deposit) => {
                    println!("    Type: Deposit");
                    println!("    Hotkey: {}", hex::encode(deposit.hotkey.as_slice()));
                    println!(
                        "    Executor ID: {}",
                        hex::encode(deposit.executorId.as_slice())
                    );
                    println!("    Miner: {}", deposit.miner);
                    println!("    Amount: {} in wei", deposit.amount);
                }
                CollateralEvent::Reclaimed(reclaimed) => {
                    println!("    Type: Reclaimed");
                    println!("    Request ID: {}", reclaimed.reclaimRequestId);
                    println!("    Hotkey: {}", hex::encode(reclaimed.hotkey.as_slice()));
                    println!(
                        "    Executor ID: {}",
                        hex::encode(reclaimed.executorId.as_slice())
                    );
                    println!("    Miner: {}", reclaimed.miner);
                    println!("    Amount: {} in wei", reclaimed.amount);
                }
                CollateralEvent::Slashed(slashed) => {
                    println!("    Type: Slashed");
                    println!("    Hotkey: {}", hex::encode(slashed.hotkey.as_slice()));
                    println!(
                        "    Executor ID: {}",
                        hex::encode(slashed.executorId.as_slice())
                    );
                    println!("    Miner: {}", slashed.miner);
                    println!("    Amount: {} in wei", slashed.amount);
                    println!("    URL: {}", slashed.url);
                    println!(
                        "    URL Content MD5: {}",
                        hex::encode(slashed.urlContentMd5Checksum.as_slice())
                    );
                }
            }
        }
    }
}

fn print_events_json(events: &HashMap<u64, Vec<CollateralEvent>>) -> Result<()> {
    let mut json_events = serde_json::Map::new();

    for (block_number, block_events) in events {
        let mut json_block_events = Vec::new();

        for event in block_events {
            let json_event = match event {
                CollateralEvent::Deposit(deposit) => {
                    serde_json::json!({
                        "type": "Deposit",
                        "hotkey": hex::encode(deposit.hotkey.as_slice()),
                        "executorId": hex::encode(deposit.executorId.as_slice()),
                        "miner": deposit.miner.to_string(),
                        "amount": deposit.amount.to_string()
                    })
                }
                CollateralEvent::Reclaimed(reclaimed) => {
                    serde_json::json!({
                        "type": "Reclaimed",
                        "reclaimRequestId": reclaimed.reclaimRequestId.to_string(),
                        "hotkey": hex::encode(reclaimed.hotkey.as_slice()),
                        "executorId": hex::encode(reclaimed.executorId.as_slice()),
                        "miner": reclaimed.miner.to_string(),
                        "amount": reclaimed.amount.to_string()
                    })
                }
                CollateralEvent::Slashed(slashed) => {
                    serde_json::json!({
                        "type": "Slashed",
                        "hotkey": hex::encode(slashed.hotkey.as_slice()),
                        "executorId": hex::encode(slashed.executorId.as_slice()),
                        "miner": slashed.miner.to_string(),
                        "amount": slashed.amount.to_string(),
                        "url": slashed.url,
                        "urlContentMd5Checksum": hex::encode(slashed.urlContentMd5Checksum.as_slice())
                    })
                }
            };
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
