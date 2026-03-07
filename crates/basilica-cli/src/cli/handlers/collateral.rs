use alloy_primitives::{Address, U256};
use color_eyre::eyre::eyre;
use console::style;
use std::path::Path;
use std::str::FromStr;
use tabled::settings::Style;
use tabled::{Table, Tabled};

use crate::cli::commands::CollateralAction;
use crate::error::CliError;
use crate::progress::{complete_spinner_and_clear, create_spinner};

use collateral_contract::config::{CollateralNetworkConfig, Network};

/// Resolved collateral parameters from CLI flags and config.
struct CollateralParams {
    private_key: String,
    network_config: CollateralNetworkConfig,
    hotkey: Option<String>,
    alpha_hotkey: Option<String>,
}

pub async fn handle_collateral(action: &CollateralAction, key_file: &Path) -> Result<(), CliError> {
    match action {
        CollateralAction::Balance => {
            let params = resolve_params(key_file, None)?;
            handle_balance(&params).await
        }
        CollateralAction::Deposit {
            hotkey,
            node_ip,
            amount,
            yes,
        } => {
            let params = resolve_params(key_file, Some(hotkey))?;
            handle_deposit(&params, node_ip, *amount, *yes).await
        }
        CollateralAction::Status { hotkey, node_id } => {
            let params = resolve_params(key_file, hotkey.as_deref())?;
            handle_status(&params, node_id.as_deref()).await
        }
        CollateralAction::Reclaim { hotkey, node_id } => {
            let params = resolve_params(key_file, Some(hotkey))?;
            handle_reclaim(&params, node_id).await
        }
        CollateralAction::Finalize { request_id } => {
            let params = resolve_params(key_file, None)?;
            handle_finalize(&params, request_id).await
        }
    }
}

// ---------------------------------------------------------------------------
// Param resolution
// ---------------------------------------------------------------------------

fn resolve_params(key_file: &Path, cli_hotkey: Option<&str>) -> Result<CollateralParams, CliError> {
    let private_key = read_private_key(key_file)?;

    // Resolve network: env var → default ("mainnet")
    let network_str =
        std::env::var("BASILICA_COLLATERAL_NETWORK").unwrap_or_else(|_| "mainnet".to_string());
    let network: Network = network_str.parse().map_err(|e| eyre!("{}", e))?;

    // Resolve contract address: env var → built-in default
    let contract_address = std::env::var("BASILICA_COLLATERAL_CONTRACT_ADDRESS").ok();

    let network_config = CollateralNetworkConfig::from_network(&network, contract_address, None)
        .map_err(|e| eyre!("{}", e))?;

    let hotkey = cli_hotkey.map(|s| s.to_string());

    // Resolve alpha_hotkey: env var → fall back to hotkey
    let alpha_hotkey = std::env::var("BASILICA_COLLATERAL_ALPHA_HOTKEY")
        .ok()
        .or_else(|| hotkey.clone());

    Ok(CollateralParams {
        private_key,
        network_config,
        hotkey,
        alpha_hotkey,
    })
}

fn read_private_key(path: &Path) -> Result<String, CliError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| eyre!("Failed to read key file {}: {}", path.display(), e))?;
    let key = content.trim().to_string();
    if key.is_empty() {
        return Err(eyre!("Key file is empty: {}", path.display()).into());
    }
    Ok(key)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_hotkey(input: &str) -> Result<[u8; 32], CliError> {
    let hex_str = input.strip_prefix("0x").unwrap_or(input);
    let bytes = hex::decode(hex_str).map_err(|e| eyre!("Invalid hotkey hex: {}", e))?;
    if bytes.len() != 32 {
        return Err(eyre!("Hotkey must be 32 bytes, got {}", bytes.len()).into());
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

fn parse_node_id(input: &str) -> Result<[u8; 16], CliError> {
    let id = uuid::Uuid::parse_str(input).map_err(|e| eyre!("Invalid node UUID: {}", e))?;
    Ok(*id.as_bytes())
}

fn resolve_node_id_from_ip(ip: &str) -> Result<[u8; 16], CliError> {
    let node_id = basilica_common::node_identity::NodeId::new(ip)
        .map_err(|e| eyre!("Failed to derive node ID from IP '{}': {}", ip, e))?;
    Ok(*node_id.uuid.as_bytes())
}

fn alpha_to_rao(alpha: f64) -> Result<U256, CliError> {
    if alpha <= 0.0 {
        return Err(eyre!("Amount must be positive, got {}", alpha).into());
    }
    let rao = (alpha * 1e9).round() as u128;
    if rao == 0 {
        return Err(eyre!("Amount too small").into());
    }
    Ok(U256::from(rao))
}

fn rao_to_alpha(rao: U256) -> f64 {
    let rao_u128: u128 = rao.try_into().unwrap_or(u128::MAX);
    rao_u128 as f64 / 1e9
}

fn require_hotkey(params: &CollateralParams) -> Result<[u8; 32], CliError> {
    let hk = params
        .hotkey
        .as_deref()
        .ok_or_else(|| eyre!("--hotkey is required"))?;
    parse_hotkey(hk)
}

fn require_alpha_hotkey(params: &CollateralParams) -> Result<[u8; 32], CliError> {
    let ahk = params.alpha_hotkey.as_deref().ok_or_else(|| {
        eyre!("alpha_hotkey is required. Set BASILICA_COLLATERAL_ALPHA_HOTKEY env var")
    })?;
    parse_hotkey(ahk)
}

fn evm_address_from_private_key(private_key: &str) -> Result<Address, CliError> {
    collateral_contract::address_from_private_key(private_key)
        .map_err(|e| eyre!("Invalid private key: {}", e).into())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handle_balance(params: &CollateralParams) -> Result<(), CliError> {
    let spinner = create_spinner("Querying balances...");

    let evm_address = evm_address_from_private_key(&params.private_key)?;

    // Alpha balance (requires alpha_hotkey and the contract's netuid)
    let alpha_balance = if params.alpha_hotkey.is_some() {
        let alpha_hotkey = require_alpha_hotkey(params)?;
        let netuid = collateral_contract::netuid(&params.network_config)
            .await
            .map_err(|e| eyre!("Failed to get netuid: {}", e))?;
        let coldkey = collateral_contract::derive_coldkey(evm_address, &params.network_config)
            .await
            .map_err(|e| eyre!("Failed to derive coldkey: {}", e))?;
        let balance = collateral_contract::get_alpha_balance(
            alpha_hotkey,
            coldkey,
            netuid,
            &params.network_config,
        )
        .await
        .map_err(|e| eyre!("Failed to get alpha balance: {}", e))?;
        Some(balance)
    } else {
        None
    };

    complete_spinner_and_clear(spinner);

    println!("{}", style("Collateral Balances").bold());
    println!("  EVM Address:   {}", evm_address);
    if let Some(alpha) = alpha_balance {
        println!("  Alpha Staked:  {:.2} alpha", rao_to_alpha(alpha));
    } else {
        println!(
            "  Alpha Staked:  {} (set BASILICA_COLLATERAL_ALPHA_HOTKEY env var)",
            style("N/A").dim()
        );
    }

    Ok(())
}

async fn handle_deposit(
    params: &CollateralParams,
    ip: &str,
    amount: f64,
    yes: bool,
) -> Result<(), CliError> {
    let hotkey = require_hotkey(params)?;
    let alpha_hotkey = require_alpha_hotkey(params)?;
    let node_id_bytes = resolve_node_id_from_ip(ip)?;
    let node_uuid = uuid::Uuid::from_bytes(node_id_bytes);
    let rao_amount = alpha_to_rao(amount)?;

    if !yes {
        println!("{}", style("Deposit Summary").bold());
        println!("  Hotkey:       0x{}", hex::encode(hotkey));
        println!("  Node:         {} ({})", ip, node_uuid);
        println!("  Alpha Hotkey: 0x{}", hex::encode(alpha_hotkey));
        println!("  Amount:       {:.2} alpha", amount);
        println!();

        let confirm = dialoguer::Confirm::new()
            .with_prompt("Proceed with deposit?")
            .default(false)
            .interact()
            .map_err(|e| eyre!("Prompt error: {}", e))?;

        if !confirm {
            println!("Deposit cancelled.");
            return Ok(());
        }
    }

    let spinner = create_spinner("Depositing alpha...");

    collateral_contract::deposit(
        &params.private_key,
        hotkey,
        node_id_bytes,
        alpha_hotkey,
        rao_amount,
        &params.network_config,
    )
    .await
    .map_err(|e| eyre!("Deposit failed: {}", e))?;

    complete_spinner_and_clear(spinner);
    println!(
        "{} Deposited {:.2} alpha for node {} ({})",
        style("✓").green().bold(),
        amount,
        ip,
        node_uuid
    );

    Ok(())
}

#[derive(Tabled)]
struct ReclaimRow {
    #[tabled(rename = "Request ID")]
    request_id: String,
    #[tabled(rename = "Hotkey")]
    hotkey: String,
    #[tabled(rename = "Node ID")]
    node_id: String,
    #[tabled(rename = "Alpha")]
    alpha_amount: String,
    #[tabled(rename = "Finalizable At")]
    finalizable_at: String,
    #[tabled(rename = "Status")]
    status: String,
}

#[derive(Tabled)]
struct CollateralRow {
    #[tabled(rename = "Hotkey")]
    hotkey: String,
    #[tabled(rename = "Node ID")]
    node_id: String,
    #[tabled(rename = "Miner")]
    miner: String,
    #[tabled(rename = "Alpha")]
    alpha_collateral: String,
}

async fn handle_status(params: &CollateralParams, node_id: Option<&str>) -> Result<(), CliError> {
    let spinner = create_spinner("Querying collateral status...");

    if let Some(node_id_str) = node_id {
        // Specific node query
        let hotkey = require_hotkey(params)?;
        let node_id_bytes = parse_node_id(node_id_str)?;

        let (_tao, alpha) =
            collateral_contract::collaterals(hotkey, node_id_bytes, &params.network_config)
                .await
                .map_err(|e| eyre!("Failed to query collateral: {}", e))?;

        complete_spinner_and_clear(spinner);

        println!("{}", style("Collateral Status").bold());
        println!("  Hotkey:     0x{}", hex::encode(hotkey));
        println!("  Node ID:    {}", node_id_str);
        println!("  Alpha:      {:.2} alpha", rao_to_alpha(alpha));
    } else {
        // All nodes - optionally filter by hotkey
        let all = collateral_contract::get_all_collaterals(&params.network_config)
            .await
            .map_err(|e| eyre!("Failed to query all collaterals: {}", e))?;

        complete_spinner_and_clear(spinner);

        if all.is_empty() {
            println!("No collateral found.");
        } else {
            let rows: Vec<CollateralRow> = all
                .iter()
                .map(|n| {
                    let node_uuid = uuid::Uuid::from_bytes(n.node_id);
                    CollateralRow {
                        hotkey: format!(
                            "0x{}..{}",
                            &hex::encode(&n.hotkey[..2]),
                            &hex::encode(&n.hotkey[30..])
                        ),
                        node_id: node_uuid.to_string(),
                        miner: format!("{}", n.miner),
                        alpha_collateral: format!("{:.2} alpha", rao_to_alpha(n.alpha_collateral)),
                    }
                })
                .collect();

            println!("{}", style("Collateral Status").bold());
            println!("{}", Table::new(rows).with(Style::modern()));
        }
    }

    // Pending reclaims section
    let reclaim_spinner = create_spinner("Querying pending reclaims...");

    let all_reclaims = collateral_contract::get_all_reclaims(&params.network_config)
        .await
        .map_err(|e| eyre!("Failed to query reclaims: {}", e))?;

    complete_spinner_and_clear(reclaim_spinner);

    println!();
    if all_reclaims.is_empty() {
        println!("No pending reclaims.");
    } else {
        let now = chrono::Utc::now();
        let rows: Vec<ReclaimRow> = all_reclaims
            .iter()
            .map(|r| {
                let node_uuid = uuid::Uuid::from_bytes(r.node_id);
                let finalizable_at =
                    chrono::DateTime::from_timestamp(r.deny_timeout as i64, 0).unwrap_or_default();
                let remaining = finalizable_at.signed_duration_since(now);

                let status = if remaining.num_seconds() <= 0 {
                    style("Ready").green().to_string()
                } else {
                    let hours = remaining.num_hours();
                    let mins = remaining.num_minutes() % 60;
                    format!("Waiting ({}h {}m remaining)", hours, mins)
                };

                ReclaimRow {
                    request_id: r.reclaim_request_id.to_string(),
                    hotkey: format!(
                        "0x{}..{}",
                        &hex::encode(&r.hotkey[..2]),
                        &hex::encode(&r.hotkey[30..])
                    ),
                    node_id: node_uuid.to_string(),
                    alpha_amount: format!("{:.2} alpha", rao_to_alpha(r.alpha_amount)),
                    finalizable_at: finalizable_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    status,
                }
            })
            .collect();

        println!("{}", style("Pending Reclaims").bold());
        println!("{}", Table::new(rows).with(Style::modern()));
    }

    Ok(())
}

async fn handle_reclaim(params: &CollateralParams, node_id: &str) -> Result<(), CliError> {
    let hotkey = require_hotkey(params)?;
    let node_id_bytes = parse_node_id(node_id)?;

    let spinner = create_spinner("Initiating collateral reclaim...");

    let reclaim_info = collateral_contract::reclaim_collateral(
        &params.private_key,
        hotkey,
        node_id_bytes,
        &params.network_config,
    )
    .await
    .map_err(|e| eyre!("Reclaim failed: {}", e))?;

    complete_spinner_and_clear(spinner);
    println!(
        "{} Reclaim initiated for node {}",
        style("✓").green().bold(),
        node_id
    );

    let request_id = reclaim_info.reclaim_request_id;
    let timeout_secs = reclaim_info.deny_timeout;
    let finalizable_at =
        chrono::DateTime::from_timestamp(timeout_secs as i64, 0).unwrap_or_default();
    let now = chrono::Utc::now();
    let remaining = finalizable_at.signed_duration_since(now);

    println!();
    println!("{}", style("Reclaim Details").bold());
    println!("  Request ID:      {}", request_id);
    println!(
        "  Alpha amount:    {:.2} alpha",
        rao_to_alpha(reclaim_info.alpha_amount)
    );
    println!(
        "  Finalizable at:  {} UTC",
        finalizable_at.format("%Y-%m-%d %H:%M:%S")
    );

    if remaining.num_seconds() > 0 {
        let hours = remaining.num_hours();
        let mins = remaining.num_minutes() % 60;
        println!("  Time remaining:  {}h {}m", hours, mins);
    } else {
        println!("  Time remaining:  {}", style("Ready to finalize").green());
    }

    println!();
    println!(
        "Run {} after the timeout to complete the reclaim.",
        style(format!(
            "basilica collateral finalize --request-id {}",
            request_id
        ))
        .cyan()
    );

    Ok(())
}

async fn handle_finalize(params: &CollateralParams, request_id: &str) -> Result<(), CliError> {
    let id = U256::from_str(request_id).map_err(|e| eyre!("Invalid request ID: {}", e))?;

    let spinner = create_spinner("Finalizing reclaim...");

    collateral_contract::finalize_reclaim(&params.private_key, id, &params.network_config)
        .await
        .map_err(|e| eyre!("Finalize failed: {}", e))?;

    complete_spinner_and_clear(spinner);
    println!(
        "{} Reclaim {} finalized",
        style("✓").green().bold(),
        request_id
    );

    Ok(())
}
