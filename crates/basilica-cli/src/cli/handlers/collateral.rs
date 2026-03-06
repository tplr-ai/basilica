use alloy_primitives::{Address, U256};
use color_eyre::eyre::eyre;
use console::style;
use std::path::Path;
use std::str::FromStr;
use tabled::{Table, Tabled};

use crate::cli::commands::CollateralAction;
use crate::config::CliConfig;
use crate::error::CliError;
use crate::progress::{complete_spinner_and_clear, create_spinner};

use collateral_contract::config::CollateralNetworkConfig;

/// Resolved collateral parameters from CLI flags and config.
struct CollateralParams {
    private_key: String,
    network_config: CollateralNetworkConfig,
    hotkey: Option<String>,
    alpha_hotkey: Option<String>,
}

pub async fn handle_collateral(
    action: &CollateralAction,
    key_file: Option<&Path>,
    network: Option<&str>,
    contract_address: Option<&str>,
    config: &CliConfig,
) -> Result<(), CliError> {
    match action {
        CollateralAction::Balance { alpha_hotkey } => {
            let params = resolve_params(
                key_file,
                network,
                contract_address,
                None,
                alpha_hotkey.as_deref(),
                config,
            )?;
            handle_balance(&params).await
        }
        CollateralAction::Deposit {
            hotkey,
            node_id,
            alpha_hotkey,
            amount,
            yes,
        } => {
            let params = resolve_params(
                key_file,
                network,
                contract_address,
                hotkey.as_deref(),
                alpha_hotkey.as_deref(),
                config,
            )?;
            handle_deposit(&params, node_id, *amount, *yes).await
        }
        CollateralAction::Status { hotkey, node_id } => {
            let params = resolve_params(
                key_file,
                network,
                contract_address,
                hotkey.as_deref(),
                None,
                config,
            )?;
            handle_status(&params, node_id.as_deref()).await
        }
        CollateralAction::Withdraw {
            hotkey,
            node_id,
            url,
            url_hash,
        } => {
            let params = resolve_params(
                key_file,
                network,
                contract_address,
                hotkey.as_deref(),
                None,
                config,
            )?;
            handle_withdraw(&params, node_id, url, url_hash).await
        }
        CollateralAction::Finalize { request_id } => {
            let params = resolve_params(key_file, network, contract_address, None, None, config)?;
            handle_finalize(&params, request_id).await
        }
    }
}

// ---------------------------------------------------------------------------
// Param resolution
// ---------------------------------------------------------------------------

fn resolve_params(
    cli_key_file: Option<&Path>,
    cli_network: Option<&str>,
    cli_contract_address: Option<&str>,
    cli_hotkey: Option<&str>,
    cli_alpha_hotkey: Option<&str>,
    config: &CliConfig,
) -> Result<CollateralParams, CliError> {
    let collateral_cfg = config.collateral.clone().unwrap_or_default();

    let private_key = read_private_key(cli_key_file, collateral_cfg.private_key_file.as_deref())?;

    let network_config = collateral_cfg.to_network_config(cli_network, cli_contract_address)?;

    let hotkey = cli_hotkey.map(|s| s.to_string()).or(collateral_cfg.hotkey);

    let alpha_hotkey = cli_alpha_hotkey
        .map(|s| s.to_string())
        .or(collateral_cfg.alpha_hotkey)
        .or_else(|| hotkey.clone());

    Ok(CollateralParams {
        private_key,
        network_config,
        hotkey,
        alpha_hotkey,
    })
}

/// Read the private key from a file. CLI --key-file takes priority over config.
fn read_private_key(
    cli_key_file: Option<&Path>,
    config_key_file: Option<&Path>,
) -> Result<String, CliError> {
    let path = cli_key_file.or(config_key_file).ok_or_else(|| {
        eyre!("No key file provided. Use --key-file or set collateral.private_key_file in config")
    })?;

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

fn parse_hash(input: &str) -> Result<[u8; 32], CliError> {
    let hex_str = input.strip_prefix("0x").unwrap_or(input);
    let bytes = hex::decode(hex_str).map_err(|e| eyre!("Invalid hash hex: {}", e))?;
    if bytes.len() != 32 {
        return Err(eyre!("Hash must be 32 bytes, got {}", bytes.len()).into());
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

fn alpha_to_rao(alpha: f64) -> Result<U256, CliError> {
    if alpha <= 0.0 {
        return Err(eyre!("Amount must be positive, got {}", alpha).into());
    }
    let rao = (alpha * 1e9).round() as u128;
    if rao == 0 {
        return Err(eyre!("Amount too small, rounds to 0 RAO").into());
    }
    Ok(U256::from(rao))
}

fn rao_to_alpha(rao: U256) -> f64 {
    let rao_u128: u128 = rao.try_into().unwrap_or(u128::MAX);
    rao_u128 as f64 / 1e9
}

fn wei_to_tao(wei: U256) -> f64 {
    let wei_u128: u128 = wei.try_into().unwrap_or(u128::MAX);
    wei_u128 as f64 / 1e18
}

fn require_hotkey(params: &CollateralParams) -> Result<[u8; 32], CliError> {
    let hk = params.hotkey.as_deref().ok_or_else(|| {
        eyre!("--hotkey is required. Provide it via CLI flag or collateral.hotkey in config")
    })?;
    parse_hotkey(hk)
}

fn require_alpha_hotkey(params: &CollateralParams) -> Result<[u8; 32], CliError> {
    let ahk = params.alpha_hotkey.as_deref().ok_or_else(|| {
        eyre!(
            "--alpha-hotkey is required. Provide it via CLI flag or collateral.alpha_hotkey in config"
        )
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

    // TAO balance (wei)
    let tao_balance = collateral_contract::get_tao_balance(evm_address, &params.network_config)
        .await
        .map_err(|e| eyre!("Failed to get TAO balance: {}", e))?;

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
    println!("  TAO Balance:   {:.9} TAO", wei_to_tao(tao_balance));
    if let Some(alpha) = alpha_balance {
        println!("  Alpha Staked:  {:.9} alpha", rao_to_alpha(alpha));
    } else {
        println!(
            "  Alpha Staked:  {} (provide --alpha-hotkey to query)",
            style("N/A").dim()
        );
    }

    Ok(())
}

async fn handle_deposit(
    params: &CollateralParams,
    node_id: &str,
    amount: f64,
    yes: bool,
) -> Result<(), CliError> {
    let hotkey = require_hotkey(params)?;
    let alpha_hotkey = require_alpha_hotkey(params)?;
    let node_id_bytes = parse_node_id(node_id)?;
    let rao_amount = alpha_to_rao(amount)?;

    if !yes {
        println!("{}", style("Deposit Summary").bold());
        println!("  Hotkey:       0x{}", hex::encode(hotkey));
        println!("  Node ID:      {}", node_id);
        println!("  Alpha Hotkey: 0x{}", hex::encode(alpha_hotkey));
        println!("  Amount:       {:.9} alpha ({} RAO)", amount, rao_amount);
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
        "{} Deposited {:.9} alpha for node {}",
        style("✓").green().bold(),
        amount,
        node_id
    );

    Ok(())
}

#[derive(Tabled)]
struct CollateralRow {
    #[tabled(rename = "Hotkey")]
    hotkey: String,
    #[tabled(rename = "Node ID")]
    node_id: String,
    #[tabled(rename = "Miner")]
    miner: String,
    #[tabled(rename = "TAO (wei)")]
    tao_collateral: String,
    #[tabled(rename = "Alpha (RAO)")]
    alpha_collateral: String,
}

async fn handle_status(params: &CollateralParams, node_id: Option<&str>) -> Result<(), CliError> {
    let spinner = create_spinner("Querying collateral status...");

    if let Some(node_id_str) = node_id {
        // Specific node query
        let hotkey = require_hotkey(params)?;
        let node_id_bytes = parse_node_id(node_id_str)?;

        let (tao, alpha) =
            collateral_contract::collaterals(hotkey, node_id_bytes, &params.network_config)
                .await
                .map_err(|e| eyre!("Failed to query collateral: {}", e))?;

        complete_spinner_and_clear(spinner);

        println!("{}", style("Collateral Status").bold());
        println!("  Hotkey:     0x{}", hex::encode(hotkey));
        println!("  Node ID:    {}", node_id_str);
        println!("  TAO:        {:.9} TAO ({} wei)", wei_to_tao(tao), tao);
        println!(
            "  Alpha:      {:.9} alpha ({} RAO)",
            rao_to_alpha(alpha),
            alpha
        );
    } else {
        // All nodes - optionally filter by hotkey
        let all = collateral_contract::get_all_collaterals(&params.network_config)
            .await
            .map_err(|e| eyre!("Failed to query all collaterals: {}", e))?;

        complete_spinner_and_clear(spinner);

        let filtered: Vec<_> = if let Some(ref hk) = params.hotkey {
            let hotkey_bytes = parse_hotkey(hk)?;
            all.into_iter()
                .filter(|n| n.hotkey == hotkey_bytes)
                .collect()
        } else {
            all
        };

        if filtered.is_empty() {
            println!("No collateral found.");
            return Ok(());
        }

        let rows: Vec<CollateralRow> = filtered
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
                    tao_collateral: format!("{:.4} TAO", wei_to_tao(n.tao_collateral)),
                    alpha_collateral: format!("{:.4} alpha", rao_to_alpha(n.alpha_collateral)),
                }
            })
            .collect();

        println!("{}", style("Collateral Status").bold());
        println!("{}", Table::new(rows));
    }

    Ok(())
}

async fn handle_withdraw(
    params: &CollateralParams,
    node_id: &str,
    url: &str,
    url_hash: &str,
) -> Result<(), CliError> {
    let hotkey = require_hotkey(params)?;
    let node_id_bytes = parse_node_id(node_id)?;
    let hash_bytes = parse_hash(url_hash)?;

    let spinner = create_spinner("Initiating collateral reclaim...");

    collateral_contract::reclaim_collateral(
        &params.private_key,
        hotkey,
        node_id_bytes,
        url,
        hash_bytes,
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
