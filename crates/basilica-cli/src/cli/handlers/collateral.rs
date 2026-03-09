use alloy_primitives::{Address, U256};
use color_eyre::eyre::eyre;
use console::style;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use std::path::Path;
use std::str::FromStr;
use tabled::settings::Style;
use tabled::{Table, Tabled};

use crate::cli::commands::{CollateralAction, SendToken};
use crate::error::CliError;
use crate::progress::{complete_spinner_and_clear, create_spinner};

use collateral_contract::config::{CollateralNetworkConfig, Network};

/// Resolved collateral parameters from CLI flags and config.
struct CollateralParams {
    private_key: String,
    network_config: CollateralNetworkConfig,
    hotkey: Option<String>,
}

pub async fn handle_collateral(action: &CollateralAction, key_file: &Path) -> Result<(), CliError> {
    match action {
        CollateralAction::Balance => {
            let params = resolve_params(key_file, None)?;
            handle_balance(&params).await
        }
        CollateralAction::Receive => {
            let params = resolve_params(key_file, None)?;
            handle_receive(&params).await
        }
        CollateralAction::Deposit {
            hotkey,
            node_ip,
            amount,
            yes,
        } => {
            let params = resolve_params(key_file, hotkey.as_deref())?;
            handle_deposit(&params, node_ip.as_deref(), *amount, *yes).await
        }
        CollateralAction::Status { hotkey, node_id } => {
            let params = resolve_params(key_file, hotkey.as_deref())?;
            handle_status(&params, node_id.as_deref()).await
        }
        CollateralAction::ReclaimStart => {
            let params = resolve_params(key_file, None)?;
            handle_reclaim(&params).await
        }
        CollateralAction::ReclaimFinalize { request_id } => {
            let params = resolve_params(key_file, None)?;
            handle_finalize(&params, request_id.as_deref()).await
        }
        CollateralAction::Send {
            to,
            amount,
            token,
            yes,
        } => {
            let params = resolve_params(key_file, None)?;
            handle_send(&params, to.as_deref(), *amount, *token, *yes).await
        }
    }
}

// ---------------------------------------------------------------------------
// Param resolution
// ---------------------------------------------------------------------------

fn resolve_params(key_file: &Path, cli_hotkey: Option<&str>) -> Result<CollateralParams, CliError> {
    let private_key = read_private_key(key_file)?;

    // Resolve network: env var -> default ("mainnet")
    let network_str =
        std::env::var("BASILICA_COLLATERAL_NETWORK").unwrap_or_else(|_| "mainnet".to_string());
    let network: Network = network_str.parse().map_err(|e| eyre!("{}", e))?;

    // Resolve contract address: env var -> built-in default
    let contract_address = std::env::var("BASILICA_COLLATERAL_CONTRACT_ADDRESS").ok();

    let network_config = CollateralNetworkConfig::from_network(&network, contract_address, None)
        .map_err(|e| eyre!("{}", e))?;

    let hotkey = cli_hotkey.map(|s| s.to_string());

    Ok(CollateralParams {
        private_key,
        network_config,
        hotkey,
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

fn wei_to_tao(wei: U256) -> f64 {
    let wei_u128: u128 = wei.try_into().unwrap_or(u128::MAX);
    wei_u128 as f64 / 1e18
}

fn require_hotkey(params: &CollateralParams) -> Result<[u8; 32], CliError> {
    let hk = params
        .hotkey
        .as_deref()
        .ok_or_else(|| eyre!("--hotkey is required"))?;
    parse_ss58_address(hk)
}

fn evm_address_from_private_key(private_key: &str) -> Result<Address, CliError> {
    collateral_contract::address_from_private_key(private_key)
        .map_err(|e| eyre!("Invalid private key: {}", e).into())
}

fn prompt_input(prompt: &str) -> Result<String, CliError> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact_text()
        .map_err(|e| eyre!("Prompt error: {}", e).into())
}

fn prompt_amount(prompt: &str) -> Result<f64, CliError> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .validate_with(|input: &f64| {
            if *input > 0.0 {
                Ok(())
            } else {
                Err("Amount must be positive")
            }
        })
        .interact_text()
        .map_err(|e| eyre!("Prompt error: {}", e).into())
}

fn format_time_remaining(remaining: &chrono::Duration) -> String {
    if remaining.num_seconds() <= 0 {
        "Ready".to_string()
    } else {
        let hours = remaining.num_hours();
        let mins = remaining.num_minutes() % 60;
        let secs = remaining.num_seconds() % 60;
        if hours > 0 {
            format!("{}h {}m remaining", hours, mins)
        } else if mins > 0 {
            format!("{}m {}s remaining", mins, secs)
        } else {
            format!("{}s remaining", secs)
        }
    }
}

fn hotkey_to_ss58(pubkey: &[u8; 32]) -> String {
    use sp_core::crypto::{AccountId32, Ss58Codec};
    AccountId32::new(*pubkey).to_ss58check()
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handle_balance(params: &CollateralParams) -> Result<(), CliError> {
    let spinner = create_spinner("Querying balances...");

    let evm_address = evm_address_from_private_key(&params.private_key)?;

    let tao_balance = collateral_contract::get_tao_balance(evm_address, &params.network_config)
        .await
        .map_err(|e| eyre!("Failed to get TAO balance: {}", e))?;

    // Alpha balance (fetch validator hotkey from contract)
    let alpha_hotkey = collateral_contract::validator_hotkey(&params.network_config)
        .await
        .map_err(|e| eyre!("Failed to get validator hotkey: {}", e))?;
    let netuid = collateral_contract::netuid(&params.network_config)
        .await
        .map_err(|e| eyre!("Failed to get netuid: {}", e))?;
    let coldkey = collateral_contract::derive_coldkey(evm_address, &params.network_config)
        .await
        .map_err(|e| eyre!("Failed to derive coldkey: {}", e))?;
    let alpha_balance = collateral_contract::get_alpha_balance(
        alpha_hotkey,
        coldkey,
        netuid,
        &params.network_config,
    )
    .await
    .map_err(|e| eyre!("Failed to get alpha balance: {}", e))?;

    complete_spinner_and_clear(spinner);

    println!("{}", style("Collateral Balances").bold());
    println!("  EVM Address:   {}", evm_address);
    println!("  TAO Balance:   {:.4} TAO", wei_to_tao(tao_balance));
    println!("  Alpha Staked:  {:.2} alpha", rao_to_alpha(alpha_balance));

    Ok(())
}

async fn handle_receive(params: &CollateralParams) -> Result<(), CliError> {
    use sp_core::crypto::{AccountId32, Ss58Codec};

    let spinner = create_spinner("Deriving SS58 address...");

    let evm_address = evm_address_from_private_key(&params.private_key)?;
    let coldkey_bytes = collateral_contract::derive_coldkey(evm_address, &params.network_config)
        .await
        .map_err(|e| eyre!("Failed to derive coldkey: {}", e))?;

    let account = AccountId32::new(coldkey_bytes);
    let ss58 = account.to_ss58check();

    complete_spinner_and_clear(spinner);

    println!("{}", style("Receive Address").bold());
    println!("  EVM Address:  {}", evm_address);
    println!("  SS58 Address: {}", ss58);
    println!();
    println!(
        "Send TAO or alpha to {} to fund this EVM wallet.",
        style(&ss58).cyan()
    );

    Ok(())
}

async fn handle_deposit(
    params: &CollateralParams,
    ip: Option<&str>,
    amount: Option<f64>,
    yes: bool,
) -> Result<(), CliError> {
    // Prompt for missing params
    let hotkey = match &params.hotkey {
        Some(hk) => parse_ss58_address(hk)?,
        None => {
            let hk = prompt_input("Miner's Bittensor hotkey")?;
            parse_ss58_address(&hk)?
        }
    };

    let ip_str = match ip {
        Some(ip) => ip.to_string(),
        None => prompt_input("Node IP address")?,
    };

    let amount = match amount {
        Some(a) => a,
        None => prompt_amount("Amount of alpha to deposit")?,
    };

    let alpha_hotkey = collateral_contract::validator_hotkey(&params.network_config)
        .await
        .map_err(|e| eyre!("Failed to get validator hotkey: {}", e))?;
    let node_id_bytes = resolve_node_id_from_ip(&ip_str)?;
    let node_uuid = uuid::Uuid::from_bytes(node_id_bytes);
    let rao_amount = alpha_to_rao(amount)?;

    if !yes {
        println!("{}", style("Deposit Summary").bold());
        println!("  Miner Hotkey:  {}", hotkey_to_ss58(&hotkey));
        println!("  Node:          {} ({})", ip_str, node_uuid);
        println!(
            "  Staked under:  {} (validator)",
            hotkey_to_ss58(&alpha_hotkey)
        );
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
        ip_str,
        node_uuid
    );

    Ok(())
}

#[derive(Tabled)]
struct ReclaimRow {
    #[tabled(rename = "Request ID")]
    request_id: String,
    #[tabled(rename = "Miner Hotkey")]
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
    #[tabled(rename = "Miner Hotkey")]
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
        println!("  Miner Hotkey:  {}", hotkey_to_ss58(&hotkey));
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
                        hotkey: hotkey_to_ss58(&n.miner_hotkey),
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

                let time_str = format_time_remaining(&remaining);
                let status = if remaining.num_seconds() <= 0 {
                    style(&time_str).green().to_string()
                } else {
                    format!("Waiting ({})", time_str)
                };

                ReclaimRow {
                    request_id: r.reclaim_request_id.to_string(),
                    hotkey: hotkey_to_ss58(&r.miner_hotkey),
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

async fn handle_reclaim(params: &CollateralParams) -> Result<(), CliError> {
    let evm_address = evm_address_from_private_key(&params.private_key)?;

    let spinner = create_spinner("Fetching your collaterals...");
    let all = collateral_contract::get_all_collaterals(&params.network_config)
        .await
        .map_err(|e| eyre!("Failed to query collaterals: {}", e))?;
    complete_spinner_and_clear(spinner);

    let mine: Vec<_> = all.iter().filter(|c| c.miner == evm_address).collect();

    if mine.is_empty() {
        return Err(eyre!(
            "No collaterals found for your address ({}). Nothing to reclaim.",
            evm_address
        )
        .into());
    }

    let labels: Vec<String> = mine
        .iter()
        .map(|c| {
            let node_uuid = uuid::Uuid::from_bytes(c.node_id);
            format!(
                "Hotkey: {}  Node: {}  Alpha: {:.2}",
                hotkey_to_ss58(&c.miner_hotkey),
                node_uuid,
                rao_to_alpha(c.alpha_collateral),
            )
        })
        .collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select collateral to reclaim")
        .items(&labels)
        .default(0)
        .interact()
        .map_err(|e| eyre!("Prompt error: {}", e))?;

    let chosen = mine[selection];
    let hotkey = chosen.miner_hotkey;
    let node_id_bytes = chosen.node_id;
    let node_uuid = uuid::Uuid::from_bytes(node_id_bytes);

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
        node_uuid
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

    let time_str = format_time_remaining(&remaining);
    if remaining.num_seconds() > 0 {
        println!("  Time remaining:  {}", time_str);
    } else {
        println!("  Time remaining:  {}", style("Ready to finalize").green());
    }

    println!();
    println!(
        "Run {} after the timeout to complete the reclaim.",
        style(format!(
            "basilica collateral reclaim-finalize --request-id {}",
            request_id
        ))
        .cyan()
    );

    Ok(())
}

async fn handle_finalize(
    params: &CollateralParams,
    request_id: Option<&str>,
) -> Result<(), CliError> {
    let request_id_str = match request_id {
        Some(id) => id.to_string(),
        None => {
            // Fetch pending reclaims and let user select
            let evm_address = evm_address_from_private_key(&params.private_key)?;

            let spinner = create_spinner("Fetching your pending reclaims...");
            let all_reclaims = collateral_contract::get_all_reclaims(&params.network_config)
                .await
                .map_err(|e| eyre!("Failed to query reclaims: {}", e))?;
            complete_spinner_and_clear(spinner);

            let mine: Vec<_> = all_reclaims
                .iter()
                .filter(|r| r.miner == evm_address)
                .collect();

            if mine.is_empty() {
                return Err(eyre!(
                    "No pending reclaims found for your address ({}). Nothing to finalize.",
                    evm_address
                )
                .into());
            }

            let now = chrono::Utc::now();
            let labels: Vec<String> = mine
                .iter()
                .map(|r| {
                    let node_uuid = uuid::Uuid::from_bytes(r.node_id);
                    let finalizable_at = chrono::DateTime::from_timestamp(r.deny_timeout as i64, 0)
                        .unwrap_or_default();
                    let remaining = finalizable_at.signed_duration_since(now);

                    let status = format_time_remaining(&remaining);

                    format!(
                        "ID: {}  Node: {}  Alpha: {:.2}  [{}]",
                        r.reclaim_request_id,
                        node_uuid,
                        rao_to_alpha(r.alpha_amount),
                        status,
                    )
                })
                .collect();

            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select reclaim to finalize")
                .items(&labels)
                .default(0)
                .interact()
                .map_err(|e| eyre!("Prompt error: {}", e))?;

            mine[selection].reclaim_request_id.to_string()
        }
    };

    let id = U256::from_str(&request_id_str).map_err(|e| eyre!("Invalid request ID: {}", e))?;

    let spinner = create_spinner("Finalizing reclaim...");

    collateral_contract::finalize_reclaim(&params.private_key, id, &params.network_config)
        .await
        .map_err(|e| eyre!("Finalize failed: {}", e))?;

    complete_spinner_and_clear(spinner);
    println!(
        "{} Reclaim {} finalized",
        style("✓").green().bold(),
        request_id_str
    );

    Ok(())
}

fn parse_ss58_address(ss58: &str) -> Result<[u8; 32], CliError> {
    use sp_core::crypto::{AccountId32, Ss58Codec};
    let account =
        AccountId32::from_ss58check(ss58).map_err(|e| eyre!("Invalid SS58 address: {:?}", e))?;
    Ok(account.into())
}

fn tao_to_wei(amount: f64) -> Result<U256, CliError> {
    if amount <= 0.0 {
        return Err(eyre!("Amount must be positive, got {}", amount).into());
    }
    let wei = (amount * 1e18).round() as u128;
    if wei == 0 {
        return Err(eyre!("Amount too small").into());
    }
    Ok(U256::from(wei))
}

async fn handle_send(
    params: &CollateralParams,
    to: Option<&str>,
    amount: Option<f64>,
    token: Option<SendToken>,
    yes: bool,
) -> Result<(), CliError> {
    // Prompt for token type if missing
    let token = match token {
        Some(t) => t,
        None => {
            let variants = [SendToken::Tao, SendToken::Alpha];
            let labels: Vec<String> = variants.iter().map(|t| t.to_string()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Token type to send")
                .items(&labels)
                .default(0)
                .interact()
                .map_err(|e| eyre!("Prompt error: {}", e))?;
            variants[selection]
        }
    };

    // Prompt for destination if missing
    let to_str = match to {
        Some(t) => t.to_string(),
        None => prompt_input("Destination SS58 address")?,
    };

    // Prompt for amount if missing
    let amount = match amount {
        Some(a) => a,
        None => {
            let unit = match token {
                SendToken::Tao => "TAO",
                SendToken::Alpha => "alpha",
            };
            prompt_amount(&format!("Amount of {} to send", unit))?
        }
    };

    let destination = parse_ss58_address(&to_str)?;

    match token {
        SendToken::Tao => {
            let amount_wei = tao_to_wei(amount)?;

            if !yes {
                println!("{}", style("Send TAO Summary").bold());
                println!("  Destination: {}", to_str);
                println!("  Amount:      {:.4} TAO", amount);
                println!();

                let confirm = dialoguer::Confirm::new()
                    .with_prompt("Proceed with TAO transfer?")
                    .default(false)
                    .interact()
                    .map_err(|e| eyre!("Prompt error: {}", e))?;

                if !confirm {
                    println!("Transfer cancelled.");
                    return Ok(());
                }
            }

            let spinner = create_spinner("Sending TAO...");

            collateral_contract::send_tao(
                &params.private_key,
                destination,
                amount_wei,
                &params.network_config,
            )
            .await
            .map_err(|e| eyre!("TAO transfer failed: {}", e))?;

            complete_spinner_and_clear(spinner);
            println!(
                "{} Sent {:.4} TAO to {}",
                style("✓").green().bold(),
                amount,
                to_str
            );
        }
        SendToken::Alpha => {
            let hotkey = collateral_contract::validator_hotkey(&params.network_config)
                .await
                .map_err(|e| eyre!("Failed to get validator hotkey: {}", e))?;
            let netuid = collateral_contract::netuid(&params.network_config)
                .await
                .map_err(|e| eyre!("Failed to get netuid: {}", e))?;
            let amount_rao = alpha_to_rao(amount)?;

            if !yes {
                println!("{}", style("Send Alpha Summary").bold());
                println!("  Destination: {}", to_str);
                println!("  Staked under: {} (validator)", hotkey_to_ss58(&hotkey));
                println!("  Netuid:      {}", netuid);
                println!("  Amount:      {:.2} alpha", amount);
                println!();

                let confirm = dialoguer::Confirm::new()
                    .with_prompt("Proceed with alpha transfer?")
                    .default(false)
                    .interact()
                    .map_err(|e| eyre!("Prompt error: {}", e))?;

                if !confirm {
                    println!("Transfer cancelled.");
                    return Ok(());
                }
            }

            let spinner = create_spinner("Sending alpha...");

            collateral_contract::send_alpha(
                &params.private_key,
                destination,
                hotkey,
                netuid,
                amount_rao,
                &params.network_config,
            )
            .await
            .map_err(|e| eyre!("Alpha transfer failed: {}", e))?;

            complete_spinner_and_clear(spinner);
            println!(
                "{} Sent {:.2} alpha to {}",
                style("✓").green().bold(),
                amount,
                to_str
            );
        }
    }

    Ok(())
}
