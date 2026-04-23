use basilica_sdk::BasilicaClient;
use color_eyre::{eyre::eyre, Help, Result as EyreResult};
use console::style;

use crate::{
    error::CliError,
    output::{json_output, print_info, table_output},
    progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner},
};

/// Handle the main fund command - show deposit address
pub async fn handle_show_deposit_address(
    client: &BasilicaClient,
    json: bool,
) -> EyreResult<(), CliError> {
    let spinner = create_spinner("Fetching deposit account...");

    // Try to get existing deposit account first
    let account = match client.get_deposit_account().await {
        Ok(account) if account.exists => {
            complete_spinner_and_clear(spinner);
            account
        }
        Ok(_) | Err(_) => {
            // Account doesn't exist or error getting it, try to create one
            complete_spinner_and_clear(spinner.clone());

            let spinner = create_spinner("Creating deposit account...");
            match client.create_deposit_account().await {
                Ok(created) => {
                    complete_spinner_and_clear(spinner);
                    basilica_sdk::DepositAccountResponse {
                        user_id: created.user_id,
                        address: created.address,
                        exists: true,
                    }
                }
                Err(e) => {
                    complete_spinner_error(spinner, "Failed to create deposit account");
                    return Err(CliError::Internal(
                        eyre!(e)
                            .suggestion("Check your authentication and try again")
                            .note("Ensure you are logged in with 'basilica login'"),
                    ));
                }
            }
        }
    };

    if json {
        json_output(&account)?;
    } else {
        println!("{}", style("Funding method: Bittensor (TAO)").bold());
        println!();
        println!("Send TAO to your unique deposit address:");
        println!(
            "  {}: {} (ss58, network=finney)",
            style("Address").cyan(),
            style(&account.address).green().bold()
        );
        println!("  {}: 0.1 TAO", style("Min amount").cyan());
        println!("  {}: 12", style("Confirmations required").cyan());
        println!();
        println!(
            "{}",
            style("Tip: you can send multiple transactions; we'll credit after 12 confirmations.")
                .dim()
                .italic()
        );
        println!();
        println!("Track status:");
        println!("  {}", style("basilica fund list").yellow());
    }

    Ok(())
}

/// Handle the fund list subcommand - show deposit history
pub async fn handle_list_deposits(
    client: &BasilicaClient,
    limit: u32,
    offset: u32,
    json: bool,
) -> EyreResult<(), CliError> {
    let spinner = create_spinner("Loading deposit history...");

    let deposits_response = client
        .list_deposits(Some(limit), Some(offset))
        .await
        .map_err(|e| {
            complete_spinner_error(spinner.clone(), "Failed to fetch deposits");
            CliError::Internal(
                eyre!(e)
                    .suggestion("Check your authentication and try again")
                    .note("If this persists, the payments service may be temporarily unavailable"),
            )
        })?;

    complete_spinner_and_clear(spinner);

    if json {
        json_output(&deposits_response)?;
    } else if deposits_response.deposits.is_empty() {
        print_info("No deposits found for your account");
        println!();
        println!("{}", style("Quick Commands:").cyan().bold());
        println!(
            "  {} {}",
            style("basilica fund").yellow().bold(),
            style("- Add TAO credits to your account").dim()
        );
        println!(
            "  {} {}",
            style("basilica balance").yellow().bold(),
            style("- Show your current credit balance").dim()
        );
    } else {
        table_output::display_deposits(&deposits_response)?;
    }

    Ok(())
}
