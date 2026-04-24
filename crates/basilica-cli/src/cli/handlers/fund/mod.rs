//! `basilica fund` command: picks a funding method (TAO or card) and
//! delegates to the matching submodule.

pub mod card;
pub mod list;
pub mod tao;

use std::io::IsTerminal;

use basilica_sdk::BasilicaClient;
use color_eyre::{eyre::eyre, Result as EyreResult};
use dialoguer::{theme::ColorfulTheme, Select};

use crate::error::CliError;

#[derive(Debug, Clone, Copy)]
enum FundMethod {
    Tao,
    Card,
}

/// Dispatch `basilica fund` based on flags or an interactive picker.
///
/// Flag precedence: `--tao` wins if set; otherwise `--card` implies card.
/// With neither, prompt the user when stdin is a TTY; error out otherwise
/// (including `--json` mode).
pub async fn handle_fund(
    client: &BasilicaClient,
    card: bool,
    tao: bool,
    json: bool,
) -> EyreResult<(), CliError> {
    let method = select_method(card, tao, json)?;
    match method {
        FundMethod::Tao => tao::handle_show_deposit_address(client, json).await,
        FundMethod::Card => card::handle_card_funding(client, json).await,
    }
}

fn select_method(card: bool, tao: bool, json: bool) -> Result<FundMethod, CliError> {
    match (tao, card) {
        (true, true) => Err(CliError::Internal(eyre!(
            "--tao and --card cannot be combined; pass one or the other"
        ))),
        (true, false) => Ok(FundMethod::Tao),
        (false, true) => Ok(FundMethod::Card),
        (false, false) => {
            if json || !std::io::stdin().is_terminal() {
                return Err(CliError::Internal(eyre!(
                    "basilica fund requires --card or --tao when stdin is not a TTY or --json is set"
                )));
            }
            prompt_for_method()
        }
    }
}

fn prompt_for_method() -> Result<FundMethod, CliError> {
    let items = ["Bittensor (TAO)", "Card"];
    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("How do you want to fund?")
        .items(&items)
        .default(0)
        .interact()
        .map_err(|e| CliError::Internal(e.into()))?;
    Ok(if selection == 0 {
        FundMethod::Tao
    } else {
        FundMethod::Card
    })
}
