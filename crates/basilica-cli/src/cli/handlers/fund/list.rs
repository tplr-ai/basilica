//! `basilica fund list`: show TAO deposit history and card payment history
//! side by side. The two sources are fetched in parallel so a partial
//! outage of either still surfaces the data that did come back.

use basilica_sdk::types::{ListCardPurchasesResponse, ListDepositsResponse};
use basilica_sdk::BasilicaClient;
use color_eyre::{eyre::eyre, Help, Result as EyreResult};
use console::style;
use serde::Serialize;

use crate::{
    error::CliError,
    output::{json_output, print_info, print_warning, table_output},
    progress::{complete_spinner_and_clear, create_spinner},
};

type SourceResult<T> = Result<T, String>;

pub async fn handle_list_payments(
    client: &BasilicaClient,
    limit: u32,
    offset: u32,
    output: crate::cli::commands::ResolvedOutput,
) -> EyreResult<(), CliError> {
    let spinner = create_spinner("Loading payment history...");

    let card_limit = i32::try_from(limit).unwrap_or(i32::MAX);
    let card_offset = i32::try_from(offset).unwrap_or(i32::MAX);
    let (tao_result, card_result) = tokio::join!(
        client.list_deposits(Some(limit), Some(offset)),
        client.list_card_purchases(Some(card_limit), Some(card_offset)),
    );

    complete_spinner_and_clear(spinner);

    let tao = tao_result.map_err(|e| e.to_string());
    let card = card_result.map_err(|e| e.to_string());

    if output.is_json() {
        render_json(&tao, &card)
    } else {
        render_text(&tao, &card, output)
    }
}

fn render_text(
    tao: &SourceResult<ListDepositsResponse>,
    card: &SourceResult<ListCardPurchasesResponse>,
    output: crate::cli::commands::ResolvedOutput,
) -> EyreResult<(), CliError> {
    if let (Err(tao_err), Err(card_err)) = (tao, card) {
        return Err(CliError::Internal(
            eyre!("Failed to load payment history")
                .note(format!("TAO: {tao_err}"))
                .note(format!("Card: {card_err}"))
                .suggestion("Check your authentication and try again"),
        ));
    }

    let tao_empty = matches!(tao, Ok(r) if r.deposits.is_empty());
    let card_empty = matches!(card, Ok(r) if r.purchases.is_empty());

    match tao {
        Ok(resp) if !resp.deposits.is_empty() => table_output::display_deposits(resp, output)?,
        Ok(_) => {
            if card.is_err() {
                print_info("No TAO deposits found yet");
            }
        }
        Err(e) => print_warning(&format!("Could not load TAO deposits: {e}")),
    }

    match card {
        Ok(resp) if !resp.purchases.is_empty() => {
            table_output::display_card_purchases(resp, output)?
        }
        Ok(_) => {
            if tao.is_err() {
                print_info("No card payments found yet");
            }
        }
        Err(e) => print_warning(&format!("Could not load card payments: {e}")),
    }

    if tao_empty && card_empty {
        print_empty_state();
    }

    Ok(())
}

fn render_json(
    tao: &SourceResult<ListDepositsResponse>,
    card: &SourceResult<ListCardPurchasesResponse>,
) -> EyreResult<(), CliError> {
    let payload = PaymentHistoryJson {
        deposits: tao.as_ref().ok(),
        card_purchases: card.as_ref().ok(),
        errors: PaymentHistoryErrors {
            tao: tao.as_ref().err().cloned(),
            card: card.as_ref().err().cloned(),
        },
    };

    json_output(&payload)?;

    if tao.is_err() && card.is_err() {
        return Err(CliError::Internal(
            eyre!("Failed to load payment history")
                .note(format!("TAO: {}", tao.as_ref().err().unwrap()))
                .note(format!("Card: {}", card.as_ref().err().unwrap())),
        ));
    }

    Ok(())
}

fn print_empty_state() {
    print_info("No deposits or card payments found for your account");
    println!();
    println!("{}", style("Quick Commands:").cyan().bold());
    println!(
        "  {} {}",
        style("basilica fund").yellow().bold(),
        style("- Add credits (TAO or card)").dim()
    );
    println!(
        "  {} {}",
        style("basilica balance").yellow().bold(),
        style("- Show your current credit balance").dim()
    );
}

#[derive(Serialize)]
struct PaymentHistoryJson<'a> {
    deposits: Option<&'a ListDepositsResponse>,
    card_purchases: Option<&'a ListCardPurchasesResponse>,
    #[serde(skip_serializing_if = "PaymentHistoryErrors::is_empty")]
    errors: PaymentHistoryErrors,
}

#[derive(Serialize, Default)]
struct PaymentHistoryErrors {
    #[serde(skip_serializing_if = "Option::is_none")]
    tao: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    card: Option<String>,
}

impl PaymentHistoryErrors {
    fn is_empty(&self) -> bool {
        self.tao.is_none() && self.card.is_none()
    }
}
