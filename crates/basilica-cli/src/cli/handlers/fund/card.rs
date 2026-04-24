//! Card funding flow: create a Stripe-backed card purchase via
//! basilica-api, open it in the browser, and poll until it completes or
//! expires.

use std::time::{Duration, Instant};

use basilica_sdk::error::ApiError;
use basilica_sdk::types::{
    CardPurchaseResponse, CardPurchaseStatus, CardPurchaseSummary, CreateCardPurchaseRequest,
};
use basilica_sdk::BasilicaClient;
use color_eyre::{eyre::eyre, Result as EyreResult};
use console::style;
use dialoguer::{theme::ColorfulTheme, Input};
use tracing::warn;
use uuid::Uuid;

use crate::{
    error::CliError,
    output::{format_cents, json_output},
    progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner},
};

/// `amount_cents` bounds the SDK accepts before even hitting the server.
/// Matches basilica-stripe's default config of $1–$1000.
const MIN_USD: u32 = 1;
const MAX_USD: u32 = 1000;

/// How often the poll loop checks purchase status.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Hard ceiling on polling — slightly past Stripe's 30-min default expiry
/// so a session that is still `pending` gets one last tick before we give up.
const POLL_TIMEOUT: Duration = Duration::from_secs(31 * 60);

pub async fn handle_card_funding(
    client: &BasilicaClient,
    amount_usd: Option<u32>,
    json: bool,
) -> EyreResult<(), CliError> {
    let amount_usd = match amount_usd {
        Some(n) => n,
        None => prompt_amount_usd()?,
    };
    let amount_cents = u64::from(amount_usd) * 100;

    let idempotency_key = Uuid::new_v4().to_string();
    let spinner = create_spinner("Creating card purchase...");
    let purchase = match client
        .create_card_purchase(
            &CreateCardPurchaseRequest { amount_cents },
            &idempotency_key,
        )
        .await
    {
        Ok(session) => {
            complete_spinner_and_clear(spinner);
            session
        }
        Err(err) => {
            complete_spinner_error(spinner, "Failed to create card purchase");
            return Err(map_create_error(err));
        }
    };

    if json {
        json_output(&purchase)?;
        return Ok(());
    }

    print_checkout_url(&purchase);
    open_in_browser(&purchase.checkout_url);

    match wait_for_terminal_state(client, &purchase.id).await {
        PollOutcome::Completed(summary) => {
            announce_success(&summary);
            Ok(())
        }
        PollOutcome::Expired => Err(CliError::Internal(eyre!(
            "Card purchase expired before payment completed. Run 'basilica fund' again to start a new purchase."
        ))),
        PollOutcome::Timeout => Err(CliError::Internal(eyre!(
            "Timed out waiting for payment. If you completed the checkout, run 'basilica balance' to see the new balance."
        ))),
        PollOutcome::Cancelled => Err(CliError::Internal(eyre!(
            "Cancelled. If you completed the payment, run 'basilica balance' to see the new balance."
        ))),
    }
}

fn prompt_amount_usd() -> Result<u32, CliError> {
    let theme = ColorfulTheme::default();
    let amount: u32 = Input::with_theme(&theme)
        .with_prompt(format!("Amount in USD (${}-${})", MIN_USD, MAX_USD))
        .validate_with(|input: &u32| {
            if (MIN_USD..=MAX_USD).contains(input) {
                Ok(())
            } else {
                Err("Amount must be between $1 and $1000")
            }
        })
        .interact_text()
        .map_err(|e| CliError::Internal(e.into()))?;
    Ok(amount)
}

fn map_create_error(err: ApiError) -> CliError {
    match &err {
        ApiError::BadRequest { message } => CliError::Internal(eyre!(
            "Card purchase rejected: {message}. Amounts must be whole dollars between $1 and $1000."
        )),
        ApiError::ServiceUnavailable => CliError::Internal(eyre!(
            "Card funding is not available right now. Try 'basilica fund --tao' to fund with TAO instead."
        )),
        _ => CliError::Api(err),
    }
}

fn print_checkout_url(purchase: &CardPurchaseResponse) {
    println!("{}", style("Funding method: Card").bold());
    println!();
    println!("Complete your purchase at:");
    println!("  {}", style(&purchase.checkout_url).cyan().underlined());
    println!();
    println!(
        "  {}: {}",
        style("Amount").dim(),
        style(format_cents(purchase.requested_amount_cents)).bold()
    );
    println!();
}

fn open_in_browser(url: &str) {
    if let Err(err) = webbrowser::open(url) {
        warn!(%err, "failed to open browser; user can paste the URL manually");
    }
}

enum PollOutcome {
    Completed(Box<CardPurchaseSummary>),
    Expired,
    Timeout,
    Cancelled,
}

async fn wait_for_terminal_state(client: &BasilicaClient, purchase_id: &str) -> PollOutcome {
    let spinner = create_spinner("Waiting for payment...");
    let outcome = tokio::select! {
        _ = tokio::signal::ctrl_c() => PollOutcome::Cancelled,
        outcome = poll_loop(client, purchase_id) => outcome,
    };
    match &outcome {
        PollOutcome::Completed(_) => complete_spinner_and_clear(spinner),
        PollOutcome::Expired => complete_spinner_error(spinner, "Card purchase expired"),
        PollOutcome::Timeout => complete_spinner_error(spinner, "Timed out waiting for payment"),
        PollOutcome::Cancelled => complete_spinner_error(spinner, "Cancelled"),
    }
    outcome
}

async fn poll_loop(client: &BasilicaClient, purchase_id: &str) -> PollOutcome {
    let deadline = Instant::now() + POLL_TIMEOUT;
    loop {
        tokio::time::sleep(POLL_INTERVAL).await;
        if Instant::now() >= deadline {
            return PollOutcome::Timeout;
        }
        match client.get_card_purchase(purchase_id).await {
            Ok(summary) => match summary.status {
                CardPurchaseStatus::Completed => return PollOutcome::Completed(Box::new(summary)),
                CardPurchaseStatus::Expired => return PollOutcome::Expired,
                CardPurchaseStatus::Pending | CardPurchaseStatus::Unspecified => continue,
            },
            Err(err) => {
                // Transient failures (network blip, server hiccup) should not
                // abort the loop — Stripe will still move the session forward
                // and the next tick will observe the final state.
                warn!(%err, "poll: transient failure fetching session, will retry");
                continue;
            }
        }
    }
}

fn announce_success(summary: &CardPurchaseSummary) {
    let paid = summary
        .paid_amount_cents
        .unwrap_or(summary.requested_amount_cents);
    println!(
        "{} {} added — run {} to see your credit balance.",
        style("✓").green().bold(),
        style(format_cents(paid)).bold(),
        style("'basilica balance'").yellow()
    );
}
