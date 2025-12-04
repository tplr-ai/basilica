//! Balance validation for rental creation
//!
//! This module provides a validation function to check if a user has sufficient
//! balance before creating a rental. It's called from route handlers rather than
//! as middleware because it needs access to pricing information from the request body.

use crate::error::ApiError;
use basilica_billing::BillingClient;
use rust_decimal::Decimal;
use std::str::FromStr;

/// Minimum balance required to start a rental (in USD)
const MIN_BALANCE_USD: f64 = 10.0;

/// Validates that a user has sufficient balance to start a rental.
///
/// # Arguments
/// * `billing_client` - Client for billing service
/// * `user_id` - The user's ID
/// * `_hourly_cost` - The hourly cost of the rental (for future: require balance >= 1hr cost)
///
/// # Returns
/// * `Ok(())` if the user has sufficient balance
/// * `Err(ApiError::InsufficientBalance)` if balance is too low
///
/// # Graceful Degradation
/// If the billing service is unreachable or returns an error, the request is allowed
/// to proceed (fail open). This prevents billing service outages from blocking all rentals.
pub async fn validate_balance_for_rental(
    billing_client: &BillingClient,
    user_id: &str,
    _hourly_cost: Option<Decimal>,
) -> Result<(), ApiError> {
    match billing_client.get_balance(user_id).await {
        Ok(balance_response) => {
            match Decimal::from_str(&balance_response.available_balance) {
                Ok(available_balance) => {
                    let min_balance =
                        Decimal::from_f64_retain(MIN_BALANCE_USD).unwrap_or(Decimal::ZERO);

                    if available_balance < min_balance {
                        tracing::warn!(
                            "Blocking rental for user {} with insufficient balance: {} < {}",
                            user_id,
                            available_balance,
                            min_balance
                        );
                        return Err(ApiError::InsufficientBalance {
                            message: "Your account balance is below the minimum required to create rentals".to_string(),
                            current_balance: balance_response.available_balance.clone(),
                            required: MIN_BALANCE_USD.to_string(),
                        });
                    }

                    tracing::debug!(
                        "User {} has sufficient balance: {} >= {}",
                        user_id,
                        available_balance,
                        min_balance
                    );
                    Ok(())
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse balance as Decimal: {}. Allowing request to proceed.",
                        e
                    );
                    Ok(())
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                "Balance check failed for user {}: {}. Allowing request to proceed (graceful degradation).",
                user_id,
                e
            );
            Ok(())
        }
    }
}
