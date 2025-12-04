//! Balance validation for rental creation
//!
//! This module provides a validation function to check if a user has sufficient
//! balance before creating a rental. It's called from route handlers rather than
//! as middleware because it needs access to pricing information from the request body.

use crate::error::ApiError;
use basilica_billing::BillingClient;
use rust_decimal::Decimal;
use std::str::FromStr;

/// Validates that a user has sufficient balance to start a rental.
///
/// # Arguments
/// * `billing_client` - Client for billing service
/// * `user_id` - The user's ID
/// * `hourly_cost` - The total hourly cost of the rental (GPU price × GPU count)
///
/// # Returns
/// * `Ok(())` if the user has sufficient balance to cover at least 1 hour
/// * `Err(ApiError::InsufficientBalance)` if balance is too low
///
/// # Graceful Degradation
/// If the billing service is unreachable or returns an error, the request is allowed
/// to proceed (fail open). This prevents billing service outages from blocking all rentals.
pub async fn validate_balance_for_rental(
    billing_client: &BillingClient,
    user_id: &str,
    hourly_cost: Decimal,
) -> Result<(), ApiError> {
    match billing_client.get_balance(user_id).await {
        Ok(balance_response) => match Decimal::from_str(&balance_response.available_balance) {
            Ok(available_balance) => {
                if available_balance < hourly_cost {
                    tracing::warn!(
                        "Blocking rental for user {} with insufficient balance: {} < {}",
                        user_id,
                        available_balance,
                        hourly_cost
                    );
                    return Err(ApiError::InsufficientBalance {
                        message: format!(
                            "Insufficient balance to cover 1 hour of rental (${:.2}/hr)",
                            hourly_cost
                        ),
                        current_balance: balance_response.available_balance.clone(),
                        required: format!("{:.2}", hourly_cost),
                    });
                }

                tracing::debug!(
                    "User {} has sufficient balance: {} >= {}",
                    user_id,
                    available_balance,
                    hourly_cost
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
        },
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
