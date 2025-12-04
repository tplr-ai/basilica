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
/// * `Err(ApiError::Internal)` if the billing service fails or balance cannot be parsed
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
                tracing::error!("Failed to parse balance as Decimal: {}", e);
                Err(ApiError::Internal {
                    message: "Failed to parse balance".to_string(),
                })
            }
        },
        Err(e) => {
            tracing::error!("Balance check failed for user {}: {}", user_id, e);
            Err(ApiError::Internal {
                message: "Balance check failed".to_string(),
            })
        }
    }
}
