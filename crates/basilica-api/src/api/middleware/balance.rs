use crate::error::ApiError;
use basilica_billing::BillingClient;
use rust_decimal::{prelude::FromPrimitive, Decimal};
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

/// Apply a percentage markup to a per-unit rate.
/// Returns the original rate if the markup percent cannot be parsed.
pub fn apply_markup(rate: Decimal, percent: f64) -> Decimal {
    let multiplier = rust_decimal::Decimal::from_f64(1.0 + (percent / 100.0))
        .unwrap_or(rust_decimal::Decimal::ONE);
    rate * multiplier
}

/// Convenience helper: compute the hourly cost for a rental after markup.
pub fn hourly_cost_with_markup(rate_per_gpu: Decimal, gpu_count: u32, percent: f64) -> Decimal {
    apply_markup(rate_per_gpu, percent) * rust_decimal::Decimal::from(gpu_count.max(1))
}

#[cfg(test)]
mod tests {
    use super::{apply_markup, hourly_cost_with_markup};
    use rust_decimal::Decimal;
    use std::str::FromStr;

    #[test]
    fn applies_positive_markup() {
        let base = Decimal::from_str("19.20").unwrap();
        let marked = apply_markup(base, 10.0);
        assert_eq!(marked, Decimal::from_str("21.12").unwrap());
    }

    #[test]
    fn applies_negative_markup_discount() {
        let base = Decimal::from_str("10.0").unwrap();
        let marked = apply_markup(base, -20.0);
        assert_eq!(marked, Decimal::from_str("8.0").unwrap());
    }

    #[test]
    fn hourly_cost_respects_markup_and_gpu_count() {
        let base = Decimal::from_str("2.50").unwrap();
        let cost = hourly_cost_with_markup(base, 8, 10.0);
        // 2.50 * 1.1 = 2.75; 2.75 * 8 = 22.00
        assert_eq!(cost, Decimal::from_str("22.00").unwrap());
    }
}
