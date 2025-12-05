use crate::error::ApiError;
use basilica_billing::BillingClient;
use rust_decimal::prelude::FromPrimitive;
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

/// Apply a percentage markup to a per-unit rate.
///
/// # Errors
/// Returns `ApiError::Internal` if the percent is NaN, Infinity, or cannot be converted to Decimal.
pub fn apply_markup(rate: Decimal, percent: f64) -> Result<Decimal, ApiError> {
    if percent.is_nan() || percent.is_infinite() {
        tracing::error!("Invalid markup percent: {} (NaN or Infinite)", percent);
        return Err(ApiError::Internal {
            message: format!("Invalid markup percent: {}", percent),
        });
    }
    let multiplier = Decimal::from_f64(1.0 + (percent / 100.0)).ok_or_else(|| {
        tracing::error!(
            "Failed to convert markup multiplier to Decimal: percent={}",
            percent
        );
        ApiError::Internal {
            message: format!("Failed to calculate markup from percent: {}", percent),
        }
    })?;
    Ok(rate * multiplier)
}

/// Convenience helper: compute the hourly cost for a rental after markup.
///
/// # Errors
/// Returns `ApiError::Internal` if the markup calculation fails.
pub fn hourly_cost_with_markup(
    rate_per_gpu: Decimal,
    gpu_count: u32,
    percent: f64,
) -> Result<Decimal, ApiError> {
    Ok(apply_markup(rate_per_gpu, percent)? * Decimal::from(gpu_count.max(1)))
}

#[cfg(test)]
mod tests {
    use super::{apply_markup, hourly_cost_with_markup};
    use rust_decimal::Decimal;
    use std::str::FromStr;

    #[test]
    fn applies_positive_markup() {
        let base = Decimal::from_str("19.20").unwrap();
        let marked = apply_markup(base, 10.0).unwrap();
        assert_eq!(marked, Decimal::from_str("21.12").unwrap());
    }

    #[test]
    fn applies_negative_markup_discount() {
        let base = Decimal::from_str("10.0").unwrap();
        let marked = apply_markup(base, -20.0).unwrap();
        assert_eq!(marked, Decimal::from_str("8.0").unwrap());
    }

    #[test]
    fn hourly_cost_respects_markup_and_gpu_count() {
        let base = Decimal::from_str("2.50").unwrap();
        let cost = hourly_cost_with_markup(base, 8, 10.0).unwrap();
        // 2.50 * 1.1 = 2.75; 2.75 * 8 = 22.00
        assert_eq!(cost, Decimal::from_str("22.00").unwrap());
    }

    #[test]
    fn rejects_nan_markup() {
        let base = Decimal::from_str("10.0").unwrap();
        let result = apply_markup(base, f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_infinity_markup() {
        let base = Decimal::from_str("10.0").unwrap();
        let result = apply_markup(base, f64::INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_negative_infinity_markup() {
        let base = Decimal::from_str("10.0").unwrap();
        let result = apply_markup(base, f64::NEG_INFINITY);
        assert!(result.is_err());
    }
}
