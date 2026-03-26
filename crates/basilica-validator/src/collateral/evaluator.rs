use crate::collateral::grace_tracker::GracePeriodTracker;
use crate::config::collateral::CollateralConfig;
use anyhow::Result;
use chrono::Duration;
use rust_decimal::Decimal;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum CollateralState {
    Sufficient {
        current_usd: Decimal,
        minimum_usd: Decimal,
    },
    Warning {
        current_usd: Decimal,
        minimum_usd: Decimal,
    },
    Undercollateralized {
        current_usd: Decimal,
        minimum_usd: Decimal,
        grace_remaining: Duration,
    },
    Excluded {
        current_usd: Decimal,
        minimum_usd: Decimal,
        reason: String,
    },
    Unknown {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct CollateralStatus {
    pub current_alpha: Decimal,
    pub current_usd_value: Decimal,
    pub minimum_usd_required: Decimal,
    pub status: String,
    pub grace_period_remaining: Option<Duration>,
    pub action_required: Option<String>,
    pub alpha_usd_price: Option<Decimal>,
    pub price_stale: bool,
}

pub struct CollateralEvaluator {
    config: CollateralConfig,
    grace_tracker: Arc<GracePeriodTracker>,
}

impl CollateralEvaluator {
    pub fn new(config: CollateralConfig, grace_tracker: Arc<GracePeriodTracker>) -> Self {
        Self {
            config,
            grace_tracker,
        }
    }

    pub fn get_minimum_usd(&self, gpu_category: &str, gpu_count: u32) -> Decimal {
        let key = gpu_category.trim().to_uppercase();
        let per_gpu = self
            .config
            .minimum_usd_per_gpu
            .get(&key)
            .or_else(|| self.config.minimum_usd_per_gpu.get("DEFAULT"))
            .copied()
            .unwrap_or(Decimal::ZERO);
        per_gpu * Decimal::from(gpu_count)
    }

    pub async fn evaluate(
        &self,
        hotkey: &str,
        node_id: &str,
        gpu_category: &str,
        gpu_count: u32,
        collateral_alpha: Decimal,
        alpha_price_usd: Option<Decimal>,
    ) -> Result<(CollateralState, CollateralStatus)> {
        let minimum_usd = self.get_minimum_usd(gpu_category, gpu_count);

        if minimum_usd <= Decimal::ZERO {
            let reason = "minimum_usd is not configured".to_string();
            return Ok((
                CollateralState::Unknown {
                    reason: reason.clone(),
                },
                CollateralStatus {
                    current_alpha: collateral_alpha,
                    current_usd_value: Decimal::ZERO,
                    minimum_usd_required: minimum_usd,
                    status: "unknown".to_string(),
                    grace_period_remaining: None,
                    action_required: Some(reason),
                    alpha_usd_price: None,
                    price_stale: true,
                },
            ));
        }

        let (current_usd, price_stale, alpha_usd_price) = match alpha_price_usd {
            Some(price) if price > Decimal::ZERO => {
                let usd = collateral_alpha * price;
                (usd, false, Some(price))
            }
            _ => {
                if self.config.exclude_on_prolonged_price_failure {
                    return self
                        .handle_price_unavailable(hotkey, node_id, minimum_usd, collateral_alpha)
                        .await;
                }
                let reason = "Alpha price unavailable".to_string();
                return Ok((
                    CollateralState::Unknown {
                        reason: reason.clone(),
                    },
                    CollateralStatus {
                        current_alpha: collateral_alpha,
                        current_usd_value: Decimal::ZERO,
                        minimum_usd_required: minimum_usd,
                        status: "unknown".to_string(),
                        grace_period_remaining: None,
                        action_required: Some(reason),
                        alpha_usd_price: None,
                        price_stale: true,
                    },
                ));
            }
        };

        let warning_threshold = minimum_usd * self.config.warning_threshold_multiplier;

        if current_usd >= warning_threshold {
            self.grace_tracker
                .clear_undercollateralized(hotkey, node_id)
                .await?;
            return Ok((
                CollateralState::Sufficient {
                    current_usd,
                    minimum_usd,
                },
                CollateralStatus {
                    current_alpha: collateral_alpha,
                    current_usd_value: current_usd,
                    minimum_usd_required: minimum_usd,
                    status: "sufficient".to_string(),
                    grace_period_remaining: None,
                    action_required: None,
                    alpha_usd_price,
                    price_stale,
                },
            ));
        }

        if current_usd >= minimum_usd {
            self.grace_tracker
                .clear_undercollateralized(hotkey, node_id)
                .await?;
            let action_required = self.action_required_warning(
                warning_threshold,
                current_usd,
                alpha_usd_price.unwrap_or(Decimal::ZERO),
            );
            return Ok((
                CollateralState::Warning {
                    current_usd,
                    minimum_usd,
                },
                CollateralStatus {
                    current_alpha: collateral_alpha,
                    current_usd_value: current_usd,
                    minimum_usd_required: minimum_usd,
                    status: "warning".to_string(),
                    grace_period_remaining: None,
                    action_required,
                    alpha_usd_price,
                    price_stale,
                },
            ));
        }

        if self
            .grace_tracker
            .get_since(hotkey, node_id)
            .await?
            .is_none()
        {
            self.grace_tracker
                .mark_undercollateralized(hotkey, node_id)
                .await?;
        }

        let grace_remaining = self
            .grace_tracker
            .get_grace_remaining(hotkey, node_id)
            .await?
            .unwrap_or_else(Duration::zero);

        if grace_remaining <= Duration::zero() {
            return Ok((
                CollateralState::Excluded {
                    current_usd,
                    minimum_usd,
                    reason: "grace_period_expired".to_string(),
                },
                CollateralStatus {
                    current_alpha: collateral_alpha,
                    current_usd_value: current_usd,
                    minimum_usd_required: minimum_usd,
                    status: "excluded".to_string(),
                    grace_period_remaining: Some(Duration::zero()),
                    action_required: Some("Deposit collateral to restore eligibility".to_string()),
                    alpha_usd_price,
                    price_stale,
                },
            ));
        }

        let action_required = self.action_required_urgent(
            minimum_usd,
            current_usd,
            alpha_usd_price.unwrap_or(Decimal::ZERO),
        );
        Ok((
            CollateralState::Undercollateralized {
                current_usd,
                minimum_usd,
                grace_remaining,
            },
            CollateralStatus {
                current_alpha: collateral_alpha,
                current_usd_value: current_usd,
                minimum_usd_required: minimum_usd,
                status: "undercollateralized".to_string(),
                grace_period_remaining: Some(grace_remaining),
                action_required,
                alpha_usd_price,
                price_stale,
            },
        ))
    }

    fn action_required_warning(
        &self,
        warning_threshold: Decimal,
        current_usd: Decimal,
        alpha_usd_price: Decimal,
    ) -> Option<String> {
        let needed_usd = if warning_threshold > current_usd {
            warning_threshold - current_usd
        } else {
            Decimal::ZERO
        };
        if needed_usd <= Decimal::ZERO {
            return None;
        }
        if alpha_usd_price <= Decimal::ZERO {
            return Some("Alpha price unavailable; cannot estimate top-up".to_string());
        }
        let needed_alpha = (needed_usd / alpha_usd_price).round_dp(2);
        let needed_usd = needed_usd.round_dp(2);
        Some(format!(
            "Deposit {:.2} Alpha (~${:.2}) to reach safe level (1.5x minimum)",
            needed_alpha, needed_usd
        ))
    }

    fn action_required_urgent(
        &self,
        minimum_usd: Decimal,
        current_usd: Decimal,
        alpha_usd_price: Decimal,
    ) -> Option<String> {
        let needed_usd = if minimum_usd > current_usd {
            minimum_usd - current_usd
        } else {
            Decimal::ZERO
        };
        if needed_usd <= Decimal::ZERO {
            return None;
        }
        if alpha_usd_price <= Decimal::ZERO {
            return Some("Alpha price unavailable; cannot estimate top-up".to_string());
        }
        let needed_alpha = (needed_usd / alpha_usd_price).round_dp(2);
        let needed_usd = needed_usd.round_dp(2);
        Some(format!(
            "URGENT: Deposit {:.2} Alpha (~${:.2}) within grace period or node will be excluded",
            needed_alpha, needed_usd
        ))
    }

    async fn handle_price_unavailable(
        &self,
        hotkey: &str,
        node_id: &str,
        minimum_usd: Decimal,
        collateral_alpha: Decimal,
    ) -> Result<(CollateralState, CollateralStatus)> {
        if self
            .grace_tracker
            .get_since(hotkey, node_id)
            .await?
            .is_none()
        {
            // TODO: Track price-feed outage duration separately from collateral grace periods.
            self.grace_tracker
                .mark_undercollateralized(hotkey, node_id)
                .await?;
        }

        let grace_remaining = self
            .grace_tracker
            .get_grace_remaining(hotkey, node_id)
            .await?
            .unwrap_or_else(Duration::zero);

        if grace_remaining <= Duration::zero() {
            return Ok((
                CollateralState::Excluded {
                    current_usd: Decimal::ZERO,
                    minimum_usd,
                    reason: "price_unavailable".to_string(),
                },
                CollateralStatus {
                    current_alpha: collateral_alpha,
                    current_usd_value: Decimal::ZERO,
                    minimum_usd_required: minimum_usd,
                    status: "excluded".to_string(),
                    grace_period_remaining: Some(Duration::zero()),
                    action_required: Some(
                        "Alpha price unavailable; node excluded after grace period".to_string(),
                    ),
                    alpha_usd_price: None,
                    price_stale: true,
                },
            ));
        }

        Ok((
            CollateralState::Undercollateralized {
                current_usd: Decimal::ZERO,
                minimum_usd,
                grace_remaining,
            },
            CollateralStatus {
                current_alpha: collateral_alpha,
                current_usd_value: Decimal::ZERO,
                minimum_usd_required: minimum_usd,
                status: "undercollateralized".to_string(),
                grace_period_remaining: Some(grace_remaining),
                action_required: Some(
                    "Alpha price unavailable; grace period started for collateral checks"
                        .to_string(),
                ),
                alpha_usd_price: None,
                price_stale: true,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::SimplePersistence;
    use rust_decimal::Decimal;

    #[tokio::test]
    async fn test_evaluator_sufficient() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let tracker = Arc::new(GracePeriodTracker::new(persistence, Duration::hours(24)));
        let evaluator = CollateralEvaluator::new(CollateralConfig::default(), tracker);
        let (state, status) = evaluator
            .evaluate(
                "hk",
                "node",
                "H100",
                2,
                Decimal::from(200),
                Some(Decimal::ONE),
            )
            .await
            .unwrap();
        assert!(matches!(state, CollateralState::Sufficient { .. }));
        assert_eq!(status.status, "sufficient");
        assert_eq!(status.minimum_usd_required, Decimal::from(100));
    }

    #[tokio::test]
    async fn test_evaluator_undercollateralized() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let tracker = Arc::new(GracePeriodTracker::new(persistence, Duration::hours(24)));
        let evaluator = CollateralEvaluator::new(CollateralConfig::default(), tracker);
        let (_state, status) = evaluator
            .evaluate("hk", "node", "H100", 1, Decimal::ONE, Some(Decimal::ONE))
            .await
            .unwrap();
        assert_eq!(status.status, "undercollateralized");
    }
}
