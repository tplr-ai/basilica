use crate::config::collateral::CollateralConfig;
use anyhow::Result;
use rust_decimal::Decimal;

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
    pub action_required: Option<String>,
    pub alpha_usd_price: Option<Decimal>,
    pub price_stale: bool,
}

pub struct CollateralEvaluator {
    config: CollateralConfig,
}

impl CollateralEvaluator {
    pub fn new(config: CollateralConfig) -> Self {
        Self { config }
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

    pub fn evaluate(
        &self,
        _hotkey: &str,
        _node_id: &str,
        gpu_category: &str,
        gpu_count: u32,
        // Policy input is alpha collateral only. TAO is synced for observability but not used for limits.
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
                        action_required: Some(reason),
                        alpha_usd_price: None,
                        price_stale: true,
                    },
                ));
            }
        };

        let warning_threshold = minimum_usd * self.config.warning_threshold_multiplier;

        if current_usd >= warning_threshold {
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
                    action_required: None,
                    alpha_usd_price,
                    price_stale,
                },
            ));
        }

        if current_usd >= minimum_usd {
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
                    action_required,
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
            },
            CollateralStatus {
                current_alpha: collateral_alpha,
                current_usd_value: current_usd,
                minimum_usd_required: minimum_usd,
                status: "undercollateralized".to_string(),
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
            "Deposit {needed_alpha:.2} Alpha (~${needed_usd:.2}) to reach safe level ({}x minimum)",
            self.config.warning_threshold_multiplier
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
            "URGENT: Deposit {:.2} Alpha (~${:.2}) to meet minimum collateral requirement",
            needed_alpha, needed_usd
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    #[test]
    fn test_evaluator_sufficient() {
        let evaluator = CollateralEvaluator::new(CollateralConfig::default());
        let (state, status) = evaluator
            .evaluate(
                "hk",
                "node",
                "H100",
                2,
                Decimal::from(200),
                Some(Decimal::ONE),
            )
            .unwrap();
        assert!(matches!(state, CollateralState::Sufficient { .. }));
        assert_eq!(status.status, "sufficient");
        assert_eq!(status.minimum_usd_required, Decimal::from(100));
    }

    #[test]
    fn test_evaluator_undercollateralized() {
        let evaluator = CollateralEvaluator::new(CollateralConfig::default());
        let (state, status) = evaluator
            .evaluate("hk", "node", "H100", 1, Decimal::ONE, Some(Decimal::ONE))
            .unwrap();
        assert!(matches!(state, CollateralState::Undercollateralized { .. }));
        assert_eq!(status.status, "undercollateralized");
    }

    #[test]
    fn test_undercollateralized_never_becomes_excluded() {
        let evaluator = CollateralEvaluator::new(CollateralConfig::default());

        // Evaluate multiple times — should remain undercollateralized, never excluded
        for _ in 0..3 {
            let (state, status) = evaluator
                .evaluate("hk", "node", "H100", 1, Decimal::ONE, Some(Decimal::ONE))
                .unwrap();
            assert!(
                matches!(state, CollateralState::Undercollateralized { .. }),
                "expected Undercollateralized, got {:?}",
                state
            );
            assert_eq!(status.status, "undercollateralized");
        }
    }
}
