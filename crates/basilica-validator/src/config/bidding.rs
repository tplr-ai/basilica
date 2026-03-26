use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiddingConfig {
    /// Price cache TTL in seconds
    pub price_cache_ttl_secs: u64,
    /// Minimum bid as fraction of baseline (0.0-1.0)
    pub min_bid_floor_fraction: f64,
    /// RPC timestamp tolerance window in seconds (replay protection)
    pub rpc_timestamp_tolerance_secs: u64,
    /// Health check interval in seconds (returned to miners in RegisterBidResponse)
    pub health_check_interval_secs: u64,
}

impl Default for BiddingConfig {
    fn default() -> Self {
        Self {
            price_cache_ttl_secs: 60,
            min_bid_floor_fraction: 0.1,
            rpc_timestamp_tolerance_secs: 300,
            health_check_interval_secs: 60,
        }
    }
}

impl BiddingConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.min_bid_floor_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.min_bid_floor_fraction)
        {
            return Err(anyhow!(
                "min_bid_floor_fraction must be between 0.0 and 1.0"
            ));
        }
        if self.price_cache_ttl_secs == 0 {
            return Err(anyhow!("price_cache_ttl_secs must be greater than 0"));
        }
        if self.rpc_timestamp_tolerance_secs == 0 {
            return Err(anyhow!(
                "rpc_timestamp_tolerance_secs must be greater than 0"
            ));
        }
        if self.health_check_interval_secs == 0 {
            return Err(anyhow!("health_check_interval_secs must be greater than 0"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_valid_endpoint() {
        let config = BiddingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_valid_config() {
        let config = BiddingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_floor_fraction() {
        let config = BiddingConfig {
            min_bid_floor_fraction: 1.5,
            ..BiddingConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_zero_durations_invalid() {
        let config = BiddingConfig {
            price_cache_ttl_secs: 0,
            rpc_timestamp_tolerance_secs: 0,
            ..BiddingConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_health_check_interval_zero_invalid() {
        let config = BiddingConfig {
            health_check_interval_secs: 0,
            ..BiddingConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
