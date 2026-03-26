use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// Token pricing cache TTL in seconds
    #[serde(default = "default_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            cache_ttl_secs: default_cache_ttl_secs(),
        }
    }
}

impl PricingConfig {
    pub fn cache_ttl(&self) -> Duration {
        Duration::from_secs(self.cache_ttl_secs)
    }

    pub fn validate(&self) -> Result<()> {
        if self.cache_ttl_secs == 0 {
            anyhow::bail!("pricing.cache_ttl_secs must be greater than 0");
        }
        Ok(())
    }
}

fn default_cache_ttl_secs() -> u64 {
    900
}
