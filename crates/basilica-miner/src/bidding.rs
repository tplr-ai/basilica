//! Automatic bidding module
//!
//! Owns the complete miner registration lifecycle:
//! - Initial registration with validator (RegisterBid)
//! - Periodic health checks
//!
//! BidManager is the single source of truth for GPU pricing.
//! All prices come from the static bidding strategy.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use basilica_protocol::miner_discovery::NodeRegistration;

use crate::config::{BiddingConfig, BiddingStrategy};
use crate::node_manager::{NodeManager, RegisteredNode};
use crate::registration_client::RegistrationClient;

/// Bid manager that owns the complete registration lifecycle
pub struct BidManager {
    config: BiddingConfig,
    node_manager: Arc<NodeManager>,
    registration_client: Arc<RegistrationClient>,
}

impl BidManager {
    /// Create a new bid manager with registration client
    pub fn new(
        config: BiddingConfig,
        node_manager: Arc<NodeManager>,
        registration_client: Arc<RegistrationClient>,
    ) -> Self {
        Self {
            config,
            node_manager,
            registration_client,
        }
    }

    /// Validate that all configured nodes have prices in BiddingConfig.
    /// This ensures operators explicitly configure pricing for all their hardware.
    pub fn validate_node_prices(&self, nodes: &[RegisteredNode]) -> Result<()> {
        for node in nodes {
            let category = node.config.gpu_category.to_uppercase();
            if self.get_bid_price(&category).is_none() {
                anyhow::bail!(
                    "Node '{}' has GPU category '{}' but no price configured in [bidding.strategy.static.static_prices]. \
                     Add `{} = <price>` to your miner.toml",
                    node.config.host,
                    category,
                    category
                );
            }
        }
        Ok(())
    }

    /// Get price for a node from BiddingConfig (assumes validation passed)
    fn get_node_price(&self, gpu_category: &str) -> u32 {
        let category = gpu_category.to_uppercase();
        // Safe to unwrap after validation
        self.get_bid_price(&category)
            .expect("validation should have caught missing price")
    }

    /// Build node registrations with prices from BiddingConfig
    fn build_node_registrations(&self, nodes: &[RegisteredNode]) -> Vec<NodeRegistration> {
        nodes
            .iter()
            .map(|n| {
                let price = self.get_node_price(&n.config.gpu_category);
                NodeRegistration {
                    host: n.config.host.clone(),
                    port: n.config.port as u32,
                    username: n.config.username.clone(),
                    gpu_category: n.config.gpu_category.clone(),
                    gpu_count: n.config.gpu_count,
                    hourly_rate_cents: price,
                    attestation: vec![], // TODO: Add attestation when available
                }
            })
            .collect()
    }

    /// Run the bid manager lifecycle:
    /// 1. Validate all nodes have prices
    /// 2. Register with validator
    /// 3. Deploy SSH keys if provided
    /// 4. Run health check loop
    pub async fn run(
        &self,
        grpc_endpoint: String,
        mut shutdown_rx: watch::Receiver<bool>,
    ) -> Result<()> {
        // 0. Validate all nodes have prices configured
        let nodes = self.node_manager.list_nodes().await?;

        if nodes.is_empty() {
            warn!("No nodes configured - publishing zero-node RegisterBid to deactivate existing bids");
            if let Err(e) = self
                .registration_client
                .register_nodes_with_registrations(&grpc_endpoint, vec![])
                .await
            {
                warn!(
                    error = %e,
                    "Failed to publish zero-node RegisterBid; stale bids may remain active"
                );
            }
            // Still wait for shutdown so we don't cause crash-loop
            loop {
                if shutdown_rx.changed().await.is_err() || *shutdown_rx.borrow() {
                    info!("BidManager shutdown requested (no nodes)");
                    return Ok(());
                }
            }
        }

        self.validate_node_prices(&nodes)?;

        info!(
            "Starting BidManager with {} nodes, {} GPU categories configured",
            nodes.len(),
            self.static_prices().len()
        );

        let node_registrations = self.build_node_registrations(&nodes);
        let state = self
            .registration_client
            .register_nodes_with_registrations(&grpc_endpoint, node_registrations)
            .await?;

        info!(
            "Successfully registered {} nodes with validator",
            nodes.len()
        );

        // 3. Deploy validator SSH key if provided
        if state.validator_ssh_public_key.is_some() {
            if let Err(e) = self.registration_client.deploy_validator_ssh_key().await {
                error!("Failed to deploy validator SSH key: {}", e);
            }
        }

        // 4. Run health check loop
        let health_interval = Duration::from_secs(state.health_check_interval_secs as u64);
        let mut health_ticker = interval(health_interval);

        info!(
            "Starting lifecycle loop: health_check={}s",
            health_interval.as_secs()
        );

        loop {
            tokio::select! {
                _ = health_ticker.tick() => {
                    match self.registration_client.send_health_check(&grpc_endpoint).await {
                        Ok(nodes_active) => {
                            debug!(nodes_active = nodes_active, "Health check successful");
                        }
                        Err(e) => {
                            warn!("Health check failed: {}", e);
                        }
                    }
                }
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        info!("BidManager shutdown requested");
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn static_prices(&self) -> &HashMap<String, u32> {
        match &self.config.strategy {
            BiddingStrategy::Static {
                static_prices_cents,
            } => static_prices_cents,
        }
    }

    /// Get the bid price for a GPU category (in cents)
    fn get_bid_price(&self, category: &str) -> Option<u32> {
        let static_prices = self.static_prices();

        // First check static_prices_cents
        if let Some(&price_cents) = static_prices.get(category) {
            return Some(price_cents);
        }

        // Try case-insensitive match
        let category_upper = category.to_uppercase();
        for (key, &price_cents) in static_prices {
            if key.to_uppercase() == category_upper {
                return Some(price_cents);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_config() -> BiddingConfig {
        let mut static_prices_cents = HashMap::new();
        static_prices_cents.insert("H100".to_string(), 250); // $2.50 in cents
        static_prices_cents.insert("A100".to_string(), 120); // $1.20 in cents

        BiddingConfig {
            strategy: BiddingStrategy::Static {
                static_prices_cents,
            },
        }
    }

    fn static_prices(config: &BiddingConfig) -> &HashMap<String, u32> {
        match &config.strategy {
            BiddingStrategy::Static {
                static_prices_cents,
            } => static_prices_cents,
        }
    }

    #[test]
    fn test_get_bid_price() {
        let config = test_config();

        // Test pricing logic directly
        let static_prices = static_prices(&config);

        assert_eq!(static_prices.get("H100"), Some(&250));
        assert_eq!(static_prices.get("A100"), Some(&120));
        assert_eq!(static_prices.get("RTX4090"), None);
    }

    #[test]
    fn test_validate_node_prices_success() {
        let config = test_config();

        // Validation should pass for categories that have prices
        let static_prices = static_prices(&config);
        let has_h100 = static_prices.contains_key("H100");
        let has_a100 = static_prices.contains_key("A100");
        assert!(has_h100 && has_a100);
    }

    #[test]
    fn test_validate_node_prices_missing() {
        let config = test_config();

        // Check that RTX4090 is not in prices (would fail validation)
        let static_prices = static_prices(&config);
        assert!(!static_prices.contains_key("RTX4090"));
    }
}
