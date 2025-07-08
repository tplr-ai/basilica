//! # Stake Monitor Service
//!
//! Periodically fetches validator stake information from the Bittensor metagraph
//! and updates the local database cache.

use anyhow::Result;
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, error, info, warn};

use crate::config::MinerConfig;
use crate::persistence::AssignmentDb;

/// Stake monitor service that periodically updates validator stakes
pub struct StakeMonitor {
    /// Bittensor service for accessing metagraph
    bittensor_service: Arc<bittensor::Service>,
    /// Network UID
    netuid: u16,
    /// Assignment database
    assignment_db: AssignmentDb,
    /// Update interval
    update_interval: Duration,
    /// Minimum stake threshold (validators below this are ignored)
    min_stake_threshold: f64,
}

impl StakeMonitor {
    /// Create a new stake monitor
    pub async fn new(config: &MinerConfig, pool: SqlitePool) -> Result<Self> {
        // Initialize the bittensor service
        let bittensor_service = Arc::new(
            bittensor::Service::new(config.bittensor.common.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize bittensor service: {}", e))?,
        );

        let assignment_db = AssignmentDb::new(pool);

        Ok(Self {
            bittensor_service,
            netuid: config.bittensor.common.netuid,
            assignment_db,
            update_interval: Duration::from_secs(300), // 5 minutes default
            min_stake_threshold: 100.0,                // 100 TAO minimum
        })
    }

    /// Configure update interval (useful for testing)
    pub fn with_update_interval(mut self, interval: Duration) -> Self {
        self.update_interval = interval;
        self
    }

    /// Configure minimum stake threshold (useful for testing)
    pub fn with_min_stake_threshold(mut self, threshold: f64) -> Self {
        self.min_stake_threshold = threshold;
        self
    }

    /// Force an immediate stake update (useful for testing)
    pub async fn force_update(&self) -> Result<usize> {
        self.update_validator_stakes().await
    }

    /// Start the stake monitoring service
    pub async fn start(self) -> Result<()> {
        info!(
            "Starting stake monitor service (interval: {:?}, min_stake: {} TAO)",
            self.update_interval, self.min_stake_threshold
        );

        // Do an initial update
        if let Err(e) = self.update_validator_stakes().await {
            error!("Failed initial stake update: {}", e);
        }

        // Start the periodic update loop
        let mut ticker = interval(self.update_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;

            match self.update_validator_stakes().await {
                Ok(count) => {
                    info!("Successfully updated stakes for {} validators", count);
                }
                Err(e) => {
                    error!("Failed to update validator stakes: {}", e);
                }
            }
        }
    }

    /// Update validator stakes from the metagraph
    async fn update_validator_stakes(&self) -> Result<usize> {
        debug!("Fetching validator information from metagraph...");

        // Get current metagraph
        let metagraph = self
            .bittensor_service
            .get_metagraph(self.netuid)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get metagraph: {}", e))?;

        // Use discovery module to find validators
        let discovery = bittensor::NeuronDiscovery::new(&metagraph);
        let validators = discovery
            .get_validators()
            .map_err(|e| anyhow::anyhow!("Failed to discover validators: {}", e))?;

        info!("Discovered {} validators from metagraph", validators.len());

        // Calculate total stake for percentage calculation
        let total_stake: f64 = validators
            .iter()
            .map(|v| v.stake as f64 / 1e9) // Convert from nano to TAO
            .sum();

        if total_stake == 0.0 {
            warn!("Total network stake is 0, skipping update");
            return Ok(0);
        }

        // Filter and prepare validator stake data
        let mut validator_stakes = Vec::new();

        for validator in validators {
            let stake_tao = validator.stake as f64 / 1e9; // Convert to TAO

            // Skip if below threshold
            if stake_tao < self.min_stake_threshold {
                continue;
            }

            let percentage = (stake_tao / total_stake) * 100.0;

            validator_stakes.push((validator.hotkey.clone(), stake_tao, percentage));

            debug!(
                "Validator UID {}: {} ({:.2} TAO, {:.2}%)",
                validator.uid,
                &validator.hotkey[..8],
                stake_tao,
                percentage
            );
        }

        // Sort by stake descending
        validator_stakes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        info!(
            "Found {} validators with stake >= {} TAO",
            validator_stakes.len(),
            self.min_stake_threshold
        );

        // Update database
        let count = validator_stakes.len();
        self.assignment_db
            .update_validator_stakes_batch(&validator_stakes)
            .await?;

        // Log top validators
        if !validator_stakes.is_empty() {
            info!("Top 5 validators by stake:");
            for (i, (hotkey, stake, percentage)) in validator_stakes.iter().take(5).enumerate() {
                info!(
                    "  {}. {} - {:.2} TAO ({:.2}%)",
                    i + 1,
                    &hotkey[..8], // Show first 8 chars of hotkey
                    stake,
                    percentage
                );
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::SqlitePool;

    async fn setup_test_monitor() -> (StakeMonitor, SqlitePool) {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();

        // Run migrations
        let assignment_db = AssignmentDb::new(pool.clone());
        assignment_db.run_migrations().await.unwrap();

        // Create a test config that doesn't require real wallet files
        let config = create_test_config();

        // Create monitor with test settings
        let monitor = StakeMonitor::new(&config, pool.clone())
            .await
            .unwrap()
            .with_update_interval(Duration::from_secs(1))
            .with_min_stake_threshold(10.0);

        (monitor, pool)
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network access
    async fn test_stake_monitor_update() {
        let (monitor, pool) = setup_test_monitor().await;

        // Force an update
        let result = monitor.force_update().await;
        assert!(result.is_ok());

        // Check that stakes were saved
        let db = AssignmentDb::new(pool);
        let stakes = db.get_all_validator_stakes().await.unwrap();
        assert!(!stakes.is_empty());

        // Verify stake percentages add up to approximately 100%
        let total_percentage: f64 = stakes.iter().map(|s| s.percentage_of_total).sum();
        assert!((total_percentage - 100.0).abs() < 1.0); // Allow 1% tolerance
    }

    #[tokio::test]
    async fn test_stake_monitor_creation() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let assignment_db = AssignmentDb::new(pool.clone());
        assignment_db.run_migrations().await.unwrap();

        let mut config = MinerConfig::default();
        // Use an invalid chain endpoint to force failure
        config.bittensor.common.chain_endpoint = Some("ws://invalid-endpoint:9944".to_string());
        let monitor = StakeMonitor::new(&config, pool).await;

        // Should fail with invalid chain endpoint
        assert!(monitor.is_err());
    }

    fn create_test_config() -> MinerConfig {
        use crate::config::MinerBittensorConfig;
        use common::config::{BittensorConfig, DatabaseConfig};

        MinerConfig {
            bittensor: MinerBittensorConfig {
                common: BittensorConfig {
                    wallet_name: "test_wallet".to_string(),
                    hotkey_name: "test_hotkey".to_string(),
                    network: "local".to_string(),
                    netuid: 999,
                    chain_endpoint: Some("ws://127.0.0.1:9944".to_string()),
                    weight_interval_secs: 300,
                },
                coldkey_name: "test_coldkey".to_string(),
                skip_registration: true,
                ..Default::default()
            },
            database: DatabaseConfig {
                url: "sqlite::memory:".to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network access and is just a config test
    async fn test_stake_monitor_config() -> Result<()> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        let assignment_db = AssignmentDb::new(pool.clone());
        assignment_db.run_migrations().await?;

        let config = create_test_config();
        let monitor = StakeMonitor::new(&config, pool)
            .await?
            .with_update_interval(Duration::from_secs(60))
            .with_min_stake_threshold(50.0);

        assert_eq!(monitor.update_interval, Duration::from_secs(60));
        assert_eq!(monitor.min_stake_threshold, 50.0);

        Ok(())
    }
}
