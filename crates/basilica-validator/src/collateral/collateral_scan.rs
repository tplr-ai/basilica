use crate::config::collateral::CollateralConfig;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use collateral_contract::{config::CollateralNetworkConfig, CollateralEvent};
use std::sync::Arc;
use tracing::{error, info};

pub struct Collateral {
    config: crate::config::VerificationConfig,
    collateral_config: CollateralConfig,
    persistence: Arc<SimplePersistence>,
}

impl Collateral {
    pub fn new(
        config: crate::config::VerificationConfig,
        collateral_config: CollateralConfig,
        persistence: Arc<SimplePersistence>,
    ) -> Self {
        Self {
            config,
            collateral_config,
            persistence,
        }
    }

    /// Start the collateral event scan loop
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting collateral event scan loop");
        let mut interval = tokio::time::interval(self.config.collateral_event_scan_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.scan_handle_collateral_events().await {
                        error!("Collateral event scan failed: {}", e);
                    }
                }
            }
        }
    }

    pub async fn scan_handle_collateral_events(&self) -> Result<()> {
        let last_block = self.persistence.get_last_scanned_block_number().await?;
        let from_block = last_block + 1;
        let network = match self.collateral_config.network.as_str() {
            "mainnet" => collateral_contract::config::Network::Mainnet,
            "testnet" => collateral_contract::config::Network::Testnet,
            "local" => collateral_contract::config::Network::Local,
            _ => collateral_contract::config::Network::Mainnet,
        };
        let network_config = CollateralNetworkConfig::from_network(
            &network,
            Some(self.collateral_config.contract_address.clone()),
        )?;
        let (to_block, events_map) =
            collateral_contract::scan_events(from_block, &network_config).await?;

        let mut sorted_events_map = events_map.iter().collect::<Vec<_>>();

        // sort the events by block number
        sorted_events_map.sort_by(|a, b| a.0.cmp(b.0));

        for (block_number, events_vec) in sorted_events_map.iter() {
            for event in events_vec.iter() {
                match event {
                    CollateralEvent::Deposit(deposit) => {
                        self.persistence.handle_deposit(deposit).await?;
                    }
                    CollateralEvent::Reclaimed(reclaimed) => {
                        self.persistence.handle_reclaimed(reclaimed).await?;
                    }
                    CollateralEvent::Slashed(slashed) => {
                        self.persistence.handle_slashed(slashed).await?;
                    }
                }
            }
            // update the last scanned block number after handling all events in the block
            self.persistence
                .update_last_scanned_block_number(**block_number)
                .await?;
        }

        // update the last scanned block number after handling all blocks
        self.persistence
            .update_last_scanned_block_number(to_block)
            .await?;

        Ok(())
    }
}
