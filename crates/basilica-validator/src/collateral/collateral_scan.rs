use crate::collateral::event_handler::CollateralEventHandler;
use crate::config::collateral::CollateralConfig;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use collateral_contract::config::CollateralNetworkConfig;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

#[derive(Clone)]
pub struct Collateral {
    config: crate::config::VerificationConfig,
    collateral_config: CollateralConfig,
    persistence: Arc<SimplePersistence>,
    event_handler: Arc<CollateralEventHandler>,
    cancellation_token: CancellationToken,
}

impl Collateral {
    pub fn new(
        config: crate::config::VerificationConfig,
        collateral_config: CollateralConfig,
        persistence: Arc<SimplePersistence>,
    ) -> Self {
        let event_handler = Arc::new(CollateralEventHandler::new(persistence.clone()));
        Self {
            config,
            collateral_config,
            persistence,
            event_handler,
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Spawn the collateral event scan loop on a background task
    pub fn start(&self) {
        let scanner = self.clone();
        tokio::spawn(async move {
            scanner.scan_loop().await;
        });
    }

    /// Stop the collateral event scan loop
    pub fn stop(&self) {
        self.cancellation_token.cancel();
    }

    async fn scan_loop(&self) {
        info!("Starting collateral event scan loop");
        let mut interval = tokio::time::interval(self.config.collateral_event_scan_interval);

        loop {
            tokio::select! {
                _ = self.cancellation_token.cancelled() => {
                    info!("Collateral event scan loop stopped");
                    break;
                }
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
            self.collateral_config.rpc_url.clone(),
        )?;
        let (to_block, events_map) =
            collateral_contract::scan_events(from_block, &network_config).await?;

        let mut sorted_events_map = events_map.iter().collect::<Vec<_>>();

        // sort the events by block number
        sorted_events_map.sort_by(|a, b| a.0.cmp(b.0));

        for (block_number, events_vec) in sorted_events_map.iter() {
            self.event_handler
                .apply_collateral_events_for_block(**block_number, events_vec.as_slice())
                .await?;
        }

        // update the last scanned block number after handling all blocks
        self.persistence
            .update_last_scanned_block_number(to_block)
            .await?;

        Ok(())
    }
}
