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
    cancellation_token: CancellationToken,
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
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Spawn the collateral sync loop on a background task
    pub fn start(&self) {
        let scanner = self.clone();
        tokio::spawn(async move {
            scanner.sync_loop().await;
        });
    }

    /// Stop the collateral sync loop
    pub fn stop(&self) {
        self.cancellation_token.cancel();
    }

    async fn sync_loop(&self) {
        info!("Starting collateral sync loop (RPC-based)");
        let mut interval = tokio::time::interval(self.config.collateral_event_scan_interval);

        loop {
            tokio::select! {
                _ = self.cancellation_token.cancelled() => {
                    info!("Collateral sync loop stopped");
                    break;
                }
                _ = interval.tick() => {
                    if let Err(e) = self.sync_collateral_state().await {
                        error!("Collateral sync failed: {}", e);
                    }
                }
            }
        }
    }

    pub async fn sync_collateral_state(&self) -> Result<()> {
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

        let snapshot_block =
            collateral_contract::get_finalized_block_number(&network_config).await?;
        let sync_page_size = self.collateral_config.sync_page_size;

        // Fetch a block-pinned snapshot via pagination to avoid missing/duplicated rows during iteration.
        let nodes = collateral_contract::get_all_collaterals_at_block(
            &network_config,
            snapshot_block,
            sync_page_size,
        )
        .await?;
        let reclaims = collateral_contract::get_all_reclaims_at_block(
            &network_config,
            snapshot_block,
            sync_page_size,
        )
        .await?;

        info!(
            snapshot_block,
            sync_page_size,
            nodes = nodes.len(),
            reclaims = reclaims.len(),
            "Syncing collateral state from contract"
        );

        // Sync to database
        self.persistence.sync_all_collateral_nodes(&nodes).await?;
        self.persistence.sync_all_reclaims(&reclaims).await?;
        self.persistence
            .update_last_scanned_block_number(snapshot_block)
            .await?;

        Ok(())
    }
}
