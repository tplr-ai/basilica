use crate::basilica_api::BasilicaApiClient;
use crate::collateral::evaluator::{CollateralEvaluator, CollateralState};
use crate::collateral::manager::{hotkey_ss58_to_hex, node_id_to_hex, u256_to_alpha};
use crate::collateral::CollateralPreference;
use crate::config::collateral::CollateralConfig;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use collateral_contract::config::CollateralNetworkConfig;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct Collateral {
    config: crate::config::VerificationConfig,
    collateral_config: CollateralConfig,
    persistence: Arc<SimplePersistence>,
    api_client: Arc<BasilicaApiClient>,
    evaluator: Arc<CollateralEvaluator>,
    netuid: u16,
    cancellation_token: CancellationToken,
}

impl Collateral {
    pub fn new(
        config: crate::config::VerificationConfig,
        collateral_config: CollateralConfig,
        persistence: Arc<SimplePersistence>,
        api_client: Arc<BasilicaApiClient>,
        evaluator: Arc<CollateralEvaluator>,
        netuid: u16,
    ) -> Self {
        Self {
            config,
            collateral_config,
            persistence,
            api_client,
            evaluator,
            netuid,
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Spawn the collateral sync loop on a background task
    pub fn start(&self) -> JoinHandle<()> {
        let scanner = self.clone();
        tokio::spawn(async move {
            scanner.sync_loop().await;
        })
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
                    match self.sync_collateral_state().await {
                        Ok(()) => {
                            if let Err(e) = self.compute_and_store_preferences().await {
                                error!("Collateral preference computation failed: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Collateral sync failed, skipping preference computation: {}", e);
                        }
                    }
                }
            }
        }
    }

    pub async fn sync_collateral_state(&self) -> Result<()> {
        let network: collateral_contract::config::Network =
            self.collateral_config.network.parse()?;
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

    async fn compute_and_store_preferences(&self) -> Result<()> {
        let alpha_price_usd = match self.api_client.get_alpha_price_usd(self.netuid).await {
            Ok(price) => Some(price),
            Err(err) => {
                warn!("Alpha price unavailable for preference computation: {err}");
                None
            }
        };

        let nodes = self.persistence.get_all_nodes_with_gpu_info().await?;

        let mut updates: Vec<(String, CollateralPreference)> = Vec::with_capacity(nodes.len());

        for node in &nodes {
            let preference = self.compute_preference(node, alpha_price_usd).await;
            updates.push((node.node_id.clone(), preference));
        }

        let preferred_count = updates
            .iter()
            .filter(|(_, p)| *p == CollateralPreference::Preferred)
            .count();

        info!(
            total = updates.len(),
            preferred = preferred_count,
            fallback = updates.len() - preferred_count,
            "Computed collateral preferences"
        );

        self.persistence
            .batch_update_collateral_preferences(&updates)
            .await?;

        Ok(())
    }

    async fn compute_preference(
        &self,
        node: &crate::persistence::miner_nodes::NodeGpuInfo,
        alpha_price_usd: Option<Decimal>,
    ) -> CollateralPreference {
        let gpu_category = match &node.gpu_category {
            Some(cat) => cat,
            None => return CollateralPreference::Fallback,
        };

        let hotkey_hex = match hotkey_ss58_to_hex(&node.hotkey_ss58) {
            Ok(val) => val,
            Err(err) => {
                warn!(
                    node_id = %node.node_id,
                    "Failed to convert hotkey to hex: {err}"
                );
                return CollateralPreference::Fallback;
            }
        };

        let node_hex = match node_id_to_hex(&node.node_id) {
            Ok(val) => val,
            Err(err) => {
                warn!(
                    node_id = %node.node_id,
                    "Failed to convert node_id to hex: {err}"
                );
                return CollateralPreference::Fallback;
            }
        };

        let amount = self
            .persistence
            .get_alpha_collateral_amount(&hotkey_hex, &node_hex)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    node_id = %node.node_id,
                    "Failed to get collateral alpha: {e}"
                );
                None
            });
        let collateral_alpha = match amount {
            Some(val) => u256_to_alpha(val).unwrap_or(Decimal::ZERO),
            None => Decimal::ZERO,
        };

        match self.evaluator.evaluate(
            &node.hotkey_ss58,
            &node.node_id,
            gpu_category,
            node.gpu_count,
            collateral_alpha,
            alpha_price_usd,
        ) {
            Ok((state, _)) => match state {
                CollateralState::Sufficient { .. } | CollateralState::Warning { .. } => {
                    CollateralPreference::Preferred
                }
                CollateralState::Undercollateralized { .. } | CollateralState::Unknown { .. } => {
                    CollateralPreference::Fallback
                }
            },
            Err(_) => CollateralPreference::Fallback,
        }
    }
}
