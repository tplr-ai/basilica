use crate::config::collateral::CollateralConfig;
use crate::persistence::SimplePersistence;
use collateral_contract::config::CollateralNetworkConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct CollateralReconciler {
    collateral_config: CollateralConfig,
    persistence: Arc<SimplePersistence>,
    interval: Duration,
    cancellation_token: CancellationToken,
}

impl CollateralReconciler {
    pub fn new(
        collateral_config: CollateralConfig,
        persistence: Arc<SimplePersistence>,
        interval: Duration,
    ) -> Self {
        Self {
            collateral_config,
            persistence,
            interval,
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Spawn the collateral reconciliation loop on a background task
    pub fn start(&self) {
        let reconciler = self.clone();
        tokio::spawn(async move {
            reconciler.reconcile_loop().await;
        });
    }

    /// Stop the collateral reconciliation loop
    pub fn stop(&self) {
        self.cancellation_token.cancel();
    }

    async fn reconcile_loop(&self) {
        info!(
            interval_secs = self.interval.as_secs(),
            "Starting collateral reconciliation loop"
        );
        let mut interval = tokio::time::interval(self.interval);

        loop {
            tokio::select! {
                _ = self.cancellation_token.cancelled() => {
                    info!("Collateral reconciliation loop stopped");
                    break;
                }
                _ = interval.tick() => {
                    if let Err(e) = self.reconcile().await {
                        error!("Collateral reconciliation tick failed: {}", e);
                    }
                }
            }
        }
    }

    async fn reconcile(&self) -> Result<(), anyhow::Error> {
        let last_scanned_block = self.persistence.get_last_scanned_block_number().await?;

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

        let nodes = self.persistence.get_all_collateral_nodes().await?;
        if nodes.is_empty() {
            return Ok(());
        }

        info!(
            block = last_scanned_block,
            node_count = nodes.len(),
            "Reconciling collateral state against on-chain values"
        );

        let mut corrected = 0u64;
        let mut errors = 0u64;

        for node in &nodes {
            match self
                .reconcile_node(node, last_scanned_block, &network_config)
                .await
            {
                Ok(was_corrected) => {
                    if was_corrected {
                        corrected += 1;
                    }
                }
                Err(e) => {
                    errors += 1;
                    error!(
                        hotkey = %node.hotkey,
                        node_id = %node.node_id,
                        "Failed to reconcile node: {}",
                        e
                    );
                }
            }
        }

        info!(
            block = last_scanned_block,
            nodes_checked = nodes.len(),
            nodes_corrected = corrected,
            nodes_errored = errors,
            "Collateral reconciliation pass complete"
        );

        Ok(())
    }

    /// Returns `true` if a mismatch was found and the DB was corrected.
    async fn reconcile_node(
        &self,
        node: &crate::persistence::collateral_persistence::CollateralNodeRecord,
        block_number: u64,
        network_config: &CollateralNetworkConfig,
    ) -> Result<bool, anyhow::Error> {
        let hotkey = hex_to_fixed::<32>(&node.hotkey)?;
        let node_id = hex_to_fixed::<16>(&node.node_id)?;

        let (chain_tao, chain_alpha) = collateral_contract::collaterals_at_block(
            hotkey,
            node_id,
            block_number,
            network_config,
        )
        .await?;

        let mut mismatch = false;

        if node.tao_collateral != chain_tao {
            warn!(
                hotkey = %node.hotkey,
                node_id = %node.node_id,
                field = "tao_collateral",
                db_value = %node.tao_collateral,
                chain_value = %chain_tao,
                block = block_number,
                "Collateral mismatch detected — overwriting DB with on-chain value"
            );
            mismatch = true;
        }

        if node.alpha_collateral != chain_alpha {
            warn!(
                hotkey = %node.hotkey,
                node_id = %node.node_id,
                field = "alpha_collateral",
                db_value = %node.alpha_collateral,
                chain_value = %chain_alpha,
                block = block_number,
                "Collateral mismatch detected — overwriting DB with on-chain value"
            );
            mismatch = true;
        }

        if mismatch {
            self.persistence
                .reconcile_collateral(&node.hotkey, &node.node_id, chain_tao, chain_alpha)
                .await?;
        }

        Ok(mismatch)
    }
}

fn hex_to_fixed<const N: usize>(hex_str: &str) -> Result<[u8; N], anyhow::Error> {
    let bytes = hex::decode(hex_str)?;
    if bytes.len() != N {
        return Err(anyhow::anyhow!(
            "Expected {} bytes, got {} from hex '{}'",
            N,
            bytes.len(),
            hex_str
        ));
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}
