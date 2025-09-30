use crate::storage::{DepositAccountsRepo, ObservedDepositsRepo, OutboxRepo, PgRepos};
use anyhow::Result;
use async_trait::async_trait;
use basilica_common::distributed::postgres_lock::{LeaderElection, LockKey};
use bittensor::connect::{BlockchainEventHandler, BlockchainMonitor};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Payments-specific event handler for blockchain monitoring
struct PaymentsEventHandler {
    repos: PgRepos,
    known_accounts: Arc<RwLock<HashSet<String>>>,
}

impl PaymentsEventHandler {
    async fn new(repos: PgRepos) -> Result<Self> {
        let accounts = repos.list_account_hexes().await?;
        let known_accounts = Arc::new(RwLock::new(accounts.into_iter().collect()));
        Ok(Self {
            repos,
            known_accounts,
        })
    }

    async fn refresh_known_accounts(&self) -> Result<()> {
        let accounts = self.repos.list_account_hexes().await?;
        let mut known = self.known_accounts.write().await;
        *known = accounts.into_iter().collect();
        Ok(())
    }
}

#[async_trait]
impl BlockchainEventHandler for PaymentsEventHandler {
    async fn handle_transfer(
        &self,
        from: &str,
        to: &str,
        amount: &str,
        block_number: u32,
        event_index: usize,
    ) -> Result<()> {
        let known = self.known_accounts.read().await;
        if !known.contains(to) {
            return Ok(());
        }

        let txid = format!("b{}#e{}#{}", block_number, event_index, to);

        if let Err(e) = async {
            let mut tx = self.repos.begin().await?;
            self.repos
                .insert_finalized_tx(
                    &mut tx,
                    block_number as i64,
                    event_index as i32,
                    to,
                    from,
                    amount,
                )
                .await?;
            self.repos.enqueue_tx(&mut tx, to, amount, &txid).await?;
            tx.commit().await?;
            Ok::<(), anyhow::Error>(())
        }
        .await
        {
            // Don't tear down the monitor on a single failed write; log and move on.
            tracing::error!(%txid, %to, %from, %amount, block_number, event_index, err=%e, "failed to persist observed deposit");
            return Ok(());
        }

        info!(
            "Recorded deposit: {} -> {} amount: {} (txid: {})",
            from, to, amount, txid
        );

        Ok(())
    }

    async fn on_block_end(&self, block_number: u32) -> Result<()> {
        if block_number.is_multiple_of(128) {
            self.refresh_known_accounts().await?;
        }
        Ok(())
    }
}

/// Monitors blockchain for deposits to payment accounts
pub struct ChainMonitor {
    repos: PgRepos,
    endpoints: Vec<String>,
}

impl ChainMonitor {
    /// Create a new chain monitor
    pub async fn new(repos: PgRepos, endpoints: Vec<String>) -> Result<Self> {
        Ok(Self { repos, endpoints })
    }

    /// Run the monitor with leader election
    ///
    /// This uses the common distributed locking to ensure only one monitor
    /// instance is active at a time in a distributed deployment.
    pub async fn run(self) -> Result<()> {
        let election = LeaderElection::new(self.repos.pool.clone(), LockKey::PAYMENTS_MONITOR)
            .with_retry_interval(3);

        let repos = self.repos;
        let endpoints = self.endpoints.clone();

        election
            .run_as_leader(move || {
                let repos = repos.clone();
                let endpoints = endpoints.clone();

                async move {
                    let handler = PaymentsEventHandler::new(repos).await?;
                    let monitor = BlockchainMonitor::new(endpoints, handler).await?;

                    monitor.run().await?;

                    Ok(())
                }
            })
            .await
    }
}
