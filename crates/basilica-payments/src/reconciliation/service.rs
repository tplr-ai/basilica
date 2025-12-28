use crate::{
    blockchain::client::BlockchainClient,
    config::ReconciliationConfig,
    error::{PaymentsError, Result},
    metrics::PaymentsMetricsSystem,
    reconciliation::{
        SkipReason, SweepCalculator, SweepDecision, SweepStatus, SweepSummary, WalletManager,
    },
    storage::{DepositAccountsRepo, PgRepos, ReconciliationRepo},
};
use anyhow::Context;
use basilica_common::{
    crypto::Aead,
    distributed::postgres_lock::{LeaderElection, LockKey},
};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

pub struct ReconciliationService {
    repos: PgRepos,
    blockchain: Arc<BlockchainClient>,
    wallet_manager: WalletManager,
    calculator: SweepCalculator,
    config: ReconciliationConfig,
    metrics: Option<Arc<PaymentsMetricsSystem>>,
}

impl ReconciliationService {
    #[allow(clippy::result_large_err)]
    pub fn new(
        repos: PgRepos,
        blockchain: Arc<BlockchainClient>,
        aead: Arc<Aead>,
        config: ReconciliationConfig,
        metrics: Option<Arc<PaymentsMetricsSystem>>,
    ) -> Result<Self> {
        let minimum_threshold = config
            .minimum_threshold_plancks
            .parse()
            .context("Invalid minimum_threshold_plancks")
            .map_err(|e| PaymentsError::Config(e.to_string()))?;

        let target_balance = config
            .target_balance_plancks
            .parse()
            .context("Invalid target_balance_plancks")
            .map_err(|e| PaymentsError::Config(e.to_string()))?;

        let estimated_fee = config
            .estimated_fee_plancks
            .parse()
            .context("Invalid estimated_fee_plancks")
            .map_err(|e| PaymentsError::Config(e.to_string()))?;

        let calculator = SweepCalculator::new(minimum_threshold, target_balance, estimated_fee);
        let wallet_manager = WalletManager::new(aead);

        Ok(Self {
            repos,
            blockchain,
            wallet_manager,
            calculator,
            config,
            metrics,
        })
    }

    pub async fn run(self) -> ! {
        if !self.config.enabled {
            info!("Reconciliation service is disabled");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
            }
        }

        info!("Starting reconciliation service with leader election");
        info!("Cold wallet: {}", self.config.coldwallet_address_ss58);
        info!("Dry run mode: {}", self.config.dry_run_mode);

        let service = Arc::new(self);
        let election =
            LeaderElection::new(service.repos.pool.clone(), LockKey::RECONCILIATION_SERVICE)
                .with_retry_interval(5);

        election
            .run_as_leader(move || {
                let service_clone = Arc::clone(&service);
                async move {
                    info!("Reconciliation service acquired leadership");
                    service_clone.sweep_loop().await
                }
            })
            .await
    }

    async fn sweep_loop(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let interval = tokio::time::Duration::from_secs(self.config.sweep_interval_seconds);
        let mut ticker = tokio::time::interval(interval);

        loop {
            ticker.tick().await;

            // HIGH-07: Reconcile stale sweeps before starting new cycle
            if let Err(e) = self.reconcile_stale_sweeps().await {
                warn!("Stale sweep reconciliation failed: {}", e);
            }

            // HIGH-08: Monitor cold wallet balance
            if let Err(e) = self.monitor_cold_wallet().await {
                warn!("Cold wallet monitoring failed: {}", e);
            }

            info!("Starting reconciliation sweep cycle");

            match self.sweep_cycle().await {
                Ok(summary) => {
                    info!(
                        "Sweep cycle completed: checked={}, swept={}, failed={}, skipped={}",
                        summary.total_checked,
                        summary.swept_count,
                        summary.failed_count,
                        summary.skipped_count
                    );

                    if let Some(ref metrics) = self.metrics {
                        let amount_tao = summary.total_amount_plancks as f64 / 1e9;
                        metrics
                            .business_metrics()
                            .record_payment_processed(amount_tao, &[("type", "reconciliation")])
                            .await;
                    }
                }
                Err(e) => {
                    error!("Sweep cycle failed: {}", e);
                }
            }
        }
    }

    async fn sweep_cycle(&self) -> Result<SweepSummary> {
        let mut summary = SweepSummary::new();
        let max_sweeps = self.config.max_sweeps_per_cycle as usize;

        let account_hexes = self
            .repos
            .list_account_hexes()
            .await
            .map_err(PaymentsError::Database)?;

        summary.total_checked = account_hexes.len();
        info!(
            "Checking {} deposit accounts (max {} sweeps per cycle)",
            account_hexes.len(),
            max_sweeps
        );

        for account_hex in account_hexes {
            // Rate limit: stop if we've reached max sweeps for this cycle
            if summary.swept_count >= max_sweeps {
                info!(
                    "Rate limit reached: {} sweeps completed, deferring remaining accounts",
                    summary.swept_count
                );
                break;
            }

            match self.sweep_account(&account_hex).await {
                Ok(Some(amount)) => {
                    summary.swept_count += 1;
                    summary.total_amount_plancks += amount;
                }
                Ok(None) => {
                    summary.skipped_count += 1;
                }
                Err(e) => {
                    let account_preview = account_hex.chars().take(8).collect::<String>();
                    error!("Failed to sweep account {}: {}", account_preview, e);
                    summary.failed_count += 1;
                }
            }
        }

        Ok(summary)
    }

    async fn sweep_account(&self, account_hex: &str) -> Result<Option<u128>> {
        if self
            .repos
            .get_recent_sweep(account_hex, self.config.sweep_interval_seconds as i64)
            .await
            .map_err(PaymentsError::Database)?
        {
            let account_preview = account_hex.chars().take(8).collect::<String>();
            debug!("Skipping {}: recent sweep exists", account_preview);
            return Ok(None);
        }

        let balance = self
            .blockchain
            .get_balance(account_hex)
            .await
            .context("Failed to get balance")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        let account_preview = account_hex.chars().take(8).collect::<String>();
        let balance_tao = balance as f64 / 1e9;
        info!(
            "Account {} balance: {} TAO ({} plancks)",
            account_preview, balance_tao, balance
        );

        let decision = self.calculator.calculate(balance);

        match decision {
            SweepDecision::Sweep { amount_plancks } => {
                let account_preview = account_hex.chars().take(8).collect::<String>();
                info!(
                    "Sweeping {} from {}: {} plancks",
                    amount_plancks / 1_000_000_000,
                    account_preview,
                    amount_plancks
                );

                self.execute_sweep(account_hex, balance, amount_plancks)
                    .await?;

                Ok(Some(amount_plancks))
            }
            SweepDecision::Skip { reason } => {
                let account_preview = account_hex.chars().take(8).collect::<String>();
                let balance_tao = balance as f64 / 1e9;
                match reason {
                    SkipReason::BelowThreshold => {
                        info!(
                            "Skipping {}: balance {} TAO below threshold 0.01 TAO",
                            account_preview, balance_tao
                        );
                    }
                    SkipReason::InsufficientForFees => {
                        info!(
                            "Skipping {}: balance {} TAO insufficient after reserve (need > 0.01 TAO)",
                            account_preview, balance_tao
                        );
                    }
                    SkipReason::RecentSweep => {
                        info!("Skipping {}: recent sweep exists", account_preview);
                    }
                }
                Ok(None)
            }
        }
    }

    async fn execute_sweep(
        &self,
        account_hex: &str,
        initial_balance: u128,
        initial_sweep_amount: u128,
    ) -> Result<()> {
        let account_preview = account_hex.chars().take(8).collect::<String>();

        // HIGH-03: Check for existing pending/submitted sweeps (idempotency)
        if let Some(existing_id) = self
            .repos
            .get_pending_sweep(account_hex)
            .await
            .map_err(PaymentsError::Database)?
        {
            warn!(
                "Skipping sweep for {}: pending sweep {} already exists",
                account_preview, existing_id
            );
            return Err(PaymentsError::Reconciliation(format!(
                "Pending sweep {} exists for account",
                existing_id
            )));
        }

        let account_data = self
            .repos
            .get_by_account_hex(account_hex)
            .await
            .map_err(PaymentsError::Database)?
            .ok_or_else(|| PaymentsError::Reconciliation("Account not found".into()))?;

        let (hotwallet_ss58, _, _, encrypted_mnemonic) = account_data;
        let hotwallet_preview = hotwallet_ss58.chars().take(10).collect::<String>();
        let coldwallet_preview = self
            .config
            .coldwallet_address_ss58
            .chars()
            .take(10)
            .collect::<String>();

        // CRITICAL-03: Re-verify balance before creating sweep record to minimize race window
        let current_balance = self
            .blockchain
            .get_balance(account_hex)
            .await
            .context("Failed to re-verify balance")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        // Recalculate sweep amount based on current balance
        let sweep_decision = self.calculator.calculate(current_balance);
        let sweep_amount = match sweep_decision {
            SweepDecision::Sweep { amount_plancks } => amount_plancks,
            SweepDecision::Skip { reason } => {
                info!(
                    "Balance changed for {}: was {} plancks, now {} plancks. Skipping: {}",
                    account_preview, initial_balance, current_balance, reason
                );
                return Ok(());
            }
        };

        // Log if sweep amount changed significantly
        if sweep_amount != initial_sweep_amount {
            info!(
                "Sweep amount adjusted for {}: {} -> {} plancks (balance: {} -> {})",
                account_preview,
                initial_sweep_amount,
                sweep_amount,
                initial_balance,
                current_balance
            );
        }

        // Create sweep record
        let mut tx = self.repos.begin().await.map_err(PaymentsError::Database)?;
        let sweep_id = self
            .repos
            .insert_sweep_tx(
                &mut tx,
                account_hex,
                &hotwallet_ss58,
                &self.config.coldwallet_address_ss58,
                &current_balance.to_string(),
                &sweep_amount.to_string(),
                &self.config.estimated_fee_plancks,
                self.config.dry_run_mode,
            )
            .await
            .map_err(PaymentsError::Database)?;
        tx.commit().await.map_err(PaymentsError::Database)?;

        // Dry run mode - simulate without blockchain transaction
        if self.config.dry_run_mode {
            info!(
                "DRY RUN: Would sweep {} plancks from {} to {}",
                sweep_amount, hotwallet_preview, coldwallet_preview
            );
            let mut tx = self.repos.begin().await.map_err(PaymentsError::Database)?;
            self.repos
                .update_sweep_status_tx(
                    &mut tx,
                    sweep_id,
                    SweepStatus::Confirmed,
                    Some("DRY_RUN"),
                    None,
                    Some(&current_balance.saturating_sub(sweep_amount).to_string()),
                    None,
                )
                .await
                .map_err(PaymentsError::Database)?;
            tx.commit().await.map_err(PaymentsError::Database)?;
            return Ok(());
        }

        // Decrypt keypair for signing
        let keypair = self
            .wallet_manager
            .decrypt_and_create_keypair(&encrypted_mnemonic)?;

        // Execute blockchain transfer
        let receipt = self
            .blockchain
            .transfer(&keypair, &self.config.coldwallet_address_ss58, sweep_amount)
            .await;

        // Update sweep status based on result
        match receipt {
            Ok(receipt) => {
                info!(
                    "Transfer finalized: tx_hash={}, block_number={:?}",
                    receipt.tx_hash, receipt.block_number
                );

                let balance_after = match self.blockchain.get_balance(account_hex).await {
                    Ok(balance) => balance,
                    Err(e) => {
                        warn!(
                            "Failed to get post-sweep balance for {}: {} (sweep succeeded)",
                            account_preview, e
                        );
                        0
                    }
                };

                let mut tx = self.repos.begin().await.map_err(PaymentsError::Database)?;
                self.repos
                    .update_sweep_status_tx(
                        &mut tx,
                        sweep_id,
                        SweepStatus::Confirmed,
                        Some(&receipt.tx_hash),
                        receipt.block_number,
                        Some(&balance_after.to_string()),
                        None,
                    )
                    .await
                    .map_err(PaymentsError::Database)?;
                tx.commit().await.map_err(PaymentsError::Database)?;

                Ok(())
            }
            Err(e) => {
                warn!("Transfer failed for {}: {}", account_preview, e);

                let mut tx = self.repos.begin().await.map_err(PaymentsError::Database)?;
                self.repos
                    .update_sweep_status_tx(
                        &mut tx,
                        sweep_id,
                        SweepStatus::Failed,
                        None,
                        None,
                        None,
                        Some(&e.to_string()),
                    )
                    .await
                    .map_err(PaymentsError::Database)?;
                tx.commit().await.map_err(PaymentsError::Database)?;

                Err(e)
            }
        }
    }

    /// HIGH-08: Monitor cold wallet balance for operational visibility
    async fn monitor_cold_wallet(&self) -> Result<()> {
        // Convert SS58 address to account hex for balance query
        use sp_core::crypto::Ss58Codec;

        let coldwallet_public =
            sp_core::sr25519::Public::from_ss58check(&self.config.coldwallet_address_ss58)
                .map_err(|e| PaymentsError::Config(format!("Invalid cold wallet SS58: {}", e)))?;

        let coldwallet_hex = hex::encode(coldwallet_public.0);

        let balance = self
            .blockchain
            .get_balance(&coldwallet_hex)
            .await
            .map_err(|e| {
                warn!("Failed to query cold wallet balance: {}", e);
                e
            })?;

        let balance_tao = balance as f64 / 1e9;
        let wallet_preview = self
            .config
            .coldwallet_address_ss58
            .chars()
            .take(10)
            .collect::<String>();

        info!(
            "Cold wallet {} balance: {:.4} TAO ({} plancks)",
            wallet_preview, balance_tao, balance
        );

        // Emit metrics if available
        if let Some(ref metrics) = self.metrics {
            metrics
                .business_metrics()
                .record_payment_processed(balance_tao, &[("type", "cold_wallet_balance")])
                .await;
        }

        Ok(())
    }

    /// HIGH-07: Reconcile sweeps stuck in pending/submitted state
    async fn reconcile_stale_sweeps(&self) -> Result<()> {
        // Consider sweeps stale after 2x the sweep interval
        let stale_threshold = (self.config.sweep_interval_seconds * 2) as i64;

        let stale_sweeps = self
            .repos
            .list_stale_sweeps(stale_threshold)
            .await
            .map_err(PaymentsError::Database)?;

        if stale_sweeps.is_empty() {
            return Ok(());
        }

        info!(
            "Found {} stale sweeps to reconcile (older than {}s)",
            stale_sweeps.len(),
            stale_threshold
        );

        for sweep in stale_sweeps {
            let account_preview = sweep.account_hex.chars().take(8).collect::<String>();

            // Check current balance on-chain
            let current_balance = match self.blockchain.get_balance(&sweep.account_hex).await {
                Ok(balance) => balance,
                Err(e) => {
                    warn!(
                        "Failed to get balance for stale sweep {} (account {}): {}",
                        sweep.id, account_preview, e
                    );
                    continue;
                }
            };

            // Parse stored values - skip if parsing fails to avoid incorrect reconciliation
            let balance_before: u128 = match sweep.balance_before_plancks.parse() {
                Ok(v) => v,
                Err(e) => {
                    error!(
                        "Failed to parse balance_before for sweep {}: {}",
                        sweep.id, e
                    );
                    continue;
                }
            };
            let sweep_amount: u128 = match sweep.sweep_amount_plancks.parse() {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to parse sweep_amount for sweep {}: {}", sweep.id, e);
                    continue;
                }
            };

            // If balance dropped significantly, the sweep likely succeeded
            let expected_balance_after = balance_before.saturating_sub(sweep_amount);
            let balance_diff = balance_before.saturating_sub(current_balance);

            // Consider sweep successful if balance dropped by at least 80% of sweep amount
            let sweep_likely_succeeded = balance_diff >= (sweep_amount * 80 / 100);

            let mut tx = self.repos.begin().await.map_err(PaymentsError::Database)?;

            if sweep_likely_succeeded {
                info!(
                    "Stale sweep {} likely succeeded: balance dropped from {} to {} plancks",
                    sweep.id, balance_before, current_balance
                );
                self.repos
                    .update_sweep_status_tx(
                        &mut tx,
                        sweep.id,
                        SweepStatus::Confirmed,
                        sweep.tx_hash.as_deref(),
                        sweep.block_number,
                        Some(&current_balance.to_string()),
                        Some("Reconciled: balance confirms sweep succeeded"),
                    )
                    .await
                    .map_err(PaymentsError::Database)?;
            } else {
                info!(
                    "Stale sweep {} likely failed: balance {} vs expected {} plancks",
                    sweep.id, current_balance, expected_balance_after
                );
                self.repos
                    .update_sweep_status_tx(
                        &mut tx,
                        sweep.id,
                        SweepStatus::Failed,
                        sweep.tx_hash.as_deref(),
                        sweep.block_number,
                        Some(&current_balance.to_string()),
                        Some("Reconciled: timed out, marked failed for retry"),
                    )
                    .await
                    .map_err(PaymentsError::Database)?;
            }

            tx.commit().await.map_err(PaymentsError::Database)?;
        }

        Ok(())
    }
}
