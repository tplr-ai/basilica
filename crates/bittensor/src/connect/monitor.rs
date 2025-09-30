//! Generic blockchain monitoring utilities
//!
//! This module provides reusable blockchain monitoring functionality that can be used
//! by various services to watch for on-chain events.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use subxt::{OnlineClient, PolkadotConfig};
use tracing::{debug, info, warn};

use crate::connect::{ConnectionPool, ConnectionPoolBuilder, ConnectionPoolTrait, HealthChecker};

/// Handler for blockchain events
///
/// Implement this trait to handle specific blockchain events in your service
#[async_trait]
pub trait BlockchainEventHandler: Send + Sync {
    /// Handle a Balance.Transfer event
    ///
    /// # Arguments
    /// * `from` - Source account (hex encoded)
    /// * `to` - Destination account (hex encoded)
    /// * `amount` - Transfer amount as string
    /// * `block_number` - Block number where event occurred
    /// * `event_index` - Event index within the block
    async fn handle_transfer(
        &self,
        from: &str,
        to: &str,
        amount: &str,
        block_number: u32,
        event_index: usize,
    ) -> Result<()>;

    /// Called when starting to process a new block
    async fn on_block_start(&self, block_number: u32) -> Result<()> {
        let _ = block_number;
        Ok(())
    }

    /// Called after processing all events in a block
    async fn on_block_end(&self, block_number: u32) -> Result<()> {
        let _ = block_number;
        Ok(())
    }
}

/// Generic blockchain monitor
///
/// Monitors blockchain for events and delegates handling to the provided handler
pub struct BlockchainMonitor<H, P>
where
    H: BlockchainEventHandler,
    P: ConnectionPoolTrait + Send + Sync + 'static,
{
    pool: Arc<P>,
    handler: H,
    health_monitor: Option<tokio::task::JoinHandle<()>>,
}

impl<H> BlockchainMonitor<H, ConnectionPool>
where
    H: BlockchainEventHandler,
{
    /// Create a new blockchain monitor backed by a connection pool and health checks
    ///
    /// # Arguments
    /// * `endpoints` - One or more WebSocket endpoints for failover
    /// * `handler` - Event handler implementation
    pub async fn new(endpoints: Vec<String>, handler: H) -> Result<Self>
    where
        H: BlockchainEventHandler + 'static,
    {
        if endpoints.is_empty() {
            anyhow::bail!("no endpoints provided to BlockchainMonitor::new");
        }

        let max = endpoints.len().min(3);
        let pool = Arc::new(
            ConnectionPoolBuilder::new(endpoints)
                .max_connections(max)
                .build(),
        );

        Self::with_pool(pool, handler).await
    }
}

impl<H, P> BlockchainMonitor<H, P>
where
    H: BlockchainEventHandler,
    P: ConnectionPoolTrait + Send + Sync + 'static,
{
    /// Construct a monitor from any pool implementing ConnectionPoolTrait
    pub async fn with_pool(pool: Arc<P>, handler: H) -> Result<Self>
    where
        H: BlockchainEventHandler + 'static,
    {
        // Initialize connections (best-effort)
        if let Err(e) = pool.reconnect_with_backoff().await {
            warn!(
                "BlockchainMonitor: initial connection attempt failed: {}",
                e
            );
        }

        // Start health monitoring in background
        let checker = Arc::new(
            HealthChecker::new()
                .with_interval(Duration::from_secs(60))
                .with_timeout(Duration::from_secs(5))
                .with_failure_threshold(3),
        );
        let health_monitor = Some(checker.start_monitoring(Arc::clone(&pool)));

        Ok(Self {
            pool,
            handler,
            health_monitor,
        })
    }

    /// Run the monitor, subscribing to finalized blocks with automatic reconnection
    ///
    /// This runs indefinitely, re-subscribing on errors or disconnects.
    pub async fn run(self) -> Result<()> {
        info!("Starting blockchain monitor for finalized blocks (with failover)");

        let mut attempts: u32 = 0;
        loop {
            // Get healthy client and subscribe
            let client = match self.pool.get_healthy_client().await {
                Ok(c) => c,
                Err(e) => {
                    attempts = attempts.saturating_add(1);
                    warn!("No healthy chain client (attempt {}): {}", attempts, e);
                    tokio::time::sleep(self.retry_delay(attempts)).await;
                    continue;
                }
            };

            info!("Subscribed to finalized blocks");
            let subscribe_result = client.blocks().subscribe_finalized().await;
            let mut sub = match subscribe_result {
                Ok(s) => s,
                Err(e) => {
                    attempts = attempts.saturating_add(1);
                    warn!(
                        "Failed to subscribe to finalized blocks (attempt {}): {}",
                        attempts, e
                    );
                    tokio::time::sleep(self.retry_delay(attempts)).await;
                    continue;
                }
            };

            // Reset attempts after successful subscribe
            attempts = 0;

            // Stream loop; break on error to resubscribe
            while let Some(block_result) = sub.next().await {
                match block_result {
                    Ok(block) => {
                        if let Err(e) = self.process_block(block).await {
                            warn!("Error processing block: {}", e);
                        }
                    }
                    Err(e) => {
                        warn!("Block subscription error: {} (will resubscribe)", e);
                        break; // Break to outer loop to resubscribe
                    }
                }
            }

            debug!("Subscription ended; attempting to resubscribe");
        }
    }

    fn retry_delay(&self, attempt: u32) -> Duration {
        let base = Duration::from_millis(500);
        let max = Duration::from_secs(10);
        let exp = base * 2u32.saturating_pow(attempt.min(5).saturating_sub(1));
        exp.min(max)
    }

    /// Process a single block
    async fn process_block(
        &self,
        block: subxt::blocks::Block<PolkadotConfig, OnlineClient<PolkadotConfig>>,
    ) -> Result<()> {
        let block_number = block.number();

        self.handler.on_block_start(block_number).await?;

        let events = match block.events().await {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to get events for block {}: {}", block_number, e);
                return Ok(());
            }
        };

        for (idx, ev_result) in events.iter().enumerate() {
            let ev = match ev_result {
                Ok(e) => e,
                Err(e) => {
                    debug!("Skipping event due to error: {}", e);
                    continue;
                }
            };

            // We're interested in Balance.Transfer events
            if ev.pallet_name() == "Balances" && ev.variant_name() == "Transfer" {
                if let Some((from, to, amount)) = Self::extract_transfer_details(&ev) {
                    self.handler
                        .handle_transfer(&from, &to, &amount, block_number, idx)
                        .await?;
                }
            }
        }

        self.handler.on_block_end(block_number).await?;
        Ok(())
    }

    /// Extract transfer details from an event
    fn extract_transfer_details(
        ev: &subxt::events::EventDetails<PolkadotConfig>,
    ) -> Option<(String, String, String)> {
        let fields = ev.field_values().ok()?;

        match fields {
            subxt::ext::scale_value::Composite::Named(fields) => {
                Self::extract_named_transfer_fields(fields)
            }
            subxt::ext::scale_value::Composite::Unnamed(fields) => {
                Self::extract_unnamed_transfer_fields(&fields)
            }
        }
    }

    fn extract_named_transfer_fields(
        fields: Vec<(String, subxt::ext::scale_value::Value<u32>)>,
    ) -> Option<(String, String, String)> {
        let mut from = None;
        let mut to = None;
        let mut amount = None;

        for (name, value) in fields {
            match name.as_str() {
                "from" => from = extract_account_hex(&value),
                "to" => to = extract_account_hex(&value),
                "amount" => amount = Some(value.to_string()),
                _ => {}
            }
        }

        match (from, to, amount) {
            (Some(f), Some(t), Some(a)) => Some((f, t, a)),
            _ => None,
        }
    }

    fn extract_unnamed_transfer_fields(
        fields: &[subxt::ext::scale_value::Value<u32>],
    ) -> Option<(String, String, String)> {
        if fields.len() < 3 {
            return None;
        }
        let from = extract_account_hex(&fields[0])?;
        let to = extract_account_hex(&fields[1])?;
        let amount = fields[2].to_string();
        Some((from, to, amount))
    }
}

impl<H, P> Drop for BlockchainMonitor<H, P>
where
    H: BlockchainEventHandler,
    P: ConnectionPoolTrait + Send + Sync + 'static,
{
    fn drop(&mut self) {
        if let Some(handle) = self.health_monitor.take() {
            handle.abort();
        }
    }
}

/// Extract account ID as hex string from a Value
pub fn extract_account_hex(value: &subxt::ext::scale_value::Value<u32>) -> Option<String> {
    let bytes = extract_account_bytes(value)?;
    Some(to_hex(&bytes))
}

/// Extract account ID bytes from a Value
pub fn extract_account_bytes(value: &subxt::ext::scale_value::Value<u32>) -> Option<Vec<u8>> {
    let s = value.to_string();

    // Handle hex string format (0x...)
    if s.starts_with("0x") && s.len() == 66 {
        hex::decode(&s[2..]).ok()
    } else if s.len() == 64 {
        // Raw hex without 0x prefix
        hex::decode(&s).ok()
    } else {
        None
    }
}

/// Convert bytes to hex string
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
