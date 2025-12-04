use crate::error::Result;
use crate::storage::events::{
    BatchRepository, BatchStatus, BatchType, BillingEvent, EventRepository, EventStatistics,
    ProcessingBatch, UsageEvent,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

pub struct EventStore {
    event_repository: Arc<dyn EventRepository + Send + Sync>,
    batch_repository: Arc<dyn BatchRepository + Send + Sync>,
    batch_size: usize,
    retention_days: u32,
}

impl EventStore {
    pub fn new(
        event_repository: Arc<dyn EventRepository + Send + Sync>,
        batch_repository: Arc<dyn BatchRepository + Send + Sync>,
        batch_size: usize,
        retention_days: u32,
    ) -> Self {
        Self {
            event_repository,
            batch_repository,
            batch_size,
            retention_days,
        }
    }

    /// Get a reference to the underlying event repository
    pub fn event_repository(&self) -> &dyn EventRepository {
        self.event_repository.as_ref()
    }

    /// Append a usage event to the store
    pub async fn append_usage_event(&self, event: &UsageEvent) -> Result<Uuid> {
        debug!("Appending usage event: {}", event.event_id);
        self.event_repository.append_usage_event(event).await
    }

    /// Append multiple usage events in a batch
    pub async fn append_usage_events_batch(&self, events: &[UsageEvent]) -> Result<Vec<Uuid>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }

        info!("Appending {} usage events in batch", events.len());
        self.event_repository
            .append_usage_events_batch(events)
            .await
    }

    /// Append a billing event
    pub async fn append_billing_event(&self, event: &BillingEvent) -> Result<Uuid> {
        debug!(
            "Appending billing event: {} - {}",
            event.event_type, event.event_id
        );
        self.event_repository.append_billing_event(event).await
    }

    /// Get unprocessed events for processing
    pub async fn get_unprocessed_events(&self, limit: Option<i64>) -> Result<Vec<UsageEvent>> {
        let actual_limit = limit.or(Some(self.batch_size as i64));
        self.event_repository
            .get_unprocessed_events(actual_limit)
            .await
    }

    /// Mark events as processed
    pub async fn mark_events_processed(&self, event_ids: &[Uuid], batch_id: Uuid) -> Result<()> {
        if !event_ids.is_empty() {
            debug!(
                "Marking {} events as processed in batch {}",
                event_ids.len(),
                batch_id
            );
        }
        self.event_repository
            .mark_events_processed(event_ids, batch_id)
            .await
    }

    /// Get events for a specific rental
    pub async fn get_rental_events(
        &self,
        rental_id: Uuid,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>> {
        self.event_repository
            .get_rental_events(rental_id, start_time, end_time)
            .await
    }

    /// Store a generic event
    pub async fn store_event(
        &self,
        entity_id: String,
        event_type: String,
        event_data: serde_json::Value,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid> {
        let event = BillingEvent {
            event_id: Uuid::new_v4(),
            event_type,
            entity_type: "rental".to_string(),
            entity_id,
            user_id: None,
            event_data,
            metadata,
            created_by: "telemetry".to_string(),
            created_at: Utc::now(),
        };

        self.append_billing_event(&event).await
    }

    /// Get events by entity ID
    pub async fn get_events_by_entity(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<BillingEvent>> {
        self.event_repository
            .get_events_by_entity(entity_id, limit)
            .await
    }

    /// Get event statistics
    pub async fn get_event_statistics(&self) -> Result<EventStatistics> {
        self.event_repository.get_event_statistics().await
    }

    /// Clean up old events based on retention policy
    pub async fn cleanup_old_events(&self) -> Result<u64> {
        let count = self
            .event_repository
            .cleanup_old_events(self.retention_days)
            .await?;
        if count > 0 {
            info!("Archived {} old events", count);
        }
        Ok(count)
    }

    /// Create a new processing batch
    pub async fn create_batch(&self, batch_type: BatchType) -> Result<ProcessingBatch> {
        self.batch_repository.create_batch(batch_type).await
    }

    /// Update batch status
    pub async fn update_batch_status(
        &self,
        batch_id: Uuid,
        status: BatchStatus,
        events_count: Option<i32>,
    ) -> Result<()> {
        self.batch_repository
            .update_batch_status(batch_id, status, events_count)
            .await
    }

    /// Complete a batch
    pub async fn complete_batch(
        &self,
        batch_id: Uuid,
        processed_count: i32,
        failed_count: i32,
    ) -> Result<()> {
        self.batch_repository
            .complete_batch(batch_id, processed_count, failed_count)
            .await
    }

    /// Get a batch by ID
    pub async fn get_batch(&self, batch_id: Uuid) -> Result<Option<ProcessingBatch>> {
        self.batch_repository.get_batch(batch_id).await
    }
}

#[async_trait]
pub trait EventStoreOperations: Send + Sync {
    async fn append_event(&self, event: &UsageEvent) -> Result<Uuid>;
    async fn get_unprocessed(&self, limit: Option<i64>) -> Result<Vec<UsageEvent>>;
    async fn mark_processed(&self, event_ids: &[Uuid], batch_id: Uuid) -> Result<()>;
}

#[async_trait]
impl EventStoreOperations for EventStore {
    async fn append_event(&self, event: &UsageEvent) -> Result<Uuid> {
        self.append_usage_event(event).await
    }

    async fn get_unprocessed(&self, limit: Option<i64>) -> Result<Vec<UsageEvent>> {
        self.get_unprocessed_events(limit).await
    }

    async fn mark_processed(&self, event_ids: &[Uuid], batch_id: Uuid) -> Result<()> {
        self.mark_events_processed(event_ids, batch_id).await
    }
}
