use crate::error::{BillingError, Result};
use crate::storage::rds::RdsConnection;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use sqlx::Row;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Telemetry,
    StatusChange,
    CostUpdate,
    RentalStart,
    RentalEnd,
    ResourceUpdate,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Telemetry => write!(f, "telemetry"),
            EventType::StatusChange => write!(f, "status_change"),
            EventType::CostUpdate => write!(f, "cost_update"),
            EventType::RentalStart => write!(f, "rental_start"),
            EventType::RentalEnd => write!(f, "rental_end"),
            EventType::ResourceUpdate => write!(f, "resource_update"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub event_id: Uuid,
    pub rental_id: Uuid,
    pub user_id: String,
    pub node_id: String,
    pub validator_id: String,
    pub event_type: EventType,
    pub event_data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub processed: bool,
    pub processed_at: Option<DateTime<Utc>>,
    pub batch_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingEvent {
    pub event_id: Uuid,
    pub event_type: String,
    pub entity_type: String,
    pub entity_id: String,
    pub user_id: Option<Uuid>,
    pub event_data: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct EventStatistics {
    pub unprocessed_count: u64,
    pub processed_count: u64,
    pub total_count: u64,
    pub oldest_event: Option<DateTime<Utc>>,
    pub newest_event: Option<DateTime<Utc>>,
}

#[async_trait]
pub trait EventRepository: Send + Sync {
    async fn append_usage_event(&self, event: &UsageEvent) -> Result<Uuid>;
    async fn append_usage_events_batch(&self, events: &[UsageEvent]) -> Result<Vec<Uuid>>;
    async fn get_unprocessed_events(&self, limit: Option<i64>) -> Result<Vec<UsageEvent>>;
    async fn mark_events_processed(&self, event_ids: &[Uuid], batch_id: Uuid) -> Result<()>;
    async fn get_rental_events(
        &self,
        rental_id: Uuid,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>>;

    async fn append_billing_event(&self, event: &BillingEvent) -> Result<Uuid>;
    async fn get_events_by_entity(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<BillingEvent>>;

    async fn get_event_statistics(&self) -> Result<EventStatistics>;
    async fn cleanup_old_events(&self, retention_days: u32) -> Result<u64>;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BatchType {
    UsageAggregation,
    BillingCalculation,
    TelemetryProcessing,
}

impl std::fmt::Display for BatchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchType::UsageAggregation => write!(f, "usage_aggregation"),
            BatchType::BillingCalculation => write!(f, "billing_calculation"),
            BatchType::TelemetryProcessing => write!(f, "telemetry_processing"),
        }
    }
}

impl BatchType {
    fn from_str(s: &str) -> Self {
        match s {
            "usage_aggregation" => BatchType::UsageAggregation,
            "billing_calculation" => BatchType::BillingCalculation,
            "telemetry_processing" => BatchType::TelemetryProcessing,
            _ => BatchType::UsageAggregation,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl std::fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchStatus::Pending => write!(f, "pending"),
            BatchStatus::Processing => write!(f, "processing"),
            BatchStatus::Completed => write!(f, "completed"),
            BatchStatus::Failed => write!(f, "failed"),
        }
    }
}

impl BatchStatus {
    fn from_str(s: &str) -> Self {
        match s {
            "pending" => BatchStatus::Pending,
            "processing" => BatchStatus::Processing,
            "completed" => BatchStatus::Completed,
            "failed" => BatchStatus::Failed,
            _ => BatchStatus::Pending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingBatch {
    pub batch_id: Uuid,
    pub batch_type: BatchType,
    pub status: BatchStatus,
    pub events_count: i32,
    pub events_processed: i32,
    pub events_failed: i32,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[async_trait]
pub trait BatchRepository: Send + Sync {
    async fn create_batch(&self, batch_type: BatchType) -> Result<ProcessingBatch>;
    async fn update_batch_status(
        &self,
        batch_id: Uuid,
        status: BatchStatus,
        events_count: Option<i32>,
    ) -> Result<()>;
    async fn complete_batch(
        &self,
        batch_id: Uuid,
        processed_count: i32,
        failed_count: i32,
    ) -> Result<()>;
    async fn get_batch(&self, batch_id: Uuid) -> Result<Option<ProcessingBatch>>;
}

pub struct SqlEventRepository {
    connection: Arc<RdsConnection>,
}

impl SqlEventRepository {
    pub fn new(connection: Arc<RdsConnection>) -> Self {
        Self { connection }
    }

    fn validate_event_ids(event: &UsageEvent) -> Result<()> {
        if event.event_id.is_nil() {
            return Err(BillingError::ValidationError {
                field: "event_id".to_string(),
                message: "event_id cannot be nil for usage event".to_string(),
            });
        }
        if event.user_id.is_empty() {
            return Err(BillingError::ValidationError {
                field: "user_id".to_string(),
                message: "user_id cannot be empty for usage event".to_string(),
            });
        }
        if event.node_id.is_empty() {
            return Err(BillingError::ValidationError {
                field: "node_id".to_string(),
                message: "node_id cannot be empty for usage event".to_string(),
            });
        }
        if event.validator_id.is_empty() {
            return Err(BillingError::ValidationError {
                field: "validator_id".to_string(),
                message: "validator_id cannot be empty for usage event".to_string(),
            });
        }
        Ok(())
    }

    fn parse_event_type(event_type_str: &str) -> EventType {
        match event_type_str {
            "telemetry" => EventType::Telemetry,
            "status_change" => EventType::StatusChange,
            "cost_update" => EventType::CostUpdate,
            "rental_start" => EventType::RentalStart,
            "rental_end" => EventType::RentalEnd,
            "resource_update" => EventType::ResourceUpdate,
            _ => EventType::Telemetry,
        }
    }
}

#[async_trait]
impl EventRepository for SqlEventRepository {
    async fn append_usage_event(&self, event: &UsageEvent) -> Result<Uuid> {
        Self::validate_event_ids(event)?;

        sqlx::query(
            r#"
            INSERT INTO billing.usage_events (
                event_id, rental_id, user_id, node_id, validator_id, event_type,
                event_data, timestamp, processed
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(event.event_id)
        .bind(event.rental_id)
        .bind(&event.user_id)
        .bind(&event.node_id)
        .bind(&event.validator_id)
        .bind(event.event_type.to_string())
        .bind(&event.event_data)
        .bind(event.timestamp)
        .bind(false)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to append usage event".to_string(),
            source: Box::new(e),
        })?;

        Ok(event.event_id)
    }

    async fn append_usage_events_batch(&self, events: &[UsageEvent]) -> Result<Vec<Uuid>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }

        let mut tx =
            self.connection
                .pool()
                .begin()
                .await
                .map_err(|e| BillingError::EventStoreError {
                    message: "Failed to begin transaction".to_string(),
                    source: Box::new(e),
                })?;

        let mut event_ids = Vec::with_capacity(events.len());

        for event in events {
            Self::validate_event_ids(event)?;

            sqlx::query(
                r#"
                INSERT INTO billing.usage_events (
                    event_id, rental_id, user_id, node_id, validator_id, event_type,
                    event_data, timestamp, processed
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                "#,
            )
            .bind(event.event_id)
            .bind(event.rental_id)
            .bind(&event.user_id)
            .bind(&event.node_id)
            .bind(&event.validator_id)
            .bind(event.event_type.to_string())
            .bind(&event.event_data)
            .bind(event.timestamp)
            .bind(false)
            .execute(&mut *tx)
            .await
            .map_err(|e| BillingError::EventStoreError {
                message: "Failed to insert event in batch".to_string(),
                source: Box::new(e),
            })?;

            event_ids.push(event.event_id);
        }

        tx.commit()
            .await
            .map_err(|e| BillingError::EventStoreError {
                message: "Failed to commit batch transaction".to_string(),
                source: Box::new(e),
            })?;

        Ok(event_ids)
    }

    async fn get_unprocessed_events(&self, limit: Option<i64>) -> Result<Vec<UsageEvent>> {
        let actual_limit = limit.unwrap_or(1000);

        let rows = sqlx::query(
            r#"
            SELECT
                event_id, rental_id, user_id, node_id, validator_id, event_type,
                event_data, timestamp, processed, processed_at, batch_id
            FROM billing.usage_events
            WHERE processed = false
            ORDER BY timestamp ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(actual_limit)
        .fetch_all(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to fetch unprocessed events".to_string(),
            source: Box::new(e),
        })?;

        let events = rows
            .into_iter()
            .map(|row| {
                let event_type_str: String = row.get("event_type");
                UsageEvent {
                    event_id: row.get("event_id"),
                    rental_id: row.get("rental_id"),
                    user_id: row.get("user_id"),
                    node_id: row.get("node_id"),
                    validator_id: row.get("validator_id"),
                    event_type: Self::parse_event_type(&event_type_str),
                    event_data: row.get("event_data"),
                    timestamp: row.get("timestamp"),
                    processed: row.get("processed"),
                    processed_at: row.get("processed_at"),
                    batch_id: row.get("batch_id"),
                }
            })
            .collect();

        Ok(events)
    }

    async fn mark_events_processed(&self, event_ids: &[Uuid], batch_id: Uuid) -> Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }

        let event_ids_vec: Vec<Uuid> = event_ids.to_vec();

        sqlx::query(
            r#"
            UPDATE billing.usage_events
            SET processed = true,
                processed_at = NOW(),
                batch_id = $1
            WHERE event_id = ANY($2)
            "#,
        )
        .bind(batch_id)
        .bind(&event_ids_vec)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to mark events as processed".to_string(),
            source: Box::new(e),
        })?;

        Ok(())
    }

    async fn get_rental_events(
        &self,
        rental_id: Uuid,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>> {
        let start = start_time.unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
        let end = end_time.unwrap_or_else(Utc::now);

        let rows = sqlx::query(
            r#"
            SELECT
                event_id, rental_id, user_id, node_id, validator_id, event_type,
                event_data, timestamp, processed, processed_at, batch_id
            FROM billing.usage_events
            WHERE rental_id = $1
                AND timestamp >= $2
                AND timestamp <= $3
            ORDER BY timestamp ASC
            "#,
        )
        .bind(rental_id)
        .bind(start)
        .bind(end)
        .fetch_all(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: format!("Failed to fetch events for rental {}", rental_id),
            source: Box::new(e),
        })?;

        let events = rows
            .into_iter()
            .map(|row| {
                let event_type_str: String = row.get("event_type");
                UsageEvent {
                    event_id: row.get("event_id"),
                    rental_id: row.get("rental_id"),
                    user_id: row.get("user_id"),
                    node_id: row.get("node_id"),
                    validator_id: row.get("validator_id"),
                    event_type: Self::parse_event_type(&event_type_str),
                    event_data: row.get("event_data"),
                    timestamp: row.get("timestamp"),
                    processed: row.get("processed"),
                    processed_at: row.get("processed_at"),
                    batch_id: row.get("batch_id"),
                }
            })
            .collect();

        Ok(events)
    }

    async fn append_billing_event(&self, event: &BillingEvent) -> Result<Uuid> {
        let event_id = event.event_id;

        sqlx::query(
            r#"
            INSERT INTO billing.billing_events (
                event_id, event_type, entity_type, entity_id,
                user_id, event_data, metadata, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(event_id)
        .bind(&event.event_type)
        .bind(&event.entity_type)
        .bind(&event.entity_id)
        .bind(event.user_id)
        .bind(&event.event_data)
        .bind(&event.metadata)
        .bind(&event.created_by)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to append billing event".to_string(),
            source: Box::new(e),
        })?;

        Ok(event_id)
    }

    async fn get_events_by_entity(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<BillingEvent>> {
        let pool = self.connection.get_pool().await?;

        let query = if let Some(limit) = limit {
            sqlx::query(
                r#"
                SELECT event_id, event_type, entity_type, entity_id, user_id,
                       event_data, metadata, created_by, created_at
                FROM billing.billing_events
                WHERE entity_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                "#,
            )
            .bind(entity_id)
            .bind(limit as i64)
        } else {
            sqlx::query(
                r#"
                SELECT event_id, event_type, entity_type, entity_id, user_id,
                       event_data, metadata, created_by, created_at
                FROM billing.billing_events
                WHERE entity_id = $1
                ORDER BY created_at DESC
                "#,
            )
            .bind(entity_id)
        };

        let rows = query.fetch_all(&pool).await?;

        let events = rows
            .into_iter()
            .map(|row| BillingEvent {
                event_id: row.get("event_id"),
                event_type: row.get("event_type"),
                entity_type: row.get("entity_type"),
                entity_id: row.get("entity_id"),
                user_id: row.get("user_id"),
                event_data: row.get("event_data"),
                metadata: row.get("metadata"),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
            })
            .collect();

        Ok(events)
    }

    async fn get_event_statistics(&self) -> Result<EventStatistics> {
        let stats = sqlx::query(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE processed = false) as unprocessed_count,
                COUNT(*) FILTER (WHERE processed = true) as processed_count,
                COUNT(*) as total_count,
                MIN(timestamp) as oldest_event,
                MAX(timestamp) as newest_event
            FROM billing.usage_events
            "#,
        )
        .fetch_one(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to get event statistics".to_string(),
            source: Box::new(e),
        })?;

        Ok(EventStatistics {
            unprocessed_count: stats
                .get::<Option<i64>, _>("unprocessed_count")
                .unwrap_or(0) as u64,
            processed_count: stats.get::<Option<i64>, _>("processed_count").unwrap_or(0) as u64,
            total_count: stats.get::<Option<i64>, _>("total_count").unwrap_or(0) as u64,
            oldest_event: stats.get("oldest_event"),
            newest_event: stats.get("newest_event"),
        })
    }

    async fn cleanup_old_events(&self, retention_days: u32) -> Result<u64> {
        let cutoff_date = Utc::now() - chrono::Duration::days(retention_days as i64);

        let archived = sqlx::query(
            r#"
            WITH archived AS (
                INSERT INTO billing.usage_events_archive
                SELECT * FROM billing.usage_events
                WHERE timestamp < $1 AND processed = true
                RETURNING 1
            )
            SELECT COUNT(*) as count FROM archived
            "#,
        )
        .bind(cutoff_date)
        .fetch_one(self.connection.pool())
        .await
        .map_err(|e| BillingError::EventStoreError {
            message: "Failed to archive old events".to_string(),
            source: Box::new(e),
        })?;

        let count: i64 = archived.get("count");
        Ok(count as u64)
    }
}

pub struct SqlBatchRepository {
    connection: Arc<RdsConnection>,
}

impl SqlBatchRepository {
    pub fn new(connection: Arc<RdsConnection>) -> Self {
        Self { connection }
    }
}

#[async_trait]
impl BatchRepository for SqlBatchRepository {
    async fn create_batch(&self, batch_type: BatchType) -> Result<ProcessingBatch> {
        let batch_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO billing.processing_batches (
                batch_id, batch_type, status
            ) VALUES ($1, $2, $3)
            "#,
        )
        .bind(batch_id)
        .bind(batch_type.to_string())
        .bind(BatchStatus::Pending.to_string())
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "create_batch".to_string(),
            source: Box::new(e),
        })?;

        Ok(ProcessingBatch {
            batch_id,
            batch_type,
            status: BatchStatus::Pending,
            events_count: 0,
            events_processed: 0,
            events_failed: 0,
            started_at: None,
            completed_at: None,
            error_message: None,
            metadata: None,
        })
    }

    async fn update_batch_status(
        &self,
        batch_id: Uuid,
        status: BatchStatus,
        events_count: Option<i32>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE billing.processing_batches
            SET status = $1,
                events_count = COALESCE($2, events_count),
                started_at = CASE WHEN $1 = 'processing' THEN NOW() ELSE started_at END
            WHERE batch_id = $3
            "#,
        )
        .bind(status.to_string())
        .bind(events_count)
        .bind(batch_id)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "update_batch_status".to_string(),
            source: Box::new(e),
        })?;

        Ok(())
    }

    async fn complete_batch(
        &self,
        batch_id: Uuid,
        processed_count: i32,
        failed_count: i32,
    ) -> Result<()> {
        let status = if failed_count == 0 {
            BatchStatus::Completed
        } else {
            BatchStatus::Failed
        };

        sqlx::query(
            r#"
            UPDATE billing.processing_batches
            SET status = $1,
                events_processed = $2,
                events_failed = $3,
                completed_at = NOW()
            WHERE batch_id = $4
            "#,
        )
        .bind(status.to_string())
        .bind(processed_count)
        .bind(failed_count)
        .bind(batch_id)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "complete_batch".to_string(),
            source: Box::new(e),
        })?;

        Ok(())
    }

    async fn get_batch(&self, batch_id: Uuid) -> Result<Option<ProcessingBatch>> {
        let row = sqlx::query(
            r#"
            SELECT batch_id, batch_type, status, events_count, events_processed,
                   events_failed, started_at, completed_at, error_message, metadata
            FROM billing.processing_batches
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .fetch_optional(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "get_batch".to_string(),
            source: Box::new(e),
        })?;

        if let Some(row) = row {
            let batch_type_str: String = row.get("batch_type");
            let status_str: String = row.get("status");

            Ok(Some(ProcessingBatch {
                batch_id: row.get("batch_id"),
                batch_type: BatchType::from_str(&batch_type_str),
                status: BatchStatus::from_str(&status_str),
                events_count: row.get("events_count"),
                events_processed: row.get("events_processed"),
                events_failed: row.get("events_failed"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
                error_message: row.get("error_message"),
                metadata: row.get("metadata"),
            }))
        } else {
            Ok(None)
        }
    }
}
