use crate::domain::types::{RentalId, UsageMetrics};
use crate::error::{BillingError, Result};
use crate::storage::events::{EventType, UsageEvent};
use crate::storage::rds::RdsConnection;
use crate::storage::{RentalRepository, SqlRentalRepository};

use basilica_protocol::billing::TelemetryData;
use chrono;
use rust_decimal::prelude::*;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, error, warn};
use uuid::Uuid;

pub struct TelemetryProcessor {
    event_store: Arc<crate::domain::events::EventStore>,
    rental_repository: Arc<dyn RentalRepository + Send + Sync>,
}

impl TelemetryProcessor {
    pub fn new(rds_connection: Arc<RdsConnection>) -> Self {
        let rental_repository = Arc::new(SqlRentalRepository::new(rds_connection.clone()));

        let event_repository = Arc::new(crate::storage::events::SqlEventRepository::new(
            rds_connection.clone(),
        ));
        let batch_repository = Arc::new(crate::storage::events::SqlBatchRepository::new(
            rds_connection.clone(),
        ));

        Self {
            event_store: Arc::new(crate::domain::events::EventStore::new(
                event_repository,
                batch_repository,
                1000,
                30,
            )),
            rental_repository,
        }
    }

    /// Process a single telemetry data point
    pub async fn process_telemetry(&self, data: TelemetryData) -> Result<()> {
        debug!(
            "Processing telemetry for rental {} from node {}",
            data.rental_id, data.node_id
        );

        let rental_id =
            RentalId::from_str(&data.rental_id).map_err(|e| BillingError::ValidationError {
                field: "rental_id".to_string(),
                message: format!("Invalid rental ID: {}", e),
            })?;

        let rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await
            .map_err(|e| {
                warn!("Failed to fetch rental {} for telemetry: {}", rental_id, e);
                BillingError::ValidationError {
                    field: "rental_id".to_string(),
                    message: format!("Failed to fetch rental: {}", e),
                }
            })?
            .ok_or_else(|| {
                warn!("Rental {} not found for telemetry", rental_id);
                BillingError::ValidationError {
                    field: "rental_id".to_string(),
                    message: format!("Rental not found: {}", rental_id),
                }
            })?;

        let usage_metrics = if let Some(ref usage) = data.resource_usage {
            UsageMetrics {
                cpu_hours: Decimal::from_f64(usage.cpu_percent / 100.0).unwrap_or(Decimal::ZERO),
                memory_gb_hours: Decimal::from(usage.memory_mb) / Decimal::from(1024),
                gpu_hours: usage
                    .gpu_usage
                    .iter()
                    .map(|gpu| {
                        Decimal::from_f64(gpu.utilization_percent / 100.0).unwrap_or(Decimal::ZERO)
                    })
                    .sum(),
                network_gb: (Decimal::from(usage.network_rx_bytes + usage.network_tx_bytes))
                    / Decimal::from(1_073_741_824u64),
                storage_gb_hours: Decimal::ZERO, // Not tracked yet
                disk_io_gb: (Decimal::from(usage.disk_read_bytes + usage.disk_write_bytes))
                    / Decimal::from(1_073_741_824u64),
            }
        } else {
            UsageMetrics::zero()
        };

        let telemetry_event = UsageEvent {
            event_id: Uuid::new_v4(),
            rental_id: rental_id.as_uuid(),
            user_id: rental.user_id.to_string(),
            node_id: data.node_id.clone(),
            validator_id: rental.validator_id.clone(),
            event_type: EventType::Telemetry,
            event_data: json!({
                "cpu_percent": usage_metrics.cpu_hours.to_f64(),
                "memory_mb": data.resource_usage.as_ref().map(|u| u.memory_mb).unwrap_or(0),
                "network_rx_bytes": data.resource_usage.as_ref().map(|u| u.network_rx_bytes).unwrap_or(0),
                "network_tx_bytes": data.resource_usage.as_ref().map(|u| u.network_tx_bytes).unwrap_or(0),
                "disk_read_bytes": data.resource_usage.as_ref().map(|u| u.disk_read_bytes).unwrap_or(0),
                "disk_write_bytes": data.resource_usage.as_ref().map(|u| u.disk_write_bytes).unwrap_or(0),
                "gpu_metrics": data.resource_usage.as_ref()
                    .map(|u| json!({
                        "gpu_count": u.gpu_usage.len(),
                        "utilization": u.gpu_usage.iter().map(|g| g.utilization_percent).collect::<Vec<_>>(),
                        "memory_used": u.gpu_usage.iter().map(|g| g.memory_used_mb).collect::<Vec<_>>(),
                    })),
                "custom_metrics": data.custom_metrics,
            }),
            timestamp: chrono::Utc::now(),
            processed: false,
            processed_at: None,
            batch_id: None,
        };

        self.event_store
            .append_usage_event(&telemetry_event)
            .await
            .map_err(|e| {
                error!("Failed to store telemetry event: {}", e);
                BillingError::TelemetryError {
                    source: Box::new(e),
                }
            })?;

        Ok(())
    }

    /// Process a batch of telemetry data
    pub async fn process_batch(&self, batch: Vec<TelemetryData>) -> Result<Vec<Result<()>>> {
        let mut results = Vec::with_capacity(batch.len());

        for data in batch {
            results.push(self.process_telemetry(data).await);
        }

        Ok(results)
    }

    /// Get aggregated metrics for a rental
    pub async fn get_rental_metrics(&self, rental_id: &RentalId) -> Result<UsageMetrics> {
        let events = self
            .event_store
            .get_events_by_entity(&rental_id.to_string(), None)
            .await
            .map_err(|e| BillingError::EventStoreError {
                message: format!("Failed to get events for rental {}", rental_id),
                source: Box::new(e),
            })?;

        let mut total_metrics = UsageMetrics::zero();

        for event in events {
            if event.event_type == "telemetry_update" {
                if let Ok(data) = serde_json::from_value::<serde_json::Value>(event.event_data) {
                    total_metrics.cpu_hours +=
                        Decimal::from_f64(data["cpu_percent"].as_f64().unwrap_or(0.0))
                            .unwrap_or(Decimal::ZERO);

                    total_metrics.memory_gb_hours +=
                        Decimal::from_f64(data["memory_gb"].as_f64().unwrap_or(0.0))
                            .unwrap_or(Decimal::ZERO);

                    total_metrics.gpu_hours +=
                        Decimal::from_f64(data["gpu_hours"].as_f64().unwrap_or(0.0))
                            .unwrap_or(Decimal::ZERO);

                    total_metrics.network_gb +=
                        Decimal::from_f64(data["network_gb"].as_f64().unwrap_or(0.0))
                            .unwrap_or(Decimal::ZERO);
                }
            }
        }

        Ok(total_metrics)
    }
}

use std::str::FromStr;
