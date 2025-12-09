use crate::domain::processor::{
    CostUpdateData, EventHandlers, RentalEndData, StatusChangeData, TelemetryData,
};
use crate::domain::rentals::{finalize_rental_core, CreateCommunityRentalParams, Rental};
use crate::domain::types::{
    CreditBalance, RentalId, RentalState, ResourceSpec, UsageMetrics, UserId,
};
use crate::error::{BillingError, Result};
use crate::storage::{
    rentals::RentalRepository, BillingEvent, EventRepository, SqlCreditRepository,
    SqlRentalRepository, UsageEvent,
};
use async_trait::async_trait;
use chrono::Utc;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Concrete implementation of EventHandlers for billing operations
pub struct BillingEventHandlers {
    rental_repository: Arc<SqlRentalRepository>,
    credit_repository: Arc<SqlCreditRepository>,
    event_repository: Arc<dyn EventRepository + Send + Sync>,
}

impl BillingEventHandlers {
    pub fn new(
        rental_repository: Arc<SqlRentalRepository>,
        credit_repository: Arc<SqlCreditRepository>,
        event_repository: Arc<dyn EventRepository + Send + Sync>,
    ) -> Self {
        Self {
            rental_repository,
            credit_repository,
            event_repository,
        }
    }

    /// Parse telemetry data from event JSON
    fn parse_telemetry_data(event_data: &serde_json::Value) -> Result<TelemetryData> {
        let telemetry = TelemetryData {
            gpu_hours: event_data
                .get("gpu_hours")
                .and_then(|v| v.as_f64())
                .and_then(Decimal::from_f64),
            cpu_percent: event_data
                .get("cpu_percent")
                .and_then(|v| v.as_f64())
                .map(|v| Decimal::from_f64_retain(v).unwrap_or(Decimal::ZERO)),
            memory_mb: event_data.get("memory_mb").and_then(|v| v.as_u64()),
            network_rx_bytes: event_data.get("network_rx_bytes").and_then(|v| v.as_u64()),
            network_tx_bytes: event_data.get("network_tx_bytes").and_then(|v| v.as_u64()),
            disk_read_bytes: event_data.get("disk_read_bytes").and_then(|v| v.as_u64()),
            disk_write_bytes: event_data.get("disk_write_bytes").and_then(|v| v.as_u64()),
            gpu_metrics: event_data.get("gpu_metrics").cloned(),
            custom_metrics: event_data.get("custom_metrics").cloned(),
        };
        Ok(telemetry)
    }

    /// Convert telemetry data to usage metrics
    fn telemetry_to_usage_metrics(telemetry: &TelemetryData) -> UsageMetrics {
        let gpu_count = telemetry
            .gpu_metrics
            .as_ref()
            .and_then(|m| m.get("gpu_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        UsageMetrics {
            gpu_hours: telemetry.gpu_hours.unwrap_or(Decimal::ZERO),
            gpu_count,
            cpu_hours: telemetry.cpu_percent.unwrap_or(Decimal::ZERO) / Decimal::from(100),
            memory_gb_hours: telemetry
                .memory_mb
                .map(|mb| Decimal::from(mb) / Decimal::from(1024))
                .unwrap_or(Decimal::ZERO),
            network_gb: (telemetry.network_rx_bytes.unwrap_or(0)
                + telemetry.network_tx_bytes.unwrap_or(0))
            .checked_div(1_073_741_824)
            .map(Decimal::from)
            .unwrap_or(Decimal::ZERO),
            storage_gb_hours: Decimal::ZERO,
            disk_io_gb: (telemetry.disk_read_bytes.unwrap_or(0)
                + telemetry.disk_write_bytes.unwrap_or(0))
            .checked_div(1_073_741_824)
            .map(Decimal::from)
            .unwrap_or(Decimal::ZERO),
        }
    }

    /// Record a billing event for audit trail
    async fn record_billing_event(
        &self,
        event_type: &str,
        entity_id: &str,
        user_id: Option<Uuid>,
        event_data: serde_json::Value,
    ) -> Result<()> {
        let billing_event = BillingEvent {
            event_id: Uuid::new_v4(),
            event_type: event_type.to_string(),
            entity_type: "rental".to_string(),
            entity_id: entity_id.to_string(),
            user_id,
            event_data,
            metadata: None,
            created_by: "billing_handler".to_string(),
            created_at: Utc::now(),
        };

        self.event_repository
            .append_billing_event(&billing_event)
            .await?;

        Ok(())
    }

    /// Finalize a rental - set terminal state, update timestamps, and emit rental_end event.
    /// This is called when a rental reaches a terminal state (e.g., insufficient credits).
    pub async fn finalize_rental_internal(
        &self,
        rental: &mut Rental,
        target_state: RentalState,
        end_time: chrono::DateTime<Utc>,
        termination_reason: Option<&str>,
    ) -> Result<()> {
        finalize_rental_core(
            rental,
            target_state,
            end_time,
            termination_reason,
            self.rental_repository.as_ref(),
            self.event_repository.as_ref(),
        )
        .await?;
        Ok(())
    }
}

#[async_trait]
impl EventHandlers for BillingEventHandlers {
    async fn process_telemetry_event(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing telemetry event {} for rental {}",
            event.event_id, event.rental_id
        );

        let rental_id = RentalId::from_uuid(event.rental_id);

        let telemetry = Self::parse_telemetry_data(&event.event_data)?;
        let usage_metrics = Self::telemetry_to_usage_metrics(&telemetry);
        let mut rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        if rental.state == RentalState::Pending {
            info!(
                "Auto-activating rental {} on first telemetry receipt",
                rental_id
            );
            rental.state = RentalState::Active;
            rental.actual_start_time = Some(Utc::now());
            self.rental_repository.update_rental(&rental).await?;

            self.record_billing_event(
                "rental_auto_activated",
                &rental_id.to_string(),
                rental.user_id.as_uuid().ok(),
                serde_json::json!({
                    "reason": "first_telemetry_received",
                    "timestamp": Utc::now(),
                }),
            )
            .await?;
        } else if rental.state != RentalState::Active {
            debug!(
                "Skipping telemetry for rental {} in state {:?}",
                rental_id, rental.state
            );
            return Ok(());
        }

        // Use marketplace-2-compute pricing formula
        use crate::domain::cost_calculator::calculate_marketplace_cost;

        let cost_breakdown = calculate_marketplace_cost(
            usage_metrics.gpu_hours,
            rental.base_price_per_gpu,
            rental.gpu_count,
        );

        let incremental_cost = cost_breakdown.total_cost;

        let mut tx = self.credit_repository.pool().begin().await.map_err(|e| {
            BillingError::DatabaseError {
                operation: "begin_billing_transaction".to_string(),
                source: Box::new(e),
            }
        })?;

        let rental_id_str = rental.id.as_uuid().to_string();
        match self
            .credit_repository
            .deduct_credits_tx(
                &mut tx,
                &rental.user_id,
                incremental_cost,
                Some(&rental_id_str),
                Some("rental"),
                Some("Incremental rental usage charge"),
            )
            .await
        {
            Ok(()) => {
                rental.actual_cost = rental.actual_cost.add(incremental_cost);

                if let Err(e) = self
                    .rental_repository
                    .update_rental_tx(&mut tx, &rental)
                    .await
                {
                    error!(
                        "Failed to update rental in transaction: {}. Rolling back.",
                        e
                    );
                    return Err(e);
                }

                tx.commit().await.map_err(|e| BillingError::DatabaseError {
                    operation: "commit_billing_transaction".to_string(),
                    source: Box::new(e),
                })?;

                self.record_billing_event(
                    "telemetry_processed",
                    &rental_id.to_string(),
                    rental.user_id.as_uuid().ok(),
                    serde_json::json!({
                        "usage_metrics": usage_metrics,
                        "incremental_cost": incremental_cost.to_string(),
                        "total_cost": rental.actual_cost.to_string(),
                        "credits_deducted": true,
                        "timestamp": event.timestamp,
                    }),
                )
                .await?;

                info!(
                    "Processed telemetry for rental {} - incremental cost: {}, total cost: {}, transaction committed",
                    rental_id, incremental_cost, rental.actual_cost
                );
            }
            Err(e) => {
                error!(
                    "Failed to deduct credits for rental {}: {}. Transaction auto-rolled back.",
                    rental_id, e
                );

                // Use specific state for insufficient credits to enable targeted handling
                let (new_state, event_type, termination_reason) =
                    if matches!(&e, BillingError::InsufficientCredits { .. }) {
                        (
                            RentalState::FailedInsufficientCredits,
                            "rental_failed_insufficient_credits",
                            "insufficient_credits",
                        )
                    } else {
                        (
                            RentalState::Failed,
                            "telemetry_billing_failed",
                            "billing_error",
                        )
                    };

                let end_time = Utc::now();

                // Fully finalize the rental - sets state, timestamps, and emits rental_end event
                self.finalize_rental_internal(
                    &mut rental,
                    new_state,
                    end_time,
                    Some(termination_reason),
                )
                .await?;

                // Also record the billing event with error details for audit
                self.record_billing_event(
                    event_type,
                    &rental_id.to_string(),
                    rental.user_id.as_uuid().ok(),
                    serde_json::json!({
                        "usage_metrics": usage_metrics,
                        "attempted_cost": incremental_cost.to_string(),
                        "error": e.to_string(),
                        "rental_state": rental.state.to_string(),
                        "timestamp": event.timestamp,
                    }),
                )
                .await?;

                return Err(e);
            }
        }

        Ok(())
    }

    async fn process_status_change(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing status change event {} for rental {}",
            event.event_id, event.rental_id
        );

        let status_data: StatusChangeData = serde_json::from_value(event.event_data.clone())
            .map_err(|e| BillingError::ValidationError {
                field: "event_data".to_string(),
                message: format!("Invalid status change data: {}", e),
            })?;

        let rental_id = RentalId::from_uuid(event.rental_id);

        let mut rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        let new_state = match status_data.new_status.as_str() {
            "pending" => RentalState::Pending,
            "active" => RentalState::Active,
            "completed" => RentalState::Completed,
            "failed" => RentalState::Failed,
            _ => {
                warn!("Unknown rental state: {}", status_data.new_status);
                return Ok(());
            }
        };

        let old_state = rental.state;
        rental.state = new_state;

        match (&old_state, &new_state) {
            (RentalState::Pending, RentalState::Active) => {
                rental.actual_start_time = Some(Utc::now());
            }
            (RentalState::Active, RentalState::Completed)
            | (RentalState::Active, RentalState::Failed) => {
                rental.actual_end_time = Some(Utc::now());

                info!(
                    "Rental {} completed/failed with total cost: {} (credits deducted incrementally)",
                    rental_id, rental.actual_cost
                );
            }
            _ => {
                debug!(
                    "Status change from {:?} to {:?} for rental {}",
                    old_state, new_state, rental_id
                );
            }
        }

        self.rental_repository.update_rental(&rental).await?;

        self.record_billing_event(
            "status_changed",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "old_status": status_data.old_status,
                "new_status": status_data.new_status,
                "reason": status_data.reason,
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Processed status change for rental {} from {} to {}",
            rental_id, status_data.old_status, status_data.new_status
        );

        Ok(())
    }

    async fn process_cost_update(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing cost update event {} for rental {}",
            event.event_id, event.rental_id
        );

        let cost_data: CostUpdateData =
            serde_json::from_value(event.event_data.clone()).map_err(|e| {
                BillingError::ValidationError {
                    field: "event_data".to_string(),
                    message: format!("Invalid cost update data: {}", e),
                }
            })?;

        let rental_id = RentalId::from_uuid(event.rental_id);

        let mut rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        let new_price = cost_data.price_per_gpu;
        if new_price <= Decimal::ZERO {
            return Err(BillingError::ValidationError {
                field: "price_per_gpu".to_string(),
                message: "price_per_gpu must be greater than zero".to_string(),
            });
        }

        let old_price = rental.base_price_per_gpu;

        if new_price == old_price {
            info!(
                "Cost update for rental {} ignored: price unchanged at {}",
                rental_id, new_price
            );
            return Ok(());
        }

        rental.base_price_per_gpu = new_price;

        self.rental_repository.update_rental(&rental).await?;

        self.record_billing_event(
            "cost_updated",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "old_price_per_gpu": old_price.to_string(),
                "new_price_per_gpu": new_price.to_string(),
                "reason": cost_data.reason,
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Updated price for rental {} from {} to {}",
            rental_id, old_price, new_price
        );

        Ok(())
    }

    async fn process_rental_start(&self, event: &UsageEvent) -> Result<()> {
        info!(
            "Processing rental start event {} for rental {}",
            event.event_id, event.rental_id
        );

        let rental_id = RentalId::from_uuid(event.rental_id);

        if let Ok(Some(_existing)) = self.rental_repository.get_rental(&rental_id).await {
            debug!("Rental {} already exists, skipping start event", rental_id);
            return Ok(());
        }

        let user_id = UserId::new(event.user_id.clone());

        // Extract and validate marketplace pricing from event data
        let base_price_per_gpu = event
            .event_data
            .get("base_price_per_gpu")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .ok_or_else(|| {
                warn!(
                    rental_id = %event.rental_id,
                    event_id = %event.event_id,
                    "Missing or invalid base_price_per_gpu in rental start event"
                );
                BillingError::ValidationError {
                    field: "base_price_per_gpu".to_string(),
                    message: "Missing or invalid base_price_per_gpu in rental start event data"
                        .to_string(),
                }
            })?;

        let gpu_count = event
            .event_data
            .get("gpu_count")
            .and_then(|v| v.as_u64())
            .filter(|&count| count > 0)
            .ok_or_else(|| {
                warn!(
                    rental_id = %event.rental_id,
                    event_id = %event.event_id,
                    "Missing or invalid gpu_count in rental start event"
                );
                BillingError::ValidationError {
                    field: "gpu_count".to_string(),
                    message:
                        "Missing or invalid gpu_count (must be > 0) in rental start event data"
                            .to_string(),
                }
            })? as u32;

        let resource_spec = ResourceSpec {
            gpu_specs: vec![],
            cpu_cores: 1,
            memory_gb: 1,
            storage_gb: 10,
            disk_iops: 1000,
            network_bandwidth_mbps: 100,
        };

        // Extract and validate required fields for community rentals
        let validator_id =
            event
                .validator_id
                .clone()
                .ok_or_else(|| BillingError::ValidationError {
                    field: "validator_id".to_string(),
                    message: "validator_id is required for community rentals".to_string(),
                })?;

        let miner_uid = event
            .event_data
            .get("miner_uid")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| BillingError::ValidationError {
                field: "miner_uid".to_string(),
                message: "miner_uid is required for community rentals".to_string(),
            })?;

        let miner_hotkey = event
            .event_data
            .get("miner_hotkey")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| BillingError::ValidationError {
                field: "miner_hotkey".to_string(),
                message: "miner_hotkey is required for community rentals".to_string(),
            })?;

        let mut rental = Rental::new_community(CreateCommunityRentalParams {
            user_id: user_id.clone(),
            node_id: event.node_id.clone(),
            validator_id,
            miner_uid,
            miner_hotkey,
            resource_spec,
            base_price_per_gpu,
            gpu_count,
        });
        rental.id = rental_id;

        rental.state = RentalState::Active;
        rental.actual_start_time = Some(Utc::now());

        self.rental_repository.create_rental(&rental).await?;

        self.record_billing_event(
            "rental_started",
            &rental_id.to_string(),
            user_id.as_uuid().ok(),
            serde_json::json!({
                "node_id": event.node_id,
                "validator_id": event.validator_id,
                "billing_model": "marketplace",
                "base_price_per_gpu": rental.base_price_per_gpu.to_string(),
                "gpu_count": rental.gpu_count,
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Started rental {} for user {} at ${}/GPU × {} GPUs",
            rental_id, user_id, rental.base_price_per_gpu, rental.gpu_count
        );

        Ok(())
    }

    async fn process_rental_end(&self, event: &UsageEvent) -> Result<()> {
        info!(
            "Processing rental end event {} for rental {}",
            event.event_id, event.rental_id
        );

        let end_data: RentalEndData =
            serde_json::from_value(event.event_data.clone()).map_err(|e| {
                BillingError::ValidationError {
                    field: "event_data".to_string(),
                    message: format!("Invalid rental end data: {}", e),
                }
            })?;

        let rental_id = RentalId::from_uuid(event.rental_id);

        let mut rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        if rental.state == RentalState::Completed
            || rental.state == RentalState::Failed
            || rental.state == RentalState::FailedInsufficientCredits
        {
            debug!("Rental {} already ended, skipping", rental_id);
            return Ok(());
        }

        let client_provided_cost = CreditBalance::from_decimal(end_data.final_cost);
        let server_tracked_cost = rental.actual_cost;

        if (server_tracked_cost.as_decimal() - client_provided_cost.as_decimal()).abs()
            > Decimal::from_f64(0.01).unwrap()
        {
            warn!(
                "Client-provided final_cost ({}) differs from server-tracked cost ({}) for rental {}",
                client_provided_cost, server_tracked_cost, rental_id
            );
        }

        rental.state = if end_data.termination_reason.as_deref() == Some("failed") {
            RentalState::Failed
        } else {
            RentalState::Completed
        };
        rental.actual_end_time = Some(Utc::now());

        self.rental_repository.update_rental(&rental).await?;

        self.record_billing_event(
            "rental_ended",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "server_tracked_cost": server_tracked_cost.to_string(),
                "client_provided_cost": client_provided_cost.to_string(),
                "end_time": end_data.end_time,
                "termination_reason": end_data.termination_reason,
                "billing_model": "incremental",
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Ended rental {} with final cost {} (client provided: {}, reason: {:?}, credits deducted incrementally)",
            rental_id, server_tracked_cost, client_provided_cost, end_data.termination_reason
        );

        Ok(())
    }

    async fn process_resource_update(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing resource update event {} for rental {}",
            event.event_id, event.rental_id
        );

        let rental_id = RentalId::from_uuid(event.rental_id);

        let rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        let new_resources = event.event_data.get("resources").cloned();

        if let Some(resources) = new_resources {
            // In marketplace-2-compute, pricing is fixed at rental creation
            // Resource changes don't affect the pricing model
            info!(
                "Resource update event for rental {} (marketplace pricing unchanged)",
                rental_id
            );

            self.rental_repository.update_rental(&rental).await?;

            self.record_billing_event(
                "resource_updated",
                &rental_id.to_string(),
                rental.user_id.as_uuid().ok(),
                serde_json::json!({
                    "resources": resources,
                    "base_price_per_gpu": rental.base_price_per_gpu.to_string(),
                    "gpu_count": rental.gpu_count,
                    "timestamp": event.timestamp,
                }),
            )
            .await?;

            info!("Updated resources for rental {}", rental_id);
        }

        Ok(())
    }
}
