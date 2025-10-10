use crate::domain::processor::{
    CostUpdateData, EventHandlers, RentalEndData, StatusChangeData, TelemetryData,
};
use crate::domain::rentals::Rental;
use crate::domain::types::{
    CreditBalance, PackageId, RentalId, RentalState, ResourceSpec, UsageMetrics, UserId,
};
use crate::error::{BillingError, Result};
use crate::storage::{
    BillingEvent, CreditRepository, EventRepository, PackageRepository, RentalRepository,
    UsageEvent, UsageRepository,
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
    rental_repository: Arc<dyn RentalRepository + Send + Sync>,
    credit_repository: Arc<dyn CreditRepository + Send + Sync>,
    usage_repository: Arc<dyn UsageRepository + Send + Sync>,
    package_repository: Arc<dyn PackageRepository + Send + Sync>,
    event_repository: Arc<dyn EventRepository + Send + Sync>,
}

impl BillingEventHandlers {
    pub fn new(
        rental_repository: Arc<dyn RentalRepository + Send + Sync>,
        credit_repository: Arc<dyn CreditRepository + Send + Sync>,
        usage_repository: Arc<dyn UsageRepository + Send + Sync>,
        package_repository: Arc<dyn PackageRepository + Send + Sync>,
        event_repository: Arc<dyn EventRepository + Send + Sync>,
    ) -> Self {
        Self {
            rental_repository,
            credit_repository,
            usage_repository,
            package_repository,
            event_repository,
        }
    }

    /// Parse telemetry data from event JSON
    fn parse_telemetry_data(event_data: &serde_json::Value) -> Result<TelemetryData> {
        let telemetry = TelemetryData {
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
        UsageMetrics {
            cpu_hours: telemetry.cpu_percent.unwrap_or(Decimal::ZERO) / Decimal::from(100),
            memory_gb_hours: telemetry
                .memory_mb
                .map(|mb| Decimal::from(mb) / Decimal::from(1024))
                .unwrap_or(Decimal::ZERO),
            gpu_hours: telemetry
                .gpu_metrics
                .as_ref()
                .and_then(|m| m.get("utilization"))
                .and_then(|v| v.as_f64())
                .map(|v| Decimal::from_f64_retain(v / 100.0).unwrap_or(Decimal::ZERO))
                .unwrap_or(Decimal::ONE), // Assume full GPU usage if no metrics
            network_gb: (telemetry.network_rx_bytes.unwrap_or(0)
                + telemetry.network_tx_bytes.unwrap_or(0))
            .checked_div(1_073_741_824)
            .map(Decimal::from)
            .unwrap_or(Decimal::ZERO),
            storage_gb_hours: Decimal::ZERO, // Not tracked in telemetry yet
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
}

#[async_trait]
impl EventHandlers for BillingEventHandlers {
    async fn process_telemetry_event(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing telemetry event {} for rental {}",
            event.event_id, event.rental_id
        );

        let telemetry = Self::parse_telemetry_data(&event.event_data)?;
        let usage_metrics = Self::telemetry_to_usage_metrics(&telemetry);

        let rental_id = RentalId::from_uuid(event.rental_id);
        let rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        if rental.state != RentalState::Active {
            debug!("Skipping telemetry for non-active rental {}", rental_id);
            return Ok(());
        }

        let package = self
            .package_repository
            .get_package(&rental.package_id)
            .await?;

        let cost_breakdown = package.calculate_cost(&usage_metrics);

        let user_id = UserId::new(event.user_id.clone());
        self.usage_repository
            .update_usage(
                &rental_id,
                &user_id,
                &usage_metrics,
                cost_breakdown.total_cost,
            )
            .await?;

        self.record_billing_event(
            "telemetry_processed",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "usage_metrics": usage_metrics,
                "cost": cost_breakdown.total_cost.to_string(),
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Processed telemetry for rental {} - cost: {}",
            rental_id, cost_breakdown.total_cost
        );

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
                if let Some(reservation_id) = &rental.reservation_id {
                    debug!(
                        "Rental {} activated with reservation {}",
                        rental_id, reservation_id
                    );
                }
                rental.actual_start_time = Some(Utc::now());
            }
            (RentalState::Active, RentalState::Completed)
            | (RentalState::Active, RentalState::Failed) => {
                rental.actual_end_time = Some(Utc::now());

                let usage = self
                    .usage_repository
                    .get_usage_for_rental(&rental_id)
                    .await?;

                let package = self
                    .package_repository
                    .get_package(&rental.package_id)
                    .await?;

                let final_cost = package.calculate_cost(&usage);
                rental.actual_cost = final_cost.total_cost;

                if let Err(e) = self
                    .credit_repository
                    .deduct_credits(&rental.user_id, final_cost.total_cost)
                    .await
                {
                    error!("Failed to deduct credits for rental {}: {}", rental_id, e);
                }

                if let Some(reservation_id) = &rental.reservation_id {
                    if let Err(e) = self
                        .credit_repository
                        .release_reservation(reservation_id)
                        .await
                    {
                        warn!("Failed to release reservation {}: {}", reservation_id, e);
                    }
                }
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

        let new_cost = CreditBalance::from_decimal(cost_data.total_cost);
        rental.actual_cost = new_cost;

        self.rental_repository.update_rental(&rental).await?;

        self.record_billing_event(
            "cost_updated",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "total_cost": cost_data.total_cost,
                "hourly_rate": cost_data.hourly_rate,
                "duration_hours": cost_data.duration_hours,
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!("Updated cost for rental {} to {}", rental_id, new_cost);

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

        let package_id = event
            .event_data
            .get("package_id")
            .and_then(|v| v.as_str())
            .map(|s| PackageId::new(s.to_string()))
            .unwrap_or_else(PackageId::h100); // Default to H100 if not specified

        let package = self.package_repository.get_package(&package_id).await?;

        let estimated_usage = UsageMetrics {
            gpu_hours: Decimal::ONE,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::ZERO,
            network_gb: Decimal::ZERO,
            disk_io_gb: Decimal::ZERO,
        };

        let estimated_cost = package.calculate_cost(&estimated_usage).total_cost;

        let reservation = self
            .credit_repository
            .reserve_credits(&user_id, estimated_cost, &rental_id)
            .await?;

        let resource_spec = ResourceSpec {
            gpu_specs: vec![],
            cpu_cores: 1,
            memory_gb: 1,
            storage_gb: 10,
            disk_iops: 1000,
            network_bandwidth_mbps: 100,
        };

        let mut rental = Rental::new(
            user_id.clone(),
            event.node_id.clone(),
            event.validator_id.clone(),
            package_id,
            resource_spec,
            Some(reservation.id),
        );
        rental.id = rental_id;

        rental.state = RentalState::Active;
        rental.actual_start_time = Some(Utc::now());

        self.rental_repository.create_rental(&rental).await?;

        self.usage_repository
            .initialize_rental(&rental_id, &user_id)
            .await?;

        self.record_billing_event(
            "rental_started",
            &rental_id.to_string(),
            user_id.as_uuid().ok(),
            serde_json::json!({
                "package_id": rental.package_id.to_string(),
                "node_id": event.node_id,
                "validator_id": event.validator_id,
                "estimated_cost": estimated_cost.to_string(),
                "reservation_id": reservation.id.to_string(),
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Started rental {} for user {} with package {}",
            rental_id, user_id, rental.package_id
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

        if rental.state == RentalState::Completed || rental.state == RentalState::Failed {
            debug!("Rental {} already ended, skipping", rental_id);
            return Ok(());
        }

        let usage = self
            .usage_repository
            .get_usage_for_rental(&rental_id)
            .await?;

        let package = self
            .package_repository
            .get_package(&rental.package_id)
            .await?;

        let cost_breakdown = package.calculate_cost(&usage);
        let computed_cost = cost_breakdown.total_cost;

        let client_provided_cost = CreditBalance::from_decimal(end_data.final_cost);
        if (computed_cost.as_decimal() - client_provided_cost.as_decimal()).abs()
            > Decimal::from_f64(0.01).unwrap()
        {
            warn!(
                "Client-provided final_cost ({}) differs from computed cost ({}) for rental {}",
                client_provided_cost, computed_cost, rental_id
            );
        }

        rental.state = if end_data.termination_reason.as_deref() == Some("failed") {
            RentalState::Failed
        } else {
            RentalState::Completed
        };
        // Use server time as authoritative end timestamp
        rental.actual_end_time = Some(Utc::now());
        rental.actual_cost = computed_cost;

        let charge_result = self
            .credit_repository
            .deduct_credits(&rental.user_id, computed_cost)
            .await;

        if let Err(e) = &charge_result {
            error!(
                "Failed to charge final cost for rental {}: {}",
                rental_id, e
            );
        }

        if let Some(reservation_id) = &rental.reservation_id {
            if let Err(e) = self
                .credit_repository
                .release_reservation(reservation_id)
                .await
            {
                warn!(
                    "Failed to release reservation {} for rental {}: {}",
                    reservation_id, rental_id, e
                );
            }
        }

        self.rental_repository.update_rental(&rental).await?;

        self.record_billing_event(
            "rental_ended",
            &rental_id.to_string(),
            rental.user_id.as_uuid().ok(),
            serde_json::json!({
                "computed_cost": computed_cost.to_string(),
                "client_provided_cost": client_provided_cost.to_string(),
                "end_time": end_data.end_time,
                "termination_reason": end_data.termination_reason,
                "usage_metrics": usage,
                "charge_success": charge_result.is_ok(),
                "timestamp": event.timestamp,
            }),
        )
        .await?;

        info!(
            "Ended rental {} with computed cost {} (client provided: {}, reason: {:?})",
            rental_id, computed_cost, client_provided_cost, end_data.termination_reason
        );

        Ok(())
    }

    async fn process_resource_update(&self, event: &UsageEvent) -> Result<()> {
        debug!(
            "Processing resource update event {} for rental {}",
            event.event_id, event.rental_id
        );

        let rental_id = RentalId::from_uuid(event.rental_id);

        let mut rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })?;

        let new_resources = event.event_data.get("resources").cloned();

        if let Some(resources) = new_resources {
            if let Some(gpu_model) = resources.get("gpu_model").and_then(|v| v.as_str()) {
                let new_package = self
                    .package_repository
                    .find_package_for_gpu_model(gpu_model)
                    .await?;

                if new_package.id != rental.package_id {
                    info!(
                        "Updating rental {} package from {} to {} due to GPU change",
                        rental_id, rental.package_id, new_package.id
                    );
                    rental.package_id = new_package.id;
                }
            }

            self.rental_repository.update_rental(&rental).await?;

            self.record_billing_event(
                "resource_updated",
                &rental_id.to_string(),
                rental.user_id.as_uuid().ok(),
                serde_json::json!({
                    "resources": resources,
                    "package_id": rental.package_id.to_string(),
                    "timestamp": event.timestamp,
                }),
            )
            .await?;

            info!("Updated resources for rental {}", rental_id);
        }

        Ok(())
    }
}
