use crate::domain::events::EventStore;
use crate::domain::{
    credits::{CreditManager, CreditOperations},
    rentals::{RentalManager, RentalOperations},
    rules_engine::RulesEngine,
    types::{
        CreditBalance, GpuSpec, PackageId, RentalId, RentalState, ReservationId, ResourceSpec,
        UserId,
    },
};
use crate::error::BillingError;
use crate::storage::events::{EventType, UsageEvent};
use crate::storage::rds::RdsConnection;
use crate::storage::SqlRulesRepository;
use crate::storage::{PackageRepository, SqlPackageRepository};
use crate::storage::{RentalRepository, SqlCreditRepository, SqlRentalRepository};
use crate::storage::{SqlUserPreferencesRepository, UserPreferencesRepository};
use crate::telemetry::{TelemetryIngester, TelemetryProcessor};

use basilica_protocol::billing::{
    billing_service_server::BillingService, ActiveRental, ApplyCreditsRequest,
    ApplyCreditsResponse, FinalizeRentalRequest, FinalizeRentalResponse, GetActiveRentalsRequest,
    GetActiveRentalsResponse, GetBalanceRequest, GetBalanceResponse, GetBillingPackagesRequest,
    GetBillingPackagesResponse, IngestResponse, ReleaseReservationRequest,
    ReleaseReservationResponse, RentalStatus, ReserveCreditsRequest, ReserveCreditsResponse,
    SetUserPackageRequest, SetUserPackageResponse, TelemetryData, TrackRentalRequest,
    TrackRentalResponse, UpdateRentalStatusRequest, UpdateRentalStatusResponse, UsageDataPoint,
    UsageReportRequest, UsageReportResponse, UsageSummary,
};

use chrono::Duration;
use rust_decimal::prelude::*;
use serde_json;
use std::str::FromStr;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};
use tracing::{error, info};
use uuid;

pub struct BillingServiceImpl {
    credit_manager: Arc<dyn CreditOperations + Send + Sync>,
    rental_manager: Arc<dyn RentalOperations + Send + Sync>,
    _rules_engine: Arc<RulesEngine>,
    #[allow(dead_code)] // Used in server's consumer loop
    telemetry_processor: Arc<TelemetryProcessor>,
    telemetry_ingester: Arc<TelemetryIngester>,
    rental_repository: Arc<dyn RentalRepository + Send + Sync>,
    package_repository: Arc<dyn PackageRepository + Send + Sync>,
    user_preferences_repository: Arc<dyn UserPreferencesRepository + Send + Sync>,
    event_store: Arc<EventStore>,
}

impl BillingServiceImpl {
    pub fn new(
        rds_connection: Arc<RdsConnection>,
        telemetry_ingester: Arc<TelemetryIngester>,
        telemetry_processor: Arc<TelemetryProcessor>,
    ) -> Self {
        let credit_repository = Arc::new(SqlCreditRepository::new(rds_connection.clone()));
        let rental_repository = Arc::new(SqlRentalRepository::new(rds_connection.clone()));
        let package_repository = Arc::new(SqlPackageRepository::new(rds_connection.pool().clone()));
        let rules_repository = Arc::new(SqlRulesRepository::new(rds_connection.pool().clone()));
        let user_preferences_repository =
            Arc::new(SqlUserPreferencesRepository::new(rds_connection.clone()));

        // Create event repositories using proper pattern
        let event_repository = Arc::new(crate::storage::events::SqlEventRepository::new(
            rds_connection.clone(),
        ));
        let batch_repository = Arc::new(crate::storage::events::SqlBatchRepository::new(
            rds_connection.clone(),
        ));
        let event_store = Arc::new(crate::domain::events::EventStore::new(
            event_repository,
            batch_repository,
            1000,
            90,
        ));

        Self {
            credit_manager: Arc::new(CreditManager::new(credit_repository.clone())),
            rental_manager: Arc::new(RentalManager::new(rental_repository.clone())),
            _rules_engine: Arc::new(RulesEngine::new(
                package_repository.clone(),
                rules_repository,
            )),
            telemetry_processor,
            telemetry_ingester,
            rental_repository: rental_repository.clone(),
            package_repository: package_repository.clone(),
            user_preferences_repository: user_preferences_repository.clone(),
            event_store,
        }
    }

    fn parse_decimal(s: &str) -> crate::error::Result<Decimal> {
        Decimal::from_str(s).map_err(|e| BillingError::ValidationError {
            field: "amount".to_string(),
            message: format!("Invalid decimal value: {}", e),
        })
    }

    fn format_decimal(d: Decimal) -> String {
        let normalized = d.normalize();
        if normalized.fract().is_zero() {
            normalized.trunc().to_string()
        } else {
            let s = normalized.to_string();
            if s.contains('.') {
                s.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                s
            }
        }
    }

    fn format_credit_balance(b: CreditBalance) -> String {
        Self::format_decimal(b.as_decimal())
    }

    fn rental_status_to_domain(status: RentalStatus) -> RentalState {
        match status {
            RentalStatus::Pending => RentalState::Pending,
            RentalStatus::Active => RentalState::Active,
            RentalStatus::Stopping => RentalState::Terminating,
            RentalStatus::Stopped => RentalState::Completed,
            RentalStatus::Failed => RentalState::Failed,
            RentalStatus::Unspecified => RentalState::Pending,
        }
    }

    fn domain_status_to_proto(state: RentalState) -> RentalStatus {
        match state {
            RentalState::Pending => RentalStatus::Pending,
            RentalState::Active => RentalStatus::Active,
            RentalState::Suspended => RentalStatus::Stopping,
            RentalState::Terminating => RentalStatus::Stopping,
            RentalState::Completed => RentalStatus::Stopped,
            RentalState::Failed => RentalStatus::Failed,
        }
    }
}

#[tonic::async_trait]
impl BillingService for BillingServiceImpl {
    async fn apply_credits(
        &self,
        request: Request<ApplyCreditsRequest>,
    ) -> std::result::Result<Response<ApplyCreditsResponse>, Status> {
        let req = request.into_inner();
        let user_id = UserId::new(req.user_id);
        let amount = Self::parse_decimal(&req.amount)
            .map_err(|e| Status::invalid_argument(format!("Invalid amount: {}", e)))?;
        let credit_balance = CreditBalance::from_decimal(amount);

        info!("Applying {} credits to user {}", amount, user_id);

        let new_balance = self
            .credit_manager
            .apply_credits(&user_id, credit_balance)
            .await
            .map_err(|e| Status::internal(format!("Failed to apply credits: {}", e)))?;

        let response = ApplyCreditsResponse {
            success: true,
            new_balance: Self::format_credit_balance(new_balance),
            credit_id: req.transaction_id,
            applied_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    async fn get_balance(
        &self,
        request: Request<GetBalanceRequest>,
    ) -> std::result::Result<Response<GetBalanceResponse>, Status> {
        let req = request.into_inner();
        let user_id = UserId::new(req.user_id);

        let account = self
            .credit_manager
            .get_account(&user_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get account: {}", e)))?;

        let response = GetBalanceResponse {
            available_balance: Self::format_credit_balance(account.available_balance()),
            reserved_balance: Self::format_credit_balance(account.reserved),
            total_balance: Self::format_credit_balance(account.balance),
            last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                account.last_updated,
            ))),
        };

        Ok(Response::new(response))
    }

    async fn reserve_credits(
        &self,
        request: Request<ReserveCreditsRequest>,
    ) -> std::result::Result<Response<ReserveCreditsResponse>, Status> {
        let req = request.into_inner();
        let user_id = UserId::new(req.user_id);
        let rental_id = if req.rental_id.is_empty() {
            None
        } else {
            Some(
                RentalId::from_str(&req.rental_id)
                    .map_err(|e| Status::invalid_argument(format!("Invalid rental ID: {}", e)))?,
            )
        };
        let amount = Self::parse_decimal(&req.amount)
            .map_err(|e| Status::invalid_argument(format!("Invalid amount: {}", e)))?;
        let credit_balance = CreditBalance::from_decimal(amount);
        let proto_duration = req
            .duration
            .ok_or_else(|| Status::invalid_argument("Duration is required"))?;
        let duration = Duration::seconds(proto_duration.seconds)
            + Duration::nanoseconds(proto_duration.nanos as i64);
        let hours = duration.num_hours();

        info!(
            "Reserving {} credits for user {} for {} hours",
            amount, user_id, hours
        );

        let reservation_id = self
            .credit_manager
            .reserve_credits(&user_id, credit_balance, duration, rental_id)
            .await
            .map_err(|e| match e {
                BillingError::InsufficientBalance {
                    available,
                    required,
                } => Status::failed_precondition(format!(
                    "Insufficient balance: available={}, required={}",
                    available, required
                )),
                _ => Status::internal(format!("Failed to reserve credits: {}", e)),
            })?;

        let reservation = self
            .credit_manager
            .get_reservation(&reservation_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get reservation: {}", e)))?;

        let response = ReserveCreditsResponse {
            success: true,
            reservation_id: reservation_id.to_string(),
            reserved_amount: Self::format_credit_balance(reservation.amount),
            reserved_until: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                reservation.expires_at,
            ))),
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn release_reservation(
        &self,
        request: Request<ReleaseReservationRequest>,
    ) -> std::result::Result<Response<ReleaseReservationResponse>, Status> {
        let req = request.into_inner();
        let reservation_id = ReservationId::from_uuid(
            uuid::Uuid::parse_str(&req.reservation_id)
                .map_err(|e| Status::invalid_argument(format!("Invalid reservation ID: {}", e)))?,
        );
        let final_amount = Self::parse_decimal(&req.final_amount)
            .map_err(|e| Status::invalid_argument(format!("Invalid amount: {}", e)))?;
        let final_balance = CreditBalance::from_decimal(final_amount);

        info!(
            "Releasing reservation {} with final amount {}",
            reservation_id, final_amount
        );

        let reservation = self
            .credit_manager
            .get_reservation(&reservation_id)
            .await
            .map_err(|e| Status::not_found(format!("Reservation not found: {}", e)))?;

        let reserved_amount = reservation.amount;

        let new_balance = self
            .credit_manager
            .charge_from_reservation(&reservation_id, final_balance)
            .await
            .map_err(|e| match e {
                BillingError::ReservationNotFound { .. } => {
                    Status::not_found(format!("Reservation not found: {}", e))
                }
                BillingError::ReservationAlreadyReleased { .. } => {
                    Status::failed_precondition(format!("Reservation already released: {}", e))
                }
                _ => Status::internal(format!("Failed to release reservation: {}", e)),
            })?;

        let refunded = reserved_amount
            .subtract(final_balance)
            .unwrap_or(CreditBalance::zero());

        let response = ReleaseReservationResponse {
            success: true,
            charged_amount: Self::format_decimal(final_amount),
            refunded_amount: Self::format_credit_balance(refunded),
            new_balance: Self::format_credit_balance(new_balance),
        };

        Ok(Response::new(response))
    }

    async fn track_rental(
        &self,
        request: Request<TrackRentalRequest>,
    ) -> std::result::Result<Response<TrackRentalResponse>, Status> {
        let req = request.into_inner();
        let rental_id = RentalId::from_str(&req.rental_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid rental ID: {}", e)))?;
        let user_id = UserId::new(req.user_id);
        let hourly_rate = Self::parse_decimal(&req.hourly_rate)
            .map_err(|e| Status::invalid_argument(format!("Invalid hourly rate: {}", e)))?;
        let credit_rate = CreditBalance::from_decimal(hourly_rate);

        info!(
            "Tracking rental {} for user {} at {} credits/hour",
            rental_id, user_id, hourly_rate
        );

        let resource_spec = if let Some(spec) = req.resource_spec {
            ResourceSpec {
                gpu_specs: spec
                    .gpus
                    .into_iter()
                    .map(|gpu| GpuSpec {
                        model: gpu.model,
                        memory_mb: gpu.memory_mb,
                        count: gpu.count,
                    })
                    .collect(),
                cpu_cores: spec.cpu_cores,
                memory_gb: (spec.memory_mb / 1024) as u32,
                storage_gb: spec.disk_gb as u32,
                disk_iops: 0,
                network_bandwidth_mbps: spec.network_bandwidth_mbps,
            }
        } else {
            ResourceSpec {
                gpu_specs: vec![],
                cpu_cores: 4,
                memory_gb: 16,
                storage_gb: 100,
                disk_iops: 1000,
                network_bandwidth_mbps: 1000,
            }
        };

        // Select package based on GPU model
        let package_id = if !resource_spec.gpu_specs.is_empty() {
            PackageId::from_gpu_model(&resource_spec.gpu_specs[0].model)
        } else {
            PackageId::custom()
        };
        let package_id_str = package_id.to_string();
        let resource_spec_value =
            serde_json::to_value(&resource_spec).unwrap_or(serde_json::Value::Null);
        let validator_id = req.validator_id.clone();
        let validator_id_copy = validator_id.clone();

        let rental_id = self
            .rental_manager
            .create_rental(
                user_id.clone(),
                req.node_id.clone(),
                validator_id,
                package_id,
                resource_spec,
                None,
            )
            .await
            .map_err(|e| Status::internal(format!("Failed to create rental: {}", e)))?;

        let _created = self
            .rental_manager
            .get_rental(&rental_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get created rental: {}", e)))?;

        let proto_duration = req
            .max_duration
            .ok_or_else(|| Status::invalid_argument("Max duration is required"))?;
        let max_duration = Duration::seconds(proto_duration.seconds)
            + Duration::nanoseconds(proto_duration.nanos as i64);
        let max_duration_hours = max_duration.num_hours() as u32;
        let estimated_cost = credit_rate.multiply(Decimal::from(max_duration_hours));

        let reservation_id = self
            .credit_manager
            .reserve_credits(&user_id, estimated_cost, max_duration, Some(rental_id))
            .await
            .map_err(|e| match e {
                BillingError::InsufficientBalance { .. } => {
                    Status::failed_precondition(format!("Insufficient balance: {}", e))
                }
                _ => Status::internal(format!("Failed to reserve credits: {}", e)),
            })?;

        let rental_start_event = UsageEvent {
            event_id: uuid::Uuid::new_v4(),
            rental_id: rental_id.as_uuid(),
            user_id: user_id.to_string(),
            node_id: req.node_id.clone(),
            validator_id: validator_id_copy,
            event_type: EventType::RentalStart,
            event_data: serde_json::json!({
                "package_id": package_id_str,
                "hourly_rate": Self::format_decimal(hourly_rate),
                "resource_spec": resource_spec_value,
                "estimated_cost": Self::format_credit_balance(estimated_cost),
                "max_duration_hours": max_duration_hours,
            }),
            timestamp: chrono::Utc::now(),
            processed: false,
            processed_at: None,
            batch_id: None,
        };

        self.event_store
            .append_usage_event(&rental_start_event)
            .await
            .map_err(|e| Status::internal(format!("Failed to store rental start event: {}", e)))?;

        let response = TrackRentalResponse {
            success: true,
            tracking_id: rental_id.to_string(),
            reservation_id: reservation_id.to_string(),
            estimated_cost: Self::format_credit_balance(estimated_cost),
        };

        Ok(Response::new(response))
    }

    async fn update_rental_status(
        &self,
        request: Request<UpdateRentalStatusRequest>,
    ) -> std::result::Result<Response<UpdateRentalStatusResponse>, Status> {
        let req = request.into_inner();
        let rental_id = RentalId::from_str(&req.rental_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid rental ID: {}", e)))?;
        let new_status = Self::rental_status_to_domain(req.status());

        info!("Updating rental {} status to {}", rental_id, new_status);

        let _rental = self
            .rental_manager
            .update_status(&rental_id, new_status)
            .await
            .map_err(|e| match e {
                BillingError::RentalNotFound { .. } => {
                    Status::not_found(format!("Rental not found: {}", e))
                }
                BillingError::InvalidStateTransition { .. } => {
                    Status::failed_precondition(format!("Invalid state transition: {}", e))
                }
                _ => Status::internal(format!("Failed to update rental: {}", e)),
            })?;

        let rental = self
            .rental_manager
            .get_rental(&rental_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get rental: {}", e)))?;
        self.rental_repository
            .update_rental(&rental)
            .await
            .map_err(|e| Status::internal(format!("Failed to persist status: {}", e)))?;

        let status_change_event = UsageEvent {
            event_id: uuid::Uuid::new_v4(),
            rental_id: rental_id.as_uuid(),
            user_id: rental.user_id.to_string(),
            node_id: rental.node_id.clone(),
            validator_id: rental.validator_id.clone(),
            event_type: EventType::StatusChange,
            event_data: serde_json::json!({
                "old_status": req.status().as_str_name(),
                "new_status": new_status.to_string(),
                "reason": if req.reason.is_empty() { None } else { Some(&req.reason) },
            }),
            timestamp: chrono::Utc::now(),
            processed: false,
            processed_at: None,
            batch_id: None,
        };
        self.event_store
            .append_usage_event(&status_change_event)
            .await
            .map_err(|e| Status::internal(format!("Failed to store status change event: {}", e)))?;

        let response = UpdateRentalStatusResponse {
            success: true,
            current_cost: Self::format_credit_balance(rental.cost_breakdown.total_cost),
            updated_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    async fn get_active_rentals(
        &self,
        request: Request<GetActiveRentalsRequest>,
    ) -> std::result::Result<Response<GetActiveRentalsResponse>, Status> {
        let req = request.into_inner();

        let rentals = if let Some(filter) = req.filter {
            match filter {
                basilica_protocol::billing::get_active_rentals_request::Filter::UserId(user_id) => {
                    let uid = UserId::new(user_id);
                    self.rental_repository
                        .get_active_rentals(Some(&uid))
                        .await
                        .map_err(|e| Status::internal(format!("Failed to list rentals: {}", e)))?
                }
                basilica_protocol::billing::get_active_rentals_request::Filter::NodeId(node_id) => {
                    let all_rentals = self
                        .rental_repository
                        .get_active_rentals(None)
                        .await
                        .map_err(|e| Status::internal(format!("Failed to list rentals: {}", e)))?;

                    all_rentals
                        .into_iter()
                        .filter(|r| r.node_id == node_id)
                        .collect()
                }
                basilica_protocol::billing::get_active_rentals_request::Filter::ValidatorId(
                    validator_id,
                ) => {
                    let all_rentals = self
                        .rental_repository
                        .get_active_rentals(None)
                        .await
                        .map_err(|e| Status::internal(format!("Failed to list rentals: {}", e)))?;

                    all_rentals
                        .into_iter()
                        .filter(|r| r.validator_id == validator_id)
                        .collect()
                }
            }
        } else {
            self.rental_repository
                .get_active_rentals(None)
                .await
                .map_err(|e| Status::internal(format!("Failed to list rentals: {}", e)))?
        };

        let active_rentals: Vec<ActiveRental> = rentals
            .into_iter()
            .filter(|r| r.state.is_active())
            .map(|r| {
                // Convert ResourceSpec to proto format
                let resource_spec = Some(basilica_protocol::billing::ResourceSpec {
                    cpu_cores: r.resource_spec.cpu_cores,
                    memory_mb: (r.resource_spec.memory_gb as u64) * 1024,
                    gpus: r
                        .resource_spec
                        .gpu_specs
                        .iter()
                        .map(|gpu| basilica_protocol::billing::GpuSpec {
                            model: gpu.model.clone(),
                            memory_mb: gpu.memory_mb,
                            count: gpu.count,
                        })
                        .collect(),
                    disk_gb: r.resource_spec.storage_gb as u64,
                    network_bandwidth_mbps: r.resource_spec.network_bandwidth_mbps,
                });

                ActiveRental {
                    rental_id: r.id.to_string(),
                    user_id: r.user_id.to_string(),
                    node_id: r.node_id.clone(),
                    validator_id: r.validator_id.clone(),
                    status: Self::domain_status_to_proto(r.state).into(),
                    resource_spec,
                    hourly_rate: Self::format_credit_balance(r.cost_breakdown.base_cost),
                    current_cost: Self::format_credit_balance(r.cost_breakdown.total_cost),
                    start_time: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                        r.created_at,
                    ))),
                    last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                        r.last_updated,
                    ))),
                    metadata: std::collections::HashMap::new(),
                }
            })
            .collect();

        let response = GetActiveRentalsResponse {
            rentals: active_rentals.clone(),
            total_count: active_rentals.len() as u64,
        };

        Ok(Response::new(response))
    }

    async fn finalize_rental(
        &self,
        request: Request<FinalizeRentalRequest>,
    ) -> std::result::Result<Response<FinalizeRentalResponse>, Status> {
        let req = request.into_inner();
        let rental_id = RentalId::from_str(&req.rental_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid rental ID: {}", e)))?;
        let final_cost = Self::parse_decimal(&req.final_cost)
            .map_err(|e| Status::invalid_argument(format!("Invalid final cost: {}", e)))?;
        let final_balance = CreditBalance::from_decimal(final_cost);

        info!("Finalizing rental {} with cost {}", rental_id, final_cost);

        let rental = self
            .rental_manager
            .finalize_rental(&rental_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to finalize rental: {}", e)))?;

        let duration = rental.last_updated - rental.created_at;
        let _duration_hours =
            duration.num_hours() as f64 + (duration.num_minutes() % 60) as f64 / 60.0;

        let reservations = self
            .credit_manager
            .get_active_reservations(&rental.user_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get reservations: {}", e)))?;

        let rental_reservation = reservations.iter().find(|r| r.rental_id == Some(rental_id));

        let (charged_amount, refunded_amount) = if let Some(reservation) = rental_reservation {
            let reserved = reservation.amount;
            let refunded = reserved
                .subtract(final_balance)
                .unwrap_or(CreditBalance::zero());

            self.credit_manager
                .charge_from_reservation(&reservation.id, final_balance)
                .await
                .map_err(|e| Status::internal(format!("Failed to charge reservation: {}", e)))?;

            (final_balance, refunded)
        } else {
            self.credit_manager
                .charge_credits(&rental.user_id, final_balance)
                .await
                .map_err(|e| Status::internal(format!("Failed to charge credits: {}", e)))?;

            (final_balance, CreditBalance::zero())
        };

        let final_rental = self
            .rental_manager
            .get_rental(&rental_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get rental: {}", e)))?;
        self.rental_repository
            .update_rental(&final_rental)
            .await
            .map_err(|e| Status::internal(format!("Failed to persist rental: {}", e)))?;

        let duration_proto = prost_types::Duration {
            seconds: duration.num_seconds(),
            nanos: (duration.num_nanoseconds().unwrap_or(0) % 1_000_000_000) as i32,
        };

        let response = FinalizeRentalResponse {
            success: true,
            total_cost: Self::format_decimal(final_cost),
            duration: Some(duration_proto),
            charged_amount: Self::format_credit_balance(charged_amount),
            refunded_amount: Self::format_credit_balance(refunded_amount),
        };

        Ok(Response::new(response))
    }

    async fn ingest_telemetry(
        &self,
        request: Request<tonic::Streaming<TelemetryData>>,
    ) -> std::result::Result<Response<IngestResponse>, Status> {
        let mut stream = request.into_inner();
        let ingester = self.telemetry_ingester.clone();

        let mut events_received = 0u64;
        let mut events_processed = 0u64;
        let mut events_failed = 0u64;
        let mut last_processed = chrono::Utc::now();

        while let Some(result) = stream.next().await {
            match result {
                Ok(telemetry_data) => {
                    events_received += 1;

                    match ingester.ingest(telemetry_data).await {
                        Ok(_) => {
                            events_processed += 1;
                            last_processed = chrono::Utc::now();
                        }
                        Err(e) => {
                            error!("Failed to ingest telemetry: {}", e);
                            events_failed += 1;
                        }
                    }
                }
                Err(e) => {
                    error!("Error receiving telemetry: {}", e);
                    events_failed += 1;
                }
            }
        }

        let response = IngestResponse {
            events_received,
            events_processed,
            events_failed,
            last_processed: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                last_processed,
            ))),
        };

        Ok(Response::new(response))
    }

    async fn get_usage_report(
        &self,
        request: Request<UsageReportRequest>,
    ) -> std::result::Result<Response<UsageReportResponse>, Status> {
        let req = request.into_inner();
        let rental_id = RentalId::from_str(&req.rental_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid rental ID: {}", e)))?;

        let rental = self
            .rental_repository
            .get_rental(&rental_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get rental: {}", e)))?
            .ok_or_else(|| Status::not_found("Rental not found"))?;

        let duration = rental.last_updated - rental.created_at;
        let _duration_hours =
            duration.num_hours() as f64 + (duration.num_minutes() % 60) as f64 / 60.0;

        let events = self
            .event_store
            .get_rental_events(
                uuid::Uuid::parse_str(&rental_id.to_string())
                    .map_err(|_| Status::internal("Invalid rental ID format"))?,
                None,
                None,
            )
            .await
            .map_err(|e| Status::internal(format!("Failed to get telemetry events: {}", e)))?;

        let mut total_cpu_percent = 0.0;
        let mut total_memory_mb = 0u64;
        let mut total_network_bytes = 0u64;
        let mut total_disk_bytes = 0u64;
        let mut total_gpu_percent = 0.0;
        let mut telemetry_count = 0u64;

        let mut data_points = Vec::new();

        for event in &events {
            if let Some(cpu_percent) = event.event_data.get("cpu_percent").and_then(|v| v.as_f64())
            {
                telemetry_count += 1;

                total_cpu_percent += cpu_percent * 100.0; // Convert from hours to percent

                if let Some(memory_gb) = event.event_data.get("memory_gb").and_then(|v| v.as_f64())
                {
                    total_memory_mb += (memory_gb * 1024.0) as u64;
                }

                if let Some(network_gb) =
                    event.event_data.get("network_gb").and_then(|v| v.as_f64())
                {
                    total_network_bytes += (network_gb * 1_073_741_824.0) as u64;
                }

                if let Some(gpu_hours) = event.event_data.get("gpu_hours").and_then(|v| v.as_f64())
                {
                    total_gpu_percent += gpu_hours * 100.0; // Convert from hours to percent
                }

                // For disk I/O, check if it exists in the data
                if let Some(disk_gb) = event.event_data.get("disk_io_gb").and_then(|v| v.as_f64()) {
                    total_disk_bytes += (disk_gb * 1_073_741_824.0) as u64;
                }

                data_points.push(UsageDataPoint {
                    timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                        event.timestamp,
                    ))),
                    usage: Some(basilica_protocol::billing::ResourceUsage {
                        cpu_percent: cpu_percent * 100.0,
                        memory_mb: (event
                            .event_data
                            .get("memory_gb")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0)
                            * 1024.0) as u64,
                        network_rx_bytes: 0,
                        network_tx_bytes: (event
                            .event_data
                            .get("network_gb")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0)
                            * 1_073_741_824.0) as u64,
                        disk_read_bytes: 0,
                        disk_write_bytes: (event
                            .event_data
                            .get("disk_io_gb")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0)
                            * 1_073_741_824.0) as u64,
                        gpu_usage: vec![],
                    }),
                    // Cost calculation would be done per interval
                    cost: "0".to_string(),
                });
            }
        }

        let duration_proto = prost_types::Duration {
            seconds: duration.num_seconds(),
            nanos: (duration.num_nanoseconds().unwrap_or(0) % 1_000_000_000) as i32,
        };

        let summary = UsageSummary {
            avg_cpu_percent: if telemetry_count > 0 {
                total_cpu_percent / telemetry_count as f64
            } else {
                0.0
            },
            avg_memory_mb: if telemetry_count > 0 {
                total_memory_mb / telemetry_count
            } else {
                0
            },
            total_network_bytes,
            total_disk_bytes,
            avg_gpu_utilization: if telemetry_count > 0 {
                total_gpu_percent / telemetry_count as f64
            } else {
                0.0
            },
            duration: Some(duration_proto),
        };

        if data_points.is_empty() {
            data_points.push(UsageDataPoint {
                timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                    rental.created_at,
                ))),
                usage: None,
                cost: Self::format_credit_balance(rental.cost_breakdown.base_cost),
            });
        }

        let response = UsageReportResponse {
            rental_id: rental_id.to_string(),
            data_points,
            summary: Some(summary),
            total_cost: Self::format_credit_balance(rental.cost_breakdown.total_cost),
        };

        Ok(Response::new(response))
    }

    async fn get_billing_packages(
        &self,
        request: Request<GetBillingPackagesRequest>,
    ) -> std::result::Result<Response<GetBillingPackagesResponse>, Status> {
        let req = request.into_inner();
        let user_id = UserId::new(req.user_id);

        let packages = self
            .package_repository
            .list_packages()
            .await
            .map_err(|e| Status::internal(format!("Failed to list packages: {}", e)))?;

        let billing_packages = packages.into_iter().map(|p| p.to_proto()).collect();

        // Get the user's current package preference
        let current_package_id = match self
            .user_preferences_repository
            .get_user_package(&user_id)
            .await
        {
            Ok(Some(pref)) => pref.package_id.to_string(),
            Ok(None) | Err(_) => {
                crate::domain::packages::PricingRules::DEFAULT_PACKAGE_ID.to_string()
            }
        };

        let response = GetBillingPackagesResponse {
            packages: billing_packages,
            current_package_id,
        };

        Ok(Response::new(response))
    }

    async fn set_user_package(
        &self,
        request: Request<SetUserPackageRequest>,
    ) -> std::result::Result<Response<SetUserPackageResponse>, Status> {
        let req = request.into_inner();
        let user_id = UserId::new(req.user_id);
        let new_package_id = PackageId::new(req.package_id.clone());

        info!("Setting package {} for user {}", new_package_id, user_id);

        let _package = self
            .package_repository
            .get_package(&new_package_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get package: {}", e)))?;

        let effective_from = req.effective_from.as_ref().map(|timestamp| {
            chrono::DateTime::from_timestamp(timestamp.seconds, timestamp.nanos as u32)
                .unwrap_or_else(chrono::Utc::now)
        });

        let previous_package_id = self
            .user_preferences_repository
            .set_user_package(&user_id, &new_package_id, effective_from)
            .await
            .map_err(|e| Status::internal(format!("Failed to update user package: {}", e)))?;

        let response = SetUserPackageResponse {
            success: true,
            previous_package_id: previous_package_id
                .unwrap_or_else(PackageId::standard)
                .to_string(),
            new_package_id: new_package_id.to_string(),
            effective_from: req.effective_from,
        };

        Ok(Response::new(response))
    }
}
