use crate::domain::idempotency::generate_idempotency_key;
use crate::domain::processor::RentalEndData;
use crate::domain::types::{
    CostBreakdown, CreditBalance, RentalId, RentalState, ResourceSpec, UsageMetrics, UserId,
};
use crate::error::{BillingError, Result};
use crate::storage::events::EventType;
use crate::storage::rentals::RentalRepository;
use crate::storage::{EventRepository, UsageEvent};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Type of cloud for rental
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudType {
    /// Community cloud (validator-based) rental
    Community,
    /// Secure cloud (direct provider API) rental
    Secure,
}

impl CloudType {
    pub fn as_str(&self) -> &'static str {
        match self {
            CloudType::Community => "community",
            CloudType::Secure => "secure",
        }
    }
}

impl std::fmt::Display for CloudType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for CloudType {
    type Err = BillingError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "community" => Ok(CloudType::Community),
            "secure" => Ok(CloudType::Secure),
            _ => Err(BillingError::ValidationError {
                field: "cloud_type".to_string(),
                message: format!("unknown cloud type: {}", s),
            }),
        }
    }
}

/// Unified rental record for both community and secure cloud rentals
///
/// Type-specific data is stored in `metadata`:
/// - Community: `node_id`, `validator_id`
/// - Secure: `provider`, `provider_instance_id`, `offering_id`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rental {
    pub id: RentalId,
    pub user_id: UserId,
    pub cloud_type: CloudType,
    pub state: RentalState,
    pub resource_spec: ResourceSpec,
    pub usage_metrics: UsageMetrics,
    pub cost_breakdown: CostBreakdown,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    /// Type-specific metadata:
    /// - Community: node_id, validator_id
    /// - Secure: provider, provider_instance_id, offering_id
    pub metadata: HashMap<String, String>,
    // Aliases for compatibility
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    // Additional fields for billing handlers
    pub actual_start_time: Option<DateTime<Utc>>,
    pub actual_end_time: Option<DateTime<Utc>>,
    pub actual_cost: CreditBalance,

    // Marketplace-2-compute pricing fields
    /// Final price per GPU per hour (already includes any markup applied by API layer)
    pub base_price_per_gpu: Decimal,
    /// Number of GPUs in this rental
    pub gpu_count: u32,
}

impl Rental {
    /// Create a new community cloud rental
    pub fn new_community(params: CreateCommunityRentalParams) -> Self {
        let now = Utc::now();
        let mut metadata = HashMap::new();
        metadata.insert("node_id".to_string(), params.node_id);
        metadata.insert("validator_id".to_string(), params.validator_id);
        metadata.insert("miner_uid".to_string(), params.miner_uid.to_string());
        metadata.insert("miner_hotkey".to_string(), params.miner_hotkey);

        Self {
            id: RentalId::new(),
            user_id: params.user_id,
            cloud_type: CloudType::Community,
            state: RentalState::Pending,
            resource_spec: params.resource_spec,
            usage_metrics: UsageMetrics::zero(),
            cost_breakdown: CostBreakdown {
                base_cost: CreditBalance::zero(),
                usage_cost: CreditBalance::zero(),
                volume_discount: CreditBalance::zero(),
                discounts: CreditBalance::zero(),
                overage_charges: CreditBalance::zero(),
                total_cost: CreditBalance::zero(),
            },
            started_at: now,
            updated_at: now,
            ended_at: None,
            metadata,
            created_at: now,
            last_updated: now,
            actual_start_time: None,
            actual_end_time: None,
            actual_cost: CreditBalance::zero(),
            base_price_per_gpu: params.base_price_per_gpu,
            gpu_count: params.gpu_count,
        }
    }

    /// Create a new secure cloud rental
    pub fn new_secure(
        user_id: UserId,
        provider: String,
        provider_instance_id: String,
        offering_id: String,
        resource_spec: ResourceSpec,
        base_price_per_gpu: Decimal,
        gpu_count: u32,
    ) -> Self {
        let now = Utc::now();
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), provider);
        metadata.insert("provider_instance_id".to_string(), provider_instance_id);
        metadata.insert("offering_id".to_string(), offering_id);

        Self {
            id: RentalId::new(),
            user_id,
            cloud_type: CloudType::Secure,
            state: RentalState::Pending,
            resource_spec,
            usage_metrics: UsageMetrics::zero(),
            cost_breakdown: CostBreakdown {
                base_cost: CreditBalance::zero(),
                usage_cost: CreditBalance::zero(),
                volume_discount: CreditBalance::zero(),
                discounts: CreditBalance::zero(),
                overage_charges: CreditBalance::zero(),
                total_cost: CreditBalance::zero(),
            },
            started_at: now,
            updated_at: now,
            ended_at: None,
            metadata,
            created_at: now,
            last_updated: now,
            actual_start_time: None,
            actual_end_time: None,
            actual_cost: CreditBalance::zero(),
            base_price_per_gpu,
            gpu_count,
        }
    }

    // Accessor methods for type-specific metadata

    /// Get node_id (community cloud only)
    pub fn node_id(&self) -> Option<&str> {
        self.metadata.get("node_id").map(|s| s.as_str())
    }

    /// Get validator_id (community cloud only)
    pub fn validator_id(&self) -> Option<&str> {
        self.metadata.get("validator_id").map(|s| s.as_str())
    }

    /// Get miner_uid (community cloud only) - Bittensor miner UID
    pub fn miner_uid(&self) -> Option<u32> {
        self.metadata.get("miner_uid").and_then(|s| s.parse().ok())
    }

    /// Get miner_hotkey (community cloud only) - Bittensor miner hotkey
    pub fn miner_hotkey(&self) -> Option<&str> {
        self.metadata.get("miner_hotkey").map(|s| s.as_str())
    }

    /// Get provider (secure cloud only)
    pub fn provider(&self) -> Option<&str> {
        self.metadata.get("provider").map(|s| s.as_str())
    }

    /// Get provider_instance_id (secure cloud only)
    pub fn provider_instance_id(&self) -> Option<&str> {
        self.metadata
            .get("provider_instance_id")
            .map(|s| s.as_str())
    }

    /// Node identifier to use when emitting events.
    /// For community rentals this is the node_id; for secure rentals we
    /// fall back to provider_instance_id so validation passes and events
    /// stay consistent with rental_start/telemetry.
    pub fn event_node_id(&self) -> Option<&str> {
        self.node_id().or_else(|| self.provider_instance_id())
    }

    /// Get offering_id (secure cloud only)
    pub fn offering_id(&self) -> Option<&str> {
        self.metadata.get("offering_id").map(|s| s.as_str())
    }

    pub fn duration(&self) -> chrono::Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }

    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }

    pub fn transition_to(&mut self, new_state: RentalState) -> Result<()> {
        if !self.state.can_transition_to(new_state) {
            return Err(BillingError::InvalidStateTransition {
                from: self.state.to_string(),
                to: new_state.to_string(),
            });
        }

        self.state = new_state;
        let now = Utc::now();
        self.updated_at = now;
        self.last_updated = now;

        if new_state.is_terminal() && self.ended_at.is_none() {
            self.ended_at = Some(now);
        }

        Ok(())
    }

    pub fn update_usage(&mut self, metrics: UsageMetrics) {
        self.usage_metrics = self.usage_metrics.add(&metrics);
        self.updated_at = Utc::now();
        self.last_updated = self.updated_at;
    }

    pub fn update_cost(&mut self, cost_breakdown: CostBreakdown) {
        self.cost_breakdown = cost_breakdown;
        self.updated_at = Utc::now();
        self.last_updated = self.updated_at;
    }

    pub fn calculate_current_cost(&self, rate_per_hour: CreditBalance) -> CreditBalance {
        let hours = self.duration().num_seconds() as f64 / 3600.0;
        let hours_decimal = Decimal::from_f64(hours).unwrap_or(Decimal::ZERO);
        rate_per_hour.multiply(hours_decimal)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalStatistics {
    pub total_rentals: u64,
    pub active_rentals: u64,
    pub completed_rentals: u64,
    pub failed_rentals: u64,
    pub total_gpu_hours: Decimal,
    pub total_cost: CreditBalance,
    pub average_duration_hours: f64,
}

/// Parameters for creating a new community cloud rental
#[derive(Debug, Clone)]
pub struct CreateCommunityRentalParams {
    pub user_id: UserId,
    pub node_id: String,
    pub validator_id: String,
    /// Bittensor miner UID for payment reconciliation
    pub miner_uid: u32,
    /// Bittensor miner hotkey for payment reconciliation
    pub miner_hotkey: String,
    pub resource_spec: ResourceSpec,
    pub base_price_per_gpu: Decimal,
    pub gpu_count: u32,
}

/// Parameters for creating a new secure cloud rental
#[derive(Debug, Clone)]
pub struct CreateSecureRentalParams {
    pub user_id: UserId,
    pub provider: String,
    pub provider_instance_id: String,
    pub offering_id: String,
    pub resource_spec: ResourceSpec,
    pub base_price_per_gpu: Decimal,
    pub gpu_count: u32,
}

/// Unified parameters for creating a rental
#[derive(Debug, Clone)]
pub enum CreateRentalParams {
    Community(CreateCommunityRentalParams),
    Secure(CreateSecureRentalParams),
}

/// Rental management operations
#[async_trait]
pub trait RentalOperations: Send + Sync {
    async fn create_rental(&self, params: CreateRentalParams) -> Result<RentalId>;

    async fn get_rental(&self, rental_id: &RentalId) -> Result<Rental>;

    async fn update_rental_state(&self, rental_id: &RentalId, new_state: RentalState)
        -> Result<()>;

    async fn update_rental_usage(&self, rental_id: &RentalId, metrics: UsageMetrics) -> Result<()>;

    async fn update_rental_cost(&self, rental_id: &RentalId, cost: CostBreakdown) -> Result<()>;

    async fn get_rentals(&self, user_id: &UserId) -> Result<Vec<Rental>>;

    async fn get_all_rentals(&self) -> Result<Vec<Rental>>;

    async fn get_rental_statistics(&self, user_id: Option<&UserId>) -> Result<RentalStatistics>;

    async fn terminate_rental(&self, rental_id: &RentalId, reason: String) -> Result<()>;

    async fn update_status(&self, rental_id: &RentalId, new_state: RentalState) -> Result<Rental>;

    async fn finalize_rental(&self, rental_id: &RentalId) -> Result<Rental>;
}

pub struct RentalManager {
    repository: Arc<dyn crate::storage::rentals::RentalRepository + Send + Sync>,
}

impl RentalManager {
    pub fn new(
        repository: Arc<dyn crate::storage::rentals::RentalRepository + Send + Sync>,
    ) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl RentalOperations for RentalManager {
    async fn create_rental(&self, params: CreateRentalParams) -> Result<RentalId> {
        let rental = match params {
            CreateRentalParams::Community(p) => Rental::new_community(p),
            CreateRentalParams::Secure(p) => Rental::new_secure(
                p.user_id,
                p.provider,
                p.provider_instance_id,
                p.offering_id,
                p.resource_spec,
                p.base_price_per_gpu,
                p.gpu_count,
            ),
        };
        let rental_id = rental.id;

        self.repository.create_rental(&rental).await?;

        Ok(rental_id)
    }

    async fn get_rental(&self, rental_id: &RentalId) -> Result<Rental> {
        self.repository
            .get_rental(rental_id)
            .await?
            .ok_or_else(|| BillingError::RentalNotFound {
                id: rental_id.to_string(),
            })
    }

    async fn update_rental_state(
        &self,
        rental_id: &RentalId,
        new_state: RentalState,
    ) -> Result<()> {
        let mut rental = self.get_rental(rental_id).await?;
        rental.transition_to(new_state)?;
        self.repository.update_rental(&rental).await
    }

    async fn update_rental_usage(&self, rental_id: &RentalId, metrics: UsageMetrics) -> Result<()> {
        let mut rental = self.get_rental(rental_id).await?;
        rental.update_usage(metrics);
        self.repository.update_rental(&rental).await
    }

    async fn update_rental_cost(&self, rental_id: &RentalId, cost: CostBreakdown) -> Result<()> {
        let mut rental = self.get_rental(rental_id).await?;
        rental.update_cost(cost);
        self.repository.update_rental(&rental).await
    }

    async fn get_rentals(&self, user_id: &UserId) -> Result<Vec<Rental>> {
        self.repository.get_rentals(Some(user_id)).await
    }

    async fn get_all_rentals(&self) -> Result<Vec<Rental>> {
        self.repository.get_rentals(None).await
    }

    async fn get_rental_statistics(&self, user_id: Option<&UserId>) -> Result<RentalStatistics> {
        self.repository.get_rental_statistics(user_id).await
    }

    async fn terminate_rental(&self, rental_id: &RentalId, reason: String) -> Result<()> {
        let mut rental = self.get_rental(rental_id).await?;

        rental
            .metadata
            .insert("termination_reason".to_string(), reason);

        if rental.state.can_transition_to(RentalState::Terminating) {
            rental.transition_to(RentalState::Terminating)?;
            rental.transition_to(RentalState::Completed)?;
        }

        self.repository.update_rental(&rental).await
    }

    async fn finalize_rental(&self, rental_id: &RentalId) -> Result<Rental> {
        let mut rental = self.get_rental(rental_id).await?;

        if rental.state == RentalState::Active {
            rental.transition_to(RentalState::Terminating)?;
        }

        if rental.state == RentalState::Terminating {
            rental.transition_to(RentalState::Completed)?;
        } else if rental.state != RentalState::Completed {
            return Err(BillingError::InvalidStateTransition {
                from: rental.state.to_string(),
                to: RentalState::Completed.to_string(),
            });
        }

        self.repository.update_rental(&rental).await?;
        Ok(rental)
    }

    async fn update_status(&self, rental_id: &RentalId, new_state: RentalState) -> Result<Rental> {
        let mut rental = self.get_rental(rental_id).await?;
        rental.transition_to(new_state)?;
        self.repository.update_rental(&rental).await?;
        Ok(rental)
    }
}

/// Core rental finalization logic - shared between BillingEventHandlers and BillingServiceImpl.
///
/// This function:
/// 1. Updates rental state, ended_at, and last_updated
/// 2. Persists the rental to the database
/// 3. Emits a rental_end usage event for audit
///
/// Returns the final cost as a Decimal.
pub async fn finalize_rental_core(
    rental: &mut Rental,
    target_state: RentalState,
    end_time: DateTime<Utc>,
    termination_reason: Option<&str>,
    rental_repository: &dyn RentalRepository,
    event_repository: &dyn EventRepository,
) -> Result<Decimal> {
    info!(
        "Finalizing rental {} to state {:?} (reason: {:?})",
        rental.id, target_state, termination_reason
    );

    // Update rental state and timestamps
    rental.state = target_state;
    rental.ended_at = Some(end_time);
    rental.last_updated = end_time;

    // Persist the updated rental
    rental_repository.update_rental(rental).await?;

    // Create rental_end event for audit
    let final_cost = rental.actual_cost.as_decimal();
    let rental_end_data = RentalEndData {
        end_time,
        final_cost,
        termination_reason: termination_reason.map(|s| s.to_string()),
    };

    let mut event_data =
        serde_json::to_value(&rental_end_data).map_err(|e| BillingError::ValidationError {
            field: "rental_end_data".to_string(),
            message: format!("Failed to serialize rental end data: {}", e),
        })?;

    // Add metadata to event
    if let serde_json::Value::Object(ref mut map) = event_data {
        map.insert(
            "timestamp".to_string(),
            serde_json::Value::String(Utc::now().timestamp_millis().to_string()),
        );
        map.insert(
            "cloud_type".to_string(),
            serde_json::Value::String(rental.cloud_type.to_string()),
        );
        map.insert(
            "target_state".to_string(),
            serde_json::Value::String(format!("{:?}", target_state)),
        );
    }

    let idempotency_key = generate_idempotency_key(rental.id.as_uuid(), &event_data);

    // Secure cloud rentals do not carry node_id; use provider_instance_id instead.
    let node_id = rental
        .event_node_id()
        .ok_or_else(|| BillingError::ValidationError {
            field: "node_id".to_string(),
            message: "node_id or provider_instance_id is required for usage event".to_string(),
        })?
        .to_string();
    let validator_id = rental.validator_id().map(|s| s.to_string());

    let usage_event = UsageEvent {
        event_id: Uuid::new_v4(),
        rental_id: rental.id.as_uuid(),
        user_id: rental.user_id.as_str().to_string(),
        node_id,
        validator_id,
        event_type: EventType::RentalEnd,
        event_data,
        timestamp: Utc::now(),
        processed: false,
        processed_at: None,
        batch_id: None,
        idempotency_key: Some(idempotency_key),
    };

    event_repository.append_usage_event(&usage_event).await?;

    info!(
        "Rental {} finalized to {:?} with total_cost: {}",
        rental.id, target_state, final_cost
    );

    Ok(final_cost)
}
