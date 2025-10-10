use crate::domain::types::{
    CostBreakdown, CreditBalance, PackageId, RentalId, RentalState, ReservationId, ResourceSpec,
    UsageMetrics, UserId,
};
use crate::error::{BillingError, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rental {
    pub id: RentalId,
    pub user_id: UserId,
    pub node_id: String,
    pub validator_id: String,
    pub package_id: PackageId,
    pub reservation_id: Option<ReservationId>,
    pub state: RentalState,
    pub resource_spec: ResourceSpec,
    pub usage_metrics: UsageMetrics,
    pub cost_breakdown: CostBreakdown,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
    // Aliases for compatibility
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    // Additional fields for billing handlers
    pub actual_start_time: Option<DateTime<Utc>>,
    pub actual_end_time: Option<DateTime<Utc>>,
    pub actual_cost: CreditBalance,
}

impl Rental {
    pub fn new(
        user_id: UserId,
        node_id: String,
        validator_id: String,
        package_id: PackageId,
        resource_spec: ResourceSpec,
        reservation_id: Option<ReservationId>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: RentalId::new(),
            user_id,
            node_id,
            validator_id,
            package_id,
            reservation_id,
            state: RentalState::Pending,
            resource_spec,
            usage_metrics: UsageMetrics::zero(),
            cost_breakdown: CostBreakdown {
                base_cost: CreditBalance::zero(),
                usage_cost: CreditBalance::zero(),
                discounts: CreditBalance::zero(),
                overage_charges: CreditBalance::zero(),
                total_cost: CreditBalance::zero(),
            },
            started_at: now,
            updated_at: now,
            ended_at: None,
            metadata: HashMap::new(),
            created_at: now,
            last_updated: now,
            actual_start_time: None,
            actual_end_time: None,
            actual_cost: CreditBalance::zero(),
        }
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

/// Rental management operations
#[async_trait]
pub trait RentalOperations: Send + Sync {
    async fn create_rental(
        &self,
        user_id: UserId,
        node_id: String,
        validator_id: String,
        package_id: PackageId,
        resource_spec: ResourceSpec,
        reservation_id: Option<ReservationId>,
    ) -> Result<RentalId>;

    async fn get_rental(&self, rental_id: &RentalId) -> Result<Rental>;

    async fn update_rental_state(&self, rental_id: &RentalId, new_state: RentalState)
        -> Result<()>;

    async fn update_rental_usage(&self, rental_id: &RentalId, metrics: UsageMetrics) -> Result<()>;

    async fn update_rental_cost(&self, rental_id: &RentalId, cost: CostBreakdown) -> Result<()>;

    async fn get_active_rentals(&self, user_id: &UserId) -> Result<Vec<Rental>>;

    async fn get_all_active_rentals(&self) -> Result<Vec<Rental>>;

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
    async fn create_rental(
        &self,
        user_id: UserId,
        node_id: String,
        validator_id: String,
        package_id: PackageId,
        resource_spec: ResourceSpec,
        reservation_id: Option<ReservationId>,
    ) -> Result<RentalId> {
        let rental = Rental::new(
            user_id,
            node_id,
            validator_id,
            package_id,
            resource_spec,
            reservation_id,
        );
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

    async fn get_active_rentals(&self, user_id: &UserId) -> Result<Vec<Rental>> {
        self.repository.get_active_rentals(Some(user_id)).await
    }

    async fn get_all_active_rentals(&self) -> Result<Vec<Rental>> {
        self.repository.get_active_rentals(None).await
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
