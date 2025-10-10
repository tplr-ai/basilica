use crate::domain::{
    rentals::{Rental, RentalStatistics},
    types::{
        CostBreakdown, CreditBalance, PackageId, RentalId, RentalState, ResourceSpec, UsageMetrics,
        UserId,
    },
};
use crate::error::{BillingError, Result};
use crate::storage::rds::RdsConnection;
use async_trait::async_trait;
use sqlx::Row;
use std::sync::Arc;

#[async_trait]
pub trait RentalRepository: Send + Sync {
    async fn create_rental(&self, rental: &Rental) -> Result<()>;
    async fn get_rental(&self, id: &RentalId) -> Result<Option<Rental>>;
    async fn update_rental(&self, rental: &Rental) -> Result<()>;
    async fn get_active_rentals(&self, user_id: Option<&UserId>) -> Result<Vec<Rental>>;
    async fn get_rentals_by_state(&self, state: RentalState) -> Result<Vec<Rental>>;
    async fn get_rental_statistics(&self, user_id: Option<&UserId>) -> Result<RentalStatistics>;
}

pub struct SqlRentalRepository {
    connection: Arc<RdsConnection>,
}

impl SqlRentalRepository {
    pub fn new(connection: Arc<RdsConnection>) -> Self {
        Self { connection }
    }

    fn parse_rental_state(state_str: &str) -> RentalState {
        match state_str {
            "pending" => RentalState::Pending,
            "active" => RentalState::Active,
            "suspended" => RentalState::Suspended,
            "terminating" => RentalState::Terminating,
            "completed" => RentalState::Completed,
            "failed" => RentalState::Failed,
            _ => RentalState::Failed,
        }
    }

    fn rental_from_row(r: &sqlx::postgres::PgRow) -> Rental {
        let status_str: String = r.get("status");
        let state = Self::parse_rental_state(&status_str);

        Rental {
            id: RentalId::from_uuid(r.get("rental_id")),
            user_id: UserId::new(r.get("user_id")),
            node_id: r.get("node_id"),
            validator_id: r
                .get::<Option<String>, _>("validator_id")
                .unwrap_or_default(),
            package_id: r
                .get::<Option<String>, _>("package_id")
                .map(PackageId::new)
                .unwrap_or_else(PackageId::custom),
            reservation_id: None, // No reservation_id in rentals table
            state,
            resource_spec: serde_json::from_value(r.get("resource_spec")).unwrap_or(ResourceSpec {
                gpu_specs: vec![],
                cpu_cores: 0,
                memory_gb: 0,
                storage_gb: 0,
                disk_iops: 0,
                network_bandwidth_mbps: 0,
            }),
            usage_metrics: UsageMetrics::zero(), // Not stored in rentals table
            cost_breakdown: {
                let hourly_rate = r.get::<rust_decimal::Decimal, _>("hourly_rate");
                let total_cost = r
                    .get::<Option<rust_decimal::Decimal>, _>("total_cost")
                    .unwrap_or(rust_decimal::Decimal::ZERO);
                CostBreakdown {
                    base_cost: CreditBalance::from_decimal(hourly_rate),
                    usage_cost: CreditBalance::zero(),
                    discounts: CreditBalance::zero(),
                    overage_charges: CreditBalance::zero(),
                    total_cost: CreditBalance::from_decimal(total_cost),
                }
            },
            started_at: r.get("start_time"),
            updated_at: r.get("updated_at"),
            ended_at: r.get("end_time"),
            metadata: serde_json::from_value(r.get("metadata")).unwrap_or_default(),
            created_at: r.get("created_at"),
            last_updated: r.get("updated_at"),
            actual_start_time: Some(r.get("start_time")),
            actual_end_time: r.get("end_time"),
            actual_cost: r
                .get::<Option<rust_decimal::Decimal>, _>("total_cost")
                .map(CreditBalance::from_decimal)
                .unwrap_or_else(CreditBalance::zero),
        }
    }
}

#[async_trait]
impl RentalRepository for SqlRentalRepository {
    async fn create_rental(&self, rental: &Rental) -> Result<()> {
        let resource_spec_json = serde_json::to_value(&rental.resource_spec)?;
        let metadata_json = serde_json::to_value(&rental.metadata)?;
        let hourly_rate = rental.cost_breakdown.base_cost.as_decimal();

        sqlx::query(
            r#"
            INSERT INTO billing.rentals
            (rental_id, user_id, node_id, validator_id, package_id, status,
             resource_spec, hourly_rate, start_time, max_duration_hours, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(rental.id.as_uuid())
        .bind(rental.user_id.as_str())
        .bind(&rental.node_id)
        .bind(if rental.validator_id.is_empty() {
            None
        } else {
            Some(&rental.validator_id)
        })
        .bind(Some(rental.package_id.as_str()))
        .bind(rental.state.to_string())
        .bind(resource_spec_json)
        .bind(hourly_rate)
        .bind(rental.started_at)
        .bind(24i32) // Default max duration
        .bind(metadata_json)
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "create_rental".to_string(),
            source: Box::new(e),
        })?;

        Ok(())
    }

    async fn get_rental(&self, id: &RentalId) -> Result<Option<Rental>> {
        let row = sqlx::query(
            r#"
            SELECT rental_id, user_id, node_id, validator_id, package_id, status,
                   resource_spec, hourly_rate, start_time, end_time, total_cost,
                   metadata, created_at, updated_at
            FROM billing.rentals
            WHERE rental_id = $1
            "#,
        )
        .bind(id.as_uuid())
        .fetch_optional(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "get_rental".to_string(),
            source: Box::new(e),
        })?;

        Ok(row.map(|r| Self::rental_from_row(&r)))
    }

    async fn update_rental(&self, rental: &Rental) -> Result<()> {
        let resource_spec_json = serde_json::to_value(&rental.resource_spec)?;
        let metadata_json = serde_json::to_value(&rental.metadata)?;
        let hourly_rate = rental.cost_breakdown.base_cost.as_decimal();
        let total_cost = rental.actual_cost.as_decimal();

        let result = sqlx::query(
            r#"
            UPDATE billing.rentals
            SET status = $2, resource_spec = $3, hourly_rate = $4,
                updated_at = $5, end_time = $6, metadata = $7, total_cost = $8
            WHERE rental_id = $1
            "#,
        )
        .bind(rental.id.as_uuid())
        .bind(rental.state.to_string())
        .bind(resource_spec_json)
        .bind(hourly_rate)
        .bind(chrono::Utc::now())
        .bind(rental.ended_at)
        .bind(metadata_json)
        .bind(if total_cost == rust_decimal::Decimal::ZERO {
            None
        } else {
            Some(total_cost)
        })
        .execute(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "update_rental".to_string(),
            source: Box::new(e),
        })?;

        if result.rows_affected() == 0 {
            return Err(BillingError::RentalNotFound {
                id: rental.id.to_string(),
            });
        }

        Ok(())
    }

    async fn get_active_rentals(&self, user_id: Option<&UserId>) -> Result<Vec<Rental>> {
        let query = if let Some(uid) = user_id {
            sqlx::query(
                r#"
                SELECT rental_id, user_id, node_id, validator_id, package_id, status,
                       resource_spec, hourly_rate, start_time, end_time, total_cost,
                       metadata, created_at, updated_at
                FROM billing.rentals
                WHERE user_id = $1 AND status IN ('active', 'pending')
                ORDER BY start_time DESC
                "#,
            )
            .bind(uid.as_str())
        } else {
            sqlx::query(
                r#"
                SELECT rental_id, user_id, node_id, validator_id, package_id, status,
                       resource_spec, hourly_rate, start_time, end_time, total_cost,
                       metadata, created_at, updated_at
                FROM billing.rentals
                WHERE status IN ('active', 'pending')
                ORDER BY start_time DESC
                "#,
            )
        };

        let rows = query.fetch_all(self.connection.pool()).await.map_err(|e| {
            BillingError::DatabaseError {
                operation: "get_active_rentals".to_string(),
                source: Box::new(e),
            }
        })?;

        Ok(rows.iter().map(Self::rental_from_row).collect())
    }

    async fn get_rentals_by_state(&self, state: RentalState) -> Result<Vec<Rental>> {
        let rows = sqlx::query(
            r#"
            SELECT rental_id, user_id, node_id, validator_id, package_id, status,
                   resource_spec, hourly_rate, start_time, end_time, total_cost,
                   metadata, created_at, updated_at
            FROM billing.rentals
            WHERE status = $1
            ORDER BY start_time DESC
            "#,
        )
        .bind(state.to_string())
        .fetch_all(self.connection.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "get_rentals_by_state".to_string(),
            source: Box::new(e),
        })?;

        Ok(rows.iter().map(Self::rental_from_row).collect())
    }

    async fn get_rental_statistics(&self, user_id: Option<&UserId>) -> Result<RentalStatistics> {
        let query = if let Some(uid) = user_id {
            sqlx::query(
                r#"
                SELECT
                    COUNT(*) as total_rentals,
                    COUNT(*) FILTER (WHERE status IN ('active', 'pending')) as active_rentals,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_rentals,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_rentals,
                    COALESCE(SUM(EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) / 3600), 0) as total_gpu_hours,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(AVG(EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) / 3600), 0) as avg_duration_hours
                FROM billing.rentals
                WHERE user_id = $1
                "#,
            )
            .bind(uid.as_str())
        } else {
            sqlx::query(
                r#"
                SELECT
                    COUNT(*) as total_rentals,
                    COUNT(*) FILTER (WHERE status IN ('active', 'pending')) as active_rentals,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_rentals,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_rentals,
                    COALESCE(SUM(EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) / 3600), 0) as total_gpu_hours,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(AVG(EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) / 3600), 0) as avg_duration_hours
                FROM billing.rentals
                "#,
            )
        };

        let row = query.fetch_one(self.connection.pool()).await.map_err(|e| {
            BillingError::DatabaseError {
                operation: "get_rental_statistics".to_string(),
                source: Box::new(e),
            }
        })?;

        Ok(RentalStatistics {
            total_rentals: row.get::<i64, _>("total_rentals") as u64,
            active_rentals: row.get::<i64, _>("active_rentals") as u64,
            completed_rentals: row.get::<i64, _>("completed_rentals") as u64,
            failed_rentals: row.get::<i64, _>("failed_rentals") as u64,
            total_gpu_hours: row.get("total_gpu_hours"),
            total_cost: CreditBalance::from_decimal(
                row.get::<Option<rust_decimal::Decimal>, _>("total_cost")
                    .unwrap_or(rust_decimal::Decimal::ZERO),
            ),
            average_duration_hours: row.get::<f64, _>("avg_duration_hours"),
        })
    }
}
