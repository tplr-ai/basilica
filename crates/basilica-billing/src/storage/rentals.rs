use crate::domain::{
    rentals::{CloudType, Rental, RentalStatistics},
    types::{
        CostBreakdown, CreditBalance, RentalId, RentalState, ResourceSpec, UsageMetrics, UserId,
    },
};
use crate::error::{BillingError, Result};
use crate::storage::rds::RdsConnection;
use async_trait::async_trait;
use sqlx::{Postgres, Row, Transaction};
use std::sync::Arc;

#[async_trait]
pub trait RentalRepository: Send + Sync {
    async fn create_rental(&self, rental: &Rental) -> Result<()>;
    async fn get_rental(&self, id: &RentalId) -> Result<Option<Rental>>;
    async fn update_rental(&self, rental: &Rental) -> Result<()>;
    async fn get_rentals(&self, user_id: Option<&UserId>) -> Result<Vec<Rental>>;
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

    pub fn pool(&self) -> &sqlx::PgPool {
        self.connection.pool()
    }

    fn parse_rental_state(state_str: &str) -> RentalState {
        match state_str {
            "pending" => RentalState::Pending,
            "active" => RentalState::Active,
            "suspended" => RentalState::Suspended,
            "terminating" => RentalState::Terminating,
            "completed" => RentalState::Completed,
            "failed" => RentalState::Failed,
            "failed_insufficient_credits" => RentalState::FailedInsufficientCredits,
            _ => RentalState::Failed,
        }
    }

    fn parse_cloud_type(cloud_type_str: &str) -> CloudType {
        match cloud_type_str {
            "community" => CloudType::Community,
            "secure" => CloudType::Secure,
            _ => CloudType::Community, // Default to community for legacy data
        }
    }

    fn rental_from_row(r: &sqlx::postgres::PgRow) -> Rental {
        let status_str: String = r.get("status");
        let state = Self::parse_rental_state(&status_str);

        let cloud_type_str: String = r.get("cloud_type");
        let cloud_type = Self::parse_cloud_type(&cloud_type_str);

        let resource_spec: ResourceSpec =
            serde_json::from_value(r.get("resource_spec")).unwrap_or(ResourceSpec {
                gpu_specs: vec![],
                cpu_cores: 0,
                memory_gb: 0,
                storage_gb: 0,
                disk_iops: 0,
                network_bandwidth_mbps: 0,
            });

        // Read marketplace pricing fields (with defaults for legacy rentals)
        let base_price_per_gpu = r
            .get::<Option<rust_decimal::Decimal>, _>("base_price_per_gpu")
            .unwrap_or(rust_decimal::Decimal::ZERO);

        let gpu_count = r.get::<Option<i32>, _>("gpu_count").unwrap_or(1) as u32;

        // Metadata contains type-specific data (node_id, validator_id for community;
        // provider, provider_instance_id, offering_id for secure)
        let metadata: std::collections::HashMap<String, String> =
            serde_json::from_value(r.get("metadata")).unwrap_or_default();

        Rental {
            id: RentalId::from_uuid(r.get("rental_id")),
            user_id: UserId::new(r.get("user_id")),
            cloud_type,
            state,
            resource_spec,
            usage_metrics: UsageMetrics::zero(),
            cost_breakdown: {
                // Derive hourly rate from base_price_per_gpu * gpu_count
                let hourly_rate = base_price_per_gpu * rust_decimal::Decimal::from(gpu_count);
                let total_cost = r
                    .get::<Option<rust_decimal::Decimal>, _>("total_cost")
                    .unwrap_or(rust_decimal::Decimal::ZERO);
                CostBreakdown {
                    base_cost: CreditBalance::from_decimal(hourly_rate),
                    usage_cost: CreditBalance::zero(),
                    volume_discount: CreditBalance::zero(),
                    discounts: CreditBalance::zero(),
                    overage_charges: CreditBalance::zero(),
                    total_cost: CreditBalance::from_decimal(total_cost),
                }
            },
            started_at: r.get("start_time"),
            updated_at: r.get("updated_at"),
            ended_at: r.get("end_time"),
            metadata,
            created_at: r.get("created_at"),
            last_updated: r.get("updated_at"),
            actual_start_time: Some(r.get("start_time")),
            actual_end_time: r.get("end_time"),
            actual_cost: r
                .get::<Option<rust_decimal::Decimal>, _>("total_cost")
                .map(CreditBalance::from_decimal)
                .unwrap_or_else(CreditBalance::zero),
            // Marketplace-2-compute fields
            base_price_per_gpu,
            gpu_count,
        }
    }
}

#[async_trait]
impl RentalRepository for SqlRentalRepository {
    async fn create_rental(&self, rental: &Rental) -> Result<()> {
        let resource_spec_json = serde_json::to_value(&rental.resource_spec)?;
        let metadata_json = serde_json::to_value(&rental.metadata)?;

        sqlx::query(
            r#"
            INSERT INTO billing.rentals
            (rental_id, user_id, cloud_type, status,
             resource_spec, start_time, metadata,
             base_price_per_gpu, gpu_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(rental.id.as_uuid())
        .bind(rental.user_id.as_str())
        .bind(rental.cloud_type.as_str())
        .bind(rental.state.to_string())
        .bind(resource_spec_json)
        .bind(rental.started_at)
        .bind(metadata_json)
        .bind(rental.base_price_per_gpu)
        .bind(rental.gpu_count as i32)
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
            SELECT rental_id, user_id, cloud_type, status,
                   resource_spec, start_time, end_time, total_cost,
                   metadata, created_at, updated_at,
                   base_price_per_gpu, gpu_count
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
        let total_cost = rental.actual_cost.as_decimal();

        let result = sqlx::query(
            r#"
            UPDATE billing.rentals
            SET status = $2, resource_spec = $3,
                updated_at = $4, end_time = $5, metadata = $6,
                total_cost = $7, base_price_per_gpu = $8
            WHERE rental_id = $1
            "#,
        )
        .bind(rental.id.as_uuid())
        .bind(rental.state.to_string())
        .bind(resource_spec_json)
        .bind(chrono::Utc::now())
        .bind(rental.ended_at)
        .bind(metadata_json)
        .bind(if total_cost == rust_decimal::Decimal::ZERO {
            None
        } else {
            Some(total_cost)
        })
        .bind(rental.base_price_per_gpu)
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

    async fn get_rentals(&self, user_id: Option<&UserId>) -> Result<Vec<Rental>> {
        // Note: This returns ALL rentals for the user. Filtering by status is done
        // in the gRPC service layer based on the status_filter parameter.
        let query = if let Some(uid) = user_id {
            sqlx::query(
                r#"
                SELECT rental_id, user_id, cloud_type, status,
                       resource_spec, start_time, end_time, total_cost,
                       metadata, created_at, updated_at,
                       base_price_per_gpu, gpu_count
                FROM billing.rentals
                WHERE user_id = $1
                ORDER BY start_time DESC
                "#,
            )
            .bind(uid.as_str())
        } else {
            sqlx::query(
                r#"
                SELECT rental_id, user_id, cloud_type, status,
                       resource_spec, start_time, end_time, total_cost,
                       metadata, created_at, updated_at,
                       base_price_per_gpu, gpu_count
                FROM billing.rentals
                ORDER BY start_time DESC
                "#,
            )
        };

        let rows = query.fetch_all(self.connection.pool()).await.map_err(|e| {
            BillingError::DatabaseError {
                operation: "get_rentals".to_string(),
                source: Box::new(e),
            }
        })?;

        Ok(rows.iter().map(Self::rental_from_row).collect())
    }

    async fn get_rentals_by_state(&self, state: RentalState) -> Result<Vec<Rental>> {
        let rows = sqlx::query(
            r#"
            SELECT rental_id, user_id, cloud_type, status,
                   resource_spec, start_time, end_time, total_cost,
                   metadata, created_at, updated_at,
                   base_price_per_gpu, gpu_count
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

impl SqlRentalRepository {
    pub async fn update_rental_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        rental: &Rental,
    ) -> Result<()> {
        let resource_spec_json = serde_json::to_value(&rental.resource_spec)?;
        let metadata_json = serde_json::to_value(&rental.metadata)?;
        let total_cost = rental.actual_cost.as_decimal();

        let result = sqlx::query(
            r#"
            UPDATE billing.rentals
            SET status = $2, resource_spec = $3,
                updated_at = $4, end_time = $5, metadata = $6,
                total_cost = $7, base_price_per_gpu = $8
            WHERE rental_id = $1
            "#,
        )
        .bind(rental.id.as_uuid())
        .bind(rental.state.to_string())
        .bind(resource_spec_json)
        .bind(chrono::Utc::now())
        .bind(rental.ended_at)
        .bind(metadata_json)
        .bind(if total_cost == rust_decimal::Decimal::ZERO {
            None
        } else {
            Some(total_cost)
        })
        .bind(rental.base_price_per_gpu)
        .execute(&mut **tx)
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "update_rental_tx".to_string(),
            source: Box::new(e),
        })?;

        if result.rows_affected() == 0 {
            return Err(BillingError::RentalNotFound {
                id: rental.id.to_string(),
            });
        }

        Ok(())
    }
}
