use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::persistence::entities::{Rental, RentalStatus, VerificationLog};

/// Simplified persistence implementation for quick testing
pub struct SimplePersistence {
    pool: SqlitePool,
}

impl SimplePersistence {
    /// Get access to the underlying database pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

impl SimplePersistence {
    pub async fn new(
        database_path: &str,
        _validator_hotkey: String,
    ) -> Result<Self, anyhow::Error> {
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{database_path}")).await?;

        let instance = Self { pool };
        instance.run_migrations().await?;

        Ok(instance)
    }

    async fn run_migrations(&self) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS miners (
                id TEXT PRIMARY KEY,
                hotkey TEXT NOT NULL UNIQUE,
                endpoint TEXT NOT NULL,
                verification_score REAL DEFAULT 0.0,
                uptime_percentage REAL DEFAULT 0.0,
                last_seen TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                executor_info TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS miner_executors (
                id TEXT PRIMARY KEY,
                miner_id TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                grpc_address TEXT NOT NULL,
                gpu_count INTEGER NOT NULL,
                gpu_specs TEXT NOT NULL,
                cpu_specs TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'unknown',
                last_health_check TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS verification_requests (
                id TEXT PRIMARY KEY,
                miner_id TEXT NOT NULL,
                verification_type TEXT NOT NULL,
                executor_id TEXT,
                status TEXT DEFAULT 'scheduled',
                scheduled_at TEXT NOT NULL,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS verification_logs (
                id TEXT PRIMARY KEY,
                executor_id TEXT NOT NULL,
                validator_hotkey TEXT NOT NULL,
                verification_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                success INTEGER NOT NULL,
                details TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rentals (
                id TEXT PRIMARY KEY,
                executor_id TEXT NOT NULL,
                customer_public_key TEXT NOT NULL,
                docker_image TEXT NOT NULL,
                env_vars TEXT,
                gpu_requirements TEXT NOT NULL,
                ssh_access_info TEXT NOT NULL,
                max_duration_hours INTEGER NOT NULL,
                cost_per_hour REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                terminated_at TEXT,
                termination_reason TEXT,
                total_cost REAL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn create_verification_log(
        &self,
        log: &VerificationLog,
    ) -> Result<(), anyhow::Error> {
        let query = r#"
            INSERT INTO verification_logs (
                id, executor_id, validator_hotkey, verification_type, timestamp,
                score, success, details, duration_ms, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(log.id.to_string())
            .bind(&log.executor_id)
            .bind(&log.validator_hotkey)
            .bind(&log.verification_type)
            .bind(log.timestamp.to_rfc3339())
            .bind(log.score)
            .bind(if log.success { 1 } else { 0 })
            .bind(&serde_json::to_string(&log.details)?)
            .bind(log.duration_ms)
            .bind(&log.error_message)
            .bind(log.created_at.to_rfc3339())
            .bind(log.updated_at.to_rfc3339())
            .execute(&self.pool)
            .await?;

        tracing::info!(
            verification_id = %log.id,
            executor_id = %log.executor_id,
            success = %log.success,
            score = %log.score,
            "Verification log created"
        );

        Ok(())
    }

    /// Query verification logs with filtering and pagination
    pub async fn query_verification_logs(
        &self,
        executor_id: Option<&str>,
        success_only: Option<bool>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<VerificationLog>, anyhow::Error> {
        let mut query = String::from(
            "SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
             score, success, details, duration_ms, error_message, created_at, updated_at
             FROM verification_logs WHERE 1=1",
        );

        let mut conditions = Vec::new();

        if let Some(exec_id) = executor_id {
            conditions.push(format!("executor_id = '{exec_id}'"));
        }

        if let Some(success) = success_only {
            conditions.push(format!("success = {}", if success { 1 } else { 0 }));
        }

        if !conditions.is_empty() {
            query.push_str(" AND ");
            query.push_str(&conditions.join(" AND "));
        }

        query.push_str(" ORDER BY timestamp DESC LIMIT ? OFFSET ?");

        let rows = sqlx::query(&query)
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(&self.pool)
            .await?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(self.row_to_verification_log(row)?);
        }

        Ok(logs)
    }

    /// Get executor statistics from verification logs
    pub async fn get_executor_stats(
        &self,
        executor_id: &str,
    ) -> Result<Option<ExecutorStats>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT
                COUNT(*) as total_verifications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_verifications,
                AVG(score) as avg_score,
                AVG(duration_ms) as avg_duration_ms,
                MIN(timestamp) as first_verification,
                MAX(timestamp) as last_verification
             FROM verification_logs
             WHERE executor_id = ?",
        )
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let total: i64 = row.get("total_verifications");
            if total == 0 {
                return Ok(None);
            }

            let stats = ExecutorStats {
                executor_id: executor_id.to_string(),
                total_verifications: total as u64,
                successful_verifications: row.get::<i64, _>("successful_verifications") as u64,
                average_score: row.get("avg_score"),
                average_duration_ms: row.get("avg_duration_ms"),
                first_verification: row.get::<Option<String>, _>("first_verification").map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .unwrap()
                        .with_timezone(&Utc)
                }),
                last_verification: row.get::<Option<String>, _>("last_verification").map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .unwrap()
                        .with_timezone(&Utc)
                }),
            };

            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    /// Get available capacity based on successful verifications
    pub async fn get_available_capacity(
        &self,
        min_score: Option<f64>,
        min_success_rate: Option<f64>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<CapacityEntry>, anyhow::Error> {
        let min_score = min_score.unwrap_or(0.0);
        let min_success_rate = min_success_rate.unwrap_or(0.0);

        let rows = sqlx::query(
            "SELECT
                executor_id,
                COUNT(*) as total_verifications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_verifications,
                AVG(score) as avg_score,
                MAX(timestamp) as last_verification,
                MAX(details) as latest_details
             FROM verification_logs
             GROUP BY executor_id
             HAVING avg_score >= ?
                AND (CAST(successful_verifications AS REAL) / CAST(total_verifications AS REAL)) >= ?
             ORDER BY avg_score DESC, last_verification DESC
             LIMIT ? OFFSET ?"
        )
        .bind(min_score)
        .bind(min_success_rate)
        .bind(limit as i64)
        .bind(offset as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::new();
        for row in rows {
            let executor_id: String = row.get("executor_id");
            let total_verifications: i64 = row.get("total_verifications");
            let successful_verifications: i64 = row.get("successful_verifications");
            let avg_score: f64 = row.get("avg_score");
            let last_verification: String = row.get("last_verification");
            let latest_details: String = row.get("latest_details");

            let success_rate = if total_verifications > 0 {
                successful_verifications as f64 / total_verifications as f64
            } else {
                0.0
            };

            let details: Value = serde_json::from_str(&latest_details).unwrap_or(Value::Null);

            entries.push(CapacityEntry {
                executor_id,
                verification_score: avg_score,
                success_rate,
                last_verification: DateTime::parse_from_rfc3339(&last_verification)
                    .unwrap()
                    .with_timezone(&Utc),
                hardware_info: details,
                total_verifications: total_verifications as u64,
            });
        }

        Ok(entries)
    }

    /// Helper function to convert database row to VerificationLog
    fn row_to_verification_log(
        &self,
        row: sqlx::sqlite::SqliteRow,
    ) -> Result<VerificationLog, anyhow::Error> {
        let id_str: String = row.get("id");
        let details_str: String = row.get("details");
        let timestamp_str: String = row.get("timestamp");
        let created_at_str: String = row.get("created_at");
        let updated_at_str: String = row.get("updated_at");

        Ok(VerificationLog {
            id: Uuid::parse_str(&id_str)?,
            executor_id: row.get("executor_id"),
            validator_hotkey: row.get("validator_hotkey"),
            verification_type: row.get("verification_type"),
            timestamp: DateTime::parse_from_rfc3339(&timestamp_str)?.with_timezone(&Utc),
            score: row.get("score"),
            success: row.get::<i64, _>("success") == 1,
            details: serde_json::from_str(&details_str)?,
            duration_ms: row.get("duration_ms"),
            error_message: row.get("error_message"),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.with_timezone(&Utc),
        })
    }

    /// Create a new rental record
    pub async fn create_rental(&self, rental: &Rental) -> Result<(), anyhow::Error> {
        let query = r#"
            INSERT INTO rentals (
                id, executor_id, customer_public_key, docker_image, env_vars,
                gpu_requirements, ssh_access_info, max_duration_hours, cost_per_hour,
                status, created_at, updated_at, started_at, terminated_at,
                termination_reason, total_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let status_str = match rental.status {
            RentalStatus::Pending => "Pending",
            RentalStatus::Active => "Active",
            RentalStatus::Terminated => "Terminated",
            RentalStatus::Failed => "Failed",
        };

        sqlx::query(query)
            .bind(rental.id.to_string())
            .bind(&rental.executor_id)
            .bind(&rental.customer_public_key)
            .bind(&rental.docker_image)
            .bind(
                rental
                    .env_vars
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap()),
            )
            .bind(serde_json::to_string(&rental.gpu_requirements)?)
            .bind(serde_json::to_string(&rental.ssh_access_info)?)
            .bind(rental.max_duration_hours as i64)
            .bind(rental.cost_per_hour)
            .bind(status_str)
            .bind(rental.created_at.to_rfc3339())
            .bind(rental.updated_at.to_rfc3339())
            .bind(rental.started_at.map(|dt| dt.to_rfc3339()))
            .bind(rental.terminated_at.map(|dt| dt.to_rfc3339()))
            .bind(&rental.termination_reason)
            .bind(rental.total_cost)
            .execute(&self.pool)
            .await?;

        tracing::info!(
            rental_id = %rental.id,
            executor_id = %rental.executor_id,
            status = ?rental.status,
            "Rental created"
        );

        Ok(())
    }

    /// Get rental by ID
    pub async fn get_rental(&self, rental_id: &Uuid) -> Result<Option<Rental>, anyhow::Error> {
        let row = sqlx::query("SELECT * FROM rentals WHERE id = ?")
            .bind(rental_id.to_string())
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            Ok(Some(self.row_to_rental(row)?))
        } else {
            Ok(None)
        }
    }

    /// Update rental record
    pub async fn update_rental(&self, rental: &Rental) -> Result<(), anyhow::Error> {
        let status_str = match rental.status {
            RentalStatus::Pending => "Pending",
            RentalStatus::Active => "Active",
            RentalStatus::Terminated => "Terminated",
            RentalStatus::Failed => "Failed",
        };

        let query = r#"
            UPDATE rentals SET
                status = ?, updated_at = ?, started_at = ?,
                terminated_at = ?, termination_reason = ?, total_cost = ?
            WHERE id = ?
        "#;

        sqlx::query(query)
            .bind(status_str)
            .bind(rental.updated_at.to_rfc3339())
            .bind(rental.started_at.map(|dt| dt.to_rfc3339()))
            .bind(rental.terminated_at.map(|dt| dt.to_rfc3339()))
            .bind(&rental.termination_reason)
            .bind(rental.total_cost)
            .bind(rental.id.to_string())
            .execute(&self.pool)
            .await?;

        tracing::info!(
            rental_id = %rental.id,
            status = ?rental.status,
            "Rental updated"
        );

        Ok(())
    }

    /// Helper function to convert database row to Rental
    fn row_to_rental(&self, row: sqlx::sqlite::SqliteRow) -> Result<Rental, anyhow::Error> {
        let id_str: String = row.get("id");
        let env_vars_str: Option<String> = row.get("env_vars");
        let gpu_requirements_str: String = row.get("gpu_requirements");
        let ssh_access_info_str: String = row.get("ssh_access_info");
        let status_str: String = row.get("status");
        let created_at_str: String = row.get("created_at");
        let updated_at_str: String = row.get("updated_at");
        let started_at_str: Option<String> = row.get("started_at");
        let terminated_at_str: Option<String> = row.get("terminated_at");

        let status = match status_str.as_str() {
            "Pending" => RentalStatus::Pending,
            "Active" => RentalStatus::Active,
            "Terminated" => RentalStatus::Terminated,
            "Failed" => RentalStatus::Failed,
            _ => return Err(anyhow::anyhow!("Invalid rental status: {}", status_str)),
        };

        Ok(Rental {
            id: Uuid::parse_str(&id_str)?,
            executor_id: row.get("executor_id"),
            customer_public_key: row.get("customer_public_key"),
            docker_image: row.get("docker_image"),
            env_vars: env_vars_str.map(|s| serde_json::from_str(&s)).transpose()?,
            gpu_requirements: serde_json::from_str(&gpu_requirements_str)?,
            ssh_access_info: serde_json::from_str(&ssh_access_info_str)?,
            max_duration_hours: row.get::<i64, _>("max_duration_hours") as u32,
            cost_per_hour: row.get("cost_per_hour"),
            status,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.with_timezone(&Utc),
            started_at: started_at_str.map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .unwrap()
                    .with_timezone(&Utc)
            }),
            terminated_at: terminated_at_str.map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .unwrap()
                    .with_timezone(&Utc)
            }),
            termination_reason: row.get("termination_reason"),
            total_cost: row.get("total_cost"),
        })
    }

    /// Get registered miners with pagination
    pub async fn get_registered_miners(
        &self,
        offset: u32,
        page_size: u32,
    ) -> Result<Vec<MinerData>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT 
                id, hotkey, endpoint, verification_score, uptime_percentage,
                last_seen, registered_at, executor_info,
                (SELECT COUNT(*) FROM miner_executors WHERE miner_id = miners.id) as executor_count
             FROM miners 
             ORDER BY registered_at DESC
             LIMIT ? OFFSET ?",
        )
        .bind(page_size as i64)
        .bind(offset as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut miners = Vec::new();
        for row in rows {
            let executor_info_str: String = row.get("executor_info");
            let executor_count: i64 = row.get("executor_count");
            let last_seen_str: String = row.get("last_seen");
            let registered_at_str: String = row.get("registered_at");

            miners.push(MinerData {
                miner_id: row.get("id"),
                hotkey: row.get("hotkey"),
                endpoint: row.get("endpoint"),
                executor_count: executor_count as u32,
                verification_score: row.get("verification_score"),
                uptime_percentage: row.get("uptime_percentage"),
                last_seen: DateTime::parse_from_rfc3339(&last_seen_str)?.with_timezone(&Utc),
                registered_at: DateTime::parse_from_rfc3339(&registered_at_str)?
                    .with_timezone(&Utc),
                executor_info: serde_json::from_str(&executor_info_str)
                    .unwrap_or(Value::Object(serde_json::Map::new())),
            });
        }

        Ok(miners)
    }

    /// Register a new miner
    pub async fn register_miner(
        &self,
        miner_id: &str,
        hotkey: &str,
        endpoint: &str,
        executors: &[crate::api::types::ExecutorRegistration],
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        let executor_info = serde_json::to_string(&executors)?;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "INSERT INTO miners (id, hotkey, endpoint, last_seen, registered_at, updated_at, executor_info)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(miner_id)
        .bind(hotkey)
        .bind(endpoint)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .bind(&executor_info)
        .execute(&mut *tx)
        .await?;

        for executor in executors {
            let executor_id = Uuid::new_v4().to_string();
            let gpu_specs_json = serde_json::to_string(&executor.gpu_specs)?;
            let cpu_specs_json = serde_json::to_string(&executor.cpu_specs)?;

            sqlx::query(
                "INSERT INTO miner_executors (id, miner_id, executor_id, grpc_address, gpu_count, gpu_specs, cpu_specs, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&executor_id)
            .bind(miner_id)
            .bind(&executor.executor_id)
            .bind(&executor.grpc_address)
            .bind(executor.gpu_count as i64)
            .bind(&gpu_specs_json)
            .bind(&cpu_specs_json)
            .bind(&now)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Get miner by ID
    pub async fn get_miner_by_id(
        &self,
        miner_id: &str,
    ) -> Result<Option<MinerData>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT 
                id, hotkey, endpoint, verification_score, uptime_percentage,
                last_seen, registered_at, executor_info,
                (SELECT COUNT(*) FROM miner_executors WHERE miner_id = miners.id) as executor_count
             FROM miners 
             WHERE id = ?",
        )
        .bind(miner_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let executor_info_str: String = row.get("executor_info");
            let executor_count: i64 = row.get("executor_count");
            let last_seen_str: String = row.get("last_seen");
            let registered_at_str: String = row.get("registered_at");

            Ok(Some(MinerData {
                miner_id: row.get("id"),
                hotkey: row.get("hotkey"),
                endpoint: row.get("endpoint"),
                executor_count: executor_count as u32,
                verification_score: row.get("verification_score"),
                uptime_percentage: row.get("uptime_percentage"),
                last_seen: DateTime::parse_from_rfc3339(&last_seen_str)?.with_timezone(&Utc),
                registered_at: DateTime::parse_from_rfc3339(&registered_at_str)?
                    .with_timezone(&Utc),
                executor_info: serde_json::from_str(&executor_info_str)
                    .unwrap_or(Value::Object(serde_json::Map::new())),
            }))
        } else {
            Ok(None)
        }
    }

    /// Update miner information
    pub async fn update_miner(
        &self,
        miner_id: &str,
        request: &crate::api::types::UpdateMinerRequest,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();

        if let Some(endpoint) = &request.endpoint {
            let result = sqlx::query("UPDATE miners SET endpoint = ?, updated_at = ? WHERE id = ?")
                .bind(endpoint)
                .bind(&now)
                .bind(miner_id)
                .execute(&self.pool)
                .await?;

            if result.rows_affected() == 0 {
                return Err(anyhow::anyhow!("Miner not found"));
            }
        }

        if let Some(executors) = &request.executors {
            let executor_info = serde_json::to_string(executors)?;
            let result =
                sqlx::query("UPDATE miners SET executor_info = ?, updated_at = ? WHERE id = ?")
                    .bind(&executor_info)
                    .bind(&now)
                    .bind(miner_id)
                    .execute(&self.pool)
                    .await?;

            if result.rows_affected() == 0 {
                return Err(anyhow::anyhow!("Miner not found"));
            }
        }

        Ok(())
    }

    /// Remove miner
    pub async fn remove_miner(&self, miner_id: &str) -> Result<(), anyhow::Error> {
        let result = sqlx::query("DELETE FROM miners WHERE id = ?")
            .bind(miner_id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            Err(anyhow::anyhow!("Miner not found"))
        } else {
            Ok(())
        }
    }

    /// Get miner health status
    pub async fn get_miner_health(
        &self,
        miner_id: &str,
    ) -> Result<Option<MinerHealthData>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT executor_id, status, last_health_check, created_at
             FROM miner_executors 
             WHERE miner_id = ?",
        )
        .bind(miner_id)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let mut executor_health = Vec::new();
        let mut latest_check = Utc::now() - chrono::Duration::hours(24);

        for row in rows {
            let last_health_str: Option<String> = row.get("last_health_check");
            let created_at_str: String = row.get("created_at");

            let last_seen = if let Some(health_str) = last_health_str {
                DateTime::parse_from_rfc3339(&health_str)?.with_timezone(&Utc)
            } else {
                DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc)
            };

            if last_seen > latest_check {
                latest_check = last_seen;
            }

            executor_health.push(ExecutorHealthData {
                executor_id: row.get("executor_id"),
                status: row
                    .get::<Option<String>, _>("status")
                    .unwrap_or_else(|| "unknown".to_string()),
                last_seen,
                gpu_utilization: 75.0, // Mock data
                memory_usage: 60.0,    // Mock data
            });
        }

        Ok(Some(MinerHealthData {
            last_health_check: latest_check,
            executor_health,
        }))
    }

    /// Schedule verification for miner
    pub async fn schedule_verification(
        &self,
        miner_id: &str,
        verification_id: &str,
        verification_type: &str,
        executor_id: Option<&str>,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO verification_requests (id, miner_id, verification_type, executor_id, scheduled_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(verification_id)
        .bind(miner_id)
        .bind(verification_type)
        .bind(executor_id)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get miner executors
    pub async fn get_miner_executors(
        &self,
        miner_id: &str,
    ) -> Result<Vec<ExecutorData>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT executor_id, gpu_specs, cpu_specs, location
             FROM miner_executors 
             WHERE miner_id = ?",
        )
        .bind(miner_id)
        .fetch_all(&self.pool)
        .await?;

        let mut executors = Vec::new();
        for row in rows {
            let gpu_specs_str: String = row.get("gpu_specs");
            let cpu_specs_str: String = row.get("cpu_specs");

            let gpu_specs: Vec<crate::api::types::GpuSpec> = serde_json::from_str(&gpu_specs_str)?;
            let cpu_specs: crate::api::types::CpuSpec = serde_json::from_str(&cpu_specs_str)?;

            executors.push(ExecutorData {
                executor_id: row.get("executor_id"),
                gpu_specs,
                cpu_specs,
                location: row.get("location"),
            });
        }

        Ok(executors)
    }
}

/// Executor statistics derived from verification logs
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub executor_id: String,
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub average_score: Option<f64>,
    pub average_duration_ms: Option<f64>,
    pub first_verification: Option<DateTime<Utc>>,
    pub last_verification: Option<DateTime<Utc>>,
}

impl ExecutorStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }
}

/// Available capacity entry
#[derive(Debug, Clone)]
pub struct CapacityEntry {
    pub executor_id: String,
    pub verification_score: f64,
    pub success_rate: f64,
    pub last_verification: DateTime<Utc>,
    pub hardware_info: Value,
    pub total_verifications: u64,
}

/// Miner data for listings
#[derive(Debug, Clone)]
pub struct MinerData {
    pub miner_id: String,
    pub hotkey: String,
    pub endpoint: String,
    pub executor_count: u32,
    pub verification_score: f64,
    pub uptime_percentage: f64,
    pub last_seen: DateTime<Utc>,
    pub registered_at: DateTime<Utc>,
    pub executor_info: Value,
}

/// Miner health data
#[derive(Debug, Clone)]
pub struct MinerHealthData {
    pub last_health_check: DateTime<Utc>,
    pub executor_health: Vec<ExecutorHealthData>,
}

#[derive(Debug, Clone)]
pub struct ExecutorHealthData {
    pub executor_id: String,
    pub status: String,
    pub last_seen: DateTime<Utc>,
    pub gpu_utilization: f64,
    pub memory_usage: f64,
}

/// Executor details for miner listings
#[derive(Debug, Clone)]
pub struct ExecutorData {
    pub executor_id: String,
    pub gpu_specs: Vec<crate::api::types::GpuSpec>,
    pub cpu_specs: crate::api::types::CpuSpec,
    pub location: Option<String>,
}
