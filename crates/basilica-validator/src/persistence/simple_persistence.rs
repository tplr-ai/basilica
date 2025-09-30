use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{QueryBuilder, Row, SqlitePool};
use tracing::{info, warn};
use uuid::Uuid;

use crate::persistence::entities::{Rental, RentalStatus, VerificationLog};
use crate::persistence::ValidatorPersistence;
use crate::rental::{RentalInfo, RentalState};

/// Extract GPU memory size in GB from GPU name string
fn extract_gpu_memory_gb(gpu_name: &str) -> u32 {
    use regex::Regex;

    let re = Regex::new(r"(\d+)GB").unwrap();
    if let Some(captures) = re.captures(gpu_name) {
        captures[1].parse().unwrap_or(0)
    } else {
        0
    }
}

/// Filter criteria for querying rentals
#[derive(Default)]
pub struct RentalFilter {
    pub rental_id: Option<String>,
    pub validator_hotkey: Option<String>,
    pub exclude_states: Option<Vec<RentalState>>,
    pub order_by_created_desc: bool,
}

/// Simplified persistence implementation for quick testing
#[derive(Debug, Clone)]
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
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    #[cfg(test)]
    pub async fn for_testing() -> Result<Self, anyhow::Error> {
        let pool = SqlitePool::connect(":memory:").await?;

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA busy_timeout = 5000")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;

        let instance = Self { pool };
        instance.run_migrations().await?;

        Ok(instance)
    }

    pub async fn new(
        database_path: &str,
        _validator_hotkey: String,
    ) -> Result<Self, anyhow::Error> {
        // Create database URL with proper connection mode
        let db_url = if database_path.starts_with("sqlite:") {
            database_path.to_string()
        } else {
            format!("sqlite:{database_path}")
        };

        // Add connection mode for read-write-create if not present
        let final_url = if db_url.contains("?") {
            db_url
        } else {
            format!("{db_url}?mode=rwc")
        };

        let pool = sqlx::SqlitePool::connect(&final_url).await?;

        // Configure SQLite for better concurrency
        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA busy_timeout = 5000")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;

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
                validator_hotkey TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                container_id TEXT NOT NULL,
                ssh_session_id TEXT NOT NULL,
                ssh_credentials TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                container_spec TEXT NOT NULL,
                miner_id TEXT NOT NULL DEFAULT '',
                customer_public_key TEXT,
                docker_image TEXT,
                env_vars TEXT,
                gpu_requirements TEXT,
                ssh_access_info TEXT,
                cost_per_hour REAL,
                status TEXT,
                updated_at TEXT,
                started_at TEXT,
                terminated_at TEXT,
                termination_reason TEXT,
                total_cost REAL
            );

            CREATE TABLE IF NOT EXISTS miner_gpu_profiles (
                miner_uid INTEGER PRIMARY KEY,
                gpu_counts_json TEXT NOT NULL,
                total_score REAL NOT NULL,
                verification_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                last_successful_validation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT valid_score CHECK (total_score >= 0.0 AND total_score <= 1.0),
                CONSTRAINT valid_count CHECK (verification_count >= 0)
            );

            CREATE TABLE IF NOT EXISTS emission_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                burn_amount INTEGER NOT NULL,
                burn_percentage REAL NOT NULL,
                category_distributions_json TEXT NOT NULL,
                total_miners INTEGER NOT NULL,
                weight_set_block INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT valid_burn_percentage CHECK (burn_percentage >= 0.0 AND burn_percentage <= 100.0),
                CONSTRAINT valid_total_miners CHECK (total_miners >= 0)
            );

            CREATE TABLE IF NOT EXISTS miner_prover_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                gpu_model TEXT NOT NULL,
                gpu_count INTEGER NOT NULL,
                gpu_memory_gb REAL NOT NULL,
                attestation_valid INTEGER NOT NULL,
                verification_timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT valid_gpu_count CHECK (gpu_count >= 0),
                CONSTRAINT valid_gpu_memory CHECK (gpu_memory_gb >= 0)
            );

            CREATE TABLE IF NOT EXISTS executor_hardware_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                cpu_model TEXT,
                cpu_cores INTEGER,
                ram_gb INTEGER,
                disk_gb INTEGER,
                full_hardware_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS executor_speedtest_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                download_mbps REAL,
                upload_mbps REAL,
                test_timestamp TEXT NOT NULL,
                test_server TEXT,
                full_result_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS executor_network_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                ip_address TEXT,
                hostname TEXT,
                city TEXT,
                region TEXT,
                country TEXT,
                location TEXT,
                organization TEXT,
                postal_code TEXT,
                timezone TEXT,
                test_timestamp TEXT NOT NULL,
                full_result_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS executor_docker_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                service_active BOOLEAN NOT NULL,
                docker_version TEXT,
                images_pulled TEXT,
                dind_supported BOOLEAN DEFAULT 0,
                validation_error TEXT,
                full_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS executor_nat_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                is_accessible BOOLEAN NOT NULL,
                test_port INTEGER NOT NULL,
                test_path TEXT NOT NULL,
                container_id TEXT,
                response_content TEXT,
                test_timestamp TEXT NOT NULL,
                full_json TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS executor_storage_profile (
                miner_uid INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                total_bytes INTEGER NOT NULL,
                available_bytes INTEGER NOT NULL,
                required_bytes INTEGER NOT NULL,
                filesystem_details TEXT NOT NULL,
                collection_timestamp TEXT NOT NULL,
                full_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, executor_id)
            );

            CREATE TABLE IF NOT EXISTS weight_allocation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                miner_uid INTEGER NOT NULL,
                gpu_category TEXT NOT NULL,
                allocated_weight INTEGER NOT NULL,
                miner_score REAL NOT NULL,
                category_total_score REAL NOT NULL,
                weight_set_block INTEGER NOT NULL,
                timestamp TEXT NOT NULL,

                emission_metrics_id INTEGER,
                FOREIGN KEY (emission_metrics_id) REFERENCES emission_metrics(id),

                CONSTRAINT valid_weight CHECK (allocated_weight >= 0),
                CONSTRAINT valid_scores CHECK (miner_score >= 0.0 AND category_total_score >= 0.0)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_gpu_profiles_score ON miner_gpu_profiles(total_score DESC);
            CREATE INDEX IF NOT EXISTS idx_gpu_profiles_updated ON miner_gpu_profiles(last_updated);
            CREATE INDEX IF NOT EXISTS idx_emission_metrics_timestamp ON emission_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_emission_metrics_block ON emission_metrics(weight_set_block);
            CREATE INDEX IF NOT EXISTS idx_prover_results_miner ON miner_prover_results(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_prover_results_timestamp ON miner_prover_results(verification_timestamp);
            CREATE INDEX IF NOT EXISTS idx_weight_history_miner ON weight_allocation_history(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_weight_history_category ON weight_allocation_history(gpu_category);
            CREATE INDEX IF NOT EXISTS idx_weight_history_block ON weight_allocation_history(weight_set_block);
            CREATE INDEX IF NOT EXISTS idx_executor_hardware_miner ON executor_hardware_profile(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_executor_hardware_updated ON executor_hardware_profile(updated_at);
            CREATE INDEX IF NOT EXISTS idx_executor_speedtest_miner ON executor_speedtest_profile(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_executor_speedtest_timestamp ON executor_speedtest_profile(test_timestamp);
            CREATE INDEX IF NOT EXISTS idx_executor_network_miner ON executor_network_profile(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_executor_network_timestamp ON executor_network_profile(test_timestamp);
            CREATE INDEX IF NOT EXISTS idx_executor_network_country ON executor_network_profile(country);
            CREATE INDEX IF NOT EXISTS idx_executor_docker_miner ON executor_docker_profile(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_executor_docker_updated ON executor_docker_profile(updated_at);
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Check if last_successful_validation column exists before adding it
        let column_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_gpu_profiles')
            WHERE name = 'last_successful_validation'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !column_exists {
            // Migration to add last_successful_validation column
            sqlx::query(
                r#"
                ALTER TABLE miner_gpu_profiles
                ADD COLUMN last_successful_validation TEXT;
                "#,
            )
            .execute(&self.pool)
            .await?;

            info!("Added last_successful_validation column to miner_gpu_profiles table");
        }

        // Check if gpu_uuids column exists in miner_prover_results
        let gpu_uuid_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_prover_results')
            WHERE name = 'gpu_uuid'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !gpu_uuid_exists {
            // Migration to add gpu_uuid column to miner_prover_results
            sqlx::query(
                r#"
                ALTER TABLE miner_prover_results
                ADD COLUMN gpu_uuid TEXT;
                "#,
            )
            .execute(&self.pool)
            .await?;

            info!("Added gpu_uuid column to miner_prover_results table");
        }

        // Check if gpu_uuids column exists in miner_executors
        let gpu_uuids_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_executors')
            WHERE name = 'gpu_uuids'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !gpu_uuids_exists {
            // Migration to add gpu_uuids column to miner_executors
            sqlx::query(
                r#"
                ALTER TABLE miner_executors
                ADD COLUMN gpu_uuids TEXT;
                "#,
            )
            .execute(&self.pool)
            .await?;

            info!("Added gpu_uuids column to miner_executors table");
        }

        // Create GPU UUID assignments table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS gpu_uuid_assignments (
                gpu_uuid TEXT PRIMARY KEY,
                gpu_index INTEGER NOT NULL,
                executor_id TEXT NOT NULL,
                miner_id TEXT NOT NULL,
                gpu_name TEXT,
                gpu_memory_gb REAL,
                last_verified TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Check if gpu_memory_gb column exists in gpu_uuid_assignments
        let gpu_memory_gb_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('gpu_uuid_assignments')
            WHERE name = 'gpu_memory_gb'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !gpu_memory_gb_exists {
            // Migration to add gpu_memory_gb column to gpu_uuid_assignments
            sqlx::query(
                r#"
                ALTER TABLE gpu_uuid_assignments
                ADD COLUMN gpu_memory_gb REAL DEFAULT NULL;
                "#,
            )
            .execute(&self.pool)
            .await?;

            info!("Added gpu_memory_gb column to gpu_uuid_assignments table");
        }

        // Create indexes
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_prover_results_gpu_uuid ON miner_prover_results(gpu_uuid);
            CREATE INDEX IF NOT EXISTS idx_executors_gpu_uuids ON miner_executors(gpu_uuids);
            CREATE INDEX IF NOT EXISTS idx_gpu_assignments_executor ON gpu_uuid_assignments(executor_id);
            CREATE INDEX IF NOT EXISTS idx_gpu_assignments_miner ON gpu_uuid_assignments(miner_id);
            CREATE INDEX IF NOT EXISTS idx_gpu_assignments_miner_executor ON gpu_uuid_assignments(miner_id, executor_id);
            CREATE INDEX IF NOT EXISTS idx_miner_executors_status ON miner_executors(status);
            CREATE INDEX IF NOT EXISTS idx_miner_executors_health_check ON miner_executors(last_health_check);
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Check if miner_id column exists in rentals table
        let miner_id_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('rentals')
            WHERE name = 'miner_id'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !miner_id_exists {
            sqlx::query(
                r#"
                ALTER TABLE rentals
                ADD COLUMN miner_id TEXT NOT NULL DEFAULT '';
                "#,
            )
            .execute(&self.pool)
            .await?;

            info!("Added miner_id column to rentals table");
        }

        self.create_collateral_scanned_blocks_table().await?;
        self.add_binary_validation_columns().await?;

        // Migration to remove deprecated columns from miner_executors table
        // Check if gpu_specs column exists
        let gpu_specs_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_executors')
            WHERE name = 'gpu_specs'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if gpu_specs_exists {
            sqlx::query("ALTER TABLE miner_executors DROP COLUMN gpu_specs")
                .execute(&self.pool)
                .await?;
            info!("Dropped gpu_specs column from miner_executors table");
        }

        // Check if cpu_specs column exists
        let cpu_specs_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_executors')
            WHERE name = 'cpu_specs'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if cpu_specs_exists {
            sqlx::query("ALTER TABLE miner_executors DROP COLUMN cpu_specs")
                .execute(&self.pool)
                .await?;
            info!("Dropped cpu_specs column from miner_executors table");
        }

        // Check if location column exists
        let location_exists: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('miner_executors')
            WHERE name = 'location'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if location_exists {
            sqlx::query("ALTER TABLE miner_executors DROP COLUMN location")
                .execute(&self.pool)
                .await?;
            info!("Dropped location column from miner_executors table");
        }

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

    /// Get available executors for rental (not currently rented)
    pub async fn get_available_executors(
        &self,
        min_gpu_memory: Option<u32>,
        gpu_type: Option<String>,
        min_gpu_count: Option<u32>,
        location: Option<basilica_common::LocationProfile>,
    ) -> Result<Vec<AvailableExecutorData>, anyhow::Error> {
        // Build the base query with LEFT JOIN to find executors without active rentals
        // Also join with gpu_uuid_assignments to get actual GPU data
        // And join with hardware profile to get CPU/RAM information
        // And join with network profile to get location information
        // And join with speedtest profile to get network speed information
        let mut query_str = String::from(
            "SELECT
                me.executor_id,
                me.miner_id,
                me.status,
                me.gpu_count,
                m.verification_score,
                m.uptime_percentage,
                GROUP_CONCAT(gua.gpu_name) as gpu_names,
                ehp.cpu_model,
                ehp.cpu_cores,
                ehp.ram_gb,
                enp.city,
                enp.region,
                enp.country,
                esp.download_mbps,
                esp.upload_mbps,
                esp.test_timestamp
            FROM miner_executors me
            JOIN miners m ON me.miner_id = m.id
            LEFT JOIN rentals r ON me.executor_id = r.executor_id
                AND r.miner_id = me.miner_id
                AND r.state IN ('Active', 'Provisioning', 'active', 'provisioning')
            LEFT JOIN gpu_uuid_assignments gua ON me.executor_id = gua.executor_id AND gua.miner_id = me.miner_id
            LEFT JOIN executor_hardware_profile ehp ON me.executor_id = ehp.executor_id AND me.miner_id = 'miner_' || ehp.miner_uid
            LEFT JOIN executor_network_profile enp ON me.executor_id = enp.executor_id AND me.miner_id = 'miner_' || enp.miner_uid
            LEFT JOIN executor_speedtest_profile esp ON me.executor_id = esp.executor_id AND me.miner_id = 'miner_' || esp.miner_uid
            WHERE r.id IS NULL
                AND (me.status IS NULL OR me.status != 'offline')",
        );

        // Add location filters if specified (case-insensitive comparison)
        if let Some(ref loc) = location {
            if let Some(ref country) = loc.country {
                query_str.push_str(&format!(" AND LOWER(enp.country) = LOWER('{}')", country));
            }
            if let Some(ref region) = loc.region {
                query_str.push_str(&format!(" AND LOWER(enp.region) = LOWER('{}')", region));
            }
            if let Some(ref city) = loc.city {
                query_str.push_str(&format!(" AND LOWER(enp.city) = LOWER('{}')", city));
            }
        }

        query_str.push_str(" GROUP BY me.executor_id");

        // Add GPU count filter if specified (use HAVING since we're grouping)
        if let Some(min_count) = min_gpu_count {
            query_str.push_str(&format!(" HAVING COUNT(gua.gpu_uuid) >= {}", min_count));
        }

        let rows = sqlx::query(&query_str).fetch_all(&self.pool).await?;

        let mut executors = Vec::new();
        for row in rows {
            // Get GPU data from gpu_uuid_assignments join
            let gpu_names: Option<String> = row.get("gpu_names");

            // Parse GPU specs from gpu_uuid_assignments data only
            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    // Parse GPU names from GROUP_CONCAT result
                    for gpu_name in names.split(',') {
                        // Extract memory from GPU name
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(), // Default, could be parsed from prover results
                        });
                    }
                }
            }

            // Apply GPU memory filter if specified
            if let Some(min_memory) = min_gpu_memory {
                let meets_memory = gpu_specs.iter().any(|gpu| gpu.memory_gb >= min_memory);
                if !meets_memory && !gpu_specs.is_empty() {
                    continue;
                }
            }

            // Apply GPU type filter if specified
            if let Some(ref gpu_type_filter) = gpu_type {
                let matches_type = gpu_specs.iter().any(|gpu| {
                    gpu.name
                        .to_lowercase()
                        .contains(&gpu_type_filter.to_lowercase())
                });
                if !matches_type && !gpu_specs.is_empty() {
                    continue;
                }
            }

            // Get hardware profile data if available, otherwise use defaults
            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");

            let cpu_specs = crate::api::types::CpuSpec {
                cores: cpu_cores.unwrap_or(0) as u32,
                model: cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: ram_gb.unwrap_or(0) as u32,
            };

            // Get network profile data for location
            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");

            // Always use LocationProfile for consistent formatting
            let location_profile = basilica_common::LocationProfile::new(city, region, country);
            let location = Some(location_profile.to_string());

            // Get speed test data if available
            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp_str: Option<String> = row.get("test_timestamp");

            let speed_test_timestamp = test_timestamp_str.and_then(|ts| {
                chrono::DateTime::parse_from_rfc3339(&ts)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            executors.push(AvailableExecutorData {
                executor_id: row.get("executor_id"),
                miner_id: row.get("miner_id"),
                gpu_specs,
                cpu_specs,
                location,
                verification_score: row.get("verification_score"),
                uptime_percentage: row.get("uptime_percentage"),
                status: row.get("status"),
                download_mbps,
                upload_mbps,
                speed_test_timestamp,
            });
        }

        Ok(executors)
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

    /// Check if an executor has an active rental
    pub async fn has_active_rental(
        &self,
        executor_id: &str,
        miner_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let query = r#"
            SELECT COUNT(*) as count
            FROM rentals
            WHERE executor_id = ?
                AND miner_id = ?
                AND state = 'active'
        "#;

        let row = sqlx::query(query)
            .bind(executor_id)
            .bind(miner_id)
            .fetch_one(&self.pool)
            .await?;

        let count: i64 = row.get("count");
        Ok(count > 0)
    }

    /// Helper function to parse rental state from string
    fn parse_rental_state(state_str: &str, rental_id: &str) -> RentalState {
        match state_str {
            "provisioning" => RentalState::Provisioning,
            "active" => RentalState::Active,
            "stopping" => RentalState::Stopping,
            "stopped" => RentalState::Stopped,
            "failed" => RentalState::Failed,
            unknown => {
                warn!(
                    "Unknown rental state '{}' for rental {}, defaulting to Failed",
                    unknown, rental_id
                );
                RentalState::Failed
            }
        }
    }

    /// Helper function to parse a rental row from the database
    fn parse_rental_row(
        &self,
        row: sqlx::sqlite::SqliteRow,
        executor_details: crate::api::types::ExecutorDetails,
    ) -> Result<RentalInfo, anyhow::Error> {
        let state_str: String = row.get("state");
        let created_at_str: String = row.get("created_at");
        let container_spec_str: String = row.get("container_spec");
        let rental_id: String = row.get("id");
        let executor_id: String = row.get("executor_id");

        // Use existing parse_rental_state for consistency
        let state = Self::parse_rental_state(&state_str, &rental_id);

        Ok(RentalInfo {
            rental_id,
            validator_hotkey: row.get("validator_hotkey"),
            executor_id,
            container_id: row.get("container_id"),
            ssh_session_id: row.get("ssh_session_id"),
            ssh_credentials: row.get("ssh_credentials"),
            state,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc),
            container_spec: serde_json::from_str(&container_spec_str)?,
            miner_id: row.get::<String, _>("miner_id"),
            executor_details,
        })
    }

    /// Query rentals with flexible filtering criteria
    async fn query_rentals(&self, filter: RentalFilter) -> Result<Vec<RentalInfo>, anyhow::Error> {
        let mut builder = QueryBuilder::new("SELECT * FROM rentals");
        let mut has_where = false;

        // Build WHERE clause dynamically
        if let Some(rental_id) = filter.rental_id {
            builder.push(" WHERE id = ");
            builder.push_bind(rental_id);
            has_where = true;
        }

        if let Some(validator_hotkey) = filter.validator_hotkey {
            builder.push(if has_where { " AND " } else { " WHERE " });
            builder.push("validator_hotkey = ");
            builder.push_bind(validator_hotkey);
            has_where = true;
        }

        if let Some(exclude_states) = filter.exclude_states {
            if !exclude_states.is_empty() {
                builder.push(if has_where { " AND " } else { " WHERE " });
                builder.push("state NOT IN (");
                for (i, state) in exclude_states.iter().enumerate() {
                    if i > 0 {
                        builder.push(", ");
                    }
                    // IMPORTANT: Convert to lowercase for database
                    builder.push_bind(match state {
                        RentalState::Provisioning => "provisioning",
                        RentalState::Active => "active",
                        RentalState::Stopping => "stopping",
                        RentalState::Stopped => "stopped",
                        RentalState::Failed => "failed",
                    });
                }
                builder.push(")");
            }
        }

        if filter.order_by_created_desc {
            builder.push(" ORDER BY created_at DESC");
        }

        let query = builder.build();
        let rows = query.fetch_all(&self.pool).await?;

        // Parse all rows and fetch executor details for each
        let mut rentals = Vec::new();
        for row in rows {
            // Extract executor_id and miner_id from the row first
            let executor_id: String = row.get("executor_id");
            let miner_id: String = row.get("miner_id");

            // Fetch executor details or use defaults if not found
            let executor_details = match self.get_executor_details(&executor_id, &miner_id).await {
                Ok(Some(details)) => details,
                _ => {
                    // Default executor details if not found
                    crate::api::types::ExecutorDetails {
                        id: executor_id.clone(),
                        gpu_specs: vec![],
                        cpu_specs: crate::api::types::CpuSpec {
                            cores: 0,
                            model: "Unknown".to_string(),
                            memory_gb: 0,
                        },
                        location: None,
                        network_speed: None,
                    }
                }
            };

            rentals.push(self.parse_rental_row(row, executor_details)?);
        }

        Ok(rentals)
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

    /// Get all registered miners
    pub async fn get_all_registered_miners(&self) -> Result<Vec<MinerData>, anyhow::Error> {
        self.get_registered_miners(0, 10000).await
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
                last_seen: chrono::NaiveDateTime::parse_from_str(
                    &last_seen_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                .or_else(|_| {
                    DateTime::parse_from_rfc3339(&last_seen_str).map(|dt| dt.with_timezone(&Utc))
                })?,
                registered_at: chrono::NaiveDateTime::parse_from_str(
                    &registered_at_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                .or_else(|_| {
                    DateTime::parse_from_rfc3339(&registered_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                })?,
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

        // Validate that grpc_addresses are not already registered
        for executor in executors {
            let existing =
                sqlx::query("SELECT COUNT(*) as count FROM miner_executors WHERE grpc_address = ?")
                    .bind(&executor.grpc_address)
                    .fetch_one(&mut *tx)
                    .await?;

            let count: i64 = existing.get("count");
            if count > 0 {
                return Err(anyhow::anyhow!(
                    "Executor with grpc_address {} is already registered",
                    executor.grpc_address
                ));
            }
        }

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

            sqlx::query(
                "INSERT INTO miner_executors (id, miner_id, executor_id, grpc_address, gpu_count, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&executor_id)
            .bind(miner_id)
            .bind(&executor.executor_id)
            .bind(&executor.grpc_address)
            .bind(executor.gpu_count as i64)
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
                last_seen: chrono::NaiveDateTime::parse_from_str(
                    &last_seen_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                .or_else(|_| {
                    DateTime::parse_from_rfc3339(&last_seen_str).map(|dt| dt.with_timezone(&Utc))
                })?,
                registered_at: chrono::NaiveDateTime::parse_from_str(
                    &registered_at_str,
                    "%Y-%m-%d %H:%M:%S",
                )
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                .or_else(|_| {
                    DateTime::parse_from_rfc3339(&registered_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                })?,
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
            // When updating executors, we need to handle the miner_executors table
            let mut tx = self.pool.begin().await?;

            // First, validate that new grpc_addresses aren't already registered by other miners
            for executor in executors {
                let existing = sqlx::query(
                    "SELECT COUNT(*) as count FROM miner_executors
                     WHERE grpc_address = ? AND miner_id != ?",
                )
                .bind(&executor.grpc_address)
                .bind(miner_id)
                .fetch_one(&mut *tx)
                .await?;

                let count: i64 = existing.get("count");
                if count > 0 {
                    return Err(anyhow::anyhow!(
                        "Executor with grpc_address {} is already registered by another miner",
                        executor.grpc_address
                    ));
                }
            }

            // Delete existing executors for this miner
            sqlx::query("DELETE FROM miner_executors WHERE miner_id = ?")
                .bind(miner_id)
                .execute(&mut *tx)
                .await?;

            // Insert new executors
            for executor in executors {
                let executor_id = Uuid::new_v4().to_string();

                sqlx::query(
                    "INSERT INTO miner_executors (id, miner_id, executor_id, grpc_address, gpu_count, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&executor_id)
                .bind(miner_id)
                .bind(&executor.executor_id)
                .bind(&executor.grpc_address)
                .bind(executor.gpu_count as i64)
                .bind(&now)
                .bind(&now)
                .execute(&mut *tx)
                .await?;
            }

            // Also update the executor_info JSON in the miners table
            let executor_info = serde_json::to_string(executors)?;
            let result =
                sqlx::query("UPDATE miners SET executor_info = ?, updated_at = ? WHERE id = ?")
                    .bind(&executor_info)
                    .bind(&now)
                    .bind(miner_id)
                    .execute(&mut *tx)
                    .await?;

            if result.rows_affected() == 0 {
                tx.rollback().await?;
                return Err(anyhow::anyhow!("Miner not found"));
            }

            tx.commit().await?;
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
            "SELECT
                me.executor_id,
                GROUP_CONCAT(gua.gpu_name) as gpu_names,
                ehp.cpu_model,
                ehp.cpu_cores,
                ehp.ram_gb,
                enp.city,
                enp.region,
                enp.country
             FROM miner_executors me
             LEFT JOIN gpu_uuid_assignments gua ON me.executor_id = gua.executor_id AND gua.miner_id = me.miner_id
             LEFT JOIN executor_hardware_profile ehp ON me.executor_id = ehp.executor_id AND me.miner_id = 'miner_' || ehp.miner_uid
             LEFT JOIN executor_network_profile enp ON me.executor_id = enp.executor_id AND me.miner_id = 'miner_' || enp.miner_uid
             WHERE me.miner_id = ?
             GROUP BY me.executor_id,
                      ehp.cpu_model, ehp.cpu_cores, ehp.ram_gb,
                      enp.city, enp.region, enp.country",
        )
        .bind(miner_id)
        .fetch_all(&self.pool)
        .await?;

        let mut executors = Vec::new();
        for row in rows {
            // Get GPU data from gpu_uuid_assignments join
            let gpu_names: Option<String> = row.get("gpu_names");

            // Parse GPU specs from gpu_uuid_assignments data
            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    // Parse GPU names from GROUP_CONCAT result
                    for gpu_name in names.split(',') {
                        // Extract memory from GPU name
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(),
                        });
                    }
                }
            }

            // Get hardware profile data from executor_hardware_profile table
            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");

            let cpu_specs = crate::api::types::CpuSpec {
                cores: cpu_cores.unwrap_or(0) as u32,
                model: cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: ram_gb.unwrap_or(0) as u32,
            };

            // Get network profile data for location
            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");

            // Always use LocationProfile for consistent formatting
            let location_profile = basilica_common::LocationProfile::new(city, region, country);
            let location = Some(location_profile.to_string());

            executors.push(ExecutorData {
                executor_id: row.get("executor_id"),
                gpu_specs,
                cpu_specs,
                location,
            });
        }

        Ok(executors)
    }

    /// Get miner ID by executor ID
    pub async fn get_miner_id_by_executor(
        &self,
        executor_id: &str,
    ) -> Result<String, anyhow::Error> {
        let miner_id: String = sqlx::query(
            "SELECT miner_id FROM miner_executors \
                 WHERE executor_id = ? \
                 LIMIT 1",
        )
        .bind(executor_id)
        .fetch_one(&self.pool)
        .await?
        .get("miner_id");

        Ok(miner_id)
    }

    /// Get detailed executor information including GPU and CPU specs
    pub async fn get_executor_details(
        &self,
        executor_id: &str,
        miner_id: &str,
    ) -> Result<Option<crate::api::types::ExecutorDetails>, anyhow::Error> {
        // Get executor info with GPU data, hardware profile, network profile, and speed test data
        let row = sqlx::query(
            "SELECT
                me.executor_id,
                GROUP_CONCAT(gua.gpu_name) as gpu_names,
                ehp.cpu_model,
                ehp.cpu_cores,
                ehp.ram_gb,
                enp.city,
                enp.region,
                enp.country,
                esp.download_mbps,
                esp.upload_mbps,
                esp.test_timestamp
             FROM miner_executors me
             LEFT JOIN gpu_uuid_assignments gua ON me.executor_id = gua.executor_id AND gua.miner_id = me.miner_id
             LEFT JOIN executor_hardware_profile ehp ON me.executor_id = ehp.executor_id AND me.miner_id = 'miner_' || ehp.miner_uid
             LEFT JOIN executor_network_profile enp ON me.executor_id = enp.executor_id AND me.miner_id = 'miner_' || enp.miner_uid
             LEFT JOIN executor_speedtest_profile esp ON me.executor_id = esp.executor_id AND me.miner_id = 'miner_' || esp.miner_uid
             WHERE me.executor_id = ? AND me.miner_id = ?
             GROUP BY me.executor_id,
                      ehp.cpu_model, ehp.cpu_cores, ehp.ram_gb,
                      enp.city, enp.region, enp.country,
                      esp.download_mbps, esp.upload_mbps, esp.test_timestamp
             LIMIT 1",
        )
        .bind(executor_id)
        .bind(miner_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let executor_id: String = row.get("executor_id");

            // Get GPU data from gpu_uuid_assignments join
            let gpu_names: Option<String> = row.get("gpu_names");

            // Parse GPU specs from gpu_uuid_assignments data
            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    // Parse GPU names from GROUP_CONCAT result
                    for gpu_name in names.split(',') {
                        // Extract memory from GPU name
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(),
                        });
                    }
                }
            }

            // Get hardware profile data from joined tables
            let hw_cpu_model: Option<String> = row.get("cpu_model");
            let hw_cpu_cores: Option<i32> = row.get("cpu_cores");
            let hw_ram_gb: Option<i32> = row.get("ram_gb");

            // Get network profile data for location
            let net_city: Option<String> = row.get("city");
            let net_region: Option<String> = row.get("region");
            let net_country: Option<String> = row.get("country");

            // Get speed test data
            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp: Option<String> = row.get("test_timestamp");

            // Parse CPU specs from hardware profile data
            let cpu_specs: crate::api::types::CpuSpec = crate::api::types::CpuSpec {
                cores: hw_cpu_cores.unwrap_or(0) as u32,
                model: hw_cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: hw_ram_gb.unwrap_or(0) as u32,
            };

            // Build location string from network profile if available
            let final_location =
                if net_city.is_some() || net_region.is_some() || net_country.is_some() {
                    let loc_profile = basilica_common::LocationProfile {
                        city: net_city,
                        region: net_region,
                        country: net_country,
                    };
                    Some(loc_profile.to_string())
                } else {
                    None
                };

            // Build network speed info if speed test data is available
            let network_speed = if download_mbps.is_some() || upload_mbps.is_some() {
                Some(crate::api::types::NetworkSpeedInfo {
                    download_mbps,
                    upload_mbps,
                    test_timestamp: test_timestamp.and_then(|ts| {
                        DateTime::parse_from_rfc3339(&ts)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                })
            } else {
                None
            };

            Ok(Some(crate::api::types::ExecutorDetails {
                id: executor_id,
                gpu_specs,
                cpu_specs,
                location: final_location,
                network_speed,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get the actual gpu_count for an executor from gpu_uuid_assignments
    pub async fn get_executor_gpu_count_from_assignments(
        &self,
        miner_id: &str,
        executor_id: &str,
    ) -> Result<u32, anyhow::Error> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT gpu_uuid) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND executor_id = ?",
        )
        .bind(miner_id)
        .bind(executor_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(count as u32)
    }

    /// Get the actual gpu_memory_gb for a specific GPU index of an executor from gpu_uuid_assignments
    pub async fn get_executor_gpu_memory_gb_by_index(
        &self,
        miner_id: &str,
        executor_id: &str,
        index: u32,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND executor_id = ? AND gpu_index = ?",
        )
        .bind(miner_id)
        .bind(executor_id)
        .bind(index)
        .fetch_one(&self.pool)
        .await?;

        Ok(memory)
    }

    /// Get the actual gpu_memory_gb for a specific GPU index of an executor from gpu_uuid_assignments
    pub async fn get_executor_gpu_memory_gb_by_gpu_uuid(
        &self,
        miner_id: &str,
        executor_id: &str,
        gpu_uuid: &str,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND executor_id = ? AND gpu_uuid = ?",
        )
        .bind(miner_id)
        .bind(executor_id)
        .bind(gpu_uuid)
        .fetch_one(&self.pool)
        .await?;

        Ok(memory)
    }

    /// Get the actual gpu_memory_gb for the first GPU (index 0) of an executor from gpu_uuid_assignments
    pub async fn get_executor_first_gpu_memory_gb(
        &self,
        miner_id: &str,
        executor_id: &str,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND executor_id = ? AND gpu_index = 0",
        )
        .bind(miner_id)
        .bind(executor_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(memory)
    }

    /// Get the GPU name/model for an executor from gpu_uuid_assignments
    pub async fn get_executor_gpu_name_from_assignments(
        &self,
        miner_id: &str,
        executor_id: &str,
    ) -> Result<Option<String>, anyhow::Error> {
        let gpu_name: Option<String> = sqlx::query_scalar(
            "SELECT gpu_name FROM gpu_uuid_assignments
             WHERE miner_id = ? AND executor_id = ?
             LIMIT 1",
        )
        .bind(miner_id)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(gpu_name)
    }

    /// Get the actual gpu_count for all ONLINE executors of a miner from gpu_uuid_assignments
    pub async fn get_miner_gpu_uuid_assignments(
        &self,
        miner_id: &str,
    ) -> Result<Vec<(String, u32, String, f64)>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT
                ga.executor_id,
                COUNT(DISTINCT ga.gpu_uuid) as gpu_count,
                ga.gpu_name,
                MAX(ga.gpu_memory_gb) as gpu_memory_gb
             FROM gpu_uuid_assignments ga
             JOIN miner_executors me ON ga.executor_id = me.executor_id AND ga.miner_id = me.miner_id
             WHERE ga.miner_id = ?
                AND me.status IN ('online', 'verified')
             GROUP BY ga.executor_id, ga.gpu_name
             HAVING COUNT(DISTINCT ga.gpu_uuid) > 0",
        )
        .bind(miner_id)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            let executor_id: String = row.get("executor_id");
            let gpu_count: i64 = row.get("gpu_count");
            let gpu_name: String = row.get("gpu_name");
            let gpu_memory_gb: f64 = row.get("gpu_memory_gb");

            results.push((executor_id, gpu_count as u32, gpu_name, gpu_memory_gb));
        }

        Ok(results)
    }

    /// Get total GPU count for a miner from gpu_uuid_assignments
    pub async fn get_miner_total_gpu_count_from_assignments(
        &self,
        miner_id: &str,
    ) -> Result<u32, anyhow::Error> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT ga.gpu_uuid)
             FROM gpu_uuid_assignments ga
             INNER JOIN miner_executors me ON ga.executor_id = me.executor_id AND ga.miner_id = me.miner_id
             WHERE ga.miner_id = ?
                AND me.status IN ('online', 'verified')",
        )
        .bind(miner_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(count as u32)
    }

    /// Add binary validation tracking columns to verification_logs table
    async fn add_binary_validation_columns(&self) -> Result<(), anyhow::Error> {
        let has_last_col: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('verification_logs')
            WHERE name = 'last_binary_validation'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !has_last_col {
            sqlx::query(
                r#"
                ALTER TABLE verification_logs
                ADD COLUMN last_binary_validation TEXT;
                "#,
            )
            .execute(&self.pool)
            .await?;
        }

        let has_score_col: bool = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) > 0
            FROM pragma_table_info('verification_logs')
            WHERE name = 'last_binary_validation_score'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !has_score_col {
            sqlx::query(
                r#"
                ALTER TABLE verification_logs
                ADD COLUMN last_binary_validation_score REAL;
                "#,
            )
            .execute(&self.pool)
            .await?;
        }

        if !has_last_col || !has_score_col {
            info!("Ensured binary validation tracking columns exist on verification_logs");
        }

        Ok(())
    }

    /// Get last successful full validation data for lightweight validation
    pub async fn get_last_full_validation_data(
        &self,
        executor_id: &str,
        miner_id: &str,
    ) -> Result<
        Option<(
            f64,
            Option<super::super::miner_prover::types::ExecutorResult>,
            u64,
            bool,
        )>,
        anyhow::Error,
    > {
        // duality due to schema migration issues
        let composite_executor_id = if miner_id.starts_with("miner_") {
            format!(
                "{}__{}",
                miner_id.replacen("miner_", "miner", 1),
                executor_id
            )
        } else {
            format!("miner{}__{}", miner_id, executor_id)
        };

        let query = r#"
            SELECT score, details
            FROM verification_logs
            WHERE (executor_id = ? OR executor_id GLOB ('*__' || ?) OR executor_id = ? )
              AND success = 1
              AND verification_type = 'ssh_automation'
              AND (
                json_extract(details, '$.binary_validation_successful') = 1
                OR json_extract(details, '$.binary_validation_successful') = 'true'
              )
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(executor_id)
            .bind(executor_id)
            .bind(&composite_executor_id)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let score: f64 = row.get("score");
            let details_str: String = row.get("details");

            let details: serde_json::Value = serde_json::from_str(&details_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse details JSON: {}", e))?;

            let executor_result = details.get("executor_result").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    serde_json::from_value::<super::super::miner_prover::types::ExecutorResult>(
                        v.clone(),
                    )
                    .ok()
                }
            });

            let gpu_count = details
                .get("gpu_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            let binary_validation_successful = details
                .get("binary_validation_successful")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            Ok(Some((
                score,
                executor_result,
                gpu_count,
                binary_validation_successful,
            )))
        } else {
            Ok(None)
        }
    }

    /// Get verification count for a miner from recent successful verifications
    pub async fn get_miner_verification_count(
        &self,
        miner_id: &str,
        hours: i64,
    ) -> Result<u32, anyhow::Error> {
        let count_query = r#"
            SELECT COUNT(*) as count
            FROM verification_logs vl
            INNER JOIN miner_executors me ON vl.executor_id = me.executor_id
            WHERE me.miner_id = ?
            AND vl.success = 1
            AND vl.timestamp > datetime('now', ? || ' hours')
        "#;

        let count: i64 = sqlx::query_scalar(count_query)
            .bind(miner_id)
            .bind(format!("-{}", hours))
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);

        Ok(count as u32)
    }

    /// Get known executors from database for a miner
    pub async fn get_known_executors_for_miner(
        &self,
        miner_uid: u16,
    ) -> Result<Vec<(String, String, i32, String)>, anyhow::Error> {
        let miner_id = format!("miner_{}", miner_uid);

        let query = r#"
            SELECT executor_id, grpc_address, gpu_count, status
            FROM miner_executors
            WHERE miner_id = ?
            AND status IN ('online', 'verified')
            AND (last_health_check IS NULL OR last_health_check > datetime('now', '-1 hour'))
        "#;

        let rows = sqlx::query(query)
            .bind(&miner_id)
            .fetch_all(&self.pool)
            .await?;

        let mut known_executors = Vec::new();
        for row in rows {
            let executor_id: String = row.get("executor_id");
            let grpc_address: String = row.get("grpc_address");
            let gpu_count: i32 = row.get("gpu_count");
            let status: String = row.get("status");
            known_executors.push((executor_id, grpc_address, gpu_count, status));
        }

        Ok(known_executors)
    }

    #[allow(clippy::too_many_arguments)]
    /// Store executor hardware profile information
    #[allow(clippy::too_many_arguments)]
    pub async fn store_executor_hardware_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        cpu_model: Option<String>,
        cpu_cores: Option<i32>,
        ram_gb: Option<i32>,
        disk_gb: Option<i32>,
        full_hardware_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO executor_hardware_profile
            (miner_uid, executor_id, cpu_model, cpu_cores, ram_gb, disk_gb, full_hardware_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                cpu_model = excluded.cpu_model,
                cpu_cores = excluded.cpu_cores,
                ram_gb = excluded.ram_gb,
                disk_gb = excluded.disk_gb,
                full_hardware_json = excluded.full_hardware_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(cpu_model)
        .bind(cpu_cores)
        .bind(ram_gb)
        .bind(disk_gb)
        .bind(full_hardware_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            executor_id = executor_id,
            "Stored hardware profile for executor"
        );

        Ok(())
    }

    /// Retrieve executor hardware profile from database
    pub async fn get_executor_hardware_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<
        Option<(
            String,
            Option<String>,
            Option<i32>,
            Option<i32>,
            Option<i32>,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT cpu_model, cpu_cores, ram_gb, disk_gb, full_hardware_json
            FROM executor_hardware_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_hardware_json: String = row.get("full_hardware_json");
            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");
            let disk_gb: Option<i32> = row.get("disk_gb");

            Ok(Some((
                full_hardware_json,
                cpu_model,
                cpu_cores,
                ram_gb,
                disk_gb,
            )))
        } else {
            Ok(None)
        }
    }

    /// Store executor network speedtest profile information
    #[allow(clippy::too_many_arguments)]
    pub async fn store_executor_speedtest_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        download_mbps: Option<f64>,
        upload_mbps: Option<f64>,
        test_timestamp: &str,
        test_server: Option<String>,
        full_result_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO executor_speedtest_profile
            (miner_uid, executor_id, download_mbps, upload_mbps, test_timestamp, test_server, full_result_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                download_mbps = excluded.download_mbps,
                upload_mbps = excluded.upload_mbps,
                test_timestamp = excluded.test_timestamp,
                test_server = excluded.test_server,
                full_result_json = excluded.full_result_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(download_mbps)
        .bind(upload_mbps)
        .bind(test_timestamp)
        .bind(test_server)
        .bind(full_result_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            executor_id = executor_id,
            download_mbps = download_mbps.unwrap_or(0.0),
            upload_mbps = upload_mbps.unwrap_or(0.0),
            "Stored speedtest profile for executor"
        );

        Ok(())
    }

    /// Retrieve executor speedtest profile from database
    pub async fn get_executor_speedtest_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<Option<(String, Option<f64>, Option<f64>, String, Option<String>)>, anyhow::Error>
    {
        let row = sqlx::query(
            r#"
            SELECT download_mbps, upload_mbps, test_timestamp, test_server, full_result_json
            FROM executor_speedtest_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_result_json: String = row.get("full_result_json");
            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp: String = row.get("test_timestamp");
            let test_server: Option<String> = row.get("test_server");

            Ok(Some((
                full_result_json,
                download_mbps,
                upload_mbps,
                test_timestamp,
                test_server,
            )))
        } else {
            Ok(None)
        }
    }

    /// Store executor network geolocation profile information
    #[allow(clippy::too_many_arguments)]
    pub async fn store_executor_network_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        ip_address: Option<String>,
        hostname: Option<String>,
        city: Option<String>,
        region: Option<String>,
        country: Option<String>,
        location: Option<String>,
        organization: Option<String>,
        postal_code: Option<String>,
        timezone: Option<String>,
        test_timestamp: &str,
        full_result_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO executor_network_profile
            (miner_uid, executor_id, ip_address, hostname, city, region, country, location,
             organization, postal_code, timezone, test_timestamp, full_result_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                ip_address = excluded.ip_address,
                hostname = excluded.hostname,
                city = excluded.city,
                region = excluded.region,
                country = excluded.country,
                location = excluded.location,
                organization = excluded.organization,
                postal_code = excluded.postal_code,
                timezone = excluded.timezone,
                test_timestamp = excluded.test_timestamp,
                full_result_json = excluded.full_result_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(ip_address.clone())
        .bind(hostname)
        .bind(city.clone())
        .bind(region.clone())
        .bind(country.clone())
        .bind(location.clone())
        .bind(organization.clone())
        .bind(postal_code)
        .bind(timezone)
        .bind(test_timestamp)
        .bind(full_result_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            executor_id = executor_id,
            ip = ip_address.unwrap_or_else(|| "Unknown".to_string()),
            country = country.unwrap_or_else(|| "Unknown".to_string()),
            city = city.unwrap_or_else(|| "Unknown".to_string()),
            region = region.unwrap_or_else(|| "Unknown".to_string()),
            organization = organization.unwrap_or_else(|| "Unknown".to_string()),
            location = location.unwrap_or_else(|| "Unknown".to_string()),
            "Stored network profile for executor"
        );

        Ok(())
    }

    /// Retrieve executor network profile from database
    #[allow(clippy::type_complexity)]
    pub async fn get_executor_network_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<
        Option<(
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            String,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT ip_address, hostname, city, region, country, location, organization,
                   postal_code, timezone, test_timestamp, full_result_json
            FROM executor_network_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_result_json: String = row.get("full_result_json");
            let ip_address: Option<String> = row.get("ip_address");
            let hostname: Option<String> = row.get("hostname");
            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");
            let location: Option<String> = row.get("location");
            let organization: Option<String> = row.get("organization");
            let postal_code: Option<String> = row.get("postal_code");
            let timezone: Option<String> = row.get("timezone");
            let test_timestamp: String = row.get("test_timestamp");

            Ok(Some((
                full_result_json,
                ip_address,
                hostname,
                city,
                region,
                country,
                location,
                organization,
                postal_code,
                timezone,
                test_timestamp,
            )))
        } else {
            Ok(None)
        }
    }

    /// Store executor Docker validation profile information
    #[allow(clippy::too_many_arguments)]
    pub async fn store_executor_docker_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        service_active: bool,
        docker_version: Option<String>,
        images_pulled: Vec<String>,
        dind_supported: bool,
        validation_error: Option<String>,
        full_json: &str,
    ) -> Result<(), anyhow::Error> {
        let images_json = serde_json::to_string(&images_pulled)?;

        sqlx::query(
            r#"
            INSERT INTO executor_docker_profile
            (miner_uid, executor_id, service_active, docker_version, images_pulled,
             dind_supported, validation_error, full_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                service_active = excluded.service_active,
                docker_version = excluded.docker_version,
                images_pulled = excluded.images_pulled,
                dind_supported = excluded.dind_supported,
                validation_error = excluded.validation_error,
                full_json = excluded.full_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(service_active)
        .bind(docker_version)
        .bind(&images_json)
        .bind(dind_supported)
        .bind(validation_error)
        .bind(full_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get executor Docker validation profile
    pub async fn get_executor_docker_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<
        Option<(
            String,
            bool,
            Option<String>,
            Vec<String>,
            bool,
            Option<String>,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT service_active, docker_version, images_pulled, dind_supported, validation_error, full_json
            FROM executor_docker_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_json: String = row.get("full_json");
            let service_active: bool = row.get("service_active");
            let docker_version: Option<String> = row.get("docker_version");
            let images_pulled_json: String = row.get("images_pulled");
            let images_pulled: Vec<String> = serde_json::from_str(&images_pulled_json)?;
            let dind_supported: bool = row.get("dind_supported");
            let validation_error: Option<String> = row.get("validation_error");

            Ok(Some((
                full_json,
                service_active,
                docker_version,
                images_pulled,
                dind_supported,
                validation_error,
            )))
        } else {
            Ok(None)
        }
    }

    /// Store executor NAT validation profile information
    pub async fn store_executor_nat_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        profile: &crate::miner_prover::validation_nat::NatProfile,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO executor_nat_profile
            (miner_uid, executor_id, is_accessible, test_port, test_path, container_id,
             response_content, test_timestamp, full_json, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                is_accessible = excluded.is_accessible,
                test_port = excluded.test_port,
                test_path = excluded.test_path,
                container_id = excluded.container_id,
                response_content = excluded.response_content,
                test_timestamp = excluded.test_timestamp,
                full_json = excluded.full_json,
                error_message = excluded.error_message,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(profile.is_accessible)
        .bind(profile.test_port as i32)
        .bind(&profile.test_path)
        .bind(&profile.container_id)
        .bind(&profile.response_content)
        .bind(profile.test_timestamp.to_rfc3339())
        .bind(&profile.full_json)
        .bind(&profile.error_message)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get executor NAT validation profile
    pub async fn get_executor_nat_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<Option<crate::miner_prover::validation_nat::NatProfile>, anyhow::Error> {
        let row = sqlx::query(
            r#"
            SELECT is_accessible, test_port, test_path, container_id, response_content,
                   test_timestamp, full_json, error_message
            FROM executor_nat_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let is_accessible: bool = row.get("is_accessible");
            let test_port: i32 = row.get("test_port");
            let test_path: String = row.get("test_path");
            let container_id: Option<String> = row.get("container_id");
            let response_content: Option<String> = row.get("response_content");
            let test_timestamp_str: String = row.get("test_timestamp");
            let full_json: String = row.get("full_json");
            let error_message: Option<String> = row.get("error_message");

            let test_timestamp = chrono::DateTime::parse_from_rfc3339(&test_timestamp_str)?
                .with_timezone(&chrono::Utc);

            Ok(Some(crate::miner_prover::validation_nat::NatProfile {
                is_accessible,
                test_port: test_port as u16,
                test_path,
                container_id,
                response_content,
                test_timestamp,
                full_json,
                error_message,
            }))
        } else {
            Ok(None)
        }
    }

    /// Store executor storage validation profile information
    pub async fn store_executor_storage_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
        profile: &crate::miner_prover::validation_storage::StorageProfile,
    ) -> Result<(), anyhow::Error> {
        let filesystem_details_json = serde_json::to_string(&profile.filesystem_details)?;

        sqlx::query(
            r#"
            INSERT INTO executor_storage_profile
            (miner_uid, executor_id, total_bytes, available_bytes,
             required_bytes, filesystem_details, collection_timestamp, full_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, executor_id) DO UPDATE SET
                total_bytes = excluded.total_bytes,
                available_bytes = excluded.available_bytes,
                required_bytes = excluded.required_bytes,
                filesystem_details = excluded.filesystem_details,
                collection_timestamp = excluded.collection_timestamp,
                full_json = excluded.full_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .bind(profile.total_bytes as i64)
        .bind(profile.available_bytes as i64)
        .bind(profile.required_bytes as i64)
        .bind(&filesystem_details_json)
        .bind(profile.collection_timestamp.to_rfc3339())
        .bind(&profile.full_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get executor storage validation profile
    pub async fn get_executor_storage_profile(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<Option<crate::miner_prover::validation_storage::StorageProfile>, anyhow::Error>
    {
        let row = sqlx::query(
            r#"
            SELECT total_bytes, available_bytes, required_bytes,
                   filesystem_details, collection_timestamp, full_json
            FROM executor_storage_profile
            WHERE miner_uid = ? AND executor_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let total_bytes: i64 = row.get("total_bytes");
            let available_bytes: i64 = row.get("available_bytes");
            let required_bytes: i64 = row.get("required_bytes");
            let filesystem_details_json: String = row.get("filesystem_details");
            let collection_timestamp_str: String = row.get("collection_timestamp");
            let full_json: String = row.get("full_json");

            let filesystem_details = serde_json::from_str(&filesystem_details_json)?;
            let collection_timestamp =
                chrono::DateTime::parse_from_rfc3339(&collection_timestamp_str)?
                    .with_timezone(&chrono::Utc);

            Ok(Some(
                crate::miner_prover::validation_storage::StorageProfile {
                    total_bytes: total_bytes as u64,
                    available_bytes: available_bytes as u64,
                    required_bytes: required_bytes as u64,
                    filesystem_details,
                    collection_timestamp,
                    full_json,
                },
            ))
        } else {
            Ok(None)
        }
    }

    /// Get all executors with their GPU and rental data for metrics initialization
    /// This eliminates N+1 queries by fetching everything in a single query with joins
    pub async fn get_all_executors_for_metrics(
        &self,
    ) -> Result<Vec<ExecutorMetricData>, anyhow::Error> {
        let query = r#"
            SELECT
                me.executor_id,
                me.miner_id,
                gua.gpu_name,
                CASE WHEN r.id IS NOT NULL THEN 1 ELSE 0 END as has_active_rental
            FROM miner_executors me
            INNER JOIN gpu_uuid_assignments gua
                ON me.executor_id = gua.executor_id
                AND me.miner_id = gua.miner_id
            LEFT JOIN rentals r
                ON me.executor_id = r.executor_id
                AND me.miner_id = r.miner_id
                AND r.state = 'active'
            WHERE gua.gpu_name IS NOT NULL
            GROUP BY me.executor_id, me.miner_id
        "#;

        let rows = sqlx::query(query).fetch_all(&self.pool).await?;

        let mut executor_metrics = Vec::new();
        for row in rows {
            let executor_id: String = row.get("executor_id");
            let miner_id: String = row.get("miner_id");
            let gpu_name: Option<String> = row.get("gpu_name");
            let has_active_rental: i32 = row.get("has_active_rental");

            // Extract miner UID from miner_id string (format: "miner_{uid}")
            let miner_uid = if miner_id.starts_with("miner_") {
                miner_id
                    .strip_prefix("miner_")
                    .and_then(|uid_str| uid_str.parse::<u16>().ok())
                    .unwrap_or(0)
            } else {
                0
            };

            executor_metrics.push(ExecutorMetricData {
                executor_id,
                miner_id,
                miner_uid,
                gpu_name,
                has_active_rental: has_active_rental != 0,
            });
        }

        Ok(executor_metrics)
    }
}

#[async_trait::async_trait]
impl ValidatorPersistence for SimplePersistence {
    async fn save_rental(&self, rental: &RentalInfo) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO rentals (
                id, validator_hotkey, executor_id, container_id, ssh_session_id,
                ssh_credentials, state, created_at, container_spec, miner_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                container_id = excluded.container_id,
                ssh_session_id = excluded.ssh_session_id,
                ssh_credentials = excluded.ssh_credentials,
                miner_id = excluded.miner_id",
        )
        .bind(&rental.rental_id)
        .bind(&rental.validator_hotkey)
        .bind(&rental.executor_id)
        .bind(&rental.container_id)
        .bind(&rental.ssh_session_id)
        .bind(&rental.ssh_credentials)
        .bind(match &rental.state {
            RentalState::Provisioning => "provisioning",
            RentalState::Active => "active",
            RentalState::Stopping => "stopping",
            RentalState::Stopped => "stopped",
            RentalState::Failed => "failed",
        })
        .bind(rental.created_at.to_rfc3339())
        .bind(serde_json::to_string(&rental.container_spec)?)
        .bind(&rental.miner_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_rental(&self, rental_id: &str) -> anyhow::Result<Option<RentalInfo>> {
        let filter = RentalFilter {
            rental_id: Some(rental_id.to_string()),
            ..Default::default()
        };
        self.query_rentals(filter)
            .await
            .map(|mut rentals| rentals.pop())
    }

    async fn list_validator_rentals(
        &self,
        validator_hotkey: &str,
    ) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            validator_hotkey: Some(validator_hotkey.to_string()),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn query_non_terminated_rentals(&self) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            exclude_states: Some(vec![RentalState::Stopped, RentalState::Failed]),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn delete_rental(&self, rental_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM rentals WHERE id = ?")
            .bind(rental_id)
            .execute(&self.pool)
            .await?;

        Ok(())
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
}

/// Executor details for miner listings
#[derive(Debug, Clone)]
pub struct ExecutorData {
    pub executor_id: String,
    pub gpu_specs: Vec<crate::api::types::GpuSpec>,
    pub cpu_specs: crate::api::types::CpuSpec,
    pub location: Option<String>,
}

/// Available executor data for rental listings
#[derive(Debug, Clone)]
pub struct AvailableExecutorData {
    pub executor_id: String,
    pub miner_id: String,
    pub gpu_specs: Vec<crate::api::types::GpuSpec>,
    pub cpu_specs: crate::api::types::CpuSpec,
    pub location: Option<String>,
    pub verification_score: f64,
    pub uptime_percentage: f64,
    pub status: Option<String>,
    pub download_mbps: Option<f64>,
    pub upload_mbps: Option<f64>,
    pub speed_test_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Executor metric data for initializing metrics
#[derive(Debug, Clone)]
pub struct ExecutorMetricData {
    pub executor_id: String,
    pub miner_id: String,
    pub miner_uid: u16,
    pub gpu_name: Option<String>,
    pub has_active_rental: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{CpuSpec, ExecutorRegistration, GpuSpec, UpdateMinerRequest};

    #[tokio::test]
    async fn test_prevent_duplicate_grpc_address_registration() {
        let db_path = ":memory:";
        let persistence = SimplePersistence::new(db_path, "test_validator".to_string())
            .await
            .expect("Failed to create persistence");

        // First miner registration
        let executors1 = vec![ExecutorRegistration {
            executor_id: "exec1".to_string(),
            grpc_address: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![GpuSpec {
                name: "RTX 4090".to_string(),
                memory_gb: 24,
                compute_capability: "8.9".to_string(),
            }],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        // Register first miner successfully
        let result = persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &executors1)
            .await;
        assert!(result.is_ok());

        // Second miner trying to register with same grpc_address
        let executors2 = vec![ExecutorRegistration {
            executor_id: "exec2".to_string(),
            grpc_address: "http://192.168.1.1:8080".to_string(), // Same address!
            gpu_count: 1,
            gpu_specs: vec![GpuSpec {
                name: "RTX 3090".to_string(),
                memory_gb: 24,
                compute_capability: "8.6".to_string(),
            }],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 16,
            },
        }];

        // Should fail due to duplicate grpc_address
        let result = persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &executors2)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered"));
    }

    #[tokio::test]
    async fn test_prevent_duplicate_grpc_address_update() {
        let db_path = ":memory:";
        let persistence = SimplePersistence::new(db_path, "test_validator".to_string())
            .await
            .expect("Failed to create persistence");

        // Register first miner
        let executors1 = vec![ExecutorRegistration {
            executor_id: "exec1".to_string(),
            grpc_address: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &executors1)
            .await
            .expect("Failed to register miner1");

        // Register second miner with different address
        let executors2 = vec![ExecutorRegistration {
            executor_id: "exec2".to_string(),
            grpc_address: "http://192.168.1.2:8080".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 16,
            },
        }];

        persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &executors2)
            .await
            .expect("Failed to register miner2");

        // Try to update miner2 with miner1's grpc_address
        let update_request = UpdateMinerRequest {
            endpoint: None,
            executors: Some(vec![ExecutorRegistration {
                executor_id: "exec2_updated".to_string(),
                grpc_address: "http://192.168.1.1:8080".to_string(), // Trying to use miner1's address
                gpu_count: 1,
                gpu_specs: vec![],
                cpu_specs: CpuSpec {
                    cores: 8,
                    model: "Intel i7".to_string(),
                    memory_gb: 16,
                },
            }]),
            signature: "test_signature".to_string(),
        };

        let result = persistence.update_miner("miner2", &update_request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered by another miner"));
    }

    #[tokio::test]
    async fn test_allow_same_miner_update_with_same_grpc_address() {
        let db_path = ":memory:";
        let persistence = SimplePersistence::new(db_path, "test_validator".to_string())
            .await
            .expect("Failed to create persistence");

        // Register miner
        let executors = vec![ExecutorRegistration {
            executor_id: "exec1".to_string(),
            grpc_address: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &executors)
            .await
            .expect("Failed to register miner");

        // Update same miner with same grpc_address (should succeed)
        let update_request = UpdateMinerRequest {
            endpoint: Some("http://miner1-updated.com".to_string()),
            executors: Some(vec![ExecutorRegistration {
                executor_id: "exec1_updated".to_string(),
                grpc_address: "http://192.168.1.1:8080".to_string(), // Same address is OK for same miner
                gpu_count: 3,                                        // Updated GPU count
                gpu_specs: vec![],
                cpu_specs: CpuSpec {
                    cores: 16,
                    model: "Intel i9".to_string(),
                    memory_gb: 64, // Updated memory
                },
            }]),
            signature: "test_signature".to_string(),
        };

        let result = persistence.update_miner("miner1", &update_request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_uuid_duplicate_prevention() {
        let db_path = ":memory:";
        let persistence = SimplePersistence::new(db_path, "test_validator".to_string())
            .await
            .unwrap();

        // Register initial miner with executor
        let executor1 = ExecutorRegistration {
            executor_id: "exec1".to_string(),
            grpc_address: "http://192.168.1.100:50051".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &[executor1])
            .await
            .unwrap();

        // Manually insert GPU UUID for testing
        let gpu_uuid = "GPU-550e8400-e29b-41d4-a716-446655440000";
        sqlx::query(
            "UPDATE miner_executors SET gpu_uuids = ? WHERE miner_id = ? AND executor_id = ?",
        )
        .bind(gpu_uuid)
        .bind("miner1")
        .bind("exec1")
        .execute(&persistence.pool)
        .await
        .unwrap();

        // Register another miner with different executor
        let executor2 = ExecutorRegistration {
            executor_id: "exec2".to_string(),
            grpc_address: "http://192.168.1.101:50051".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &[executor2])
            .await
            .unwrap();

        // Verify both executors exist
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM miner_executors")
            .fetch_one(&persistence.pool)
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify only one has the GPU UUID
        let gpu_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM miner_executors WHERE gpu_uuids = ?")
                .bind(gpu_uuid)
                .fetch_one(&persistence.pool)
                .await
                .unwrap();
        assert_eq!(gpu_count, 1);
    }

    #[tokio::test]
    async fn test_hardware_profile_enrichment() {
        let db_path = ":memory:";
        let persistence = SimplePersistence::new(db_path, "test_validator".to_string())
            .await
            .expect("Failed to create persistence");

        // Register a miner with an executor
        let executor = ExecutorRegistration {
            executor_id: "exec1".to_string(),
            grpc_address: "http://192.168.1.100:50051".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner_1", "hotkey1", "http://miner1.com", &[executor])
            .await
            .unwrap();

        // Store hardware profile for the executor
        persistence
            .store_executor_hardware_profile(
                1, // miner_uid
                "exec1",
                Some("AMD EPYC 7763".to_string()),
                Some(64),
                Some(256),
                Some(1000),
                r#"{"cpu": "AMD EPYC 7763", "cores": 64, "ram": 256}"#,
            )
            .await
            .unwrap();

        // Store network profile for the executor
        persistence
            .store_executor_network_profile(
                1, // miner_uid
                "exec1",
                Some("192.168.1.100".to_string()),
                Some("exec1.example.com".to_string()),
                Some("San Francisco".to_string()),
                Some("California".to_string()),
                Some("US".to_string()),
                Some("37.7749,-122.4194".to_string()),
                Some("AS12345 Example ISP".to_string()),
                Some("94102".to_string()),
                Some("America/Los_Angeles".to_string()),
                &chrono::Utc::now().to_rfc3339(),
                r#"{"city": "San Francisco", "region": "California", "country": "US"}"#,
            )
            .await
            .unwrap();

        // Get miner executors and verify hardware profile is used
        let executors = persistence.get_miner_executors("miner_1").await.unwrap();
        assert_eq!(executors.len(), 1);

        let executor = &executors[0];
        assert_eq!(executor.executor_id, "exec1");
        assert_eq!(executor.cpu_specs.model, "AMD EPYC 7763");
        assert_eq!(executor.cpu_specs.cores, 64);
        assert_eq!(executor.cpu_specs.memory_gb, 256);
        assert_eq!(
            executor.location,
            Some("San Francisco/California/US".to_string())
        );

        // Test get_available_executors with hardware profile
        let available = persistence
            .get_available_executors(None, None, None, None)
            .await
            .unwrap();

        assert_eq!(available.len(), 1);
        let available_exec = &available[0];
        assert_eq!(available_exec.cpu_specs.model, "AMD EPYC 7763");
        assert_eq!(available_exec.cpu_specs.cores, 64);
        assert_eq!(available_exec.cpu_specs.memory_gb, 256);
        assert_eq!(
            available_exec.location,
            Some("San Francisco/California/US".to_string())
        );
    }
}
