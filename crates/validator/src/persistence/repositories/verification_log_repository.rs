use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use common::persistence::{
    Repository, DatabaseConnection, Cleanup, 
    PaginatedResponse, Pagination
};
use common::PersistenceError;
use crate::persistence::entities::{VerificationLog, ExecutorVerificationStats};

#[async_trait]
pub trait VerificationLogRepository: Repository<VerificationLog, Uuid> {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError>;

    async fn get_by_validator(
        &self,
        validator_hotkey: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError>;

    async fn get_recent_logs(
        &self,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError>;

    async fn get_executor_stats(
        &self,
        executor_id: &str,
        days: Option<i32>,
    ) -> Result<ExecutorVerificationStats, PersistenceError>;
}

pub struct SqliteVerificationLogRepository {
    pool: SqlitePool,
}

impl SqliteVerificationLogRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    async fn row_to_verification_log(row: &sqlx::sqlite::SqliteRow) -> Result<VerificationLog, PersistenceError> {
        Ok(VerificationLog {
            id: Uuid::parse_str(row.get::<String, _>("id").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            executor_id: row.get("executor_id"),
            validator_hotkey: row.get("validator_hotkey"),
            verification_type: row.get("verification_type"),
            timestamp: DateTime::parse_from_rfc3339(row.get::<String, _>("timestamp").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?
                .with_timezone(&Utc),
            score: row.get("score"),
            success: row.get::<i64, _>("success") != 0,
            details: serde_json::from_str(row.get::<String, _>("details").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            duration_ms: row.get("duration_ms"),
            error_message: row.get("error_message"),
            created_at: DateTime::parse_from_rfc3339(row.get::<String, _>("created_at").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?
                .with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339(row.get::<String, _>("updated_at").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?
                .with_timezone(&Utc),
        })
    }
}

#[async_trait]
impl Repository<VerificationLog, Uuid> for SqliteVerificationLogRepository {
    type Error = PersistenceError;
    async fn create(&self, entity: &VerificationLog) -> Result<(), PersistenceError> {
        let query = r#"
            INSERT INTO verification_logs (
                id, executor_id, validator_hotkey, verification_type, timestamp,
                score, success, details, duration_ms, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(&entity.id.to_string())
            .bind(&entity.executor_id)
            .bind(&entity.validator_hotkey)
            .bind(&entity.verification_type)
            .bind(&entity.timestamp.to_rfc3339())
            .bind(entity.score)
            .bind(if entity.success { 1 } else { 0 })
            .bind(&serde_json::to_string(&entity.details)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(entity.duration_ms)
            .bind(&entity.error_message)
            .bind(&entity.created_at.to_rfc3339())
            .bind(&entity.updated_at.to_rfc3339())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn get_by_id(&self, id: &Uuid) -> Result<Option<VerificationLog>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
                   score, success, details, duration_ms, error_message, created_at, updated_at
            FROM verification_logs 
            WHERE id = ?
        "#;

        let row = sqlx::query(query)
            .bind(&id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        match row {
            Some(row) => Ok(Some(Self::row_to_verification_log(&row).await?)),
            None => Ok(None),
        }
    }

    async fn update(&self, entity: &VerificationLog) -> Result<(), PersistenceError> {
        let query = r#"
            UPDATE verification_logs SET
                executor_id = ?, validator_hotkey = ?, verification_type = ?, timestamp = ?,
                score = ?, success = ?, details = ?, duration_ms = ?, error_message = ?, updated_at = ?
            WHERE id = ?
        "#;

        sqlx::query(query)
            .bind(&entity.executor_id)
            .bind(&entity.validator_hotkey)
            .bind(&entity.verification_type)
            .bind(&entity.timestamp.to_rfc3339())
            .bind(entity.score)
            .bind(if entity.success { 1 } else { 0 })
            .bind(&serde_json::to_string(&entity.details)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(entity.duration_ms)
            .bind(&entity.error_message)
            .bind(&Utc::now().to_rfc3339())
            .bind(&entity.id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        let query = "DELETE FROM verification_logs WHERE id = ?";

        let result = sqlx::query(query)
            .bind(&id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(result.rows_affected() > 0)
    }

    async fn exists(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        let query = "SELECT 1 FROM verification_logs WHERE id = ?";
        
        let row = sqlx::query(query)
            .bind(&id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(row.is_some())
    }

    async fn list(&self, limit: u32, offset: u32) -> Result<Vec<VerificationLog>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
                   score, success, details, duration_ms, error_message, created_at, updated_at
            FROM verification_logs 
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(Self::row_to_verification_log(&row).await?);
        }

        Ok(logs)
    }

    async fn find_all(&self, pagination: Pagination) -> Result<PaginatedResponse<VerificationLog>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
                   score, success, details, duration_ms, error_message, created_at, updated_at
            FROM verification_logs 
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(Self::row_to_verification_log(&row).await?);
        }

        let count_row = sqlx::query("SELECT COUNT(*) as count FROM verification_logs")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(logs, total_count, pagination))
    }

    async fn count(&self) -> Result<u64, PersistenceError> {
        let query = "SELECT COUNT(*) as count FROM verification_logs";

        let row = sqlx::query(query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(row.get::<i64, _>("count") as u64)
    }
}

#[async_trait]
impl VerificationLogRepository for SqliteVerificationLogRepository {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
                   score, success, details, duration_ms, error_message, created_at, updated_at
            FROM verification_logs 
            WHERE executor_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(executor_id)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(Self::row_to_verification_log(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM verification_logs WHERE executor_id = ?";
        let count_row = sqlx::query(count_query)
            .bind(executor_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(logs, total_count, pagination))
    }

    async fn get_by_validator(
        &self,
        validator_hotkey: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, validator_hotkey, verification_type, timestamp,
                   score, success, details, duration_ms, error_message, created_at, updated_at
            FROM verification_logs 
            WHERE validator_hotkey = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(validator_hotkey)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(Self::row_to_verification_log(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM verification_logs WHERE validator_hotkey = ?";
        let count_row = sqlx::query(count_query)
            .bind(validator_hotkey)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(logs, total_count, pagination))
    }

    async fn get_recent_logs(
        &self,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError> {
        self.find_all(pagination).await
    }

    async fn get_executor_stats(
        &self,
        executor_id: &str,
        days: Option<i32>,
    ) -> Result<ExecutorVerificationStats, PersistenceError> {
        let date_filter = if let Some(days) = days {
            format!("AND timestamp >= datetime('now', '-{} days')", days)
        } else {
            String::new()
        };

        let query = format!(
            r#"
            SELECT 
                COUNT(*) as total_verifications,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_verifications,
                AVG(CASE WHEN success = 1 THEN score END) as average_score,
                AVG(duration_ms) as average_duration_ms,
                MIN(timestamp) as first_verification,
                MAX(timestamp) as last_verification
            FROM verification_logs 
            WHERE executor_id = ? {}
            "#,
            date_filter
        );

        let row = sqlx::query(&query)
            .bind(executor_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(ExecutorVerificationStats {
            executor_id: executor_id.to_string(),
            total_verifications: row.get::<i64, _>("total_verifications") as u64,
            successful_verifications: row.get::<i64, _>("successful_verifications") as u64,
            average_score: row.get("average_score"),
            average_duration_ms: row.get("average_duration_ms"),
            first_verification: row.get::<Option<String>, _>("first_verification")
                .map(|s| DateTime::parse_from_rfc3339(&s).ok()?.with_timezone(&Utc))
                .flatten(),
            last_verification: row.get::<Option<String>, _>("last_verification")
                .map(|s| DateTime::parse_from_rfc3339(&s).ok()?.with_timezone(&Utc))
                .flatten(),
        })
    }
}

#[async_trait]
impl DatabaseConnection for SqliteVerificationLogRepository {
    async fn health_check(&self) -> Result<(), PersistenceError> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionFailed { source: Box::new(e) })?;
        Ok(())
    }

    async fn close(&self) -> Result<(), PersistenceError> {
        self.pool.close().await;
        Ok(())
    }
}

#[async_trait]
impl Cleanup for SqliteVerificationLogRepository {
    async fn cleanup_old_records(&self, retention_days: i64) -> Result<u64, PersistenceError> {
        let query = "DELETE FROM verification_logs WHERE timestamp < datetime('now', '-' || ? || ' days')";

        let result = sqlx::query(query)
            .bind(retention_days)
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(result.rows_affected())
    }

    async fn optimize_storage(&self) -> Result<(), PersistenceError> {
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;
        Ok(())
    }

    async fn storage_stats(&self) -> Result<common::persistence::StorageStats, PersistenceError> {
        // SQLite doesn't have built-in storage stats like PostgreSQL
        Ok(common::persistence::StorageStats {
            total_size_bytes: 0,
            data_size_bytes: 0,
            index_size_bytes: 0,
            temp_size_bytes: 0,
            table_count: 1,
            index_count: 0,
        })
    }
}