use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use common::persistence::{Repository, DatabaseConnection, Cleanup, PaginatedResponse, Pagination};
use common::PersistenceError;
use crate::persistence::entities::ChallengeResult;

#[async_trait]
pub trait ChallengeResultRepository: Repository<ChallengeResult, Uuid> {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError>;

    async fn get_by_challenge_type(
        &self,
        challenge_type: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError>;

    async fn get_successful_challenges(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError>;
}

pub struct SqliteChallengeResultRepository {
    pool: SqlitePool,
}

impl SqliteChallengeResultRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    async fn row_to_challenge_result(row: &sqlx::sqlite::SqliteRow) -> Result<ChallengeResult, PersistenceError> {
        Ok(ChallengeResult {
            id: Uuid::parse_str(row.get::<String, _>("id").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            executor_id: row.get("executor_id"),
            challenge_type: row.get("challenge_type"),
            challenge_parameters: serde_json::from_str(row.get::<String, _>("challenge_parameters").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            solution_data: row.get::<Option<String>, _>("solution_data")
                .map(|s| serde_json::from_str(&s))
                .transpose()
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            success: row.get::<i64, _>("success") != 0,
            score: row.get("score"),
            execution_time_ms: row.get("execution_time_ms"),
            verification_time_ms: row.get("verification_time_ms"),
            issued_at: DateTime::parse_from_rfc3339(row.get::<String, _>("issued_at").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?
                .with_timezone(&Utc),
            completed_at: row.get::<Option<String>, _>("completed_at")
                .map(|s| DateTime::parse_from_rfc3339(&s).ok()?.with_timezone(&Utc))
                .flatten(),
            difficulty_level: row.get("difficulty_level"),
            expected_ops: row.get("expected_ops"),
            timeout_seconds: row.get("timeout_seconds"),
            error_message: row.get("error_message"),
            error_code: row.get("error_code"),
        })
    }
}

#[async_trait]
impl Repository<ChallengeResult, Uuid> for SqliteChallengeResultRepository {
    async fn create(&self, entity: &ChallengeResult) -> Result<(), PersistenceError> {
        let query = r#"
            INSERT INTO challenge_results (
                id, executor_id, challenge_type, challenge_parameters, solution_data,
                success, score, execution_time_ms, verification_time_ms, 
                issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                error_message, error_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(&entity.id.to_string())
            .bind(&entity.executor_id)
            .bind(&entity.challenge_type)
            .bind(&serde_json::to_string(&entity.challenge_parameters)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&entity.solution_data.as_ref()
                .map(|d| serde_json::to_string(d))
                .transpose()
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(if entity.success { 1 } else { 0 })
            .bind(entity.score)
            .bind(entity.execution_time_ms)
            .bind(entity.verification_time_ms)
            .bind(&entity.issued_at.to_rfc3339())
            .bind(&entity.completed_at.map(|dt| dt.to_rfc3339()))
            .bind(entity.difficulty_level)
            .bind(entity.expected_ops)
            .bind(entity.timeout_seconds)
            .bind(&entity.error_message)
            .bind(&entity.error_code)
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn get_by_id(&self, id: &Uuid) -> Result<Option<ChallengeResult>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, challenge_type, challenge_parameters, solution_data,
                   success, score, execution_time_ms, verification_time_ms, 
                   issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                   error_message, error_code
            FROM challenge_results 
            WHERE id = ?
        "#;

        let row = sqlx::query(query)
            .bind(&id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        match row {
            Some(row) => Ok(Some(Self::row_to_challenge_result(&row).await?)),
            None => Ok(None),
        }
    }

    async fn update(&self, entity: &ChallengeResult) -> Result<(), PersistenceError> {
        let query = r#"
            UPDATE challenge_results SET
                executor_id = ?, challenge_type = ?, challenge_parameters = ?, solution_data = ?,
                success = ?, score = ?, execution_time_ms = ?, verification_time_ms = ?, 
                issued_at = ?, completed_at = ?, difficulty_level = ?, expected_ops = ?, 
                timeout_seconds = ?, error_message = ?, error_code = ?
            WHERE id = ?
        "#;

        sqlx::query(query)
            .bind(&entity.executor_id)
            .bind(&entity.challenge_type)
            .bind(&serde_json::to_string(&entity.challenge_parameters)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&entity.solution_data.as_ref()
                .map(|d| serde_json::to_string(d))
                .transpose()
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(if entity.success { 1 } else { 0 })
            .bind(entity.score)
            .bind(entity.execution_time_ms)
            .bind(entity.verification_time_ms)
            .bind(&entity.issued_at.to_rfc3339())
            .bind(&entity.completed_at.map(|dt| dt.to_rfc3339()))
            .bind(entity.difficulty_level)
            .bind(entity.expected_ops)
            .bind(entity.timeout_seconds)
            .bind(&entity.error_message)
            .bind(&entity.error_code)
            .bind(&entity.id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        let query = "DELETE FROM challenge_results WHERE id = ?";

        let result = sqlx::query(query)
            .bind(&id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(result.rows_affected() > 0)
    }

    async fn list(&self, limit: u32, offset: u32) -> Result<Vec<ChallengeResult>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, challenge_type, challenge_parameters, solution_data,
                   success, score, execution_time_ms, verification_time_ms, 
                   issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                   error_message, error_code
            FROM challenge_results 
            ORDER BY issued_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(Self::row_to_challenge_result(&row).await?);
        }

        let total_count = self.count().await?;

        Ok(PaginatedResponse::new(results, total_count, pagination))
    }

    async fn count(&self) -> Result<u64, PersistenceError> {
        let query = "SELECT COUNT(*) as count FROM challenge_results";

        let row = sqlx::query(query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(row.get::<i64, _>("count") as u64)
    }
}

#[async_trait]
impl ChallengeResultRepository for SqliteChallengeResultRepository {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, challenge_type, challenge_parameters, solution_data,
                   success, score, execution_time_ms, verification_time_ms, 
                   issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                   error_message, error_code
            FROM challenge_results 
            WHERE executor_id = ?
            ORDER BY issued_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(executor_id)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(Self::row_to_challenge_result(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM challenge_results WHERE executor_id = ?";
        let count_row = sqlx::query(count_query)
            .bind(executor_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(results, total_count, pagination))
    }

    async fn get_by_challenge_type(
        &self,
        challenge_type: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, challenge_type, challenge_parameters, solution_data,
                   success, score, execution_time_ms, verification_time_ms, 
                   issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                   error_message, error_code
            FROM challenge_results 
            WHERE challenge_type = ?
            ORDER BY issued_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(challenge_type)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(Self::row_to_challenge_result(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM challenge_results WHERE challenge_type = ?";
        let count_row = sqlx::query(count_query)
            .bind(challenge_type)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(results, total_count, pagination))
    }

    async fn get_successful_challenges(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, challenge_type, challenge_parameters, solution_data,
                   success, score, execution_time_ms, verification_time_ms, 
                   issued_at, completed_at, difficulty_level, expected_ops, timeout_seconds,
                   error_message, error_code
            FROM challenge_results 
            WHERE executor_id = ? AND success = 1
            ORDER BY issued_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(executor_id)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(Self::row_to_challenge_result(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM challenge_results WHERE executor_id = ? AND success = 1";
        let count_row = sqlx::query(count_query)
            .bind(executor_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(results, total_count, pagination))
    }
}

#[async_trait]
impl DatabaseConnection for SqliteChallengeResultRepository {
    async fn health_check(&self) -> Result<(), PersistenceError> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::ConnectionError(e.to_string()))?;
        Ok(())
    }

    async fn close(&self) -> Result<(), PersistenceError> {
        self.pool.close().await;
        Ok(())
    }
}

#[async_trait]
impl Cleanup for SqliteChallengeResultRepository {
    async fn cleanup_old_records(&self, days: u32) -> Result<u64, PersistenceError> {
        let query = "DELETE FROM challenge_results WHERE issued_at < datetime('now', '-' || ? || ' days')";

        let result = sqlx::query(query)
            .bind(days as i64)
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(result.rows_affected())
    }

    async fn vacuum(&self) -> Result<(), PersistenceError> {
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;
        Ok(())
    }
}