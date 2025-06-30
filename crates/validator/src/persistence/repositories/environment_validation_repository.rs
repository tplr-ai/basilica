use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use common::persistence::{Repository, DatabaseConnection, Cleanup, PaginatedResponse, Pagination};
use common::PersistenceError;
use crate::persistence::entities::EnvironmentValidation;

#[async_trait]
pub trait EnvironmentValidationRepository: Repository<EnvironmentValidation, Uuid> {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError>;

    async fn get_latest_by_executor(
        &self,
        executor_id: &str,
    ) -> Result<Option<EnvironmentValidation>, PersistenceError>;

    async fn get_passing_validations(
        &self,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError>;
}

pub struct SqliteEnvironmentValidationRepository {
    pool: SqlitePool,
}

impl SqliteEnvironmentValidationRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    async fn row_to_environment_validation(row: &sqlx::sqlite::SqliteRow) -> Result<EnvironmentValidation, PersistenceError> {
        Ok(EnvironmentValidation {
            id: Uuid::parse_str(row.get::<String, _>("id").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            executor_id: row.get("executor_id"),
            docker_score: row.get("docker_score"),
            gpu_score: row.get("gpu_score"),
            security_score: row.get("security_score"),
            performance_score: row.get("performance_score"),
            overall_score: row.get("overall_score"),
            issues: serde_json::from_str(row.get::<String, _>("issues").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            warnings: serde_json::from_str(row.get::<String, _>("warnings").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            environment_data: serde_json::from_str(row.get::<String, _>("environment_data").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?,
            validation_duration_ms: row.get("validation_duration_ms"),
            created_at: DateTime::parse_from_rfc3339(row.get::<String, _>("created_at").as_str())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?
                .with_timezone(&Utc),
        })
    }
}

#[async_trait]
impl Repository<EnvironmentValidation, Uuid> for SqliteEnvironmentValidationRepository {
    async fn create(&self, entity: &EnvironmentValidation) -> Result<(), PersistenceError> {
        let query = r#"
            INSERT INTO environment_validations (
                id, executor_id, docker_score, gpu_score, security_score, performance_score,
                overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(&entity.id.to_string())
            .bind(&entity.executor_id)
            .bind(entity.docker_score)
            .bind(entity.gpu_score)
            .bind(entity.security_score)
            .bind(entity.performance_score)
            .bind(entity.overall_score)
            .bind(&serde_json::to_string(&entity.issues)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&serde_json::to_string(&entity.warnings)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&serde_json::to_string(&entity.environment_data)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(entity.validation_duration_ms)
            .bind(&entity.created_at.to_rfc3339())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn get_by_id(&self, id: &Uuid) -> Result<Option<EnvironmentValidation>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, docker_score, gpu_score, security_score, performance_score,
                   overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            FROM environment_validations 
            WHERE id = ?
        "#;

        let row = sqlx::query(query)
            .bind(&id.to_string())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        match row {
            Some(row) => Ok(Some(Self::row_to_environment_validation(&row).await?)),
            None => Ok(None),
        }
    }

    async fn update(&self, entity: &EnvironmentValidation) -> Result<(), PersistenceError> {
        let query = r#"
            UPDATE environment_validations SET
                executor_id = ?, docker_score = ?, gpu_score = ?, security_score = ?, 
                performance_score = ?, overall_score = ?, issues = ?, warnings = ?, 
                environment_data = ?, validation_duration_ms = ?
            WHERE id = ?
        "#;

        sqlx::query(query)
            .bind(&entity.executor_id)
            .bind(entity.docker_score)
            .bind(entity.gpu_score)
            .bind(entity.security_score)
            .bind(entity.performance_score)
            .bind(entity.overall_score)
            .bind(&serde_json::to_string(&entity.issues)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&serde_json::to_string(&entity.warnings)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(&serde_json::to_string(&entity.environment_data)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?)
            .bind(entity.validation_duration_ms)
            .bind(&entity.id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(())
    }

    async fn delete(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        let query = "DELETE FROM environment_validations WHERE id = ?";

        let result = sqlx::query(query)
            .bind(&id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(result.rows_affected() > 0)
    }

    async fn find_all(&self, pagination: Pagination) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, docker_score, gpu_score, security_score, performance_score,
                   overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            FROM environment_validations 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut validations = Vec::new();
        for row in rows {
            validations.push(Self::row_to_environment_validation(&row).await?);
        }

        let total_count = self.count().await?;

        Ok(PaginatedResponse::new(validations, total_count, pagination))
    }

    async fn count(&self) -> Result<u64, PersistenceError> {
        let query = "SELECT COUNT(*) as count FROM environment_validations";

        let row = sqlx::query(query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        Ok(row.get::<i64, _>("count") as u64)
    }
}

#[async_trait]
impl EnvironmentValidationRepository for SqliteEnvironmentValidationRepository {
    async fn get_by_executor(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, docker_score, gpu_score, security_score, performance_score,
                   overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            FROM environment_validations 
            WHERE executor_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(executor_id)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut validations = Vec::new();
        for row in rows {
            validations.push(Self::row_to_environment_validation(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM environment_validations WHERE executor_id = ?";
        let count_row = sqlx::query(count_query)
            .bind(executor_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(validations, total_count, pagination))
    }

    async fn get_latest_by_executor(
        &self,
        executor_id: &str,
    ) -> Result<Option<EnvironmentValidation>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, docker_score, gpu_score, security_score, performance_score,
                   overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            FROM environment_validations 
            WHERE executor_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(executor_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        match row {
            Some(row) => Ok(Some(Self::row_to_environment_validation(&row).await?)),
            None => Ok(None),
        }
    }

    async fn get_passing_validations(
        &self,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError> {
        let query = r#"
            SELECT id, executor_id, docker_score, gpu_score, security_score, performance_score,
                   overall_score, issues, warnings, environment_data, validation_duration_ms, created_at
            FROM environment_validations 
            WHERE overall_score >= 0.7 AND issues = '[]'
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(pagination.limit as i64)
            .bind(pagination.offset as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let mut validations = Vec::new();
        for row in rows {
            validations.push(Self::row_to_environment_validation(&row).await?);
        }

        let count_query = "SELECT COUNT(*) as count FROM environment_validations WHERE overall_score >= 0.7 AND issues = '[]'";
        let count_row = sqlx::query(count_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PersistenceError::QueryFailed { query: "database operation".to_string() })?;

        let total_count = count_row.get::<i64, _>("count") as u64;

        Ok(PaginatedResponse::new(validations, total_count, pagination))
    }
}

#[async_trait]
impl DatabaseConnection for SqliteEnvironmentValidationRepository {
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
impl Cleanup for SqliteEnvironmentValidationRepository {
    async fn cleanup_old_records(&self, days: u32) -> Result<u64, PersistenceError> {
        let query = "DELETE FROM environment_validations WHERE created_at < datetime('now', '-' || ? || ' days')";

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