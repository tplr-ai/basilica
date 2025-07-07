//! # Assignment Database
//!
//! Database operations for manual executor assignments

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono;
use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

/// An executor assignment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorAssignment {
    pub id: i64,
    pub executor_id: String,
    pub validator_hotkey: String,
    pub assigned_at: DateTime<Utc>,
    pub assigned_by: String,
    pub notes: Option<String>,
}

/// Validator stake information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorStake {
    pub validator_hotkey: String,
    pub stake_amount: f64,
    pub percentage_of_total: f64,
    pub last_updated: DateTime<Utc>,
}

/// Assignment history record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentHistory {
    pub id: i64,
    pub executor_id: String,
    pub validator_hotkey: Option<String>,
    pub action: String,
    pub performed_at: DateTime<Utc>,
    pub performed_by: String,
}

/// Coverage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStats {
    pub total_executors: usize,
    pub assigned_executors: usize,
    pub covered_stake_percentage: f64,
    pub covered_validators: usize,
    pub total_validators: usize,
}

/// Assignment database operations
pub struct AssignmentDb {
    pool: SqlitePool,
}

impl AssignmentDb {
    /// Create a new assignment database
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get the database pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Run database migrations
    pub async fn run_migrations(&self) -> Result<()> {
        info!("Running assignment database migrations");

        // Create executor_assignments table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS executor_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executor_id TEXT NOT NULL UNIQUE,
                validator_hotkey TEXT NOT NULL,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT NOT NULL,
                notes TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create validator_stakes table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS validator_stakes (
                validator_hotkey TEXT PRIMARY KEY,
                stake_amount REAL NOT NULL,
                percentage_of_total REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create assignment_history table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS assignment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executor_id TEXT NOT NULL,
                validator_hotkey TEXT,
                action TEXT NOT NULL,
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performed_by TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        info!("Assignment database migrations completed");
        Ok(())
    }

    /// Create a new executor assignment
    pub async fn create_assignment(
        &self,
        executor_id: &str,
        validator_hotkey: &str,
        assigned_by: &str,
        notes: Option<&str>,
    ) -> Result<ExecutorAssignment> {
        debug!(
            "Creating assignment: {} -> {}",
            executor_id, validator_hotkey
        );

        // Check if executor is already assigned
        if let Some(existing) = self.get_assignment_by_executor(executor_id).await? {
            return Err(anyhow!(
                "Executor {} is already assigned to validator {}",
                executor_id,
                existing.validator_hotkey
            ));
        }

        let now = Utc::now();

        // Insert the assignment
        let result = sqlx::query(
            r#"
            INSERT INTO executor_assignments (executor_id, validator_hotkey, assigned_at, assigned_by, notes)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(executor_id)
        .bind(validator_hotkey)
        .bind(now)
        .bind(assigned_by)
        .bind(notes)
        .execute(&self.pool)
        .await?;

        // Log to history
        self.log_assignment_action(executor_id, Some(validator_hotkey), "assign", assigned_by)
            .await?;

        Ok(ExecutorAssignment {
            id: result.last_insert_rowid(),
            executor_id: executor_id.to_string(),
            validator_hotkey: validator_hotkey.to_string(),
            assigned_at: now,
            assigned_by: assigned_by.to_string(),
            notes: notes.map(String::from),
        })
    }

    /// Delete an executor assignment
    pub async fn delete_assignment(&self, executor_id: &str, unassigned_by: &str) -> Result<()> {
        debug!("Deleting assignment for executor: {}", executor_id);

        let result = sqlx::query("DELETE FROM executor_assignments WHERE executor_id = ?")
            .bind(executor_id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Err(anyhow!("No assignment found for executor {}", executor_id));
        }

        // Log to history
        self.log_assignment_action(executor_id, None, "unassign", unassigned_by)
            .await?;

        Ok(())
    }

    /// Get assignment by executor ID
    pub async fn get_assignment_by_executor(
        &self,
        executor_id: &str,
    ) -> Result<Option<ExecutorAssignment>> {
        let row = sqlx::query(
            r#"
            SELECT id, executor_id, validator_hotkey, assigned_at, assigned_by, notes
            FROM executor_assignments
            WHERE executor_id = ?
            "#,
        )
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(ExecutorAssignment {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                assigned_at: row.get("assigned_at"),
                assigned_by: row.get("assigned_by"),
                notes: row.get("notes"),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get all assignments for a validator
    pub async fn get_assignments_for_validator(
        &self,
        validator_hotkey: &str,
    ) -> Result<Vec<ExecutorAssignment>> {
        let rows = sqlx::query(
            r#"
            SELECT id, executor_id, validator_hotkey, assigned_at, assigned_by, notes
            FROM executor_assignments
            WHERE validator_hotkey = ?
            ORDER BY assigned_at DESC
            "#,
        )
        .bind(validator_hotkey)
        .fetch_all(&self.pool)
        .await?;

        let assignments = rows
            .into_iter()
            .map(|row| ExecutorAssignment {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                assigned_at: row.get("assigned_at"),
                assigned_by: row.get("assigned_by"),
                notes: row.get("notes"),
            })
            .collect();

        Ok(assignments)
    }

    /// Get all assignments
    pub async fn get_all_assignments(&self) -> Result<Vec<ExecutorAssignment>> {
        let rows = sqlx::query(
            r#"
            SELECT id, executor_id, validator_hotkey, assigned_at, assigned_by, notes
            FROM executor_assignments
            ORDER BY assigned_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let assignments = rows
            .into_iter()
            .map(|row| ExecutorAssignment {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                assigned_at: row.get("assigned_at"),
                assigned_by: row.get("assigned_by"),
                notes: row.get("notes"),
            })
            .collect();

        Ok(assignments)
    }

    /// Update validator stake information
    pub async fn update_validator_stake(
        &self,
        validator_hotkey: &str,
        stake_amount: f64,
        percentage_of_total: f64,
    ) -> Result<()> {
        let now = Utc::now();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO validator_stakes (validator_hotkey, stake_amount, percentage_of_total, last_updated)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(validator_hotkey)
        .bind(stake_amount)
        .bind(percentage_of_total)
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update validator stakes in batch
    pub async fn update_validator_stakes_batch(
        &self,
        stakes: &[(String, f64, f64)], // (hotkey, stake_amount, percentage)
    ) -> Result<()> {
        let now = Utc::now();

        for (hotkey, stake_amount, percentage) in stakes {
            sqlx::query(
                r#"
                INSERT OR REPLACE INTO validator_stakes (validator_hotkey, stake_amount, percentage_of_total, last_updated)
                VALUES (?, ?, ?, ?)
                "#,
            )
            .bind(hotkey)
            .bind(stake_amount)
            .bind(percentage)
            .bind(now)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    /// Get all validator stakes
    pub async fn get_all_validator_stakes(&self) -> Result<Vec<ValidatorStake>> {
        let rows = sqlx::query(
            r#"
            SELECT validator_hotkey, stake_amount, percentage_of_total, last_updated
            FROM validator_stakes
            ORDER BY stake_amount DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let stakes = rows
            .into_iter()
            .map(|row| ValidatorStake {
                validator_hotkey: row.get("validator_hotkey"),
                stake_amount: row.get("stake_amount"),
                percentage_of_total: row.get("percentage_of_total"),
                last_updated: row.get("last_updated"),
            })
            .collect();

        Ok(stakes)
    }

    /// Get validator stakes above threshold
    pub async fn get_validator_stakes_above(&self, min_stake: f64) -> Result<Vec<ValidatorStake>> {
        let rows = sqlx::query(
            r#"
            SELECT validator_hotkey, stake_amount, percentage_of_total, last_updated
            FROM validator_stakes
            WHERE stake_amount >= ?
            ORDER BY stake_amount DESC
            "#,
        )
        .bind(min_stake)
        .fetch_all(&self.pool)
        .await?;

        let stakes = rows
            .into_iter()
            .map(|row| ValidatorStake {
                validator_hotkey: row.get("validator_hotkey"),
                stake_amount: row.get("stake_amount"),
                percentage_of_total: row.get("percentage_of_total"),
                last_updated: row.get("last_updated"),
            })
            .collect();

        Ok(stakes)
    }

    /// Get coverage statistics
    pub async fn get_coverage_stats(&self, total_executors: usize) -> Result<CoverageStats> {
        // Get assigned executors count
        let assigned_count_row = sqlx::query("SELECT COUNT(*) as count FROM executor_assignments")
            .fetch_one(&self.pool)
            .await?;
        let assigned_executors: i64 = assigned_count_row.get("count");

        // Get total validators
        let total_validators_row = sqlx::query("SELECT COUNT(*) as count FROM validator_stakes")
            .fetch_one(&self.pool)
            .await?;
        let total_validators: i64 = total_validators_row.get("count");

        // Get covered stake percentage
        let covered_stake_row = sqlx::query(
            r#"
            SELECT COALESCE(SUM(vs.percentage_of_total), 0) as covered_percentage
            FROM validator_stakes vs
            WHERE vs.validator_hotkey IN (
                SELECT DISTINCT validator_hotkey FROM executor_assignments
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await?;
        let covered_stake_percentage: f64 = covered_stake_row.get("covered_percentage");

        // Get covered validators count
        let covered_validators_row = sqlx::query(
            r#"
            SELECT COUNT(DISTINCT validator_hotkey) as count
            FROM executor_assignments
            "#,
        )
        .fetch_one(&self.pool)
        .await?;
        let covered_validators: i64 = covered_validators_row.get("count");

        Ok(CoverageStats {
            total_executors,
            assigned_executors: assigned_executors as usize,
            covered_stake_percentage,
            covered_validators: covered_validators as usize,
            total_validators: total_validators as usize,
        })
    }

    /// Log assignment action to history
    pub async fn log_assignment_action(
        &self,
        executor_id: &str,
        validator_hotkey: Option<&str>,
        action: &str,
        performed_by: &str,
    ) -> Result<()> {
        let now = Utc::now();

        sqlx::query(
            r#"
            INSERT INTO assignment_history (executor_id, validator_hotkey, action, performed_at, performed_by)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(executor_id)
        .bind(validator_hotkey)
        .bind(action)
        .bind(now)
        .bind(performed_by)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get assignment history for an executor
    pub async fn get_assignment_history(
        &self,
        executor_id: &str,
    ) -> Result<Vec<AssignmentHistory>> {
        let rows = sqlx::query(
            r#"
            SELECT id, executor_id, validator_hotkey, action, performed_at, performed_by
            FROM assignment_history
            WHERE executor_id = ?
            ORDER BY performed_at DESC
            "#,
        )
        .bind(executor_id)
        .fetch_all(&self.pool)
        .await?;

        let history = rows
            .into_iter()
            .map(|row| AssignmentHistory {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                action: row.get("action"),
                performed_at: row.get("performed_at"),
                performed_by: row.get("performed_by"),
            })
            .collect();

        Ok(history)
    }

    /// Get recent assignment history with limit
    pub async fn get_recent_assignment_history(
        &self,
        limit: i64,
    ) -> Result<Vec<AssignmentHistory>> {
        let rows = sqlx::query(
            r#"
            SELECT id, executor_id, validator_hotkey, action, performed_at, performed_by
            FROM assignment_history
            ORDER BY performed_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let history = rows
            .into_iter()
            .map(|row| AssignmentHistory {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                action: row.get("action"),
                performed_at: row.get("performed_at"),
                performed_by: row.get("performed_by"),
            })
            .collect();

        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_test_db() -> Result<AssignmentDb> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        let db = AssignmentDb::new(pool);
        db.run_migrations().await?;
        Ok(db)
    }

    #[tokio::test]
    async fn test_create_and_get_assignment() -> Result<()> {
        let db = setup_test_db().await?;

        let assignment = db
            .create_assignment(
                "exec-1",
                "validator-hotkey",
                "test-operator",
                Some("Test note"),
            )
            .await?;

        assert_eq!(assignment.executor_id, "exec-1");
        assert_eq!(assignment.validator_hotkey, "validator-hotkey");
        assert_eq!(assignment.assigned_by, "test-operator");
        assert_eq!(assignment.notes, Some("Test note".to_string()));

        let retrieved = db.get_assignment_by_executor("exec-1").await?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.executor_id, assignment.executor_id);
        assert_eq!(retrieved.validator_hotkey, assignment.validator_hotkey);

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_assignment() -> Result<()> {
        let db = setup_test_db().await?;

        db.create_assignment("exec-1", "validator-hotkey", "test-operator", None)
            .await?;

        db.delete_assignment("exec-1", "test-operator").await?;

        let retrieved = db.get_assignment_by_executor("exec-1").await?;
        assert!(retrieved.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_validator_stakes() -> Result<()> {
        let db = setup_test_db().await?;

        db.update_validator_stake("validator-1", 1000.0, 60.0)
            .await?;
        db.update_validator_stake("validator-2", 500.0, 40.0)
            .await?;

        let stakes = db.get_all_validator_stakes().await?;
        assert_eq!(stakes.len(), 2);
        assert_eq!(stakes[0].validator_hotkey, "validator-1"); // Should be first due to higher stake
        assert_eq!(stakes[0].stake_amount, 1000.0);
        assert_eq!(stakes[0].percentage_of_total, 60.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_coverage_stats() -> Result<()> {
        let db = setup_test_db().await?;

        // Add validator stakes
        db.update_validator_stake("validator-1", 1000.0, 60.0)
            .await?;
        db.update_validator_stake("validator-2", 500.0, 40.0)
            .await?;

        // Add assignment for 60% coverage
        db.create_assignment("exec-1", "validator-1", "test", None)
            .await?;

        let stats = db.get_coverage_stats(2).await?;
        assert_eq!(stats.total_executors, 2);
        assert_eq!(stats.assigned_executors, 1);
        assert_eq!(stats.covered_stake_percentage, 60.0);
        assert_eq!(stats.covered_validators, 1);
        assert_eq!(stats.total_validators, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_history() -> Result<()> {
        let db = setup_test_db().await?;

        db.create_assignment("exec-1", "validator-1", "operator-1", None)
            .await?;
        db.delete_assignment("exec-1", "operator-1").await?;
        db.create_assignment("exec-1", "validator-2", "operator-2", None)
            .await?;

        let history = db.get_assignment_history("exec-1").await?;
        assert_eq!(history.len(), 3); // assign, unassign, assign

        assert_eq!(history[0].action, "assign"); // Most recent
        assert_eq!(history[0].validator_hotkey, Some("validator-2".to_string()));
        assert_eq!(history[1].action, "unassign");
        assert_eq!(history[2].action, "assign");
        assert_eq!(history[2].validator_hotkey, Some("validator-1".to_string()));

        Ok(())
    }
}
