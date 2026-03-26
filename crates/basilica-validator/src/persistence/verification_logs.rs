use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::Row;
use tracing::info;
use uuid::Uuid;

use crate::persistence::entities::VerificationLog;
use crate::persistence::simple_persistence::SimplePersistence;
use crate::persistence::types::{CapacityEntry, NodeStats};

impl SimplePersistence {
    pub async fn create_verification_log(
        &self,
        log: &VerificationLog,
    ) -> Result<(), anyhow::Error> {
        let query = r#"
            INSERT INTO verification_logs (
                id, node_id, validator_hotkey, verification_type, timestamp,
                score, success, details, duration_ms, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(log.id.to_string())
            .bind(&log.node_id)
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
            .execute(self.pool())
            .await?;

        info!(
            verification_id = %log.id,
            node_id = %log.node_id,
            success = %log.success,
            score = %log.score,
            "Verification log created"
        );

        Ok(())
    }

    pub async fn query_verification_logs(
        &self,
        node_id: Option<&str>,
        success_only: Option<bool>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<VerificationLog>, anyhow::Error> {
        let mut query = String::from(
            "SELECT id, node_id, validator_hotkey, verification_type, timestamp,
             score, success, details, duration_ms, error_message, created_at, updated_at
             FROM verification_logs WHERE 1=1",
        );

        let mut conditions = Vec::new();

        if let Some(exec_id) = node_id {
            conditions.push(format!("node_id = '{exec_id}'"));
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
            .fetch_all(self.pool())
            .await?;

        let mut logs = Vec::new();
        for row in rows {
            logs.push(self.row_to_verification_log(row)?);
        }

        Ok(logs)
    }

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
                node_id,
                COUNT(*) as total_verifications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_verifications,
                AVG(score) as avg_score,
                MAX(timestamp) as last_verification,
                MAX(details) as latest_details
             FROM verification_logs
             GROUP BY node_id
             HAVING avg_score >= ?
                AND (CAST(successful_verifications AS REAL) / CAST(total_verifications AS REAL)) >= ?
             ORDER BY avg_score DESC, last_verification DESC
             LIMIT ? OFFSET ?",
        )
        .bind(min_score)
        .bind(min_success_rate)
        .bind(limit as i64)
        .bind(offset as i64)
        .fetch_all(self.pool())
        .await?;

        let mut entries = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
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
                node_id,
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

    pub(crate) fn row_to_verification_log(
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
            node_id: row.get("node_id"),
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

    pub async fn get_last_full_validation_data(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<
        Option<(
            f64,
            Option<super::super::miner_prover::types::NodeResult>,
            u64,
            bool,
        )>,
        anyhow::Error,
    > {
        let composite_node_id = if miner_id.starts_with("miner_") {
            format!("{}__{}", miner_id.replacen("miner_", "miner", 1), node_id)
        } else {
            format!("miner{}__{}", miner_id, node_id)
        };

        let query = r#"
            SELECT score, details
            FROM verification_logs
            WHERE (node_id = ? OR node_id GLOB ('*__' || ?) OR node_id = ? )
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
            .bind(node_id)
            .bind(node_id)
            .bind(&composite_node_id)
            .fetch_optional(self.pool())
            .await?;

        if let Some(row) = row {
            let score: f64 = row.get("score");
            let details_str: String = row.get("details");

            let details: serde_json::Value = serde_json::from_str(&details_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse details JSON: {}", e))?;

            let node_result = details.get("node_result").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    serde_json::from_value::<super::super::miner_prover::types::NodeResult>(
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
                node_result,
                gpu_count,
                binary_validation_successful,
            )))
        } else {
            Ok(None)
        }
    }

    pub async fn get_last_full_validation_timestamp(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<Option<chrono::DateTime<chrono::Utc>>, anyhow::Error> {
        let composite_node_id = if miner_id.starts_with("miner_") {
            format!("{}__{}", miner_id.replacen("miner_", "miner", 1), node_id)
        } else {
            format!("miner{}__{}", miner_id, node_id)
        };

        let query = r#"
            SELECT timestamp
            FROM verification_logs
            WHERE (node_id = ? OR node_id GLOB ('*__' || ?) OR node_id = ? )
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
            .bind(node_id)
            .bind(node_id)
            .bind(&composite_node_id)
            .fetch_optional(self.pool())
            .await?;

        if let Some(row) = row {
            let timestamp_str: String = row.get("timestamp");
            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                .map_err(|e| anyhow::anyhow!("Invalid timestamp format: {}", e))?
                .with_timezone(&chrono::Utc);
            Ok(Some(timestamp))
        } else {
            Ok(None)
        }
    }

    pub async fn get_miner_verification_count(
        &self,
        miner_id: &str,
        hours: i64,
    ) -> Result<u32, anyhow::Error> {
        let count_query = r#"
            SELECT COUNT(*) as count
            FROM verification_logs vl
            INNER JOIN miner_nodes me ON vl.node_id = me.node_id
            WHERE me.miner_id = ?
            AND vl.success = 1
            AND vl.timestamp > datetime('now', ? || ' hours')
        "#;

        let count: i64 = sqlx::query_scalar(count_query)
            .bind(miner_id)
            .bind(format!("-{}", hours))
            .fetch_one(self.pool())
            .await
            .unwrap_or(0);

        Ok(count as u32)
    }

    pub async fn get_node_stats(&self, node_id: &str) -> Result<Option<NodeStats>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT
                COUNT(*) as total_verifications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_verifications,
                AVG(score) as avg_score,
                AVG(duration_ms) as avg_duration_ms,
                MIN(timestamp) as first_verification,
                MAX(timestamp) as last_verification
             FROM verification_logs
             WHERE node_id = ?",
        )
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(row) = row {
            let total: i64 = row.get("total_verifications");
            if total == 0 {
                return Ok(None);
            }

            let stats = NodeStats {
                node_id: node_id.to_string(),
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
}
