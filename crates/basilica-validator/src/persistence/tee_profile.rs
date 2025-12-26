//! TEE (Trusted Execution Environment) profile persistence
//!
//! Stores and retrieves TEE verification status for nodes.

use sqlx::Row;

use crate::miner_prover::types::TeeVerificationStatus;
use crate::persistence::simple_persistence::SimplePersistence;

/// TEE status data as stored in the database
#[derive(Debug, Clone)]
pub struct NodeTeeStatusRow {
    pub miner_uid: u16,
    pub node_id: String,
    pub tdx_verified: bool,
    pub tdx_quote_valid: Option<bool>,
    pub tdx_mrtd_matches: Option<bool>,
    pub tdx_mrtd_hex: Option<String>,
    pub gpu_cc_enabled: bool,
    pub gpu_cc_attestation_valid: Option<bool>,
    pub gpu_cc_model: Option<String>,
    pub gpu_cc_uuid: Option<String>,
    pub tee_verified: bool,
    pub last_verification_at: String,
    pub verification_error: Option<String>,
}

impl SimplePersistence {
    /// Store TEE verification status for a node
    #[allow(clippy::too_many_arguments)]
    pub async fn store_node_tee_status(
        &self,
        miner_uid: u16,
        node_id: &str,
        status: &TeeVerificationStatus,
    ) -> Result<(), anyhow::Error> {
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT INTO node_tee_status
            (miner_uid, node_id, tdx_verified, tdx_quote_valid, tdx_mrtd_matches,
             tdx_mrtd_hex, gpu_cc_enabled, gpu_cc_attestation_valid, gpu_cc_model,
             tee_verified, last_verification_at, verification_error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                tdx_verified = excluded.tdx_verified,
                tdx_quote_valid = excluded.tdx_quote_valid,
                tdx_mrtd_matches = excluded.tdx_mrtd_matches,
                tdx_mrtd_hex = excluded.tdx_mrtd_hex,
                gpu_cc_enabled = excluded.gpu_cc_enabled,
                gpu_cc_attestation_valid = excluded.gpu_cc_attestation_valid,
                gpu_cc_model = excluded.gpu_cc_model,
                tee_verified = excluded.tee_verified,
                last_verification_at = excluded.last_verification_at,
                verification_error = excluded.verification_error,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(status.tdx_verified)
        .bind(status.tdx_verified) // tdx_quote_valid - same as tdx_verified for now
        .bind(status.tdx_verified) // tdx_mrtd_matches - same as tdx_verified for now
        .bind(&status.mrtd_hex)
        .bind(status.gpu_cc_mode_enabled)
        .bind(status.gpu_cc_verified)
        .bind(&status.gpu_model)
        .bind(status.verified)
        .bind(&now)
        .bind(&status.error)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get TEE verification status for a node
    pub async fn get_node_tee_status(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<NodeTeeStatusRow>, anyhow::Error> {
        let row = sqlx::query(
            r#"
            SELECT miner_uid, node_id, tdx_verified, tdx_quote_valid, tdx_mrtd_matches,
                   tdx_mrtd_hex, gpu_cc_enabled, gpu_cc_attestation_valid, gpu_cc_model,
                   gpu_cc_uuid, tee_verified, last_verification_at, verification_error
            FROM node_tee_status
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(NodeTeeStatusRow {
                miner_uid: row.get::<i32, _>("miner_uid") as u16,
                node_id: row.get("node_id"),
                tdx_verified: row.get("tdx_verified"),
                tdx_quote_valid: row.get("tdx_quote_valid"),
                tdx_mrtd_matches: row.get("tdx_mrtd_matches"),
                tdx_mrtd_hex: row.get("tdx_mrtd_hex"),
                gpu_cc_enabled: row.get("gpu_cc_enabled"),
                gpu_cc_attestation_valid: row.get("gpu_cc_attestation_valid"),
                gpu_cc_model: row.get("gpu_cc_model"),
                gpu_cc_uuid: row.get("gpu_cc_uuid"),
                tee_verified: row.get("tee_verified"),
                last_verification_at: row.get("last_verification_at"),
                verification_error: row.get("verification_error"),
            }))
        } else {
            Ok(None)
        }
    }

    /// Check if a node has valid TEE attestation
    pub async fn is_node_tee_verified(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let row = sqlx::query(
            r#"
            SELECT tee_verified
            FROM node_tee_status
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row
            .map(|r| r.get::<bool, _>("tee_verified"))
            .unwrap_or(false))
    }

    /// Get all TEE-verified nodes for a miner
    pub async fn get_tee_verified_nodes(
        &self,
        miner_uid: u16,
    ) -> Result<Vec<String>, anyhow::Error> {
        let rows = sqlx::query(
            r#"
            SELECT node_id
            FROM node_tee_status
            WHERE miner_uid = ? AND tee_verified = 1
            "#,
        )
        .bind(miner_uid as i32)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("node_id")).collect())
    }

    /// Get summary of TEE status across all nodes
    pub async fn get_tee_status_summary(&self) -> Result<TeeStatusSummary, anyhow::Error> {
        let row = sqlx::query(
            r#"
            SELECT 
                COUNT(*) as total_nodes,
                SUM(CASE WHEN tee_verified = 1 THEN 1 ELSE 0 END) as tee_verified_count,
                SUM(CASE WHEN tdx_verified = 1 THEN 1 ELSE 0 END) as tdx_verified_count,
                SUM(CASE WHEN gpu_cc_enabled = 1 THEN 1 ELSE 0 END) as gpu_cc_enabled_count
            FROM node_tee_status
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(TeeStatusSummary {
            total_nodes: row.get::<i64, _>("total_nodes") as u64,
            tee_verified_count: row.get::<i64, _>("tee_verified_count") as u64,
            tdx_verified_count: row.get::<i64, _>("tdx_verified_count") as u64,
            gpu_cc_enabled_count: row.get::<i64, _>("gpu_cc_enabled_count") as u64,
        })
    }
}

/// Summary of TEE verification status across all nodes
#[derive(Debug, Clone)]
pub struct TeeStatusSummary {
    pub total_nodes: u64,
    pub tee_verified_count: u64,
    pub tdx_verified_count: u64,
    pub gpu_cc_enabled_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_get_tee_status() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let status = TeeVerificationStatus {
            verified: true,
            tdx_verified: true,
            gpu_cc_verified: true,
            mrtd_hex: Some("aabbccdd".to_string()),
            gpu_cc_mode_enabled: true,
            gpu_model: Some("H100 PCIe".to_string()),
            error: None,
        };

        // Store status
        persistence
            .store_node_tee_status(1, "node-123", &status)
            .await
            .unwrap();

        // Retrieve status
        let retrieved = persistence
            .get_node_tee_status(1, "node-123")
            .await
            .unwrap();

        assert!(retrieved.is_some());
        let row = retrieved.unwrap();
        assert_eq!(row.miner_uid, 1);
        assert_eq!(row.node_id, "node-123");
        assert!(row.tee_verified);
        assert!(row.tdx_verified);
        assert!(row.gpu_cc_enabled);
        assert_eq!(row.gpu_cc_model, Some("H100 PCIe".to_string()));
    }

    #[tokio::test]
    async fn test_is_node_tee_verified() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let status = TeeVerificationStatus {
            verified: true,
            tdx_verified: true,
            gpu_cc_verified: true,
            mrtd_hex: None,
            gpu_cc_mode_enabled: true,
            gpu_model: None,
            error: None,
        };

        persistence
            .store_node_tee_status(1, "node-123", &status)
            .await
            .unwrap();

        let is_verified = persistence
            .is_node_tee_verified(1, "node-123")
            .await
            .unwrap();
        assert!(is_verified);

        let is_not_verified = persistence
            .is_node_tee_verified(1, "node-456")
            .await
            .unwrap();
        assert!(!is_not_verified);
    }
}
