//! GPU assignment persistence operations
//!
//! This module contains all SQL operations related to GPU UUID assignments.

use crate::miner_prover::types::GpuInfo;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use sqlx::Row;
use tracing::{info, warn};

impl SimplePersistence {
    /// Clean up GPU assignments for a node
    pub async fn cleanup_gpu_assignments(
        &self,
        node_id: &str,
        miner_id: &str,
        tx: Option<&mut sqlx::Transaction<'_, sqlx::Sqlite>>,
    ) -> Result<u64> {
        let query = "DELETE FROM gpu_uuid_assignments WHERE node_id = ? AND miner_id = ?";

        let rows_affected = if let Some(transaction) = tx {
            sqlx::query(query)
                .bind(node_id)
                .bind(miner_id)
                .execute(&mut **transaction)
                .await?
                .rows_affected()
        } else {
            sqlx::query(query)
                .bind(node_id)
                .bind(miner_id)
                .execute(self.pool())
                .await?
                .rows_affected()
        };

        if rows_affected > 0 {
            info!(
                "Cleaned up {} GPU assignments for node {} (miner: {})",
                rows_affected, node_id, miner_id
            );
        }

        Ok(rows_affected)
    }

    /// Store GPU UUID assignments for a node
    pub async fn store_gpu_uuid_assignments(
        &self,
        miner_uid: u16,
        node_id: &str,
        gpu_infos: &[GpuInfo],
    ) -> Result<()> {
        let miner_id = format!("miner_{miner_uid}");
        let now = chrono::Utc::now().to_rfc3339();

        let reported_gpu_uuids: Vec<String> = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .map(|g| g.gpu_uuid.clone())
            .collect();

        if !reported_gpu_uuids.is_empty() {
            let placeholders = reported_gpu_uuids
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            let query = format!(
                "DELETE FROM gpu_uuid_assignments
                 WHERE miner_id = ? AND node_id = ?
                 AND gpu_uuid NOT IN ({placeholders})"
            );

            let mut q = sqlx::query(&query).bind(&miner_id).bind(node_id);

            for uuid in &reported_gpu_uuids {
                q = q.bind(uuid);
            }

            let deleted = q.execute(self.pool()).await?;

            if deleted.rows_affected() > 0 {
                info!(
                    miner_uid = miner_uid,
                    "Cleaned up {} stale GPU assignments for {}/{}",
                    deleted.rows_affected(),
                    miner_id,
                    node_id
                );
            }
        } else {
            let deleted_rows = self
                .cleanup_gpu_assignments(node_id, &miner_id, None)
                .await?;

            if deleted_rows > 0 {
                info!(
                    miner_uid = miner_uid,
                    "Cleaned up {} GPU assignments for {}/{} (no GPUs reported)",
                    deleted_rows,
                    miner_id,
                    node_id
                );
            }
        }

        for gpu_info in gpu_infos {
            if gpu_info.gpu_uuid.is_empty() || gpu_info.gpu_uuid == "Unknown UUID" {
                continue;
            }

            let existing = sqlx::query(
                "SELECT miner_id, node_id FROM gpu_uuid_assignments WHERE gpu_uuid = ?",
            )
            .bind(&gpu_info.gpu_uuid)
            .fetch_optional(self.pool())
            .await?;

            if let Some(row) = existing {
                let existing_miner_id: String = row.get("miner_id");
                let existing_node_id: String = row.get("node_id");

                if existing_miner_id != miner_id || existing_node_id != node_id {
                    let node_status_query =
                        "SELECT status FROM miner_nodes WHERE node_id = ? AND miner_id = ?";
                    let status_row = sqlx::query(node_status_query)
                        .bind(&existing_node_id)
                        .bind(&existing_miner_id)
                        .fetch_optional(self.pool())
                        .await?;

                    let can_reassign = if let Some(row) = status_row {
                        let status: String = row.get("status");
                        status == "offline" || status == "failed" || status == "stale"
                    } else {
                        true
                    };

                    if can_reassign {
                        info!(
                            miner_uid = miner_uid,
                            security = true,
                            gpu_uuid = %gpu_info.gpu_uuid,
                            previous_miner_id = %existing_miner_id,
                            previous_node_id = %existing_node_id,
                            new_miner_id = %miner_id,
                            new_node_id = %node_id,
                            gpu_memory_gb = %gpu_info.gpu_memory_gb,
                            action = "gpu_assignment_reassigned",
                            reassignment_reason = "previous_node_inactive",
                            "GPU {} reassigned from {}/{} to {}/{} (previous node inactive)",
                            gpu_info.gpu_uuid,
                            existing_miner_id,
                            existing_node_id,
                            miner_id,
                            node_id
                        );

                        sqlx::query(
                            "UPDATE gpu_uuid_assignments
                             SET miner_id = ?, node_id = ?, gpu_index = ?, gpu_name = ?,
                                 gpu_memory_gb = ?, last_verified = ?, updated_at = ?
                             WHERE gpu_uuid = ?",
                        )
                        .bind(&miner_id)
                        .bind(node_id)
                        .bind(gpu_info.index as i32)
                        .bind(&gpu_info.gpu_name)
                        .bind(gpu_info.gpu_memory_gb)
                        .bind(&now)
                        .bind(&now)
                        .bind(&gpu_info.gpu_uuid)
                        .execute(self.pool())
                        .await?;
                    } else {
                        warn!(
                            miner_uid = miner_uid,
                            security = true,
                            gpu_uuid = %gpu_info.gpu_uuid,
                            existing_miner_id = %existing_miner_id,
                            existing_node_id = %existing_node_id,
                            attempting_miner_id = %miner_id,
                            attempting_node_id = %node_id,
                            action = "gpu_assignment_rejected",
                            rejection_reason = "already_owned_by_active_node",
                            "GPU UUID {} still owned by active node {}/{}, rejecting claim from {}/{}",
                            gpu_info.gpu_uuid,
                            existing_miner_id,
                            existing_node_id,
                            miner_id,
                            node_id
                        );
                        continue;
                    }
                } else {
                    sqlx::query(
                        "UPDATE gpu_uuid_assignments
                         SET gpu_memory_gb = ?, last_verified = ?, updated_at = ?
                         WHERE gpu_uuid = ?",
                    )
                    .bind(gpu_info.gpu_memory_gb)
                    .bind(&now)
                    .bind(&now)
                    .bind(&gpu_info.gpu_uuid)
                    .execute(self.pool())
                    .await?;
                }
            } else {
                sqlx::query(
                    "INSERT INTO gpu_uuid_assignments
                     (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, gpu_memory_gb, last_verified, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&gpu_info.gpu_uuid)
                .bind(gpu_info.index as i32)
                .bind(node_id)
                .bind(&miner_id)
                .bind(&gpu_info.gpu_name)
                .bind(gpu_info.gpu_memory_gb)
                .bind(&now)
                .bind(&now)
                .bind(&now)
                .execute(self.pool())
                .await?;

                info!(
                    miner_uid = miner_uid,
                    security = true,
                    gpu_uuid = %gpu_info.gpu_uuid,
                    gpu_index = gpu_info.index,
                    node_id = %node_id,
                    miner_id = %miner_id,
                    gpu_name = %gpu_info.gpu_name,
                    gpu_memory_gb = %gpu_info.gpu_memory_gb,
                    action = "gpu_assignment_created",
                    "Registered new GPU {} (index {}) for {}/{}",
                    gpu_info.gpu_uuid, gpu_info.index, miner_id, node_id
                );
            }
        }

        let gpu_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM gpu_uuid_assignments WHERE miner_id = ? AND node_id = ?",
        )
        .bind(&miner_id)
        .bind(node_id)
        .fetch_one(self.pool())
        .await?;

        let current_status = sqlx::query_scalar::<_, String>(
            "SELECT status FROM miner_nodes WHERE miner_id = ? AND node_id = ?",
        )
        .bind(&miner_id)
        .bind(node_id)
        .fetch_one(self.pool())
        .await?;

        let new_status = match (current_status.as_str(), gpu_count > 0) {
            ("online", true) => "online",
            ("verified", true) => "online",
            ("online", false) => "offline",
            (_, true) => "verified",
            (_, false) => "offline",
        };

        sqlx::query(
            "UPDATE miner_nodes SET gpu_count = ?, status = ?, updated_at = datetime('now')
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(gpu_count as i32)
        .bind(new_status)
        .bind(&miner_id)
        .bind(node_id)
        .execute(self.pool())
        .await?;

        if gpu_count > 0 {
            info!(
                security = true,
                node_id = %node_id,
                miner_id = %miner_id,
                gpu_count = gpu_count,
                new_status = %new_status,
                action = "node_gpu_verification_success",
                "Node {}/{} verified with {} GPUs, status: {}",
                miner_id, node_id, gpu_count, new_status
            );
        } else {
            warn!(
                miner_uid = miner_uid,
                security = true,
                node_id = %node_id,
                miner_id = %miner_id,
                gpu_count = 0,
                new_status = %new_status,
                action = "node_gpu_verification_failure",
                "Node {}/{} has no GPUs, marking as {}",
                miner_id, node_id, new_status
            );
        }

        let expected_gpu_count = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .count() as i64;

        if gpu_count != expected_gpu_count {
            warn!(
                miner_uid = miner_uid,
                "GPU assignment mismatch for {}/{}: stored {} GPUs but expected {}",
                miner_id,
                node_id,
                gpu_count,
                expected_gpu_count
            );
        }

        if expected_gpu_count > 0 && gpu_count == 0 {
            return Err(anyhow::anyhow!(
                "GPU assignment validation failed: no valid GPU UUIDs stored despite {} GPUs reported",
                expected_gpu_count
            ));
        }

        Ok(())
    }

    /// Update last_verified timestamp for existing GPU assignments
    pub async fn update_gpu_assignment_timestamps(
        &self,
        miner_uid: u16,
        node_id: &str,
        gpu_infos: &[GpuInfo],
    ) -> Result<()> {
        let miner_id = format!("miner_{miner_uid}");
        let now = chrono::Utc::now().to_rfc3339();

        let reported_gpu_uuids: Vec<String> = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .map(|g| g.gpu_uuid.clone())
            .collect();

        if reported_gpu_uuids.is_empty() {
            return Ok(());
        }

        let placeholders = reported_gpu_uuids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");

        let query = format!(
            "UPDATE gpu_uuid_assignments
             SET last_verified = ?, updated_at = ?
             WHERE miner_id = ? AND node_id = ? AND gpu_uuid IN ({placeholders})"
        );

        let mut q = sqlx::query(&query)
            .bind(&now)
            .bind(&now)
            .bind(&miner_id)
            .bind(node_id);

        for uuid in &reported_gpu_uuids {
            q = q.bind(uuid);
        }

        let result = q.execute(self.pool()).await?;
        let updated_count = result.rows_affected();

        if updated_count > 0 {
            info!(
                security = true,
                miner_uid = miner_uid,
                node_id = %node_id,
                validation_type = "lightweight",
                updated_assignments = updated_count,
                action = "gpu_assignment_timestamp_updated",
                "Updated {} GPU assignment timestamps for {}/{} (lightweight validation)",
                updated_count, miner_id, node_id
            );
        } else if self
            .has_active_rental(node_id, &miner_id)
            .await
            .unwrap_or(false)
        {
            self.store_gpu_uuid_assignments(miner_uid, node_id, gpu_infos)
                .await?;
        }

        Ok(())
    }
}
