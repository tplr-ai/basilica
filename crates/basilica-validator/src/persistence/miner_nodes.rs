//! Miner node relationship persistence operations
//!
//! This module contains all SQL operations related to miner-node relationships.

use crate::miner_prover::types::MinerInfo;
use crate::persistence::types::{AvailableNodeData, NodeData};
use crate::persistence::SimplePersistence;
use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::Row;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, info, warn};

fn extract_gpu_memory_gb(gpu_name: &str) -> u32 {
    use regex::Regex;
    let re = Regex::new(r"(\d+)GB").unwrap();
    if let Some(captures) = re.captures(gpu_name) {
        captures[1].parse().unwrap_or(0)
    } else {
        0
    }
}

impl SimplePersistence {
    /// Ensure miner-node relationship exists
    pub async fn ensure_miner_node_relationship(
        &self,
        miner_uid: u16,
        node_id: &str,
        node_ssh_endpoint: &str,
        miner_info: &MinerInfo,
    ) -> Result<()> {
        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "Ensuring miner-node relationship for miner {} and node {} with real data",
            miner_uid,
            node_id
        );

        let miner_id = format!("miner_{miner_uid}");

        self.ensure_miner_exists_with_info(miner_info).await?;

        let query = "SELECT COUNT(*) as count FROM miner_nodes WHERE miner_id = ? AND node_id = ?";
        let row = sqlx::query(query)
            .bind(&miner_id)
            .bind(node_id)
            .fetch_one(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner-node relationship: {}", e))?;

        let count: i64 = row.get("count");

        if count == 0 {
            let existing_miner: Option<String> = sqlx::query_scalar(
                "SELECT miner_id FROM miner_nodes WHERE ssh_endpoint = ? AND miner_id != ? LIMIT 1",
            )
            .bind(node_ssh_endpoint)
            .bind(&miner_id)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check ssh_endpoint uniqueness: {}", e))?;

            if let Some(other_miner) = existing_miner {
                return Err(anyhow::anyhow!(
                    "Cannot create node relationship: ssh_endpoint {} is already registered to {}",
                    node_ssh_endpoint,
                    other_miner
                ));
            }

            let old_node_id: Option<String> = sqlx::query_scalar(
                "SELECT node_id FROM miner_nodes WHERE ssh_endpoint = ? AND miner_id = ?",
            )
            .bind(node_ssh_endpoint)
            .bind(&miner_id)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check for existing node: {}", e))?;

            if let Some(old_id) = old_node_id {
                info!(
                    "Miner {} is changing node ID from {} to {} for endpoint {}",
                    miner_id, old_id, node_id, node_ssh_endpoint
                );

                let mut tx = self.pool().begin().await?;

                sqlx::query(
                    "UPDATE gpu_uuid_assignments SET node_id = ? WHERE node_id = ? AND miner_id = ?"
                )
                .bind(node_id)
                .bind(&old_id)
                .bind(&miner_id)
                .execute(&mut *tx)
                .await?;

                sqlx::query("DELETE FROM miner_nodes WHERE node_id = ? AND miner_id = ?")
                    .bind(&old_id)
                    .bind(&miner_id)
                    .execute(&mut *tx)
                    .await?;

                tx.commit().await?;

                info!(
                    "Successfully migrated GPU assignments from node {} to {}",
                    old_id, node_id
                );
            }

            let insert_query = r#"
                INSERT OR IGNORE INTO miner_nodes (
                    id, miner_id, node_id, ssh_endpoint, gpu_count,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            "#;

            let relationship_id = format!("{miner_id}_{node_id}");

            sqlx::query(insert_query)
                .bind(&relationship_id)
                .bind(&miner_id)
                .bind(node_id)
                .bind(node_ssh_endpoint)
                .bind(0)
                .bind("online")
                .execute(self.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to insert miner-node relationship: {}", e))?;

            info!(
                miner_uid = miner_uid,
                node_id = node_id,
                "Created miner-node relationship: {} -> {} with endpoint {}",
                miner_id,
                node_id,
                node_ssh_endpoint
            );
        } else {
            debug!(
                miner_uid = miner_uid,
                node_id = node_id,
                "Miner-node relationship already exists: {} -> {}",
                miner_id,
                node_id
            );

            let duplicate_check_query: &'static str =
                "SELECT id, node_id FROM miner_nodes WHERE ssh_endpoint = ? AND id != ?";
            let relationship_id = format!("{miner_id}_{node_id}");

            let duplicates = sqlx::query(duplicate_check_query)
                .bind(node_ssh_endpoint)
                .bind(&relationship_id)
                .fetch_all(self.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to check for duplicate nodes: {}", e))?;

            if !duplicates.is_empty() {
                let duplicate_count = duplicates.len();
                warn!(
                    miner_uid = miner_uid,
                    "Found {} duplicate nodes with same ssh_endpoint {} for miner {}",
                    duplicate_count,
                    node_ssh_endpoint,
                    miner_id
                );

                for duplicate in duplicates {
                    let dup_id: String = duplicate.get("id");
                    let dup_node_id: String = duplicate.get("node_id");

                    warn!(
                        miner_uid = miner_uid,
                        "Marking duplicate node {} (id: {}) as offline with same ssh_endpoint as {} for miner {}",
                        dup_node_id, dup_id, node_id, miner_id
                    );

                    sqlx::query("UPDATE miner_nodes SET status = 'offline', last_health_check = datetime('now'), updated_at = datetime('now') WHERE id = ?")
                        .bind(&dup_id)
                        .execute(self.pool())
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to update duplicate node status: {}", e)
                        })?;

                    self.cleanup_gpu_assignments(&dup_node_id, &miner_id, None)
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to clean up GPU assignments for duplicate: {}",
                                e
                            )
                        })?;
                }

                info!(
                    miner_uid = miner_uid,
                    "Cleaned up {} duplicate nodes for miner {} with ssh_endpoint {}",
                    duplicate_count,
                    miner_id,
                    node_ssh_endpoint
                );
            }
        }

        Ok(())
    }

    /// Clean up nodes that have consecutive failed validations
    pub async fn cleanup_failed_nodes_after_failures(
        &self,
        consecutive_failures_threshold: i32,
        gpu_assignment_cleanup_ttl: Option<Duration>,
    ) -> Result<()> {
        info!(
            "Running node cleanup - checking for {} consecutive failures",
            consecutive_failures_threshold
        );

        let offline_with_gpus_query = r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            GROUP BY me.node_id, me.miner_id
        "#;

        let offline_with_gpus = sqlx::query(offline_with_gpus_query)
            .fetch_all(self.pool())
            .await?;

        let mut gpu_assignments_cleaned = 0;
        for row in offline_with_gpus {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                "Cleaning up {} GPU assignments for offline node {} (miner: {})",
                gpu_count, node_id, miner_id
            );

            let rows_cleaned = self
                .cleanup_gpu_assignments(&node_id, &miner_id, None)
                .await?;
            gpu_assignments_cleaned += rows_cleaned;
        }

        let mismatched_gpu_query = r#"
            SELECT me.node_id, me.miner_id, me.gpu_count, me.status
            FROM miner_nodes me
            WHERE me.gpu_count > 0
            AND NOT EXISTS (
                SELECT 1 FROM gpu_uuid_assignments ga
                WHERE ga.node_id = me.node_id AND ga.miner_id = me.miner_id
            )
        "#;

        let mismatched_nodes = sqlx::query(mismatched_gpu_query)
            .fetch_all(self.pool())
            .await?;

        for row in mismatched_nodes {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i32 = row.try_get("gpu_count")?;
            let status: String = row.try_get("status")?;

            warn!(
                "Node {} (miner: {}) claims {} GPUs but has no assignments, status: {}. Resetting GPU count to 0",
                node_id, miner_id, gpu_count, status
            );

            sqlx::query(
                "UPDATE miner_nodes SET gpu_count = 0, updated_at = datetime('now')
                 WHERE node_id = ? AND miner_id = ?",
            )
            .bind(&node_id)
            .bind(&miner_id)
            .execute(self.pool())
            .await?;

            if status == "online" || status == "verified" {
                sqlx::query(
                    "UPDATE miner_nodes SET status = 'offline', updated_at = datetime('now')
                     WHERE node_id = ? AND miner_id = ?",
                )
                .bind(&node_id)
                .bind(&miner_id)
                .execute(self.pool())
                .await?;

                info!(
                    "Marked node {} as offline (claimed {} GPUs but has 0 assignments)",
                    node_id, gpu_count
                );
            }
        }

        let stale_gpu_cleanup_query = r#"
            DELETE FROM gpu_uuid_assignments
            WHERE last_verified < datetime('now', '-6 hours')
            OR (
                EXISTS (
                    SELECT 1 FROM miner_nodes me
                    WHERE me.node_id = gpu_uuid_assignments.node_id
                    AND me.miner_id = gpu_uuid_assignments.miner_id
                    AND me.status = 'offline'
                    AND (
                        me.last_health_check < datetime('now', '-2 hours')
                        OR (me.last_health_check IS NULL AND me.updated_at < datetime('now', '-2 hours'))
                    )
                )
            )
        "#;

        let stale_gpu_result = sqlx::query(stale_gpu_cleanup_query)
            .execute(self.pool())
            .await?;

        if stale_gpu_result.rows_affected() > 0 {
            info!(
                security = true,
                cleaned_count = stale_gpu_result.rows_affected(),
                cleanup_reason = "stale_timeout",
                threshold_hours = 6,
                "Cleaned up {} stale GPU assignments (not verified in last 6 hours or belonging to offline nodes >2h)",
                stale_gpu_result.rows_affected()
            );
        }

        let cleanup_minutes = gpu_assignment_cleanup_ttl
            .map(|d| d.as_secs() / 60)
            .unwrap_or(120)
            .max(120);

        info!(
            "Cleaning GPU assignments from nodes offline >{} minutes",
            cleanup_minutes
        );

        let stale_offline_query = format!(
            r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            AND (
                me.last_health_check < datetime('now', '-{cleanup_minutes} minutes')
                OR (me.last_health_check IS NULL AND me.updated_at < datetime('now', '-{cleanup_minutes} minutes'))
            )
            GROUP BY me.node_id, me.miner_id
            "#
        );

        let stale_offline = sqlx::query(&stale_offline_query)
            .fetch_all(self.pool())
            .await?;

        let mut stale_gpu_cleaned = 0;
        for row in stale_offline {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                security = true,
                node_id = %node_id,
                miner_id = %miner_id,
                gpu_count = gpu_count,
                cleanup_minutes = cleanup_minutes,
                "Cleaning GPU assignments from node offline >{}min", cleanup_minutes
            );

            let cleaned = self
                .cleanup_gpu_assignments(&node_id, &miner_id, None)
                .await?;
            stale_gpu_cleaned += cleaned;
        }

        if stale_gpu_cleaned > 0 {
            info!(
                security = true,
                cleaned_count = stale_gpu_cleaned,
                cleanup_minutes = cleanup_minutes,
                "Cleaned {} GPU assignments from nodes offline >{}min",
                stale_gpu_cleaned,
                cleanup_minutes
            );
        }

        let delete_nodes_query = r#"
            WITH recent_verifications AS (
                SELECT
                    vl.node_id,
                    vl.success,
                    vl.timestamp,
                    ROW_NUMBER() OVER (PARTITION BY vl.node_id ORDER BY vl.timestamp DESC) as rn
                FROM verification_logs vl
                WHERE vl.timestamp > datetime('now', '-1 hour')
            )
            SELECT
                me.node_id,
                me.miner_id,
                me.status,
                COALESCE(SUM(CASE WHEN rv.success = 0 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as consecutive_fails,
                COALESCE(SUM(CASE WHEN rv.success = 1 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as recent_successes,
                MAX(rv.timestamp) as last_verification
            FROM miner_nodes me
            LEFT JOIN recent_verifications rv ON me.node_id = rv.node_id
            WHERE me.status = 'offline'
            GROUP BY me.node_id, me.miner_id, me.status
            HAVING consecutive_fails >= ? AND recent_successes = 0
        "#;

        let nodes_to_delete = sqlx::query(delete_nodes_query)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .fetch_all(self.pool())
            .await?;

        let mut deleted = 0;
        for row in nodes_to_delete {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let consecutive_fails: i64 = row.try_get("consecutive_fails")?;
            let last_verification: Option<String> = row.try_get("last_verification").ok();

            info!(
                "Permanently deleting node {} (miner: {}) after {} consecutive failures, last seen: {}",
                node_id, miner_id, consecutive_fails,
                last_verification.as_deref().unwrap_or("never")
            );

            let mut tx = self.pool().begin().await?;

            self.cleanup_gpu_assignments(&node_id, &miner_id, Some(&mut tx))
                .await?;

            sqlx::query("DELETE FROM miner_nodes WHERE node_id = ? AND miner_id = ?")
                .bind(&node_id)
                .bind(&miner_id)
                .execute(&mut *tx)
                .await?;

            tx.commit().await?;
            deleted += 1;
        }

        let stale_delete_query = format!(
            r#"
            DELETE FROM miner_nodes
            WHERE status = 'offline'
            AND (
                last_health_check < datetime('now', '-{} minutes')
                OR (last_health_check IS NULL AND updated_at < datetime('now', '-{} minutes'))
            )
            "#,
            cleanup_minutes, cleanup_minutes
        );

        info!(
            "Deleting stale offline nodes using {}min timeout (configurable via gpu_assignment_cleanup_ttl)",
            cleanup_minutes
        );

        let stale_result = sqlx::query(&stale_delete_query)
            .execute(self.pool())
            .await?;

        let stale_deleted = stale_result.rows_affected();

        let affected_miners_query = r#"
            SELECT DISTINCT miner_uid
            FROM miner_gpu_profiles
            WHERE miner_uid IN (
                SELECT DISTINCT CAST(SUBSTR(miner_id, 7) AS INTEGER)
                FROM miner_nodes
                WHERE status = 'offline'

                UNION

                SELECT miner_uid
                FROM miner_gpu_profiles
                WHERE gpu_counts_json <> '{}'
                AND NOT EXISTS (
                    SELECT 1 FROM miner_nodes
                    WHERE miner_id = 'miner_' || miner_gpu_profiles.miner_uid
                    AND status NOT IN ('offline', 'failed', 'stale')
                )
            )
        "#;

        let affected_miners = sqlx::query(affected_miners_query)
            .fetch_all(self.pool())
            .await?;

        for row in affected_miners {
            let miner_uid: i64 = row.try_get("miner_uid")?;
            let miner_id = format!("miner_{}", miner_uid);

            let gpu_counts = self.get_miner_gpu_uuid_assignments(&miner_id).await?;

            let mut gpu_map: std::collections::HashMap<String, u32> =
                std::collections::HashMap::new();
            for (_, count, gpu_name, _) in gpu_counts {
                let category =
                    crate::gpu::categorization::GpuCategory::from_str(&gpu_name).unwrap();
                let model = category.to_string();
                *gpu_map.entry(model).or_insert(0) += count;
            }

            let update_query = if gpu_map.is_empty() {
                r#"
                UPDATE miner_gpu_profiles
                SET gpu_counts_json = ?,
                    total_score = 0.0,
                    verification_count = 0,
                    last_successful_validation = NULL,
                    last_updated = datetime('now')
                WHERE miner_uid = ?
                "#
            } else {
                r#"
                UPDATE miner_gpu_profiles
                SET gpu_counts_json = ?,
                    last_updated = datetime('now')
                WHERE miner_uid = ?
                "#
            };

            let gpu_json = serde_json::to_string(&gpu_map)?;
            let result = sqlx::query(update_query)
                .bind(&gpu_json)
                .bind(miner_uid)
                .execute(self.pool())
                .await?;

            if result.rows_affected() > 0 {
                info!(
                    "Updated GPU profile for miner {} after cleanup: {}",
                    miner_uid, gpu_json
                );
            }
        }

        if gpu_assignments_cleaned > 0 {
            info!(
                "Deleted {} GPU assignments from offline nodes",
                gpu_assignments_cleaned
            );
        }

        if deleted > 0 {
            info!(
                "Deleted {} nodes with {} or more consecutive failures",
                deleted, consecutive_failures_threshold
            );
        }

        if stale_deleted > 0 {
            info!("Deleted {} stale offline nodes", stale_deleted);
        }

        if gpu_assignments_cleaned == 0 && deleted == 0 && stale_deleted == 0 {
            debug!("No nodes needed cleanup in this cycle");
        }

        Ok(())
    }

    /// Get available nodes for rental (not currently rented)
    pub async fn get_available_nodes(
        &self,
        min_gpu_memory: Option<u32>,
        gpu_type: Option<String>,
        min_gpu_count: Option<u32>,
        location: Option<basilica_common::LocationProfile>,
    ) -> Result<Vec<AvailableNodeData>, anyhow::Error> {
        let mut query_str = String::from(
            "SELECT
                me.node_id,
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
            FROM miner_nodes me
            JOIN miners m ON me.miner_id = m.id
            LEFT JOIN rentals r ON me.node_id = r.node_id
                AND r.miner_id = me.miner_id
                AND r.state IN ('Active', 'Provisioning', 'active', 'provisioning')
            LEFT JOIN gpu_uuid_assignments gua ON me.node_id = gua.node_id AND gua.miner_id = me.miner_id
            LEFT JOIN node_hardware_profile ehp ON me.node_id = ehp.node_id AND me.miner_id = 'miner_' || ehp.miner_uid
            LEFT JOIN node_network_profile enp ON me.node_id = enp.node_id AND me.miner_id = 'miner_' || enp.miner_uid
            LEFT JOIN node_speedtest_profile esp ON me.node_id = esp.node_id AND me.miner_id = 'miner_' || esp.miner_uid
            WHERE r.id IS NULL
                AND (me.status IS NULL OR me.status != 'offline')",
        );

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

        query_str.push_str(" GROUP BY me.node_id");

        if let Some(min_count) = min_gpu_count {
            query_str.push_str(&format!(" HAVING COUNT(gua.gpu_uuid) >= {}", min_count));
        }

        let rows = sqlx::query(&query_str).fetch_all(self.pool()).await?;

        let mut nodes = Vec::new();
        for row in rows {
            let gpu_names: Option<String> = row.get("gpu_names");

            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    for gpu_name in names.split(',') {
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(),
                        });
                    }
                }
            }

            if let Some(min_memory) = min_gpu_memory {
                let meets_memory = gpu_specs.iter().any(|gpu| gpu.memory_gb >= min_memory);
                if !meets_memory && !gpu_specs.is_empty() {
                    continue;
                }
            }

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

            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");

            let cpu_specs = crate::api::types::CpuSpec {
                cores: cpu_cores.unwrap_or(0) as u32,
                model: cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: ram_gb.unwrap_or(0) as u32,
            };

            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");

            let location_profile = basilica_common::LocationProfile::new(city, region, country);
            let location = Some(location_profile.to_string());

            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp_str: Option<String> = row.get("test_timestamp");

            let speed_test_timestamp = test_timestamp_str.and_then(|ts| {
                chrono::DateTime::parse_from_rfc3339(&ts)
                    .ok()
                    .map(|dt| dt.with_timezone(&chrono::Utc))
            });

            nodes.push(AvailableNodeData {
                node_id: row.get("node_id"),
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

        Ok(nodes)
    }

    /// Get miner nodes
    pub async fn get_miner_nodes(&self, miner_id: &str) -> Result<Vec<NodeData>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT
                me.node_id,
                GROUP_CONCAT(gua.gpu_name) as gpu_names,
                ehp.cpu_model,
                ehp.cpu_cores,
                ehp.ram_gb,
                enp.city,
                enp.region,
                enp.country
             FROM miner_nodes me
             LEFT JOIN gpu_uuid_assignments gua ON me.node_id = gua.node_id AND gua.miner_id = me.miner_id
             LEFT JOIN node_hardware_profile ehp ON me.node_id = ehp.node_id AND me.miner_id = 'miner_' || ehp.miner_uid
             LEFT JOIN node_network_profile enp ON me.node_id = enp.node_id AND me.miner_id = 'miner_' || enp.miner_uid
             WHERE me.miner_id = ?
             GROUP BY me.node_id,
                      ehp.cpu_model, ehp.cpu_cores, ehp.ram_gb,
                      enp.city, enp.region, enp.country",
        )
        .bind(miner_id)
        .fetch_all(self.pool())
        .await?;

        let mut nodes = Vec::new();
        for row in rows {
            let gpu_names: Option<String> = row.get("gpu_names");

            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    for gpu_name in names.split(',') {
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(),
                        });
                    }
                }
            }

            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");

            let cpu_specs = crate::api::types::CpuSpec {
                cores: cpu_cores.unwrap_or(0) as u32,
                model: cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: ram_gb.unwrap_or(0) as u32,
            };

            let city: Option<String> = row.get("city");
            let region: Option<String> = row.get("region");
            let country: Option<String> = row.get("country");

            let location_profile = basilica_common::LocationProfile::new(city, region, country);
            let location = Some(location_profile.to_string());

            nodes.push(NodeData {
                node_id: row.get("node_id"),
                gpu_specs,
                cpu_specs,
                location,
            });
        }

        Ok(nodes)
    }

    /// Get miner ID by node ID
    pub async fn get_miner_id_by_node(&self, node_id: &str) -> Result<String, anyhow::Error> {
        let miner_id: String = sqlx::query(
            "SELECT miner_id FROM miner_nodes \
                 WHERE node_id = ? \
                 LIMIT 1",
        )
        .bind(node_id)
        .fetch_one(self.pool())
        .await?
        .get("miner_id");

        Ok(miner_id)
    }

    /// get node ssh-endpoint by node ID, return None if not found
    pub async fn get_node_ssh_endpoint(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<Option<String>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT ssh_endpoint FROM miner_nodes \
                 WHERE node_id = ? AND miner_id = ? \
                 LIMIT 1",
        )
        .bind(node_id)
        .bind(miner_id)
        .fetch_optional(self.pool())
        .await?;

        Ok(row.map(|r| r.get("ssh_endpoint")))
    }

    /// Get detailed node information including GPU and CPU specs
    pub async fn get_node_details(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<Option<crate::api::types::NodeDetails>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT
                me.node_id,
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
             FROM miner_nodes me
             LEFT JOIN gpu_uuid_assignments gua ON me.node_id = gua.node_id AND gua.miner_id = me.miner_id
             LEFT JOIN node_hardware_profile ehp ON me.node_id = ehp.node_id AND me.miner_id = 'miner_' || ehp.miner_uid
             LEFT JOIN node_network_profile enp ON me.node_id = enp.node_id AND me.miner_id = 'miner_' || enp.miner_uid
             LEFT JOIN node_speedtest_profile esp ON me.node_id = esp.node_id AND me.miner_id = 'miner_' || esp.miner_uid
             WHERE me.node_id = ? AND me.miner_id = ?
             GROUP BY me.node_id,
                      ehp.cpu_model, ehp.cpu_cores, ehp.ram_gb,
                      enp.city, enp.region, enp.country,
                      esp.download_mbps, esp.upload_mbps, esp.test_timestamp
             LIMIT 1",
        )
        .bind(node_id)
        .bind(miner_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(row) = row {
            let node_id: String = row.get("node_id");

            let gpu_names: Option<String> = row.get("gpu_names");

            let mut gpu_specs: Vec<crate::api::types::GpuSpec> = vec![];

            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    for gpu_name in names.split(',') {
                        let memory_gb = extract_gpu_memory_gb(gpu_name);

                        gpu_specs.push(crate::api::types::GpuSpec {
                            name: gpu_name.to_string(),
                            memory_gb,
                            compute_capability: "8.0".to_string(),
                        });
                    }
                }
            }

            let hw_cpu_model: Option<String> = row.get("cpu_model");
            let hw_cpu_cores: Option<i32> = row.get("cpu_cores");
            let hw_ram_gb: Option<i32> = row.get("ram_gb");

            let net_city: Option<String> = row.get("city");
            let net_region: Option<String> = row.get("region");
            let net_country: Option<String> = row.get("country");

            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp: Option<String> = row.get("test_timestamp");

            let cpu_specs: crate::api::types::CpuSpec = crate::api::types::CpuSpec {
                cores: hw_cpu_cores.unwrap_or(0) as u32,
                model: hw_cpu_model.unwrap_or_else(|| "Unknown".to_string()),
                memory_gb: hw_ram_gb.unwrap_or(0) as u32,
            };

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

            Ok(Some(crate::api::types::NodeDetails {
                id: node_id,
                gpu_specs,
                cpu_specs,
                location: final_location,
                network_speed,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get the actual gpu_count for an node from gpu_uuid_assignments
    pub async fn get_node_gpu_count_from_assignments(
        &self,
        miner_id: &str,
        node_id: &str,
    ) -> Result<u32, anyhow::Error> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT gpu_uuid) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(node_id)
        .fetch_one(self.pool())
        .await?;

        Ok(count as u32)
    }

    /// Get the actual gpu_memory_gb for a specific GPU index of an node from gpu_uuid_assignments
    pub async fn get_node_gpu_memory_gb_by_index(
        &self,
        miner_id: &str,
        node_id: &str,
        index: u32,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND node_id = ? AND gpu_index = ?",
        )
        .bind(miner_id)
        .bind(node_id)
        .bind(index)
        .fetch_one(self.pool())
        .await?;

        Ok(memory)
    }

    /// Get the actual gpu_memory_gb for a specific GPU index of an node from gpu_uuid_assignments
    pub async fn get_node_gpu_memory_gb_by_gpu_uuid(
        &self,
        miner_id: &str,
        node_id: &str,
        gpu_uuid: &str,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND node_id = ? AND gpu_uuid = ?",
        )
        .bind(miner_id)
        .bind(node_id)
        .bind(gpu_uuid)
        .fetch_one(self.pool())
        .await?;

        Ok(memory)
    }

    /// Get the actual gpu_memory_gb for the first GPU (index 0) of an node from gpu_uuid_assignments
    pub async fn get_node_first_gpu_memory_gb(
        &self,
        miner_id: &str,
        node_id: &str,
    ) -> Result<f64, anyhow::Error> {
        let memory: f64 = sqlx::query_scalar(
            "SELECT COALESCE(gpu_memory_gb, 0.0) FROM gpu_uuid_assignments
             WHERE miner_id = ? AND node_id = ? AND gpu_index = 0",
        )
        .bind(miner_id)
        .bind(node_id)
        .fetch_one(self.pool())
        .await?;

        Ok(memory)
    }

    /// Get the GPU name/model for an node from gpu_uuid_assignments
    pub async fn get_node_gpu_name_from_assignments(
        &self,
        miner_id: &str,
        node_id: &str,
    ) -> Result<Option<String>, anyhow::Error> {
        let gpu_name: Option<String> = sqlx::query_scalar(
            "SELECT gpu_name FROM gpu_uuid_assignments
             WHERE miner_id = ? AND node_id = ?
             LIMIT 1",
        )
        .bind(miner_id)
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        Ok(gpu_name)
    }

    /// Get the actual gpu_count for all ONLINE nodes of a miner from gpu_uuid_assignments
    pub async fn get_miner_gpu_uuid_assignments(
        &self,
        miner_id: &str,
    ) -> Result<Vec<(String, u32, String, f64)>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT
                ga.node_id,
                COUNT(DISTINCT ga.gpu_uuid) as gpu_count,
                ga.gpu_name,
                MAX(ga.gpu_memory_gb) as gpu_memory_gb
             FROM gpu_uuid_assignments ga
             JOIN miner_nodes me ON ga.node_id = me.node_id AND ga.miner_id = me.miner_id
             WHERE ga.miner_id = ?
                AND me.status IN ('online', 'verified')
             GROUP BY ga.node_id, ga.gpu_name
             HAVING COUNT(DISTINCT ga.gpu_uuid) > 0",
        )
        .bind(miner_id)
        .fetch_all(self.pool())
        .await?;

        let mut results = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
            let gpu_count: i64 = row.get("gpu_count");
            let gpu_name: String = row.get("gpu_name");
            let gpu_memory_gb: f64 = row.get("gpu_memory_gb");

            results.push((node_id, gpu_count as u32, gpu_name, gpu_memory_gb));
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
             INNER JOIN miner_nodes me ON ga.node_id = me.node_id AND ga.miner_id = me.miner_id
             WHERE ga.miner_id = ?
                AND me.status IN ('online', 'verified')",
        )
        .bind(miner_id)
        .fetch_one(self.pool())
        .await?;

        Ok(count as u32)
    }

    /// Get known nodes from database for a miner
    pub async fn get_known_nodes_for_miner(
        &self,
        miner_uid: u16,
    ) -> Result<Vec<(String, String, i32, String)>, anyhow::Error> {
        let miner_id = format!("miner_{}", miner_uid);

        let query = r#"
            SELECT node_id, ssh_endpoint, gpu_count, status
            FROM miner_nodes
            WHERE miner_id = ?
            AND status IN ('online', 'verified')
            AND (last_health_check IS NULL OR last_health_check > datetime('now', '-1 hour'))
        "#;

        let rows = sqlx::query(query)
            .bind(&miner_id)
            .fetch_all(self.pool())
            .await?;

        let mut known_nodes = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
            let ssh_endpoint: String = row.get("ssh_endpoint");
            let gpu_count: i32 = row.get("gpu_count");
            let status: String = row.get("status");
            known_nodes.push((node_id, ssh_endpoint, gpu_count, status));
        }

        Ok(known_nodes)
    }
}
