//! Miner node relationship persistence operations
//!
//! This module contains all SQL operations related to miner-node relationships.

use crate::miner_prover::types::MinerInfo;
use crate::persistence::types::{AvailableNodeData, NodeData};
use crate::persistence::SimplePersistence;
use anyhow::Result;
use basilica_common::types::GpuCategory;
use chrono::{DateTime, Utc};
use sqlx::Row;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, info, warn};

/// A bid candidate sourced directly from `miner_nodes` (epoch-free).
#[derive(Debug, Clone)]
pub struct NodeBidCandidate {
    pub node_id: String,
    pub miner_id: String,
    pub miner_hotkey: String,
    pub miner_uid: i64,
    pub hourly_rate_cents: u32,
    pub gpu_count: i64,
}

/// Stored bid metadata for a registered node.
#[derive(Debug, Clone)]
pub struct RegisteredNodeBidMetadata {
    pub gpu_category: String,
    pub gpu_count: u32,
}

pub(crate) fn extract_gpu_memory_gb(gpu_name: &str) -> u32 {
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
        node_ip: &str,
        miner_info: &MinerInfo,
        hourly_rate_cents: u32,
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
            let relationship_id = format!("{miner_id}_{node_id}");
            let existing_miner: Option<String> = sqlx::query_scalar(
                "SELECT miner_id FROM miner_nodes WHERE node_ip = ? AND id != ? LIMIT 1",
            )
            .bind(node_ip)
            .bind(&relationship_id)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check node_ip uniqueness: {}", e))?;

            if let Some(other_miner) = existing_miner {
                return Err(anyhow::anyhow!(
                    "Cannot create node relationship: host {} is already registered to {}",
                    node_ip,
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
                    id, miner_id, node_id, ssh_endpoint, node_ip, gpu_count,
                    hourly_rate_cents, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            "#;

            sqlx::query(insert_query)
                .bind(&relationship_id)
                .bind(&miner_id)
                .bind(node_id)
                .bind(node_ssh_endpoint)
                .bind(node_ip)
                .bind(0)
                .bind(hourly_rate_cents as i64)
                .bind("online")
                .execute(self.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to insert miner-node relationship: {}", e))?;

            info!(
                miner_uid = miner_uid,
                node_id = node_id,
                hourly_rate_cents = hourly_rate_cents,
                "Created miner-node relationship: {} -> {} with endpoint {} and pricing {}¢/hour",
                miner_id,
                node_id,
                node_ssh_endpoint,
                hourly_rate_cents
            );
        } else {
            info!(
                miner_uid = miner_uid,
                node_id = node_id,
                hourly_rate_cents = hourly_rate_cents,
                "Miner-node relationship already exists: {} -> {}, will update pricing to {}¢/hour",
                miner_id,
                node_id,
                hourly_rate_cents
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
                        "Marking duplicate node {} (id: {}) as offline and bid-inactive with same ssh_endpoint as {} for miner {}",
                        dup_node_id, dup_id, node_id, miner_id
                    );

                    let duplicate_update = sqlx::query(
                        "UPDATE miner_nodes
                         SET status = 'offline', bid_active = 0
                         WHERE id = ? AND active_rental_id IS NULL",
                    )
                    .bind(&dup_id)
                    .execute(self.pool())
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to update duplicate node status: {}", e)
                    })?;

                    if duplicate_update.rows_affected() == 0 {
                        warn!(
                            miner_uid = miner_uid,
                            duplicate_node_id = %dup_node_id,
                            "Skipping duplicate node offlining because it has an active rental"
                        );
                        continue;
                    }

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

        // Update pricing for all nodes (new and existing) on every discovery
        let result = sqlx::query(
            "UPDATE miner_nodes
             SET hourly_rate_cents = ?
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(hourly_rate_cents as i64)
        .bind(&miner_id)
        .bind(node_id)
        .execute(self.pool())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to update node pricing: {}", e))?;

        let rows_affected = result.rows_affected();
        if rows_affected > 0 {
            info!(
                miner_uid = miner_uid,
                node_id = node_id,
                hourly_rate_cents = hourly_rate_cents,
                "Updated pricing for node {} to {}¢/hour",
                node_id,
                hourly_rate_cents
            );
        } else {
            warn!(
                miner_uid = miner_uid,
                node_id = node_id,
                "Pricing UPDATE affected 0 rows - node may not exist"
            );
        }

        Ok(())
    }

    /// Clean up nodes that have consecutive failed validations
    pub async fn cleanup_failed_nodes_after_failures(
        &self,
        consecutive_failures_threshold: i32,
        gpu_assignment_cleanup_ttl: Option<Duration>,
    ) -> Result<Vec<(String, String)>> {
        info!(
            "Running node cleanup - checking for {} consecutive failures",
            consecutive_failures_threshold
        );

        let mut removed_nodes: Vec<(String, String)> = Vec::new();

        let offline_with_gpus_query = r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            AND me.active_rental_id IS NULL
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

        // Mark nodes offline when validator node verification is stale (>30 minutes)
        let stale_health_check_result = sqlx::query(
            r#"
            UPDATE miner_nodes
            SET status = 'offline'
            WHERE status IN ('online', 'verified')
            AND active_rental_id IS NULL
            AND (
                last_node_check IS NULL
                OR datetime(last_node_check) < datetime('now', '-30 minutes')
            )
            "#,
        )
        .execute(self.pool())
        .await?;

        if stale_health_check_result.rows_affected() > 0 {
            info!(
                "Marked {} nodes offline due to stale validator node verification (>30 minutes)",
                stale_health_check_result.rows_affected()
            );
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
                    AND me.active_rental_id IS NULL
                    AND datetime(me.last_node_check) < datetime('now', '-2 hours')
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
            .unwrap_or(120);

        info!(
            "Cleaning GPU assignments from nodes offline >{} minutes",
            cleanup_minutes
        );

        let stale_offline_query = format!(
            r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            LEFT JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            AND me.active_rental_id IS NULL
            AND datetime(me.last_node_check) < datetime('now', '-{cleanup_minutes} minutes')
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
            AND me.active_rental_id IS NULL
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
            removed_nodes.push((miner_id.clone(), node_id.clone()));
        }

        let stale_delete_query = r#"
            DELETE FROM miner_nodes
            WHERE status = 'offline'
            AND active_rental_id IS NULL
            AND datetime(last_node_check) < datetime('now', '-30 minutes')
            "#;

        info!("Deleting stale offline nodes using fixed 30-minute timeout");

        let stale_nodes_query = r#"
            SELECT node_id, miner_id
            FROM miner_nodes
            WHERE status = 'offline'
            AND active_rental_id IS NULL
            AND datetime(last_node_check) < datetime('now', '-30 minutes')
            "#;

        let mut stale_tx = self.pool().begin().await?;

        let stale_rows = sqlx::query(stale_nodes_query)
            .fetch_all(&mut *stale_tx)
            .await?;

        let mut stale_pairs = Vec::with_capacity(stale_rows.len());
        for row in stale_rows {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            stale_pairs.push((miner_id, node_id));
        }

        let stale_result = sqlx::query(stale_delete_query)
            .execute(&mut *stale_tx)
            .await?;

        stale_tx.commit().await?;

        let stale_deleted = stale_result.rows_affected();

        removed_nodes.extend(stale_pairs);

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
                let category = GpuCategory::from_str(&gpu_name).unwrap();
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

        Ok(removed_nodes)
    }

    /// Get available nodes for rental (not currently rented)
    pub async fn get_available_nodes(
        &self,
        min_gpu_memory: Option<u32>,
        gpu_type: Option<String>,
        min_gpu_count: Option<u32>,
        location: Option<basilica_common::LocationProfile>,
    ) -> Result<Vec<AvailableNodeData>, anyhow::Error> {
        // A node is only rentable/visible once we have at least one validated GPU UUID assignment.
        let effective_min_gpu_count = std::cmp::max(min_gpu_count.unwrap_or(0), 1);
        let mut query_builder = sqlx::QueryBuilder::new(
            "SELECT
                me.node_id,
                me.miner_id,
                me.status,
                me.gpu_count,
                me.hourly_rate_cents,
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
            WHERE me.active_rental_id IS NULL
                AND me.bid_active = 1
                AND (me.status IS NULL OR me.status != 'offline')",
        );

        if let Some(ref loc) = location {
            if let Some(ref country) = loc.country {
                query_builder
                    .push(" AND LOWER(enp.country) = LOWER(")
                    .push_bind(country)
                    .push(")");
            }
            if let Some(ref region) = loc.region {
                query_builder
                    .push(" AND LOWER(enp.region) = LOWER(")
                    .push_bind(region)
                    .push(")");
            }
            if let Some(ref city) = loc.city {
                query_builder
                    .push(" AND LOWER(enp.city) = LOWER(")
                    .push_bind(city)
                    .push(")");
            }
        }

        query_builder.push(" GROUP BY me.node_id");

        query_builder
            .push(" HAVING COUNT(DISTINCT gua.gpu_uuid) >= ")
            .push_bind(effective_min_gpu_count);

        // TODO: Consider adding a functional index for LOWER(enp.country/region/city) if this query becomes hot.
        let rows = query_builder.build().fetch_all(self.pool()).await?;

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

            let hourly_rate_cents: Option<i64> = row.get("hourly_rate_cents");

            nodes.push(AvailableNodeData {
                node_id: row.get("node_id"),
                miner_id: row.get("miner_id"),
                gpu_specs,
                cpu_specs,
                location,
                status: row.get("status"),
                download_mbps,
                upload_mbps,
                speed_test_timestamp,
                hourly_rate_cents: hourly_rate_cents.map(|v| v as u32),
            });
        }

        Ok(nodes)
    }

    /// Get available nodes for a specific miner, filtered by GPU type/count.
    pub async fn get_available_nodes_for_miner(
        &self,
        miner_id: &str,
        gpu_category: &str,
        min_gpu_count: u32,
    ) -> Result<Vec<String>, anyhow::Error> {
        let query = r#"
            SELECT
                me.node_id,
                GROUP_CONCAT(gua.gpu_name) as gpu_names
            FROM miner_nodes me
            LEFT JOIN gpu_uuid_assignments gua ON me.node_id = gua.node_id AND gua.miner_id = me.miner_id
            WHERE me.active_rental_id IS NULL
                AND me.miner_id = ?
                AND (me.status IS NULL OR me.status != 'offline')
                AND me.bid_active = 1
            GROUP BY me.node_id
            HAVING COUNT(gua.gpu_uuid) >= ?
            "#;

        let rows = sqlx::query(query)
            .bind(miner_id)
            .bind(min_gpu_count as i64)
            .fetch_all(self.pool())
            .await?;

        let mut nodes = Vec::new();
        for row in rows {
            let gpu_names: Option<String> = row.get("gpu_names");
            if let Some(names) = gpu_names {
                if !names.is_empty() {
                    let matches_type = names
                        .split(',')
                        .any(|name| name.to_lowercase().contains(&gpu_category.to_lowercase()));
                    if !matches_type {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }

            let node_id: String = row.get("node_id");
            nodes.push(node_id);
        }

        Ok(nodes)
    }

    /// Get bid candidates across all miners, filtered by GPU category/count.
    /// Returns nodes ordered by hourly_rate_cents ASC (cheapest first).
    pub async fn get_node_bid_candidates(
        &self,
        gpu_category: &str,
        min_gpu_count: u32,
        max_hourly_rate_cents: u32,
        limit: u32,
    ) -> Result<Vec<NodeBidCandidate>, anyhow::Error> {
        let query = r#"
            SELECT
                me.node_id,
                me.miner_id,
                m.hotkey  AS miner_hotkey,
                CAST(REPLACE(m.id, 'miner_', '') AS INTEGER) AS miner_uid,
                me.hourly_rate_cents,
                COUNT(DISTINCT gua.gpu_uuid) AS gpu_count,
                GROUP_CONCAT(gua.gpu_name) AS gpu_names
            FROM miner_nodes me
            JOIN miners m ON me.miner_id = m.id
            LEFT JOIN gpu_uuid_assignments gua
                ON me.node_id = gua.node_id AND gua.miner_id = me.miner_id
            WHERE me.active_rental_id IS NULL
                AND (me.status IS NULL OR me.status != 'offline')
                AND me.bid_active = 1
                AND me.hourly_rate_cents IS NOT NULL
                AND me.hourly_rate_cents <= ?
            GROUP BY me.node_id, me.miner_id, m.hotkey, m.id, me.hourly_rate_cents
            HAVING COUNT(DISTINCT gua.gpu_uuid) >= ?
            ORDER BY me.hourly_rate_cents ASC
            LIMIT ?
            "#;

        let rows = sqlx::query(query)
            .bind(max_hourly_rate_cents as i64)
            .bind(min_gpu_count as i64)
            .bind(limit as i64)
            .fetch_all(self.pool())
            .await?;

        let mut candidates = Vec::new();
        for row in rows {
            let gpu_names: Option<String> = row.get("gpu_names");
            if let Some(names) = &gpu_names {
                if !names.is_empty() {
                    let matches_type = names
                        .split(',')
                        .any(|name| name.to_lowercase().contains(&gpu_category.to_lowercase()));
                    if !matches_type {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }

            candidates.push(NodeBidCandidate {
                node_id: row.get("node_id"),
                miner_id: row.get("miner_id"),
                miner_hotkey: row.get("miner_hotkey"),
                miner_uid: row.get("miner_uid"),
                hourly_rate_cents: row.get::<i64, _>("hourly_rate_cents") as u32,
                gpu_count: row.get("gpu_count"),
            });
        }

        Ok(candidates)
    }

    // =========================================================================
    // Node Claim/Release (active_rental_id lifecycle)
    // =========================================================================
    //
    // The `active_rental_id` column on `miner_nodes` tracks whether a node is
    // currently claimed for a rental. When NULL the node is available; when set
    // it holds the rental_id that owns the node.
    //
    // INVARIANT: active_rental_id MUST be cleared on EVERY termination path:
    //   1. stop_rental()                    -- user stops rental
    //   2. deploy_container failure         -- container deploy fails
    //   3. finalize_rental failure          -- DB save fails after deploy
    //   4. ensure_not_banned failure        -- node is banned
    //   5. require_ssh_endpoint failure     -- SSH endpoint missing
    //   6. require_node_details failure     -- node details missing
    //   7. ensure_recent_validation failure -- validation too old
    //   8. health check timeout             -- monitoring.rs
    //   9. health check error               -- monitoring.rs
    //  10. container unhealthy              -- monitoring.rs
    //  11. restart failure                  -- restart_rental() fails
    //
    // There is NO TTL. If release_node is not called, the node stays claimed
    // forever. This is by design to avoid silent data corruption.

    /// Atomically claim a node for a rental by setting active_rental_id.
    /// Returns true if the claim succeeded (node was available), false if
    /// another process already claimed it.
    ///
    /// The WHERE clause `active_rental_id IS NULL` guarantees that two
    /// concurrent callers cannot both succeed — SQLite serialises writes,
    /// so at most one UPDATE will find the row still NULL.
    pub async fn claim_node(
        &self,
        node_id: &str,
        miner_id: &str,
        rental_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let result = sqlx::query(
            r#"
            UPDATE miner_nodes
            SET active_rental_id = ?
            WHERE node_id = ? AND miner_id = ?
              AND active_rental_id IS NULL
            "#,
        )
        .bind(rental_id)
        .bind(node_id)
        .bind(miner_id)
        .execute(self.pool())
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Release a node claim by clearing active_rental_id.
    /// Only clears if the current active_rental_id matches the given rental_id,
    /// preventing one process from accidentally clearing another's claim.
    ///
    /// See the INVARIANT comment above for the full list of paths that must
    /// call this function.
    pub async fn release_node(
        &self,
        node_id: &str,
        miner_id: &str,
        rental_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let result = sqlx::query(
            r#"
            UPDATE miner_nodes
            SET active_rental_id = NULL
            WHERE node_id = ? AND miner_id = ?
              AND active_rental_id = ?
            "#,
        )
        .bind(node_id)
        .bind(miner_id)
        .bind(rental_id)
        .execute(self.pool())
        .await?;

        Ok(result.rows_affected() > 0)
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
                me.hourly_rate_cents,
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
                      me.hourly_rate_cents,
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

            let hourly_rate_cents: Option<i64> = row.get("hourly_rate_cents");

            Ok(Some(crate::api::types::NodeDetails {
                id: node_id,
                gpu_specs,
                cpu_specs,
                location: final_location,
                network_speed,
                hourly_rate_cents: hourly_rate_cents.map(|v| v as i32),
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
    ) -> Result<Vec<(String, String, String, i32, String, u32)>, anyhow::Error> {
        let miner_id = format!("miner_{}", miner_uid);

        let query = r#"
            SELECT node_id, ssh_endpoint, node_ip, gpu_count, status, hourly_rate_cents
            FROM miner_nodes
            WHERE miner_id = ?
            AND (bid_active = 1 OR active_rental_id IS NOT NULL)
        "#;

        let rows = sqlx::query(query)
            .bind(&miner_id)
            .fetch_all(self.pool())
            .await?;

        let mut known_nodes = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
            let ssh_endpoint: String = row.get("ssh_endpoint");
            let node_ip: String = row.get("node_ip");
            let gpu_count: i32 = row.get("gpu_count");
            let status: String = row.get("status");
            let hourly_rate_cents: i64 = row.try_get("hourly_rate_cents").unwrap_or(0);
            known_nodes.push((
                node_id,
                ssh_endpoint,
                node_ip,
                gpu_count,
                status,
                hourly_rate_cents as u32,
            ));
        }

        Ok(known_nodes)
    }

    /// Get hourly rate for a specific node
    pub async fn get_node_hourly_rate(&self, node_id: &str) -> Result<Option<u32>> {
        let rate_cents: Option<i64> = sqlx::query_scalar(
            "SELECT hourly_rate_cents FROM miner_nodes WHERE node_id = ? LIMIT 1",
        )
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        Ok(rate_cents.map(|v| v as u32))
    }

    /// Get bid metadata for a specific node owned by a miner.
    pub async fn get_node_bid_metadata(
        &self,
        miner_id: &str,
        node_id: &str,
    ) -> Result<Option<RegisteredNodeBidMetadata>> {
        let row = sqlx::query(
            r#"
            SELECT gpu_category, gpu_count
            FROM miner_nodes
            WHERE miner_id = ? AND node_id = ?
            LIMIT 1
            "#,
        )
        .bind(miner_id)
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        let Some(row) = row else {
            return Ok(None);
        };

        let gpu_category: Option<String> = row.try_get("gpu_category")?;
        let gpu_count: i64 = row.get("gpu_count");

        Ok(Some(RegisteredNodeBidMetadata {
            gpu_category: gpu_category.unwrap_or_default(),
            gpu_count: gpu_count as u32,
        }))
    }

    // =========================================================================
    // Miner Registration (miner→validator flow) methods
    // =========================================================================

    /// Upsert a node from RegisterBid request.
    /// Creates the node if it doesn't exist, updates it if it does.
    /// Returns true if the node was created (new), false if updated (existing).
    /// NOTE: Existing rows must not update validator-controlled liveness fields
    /// (`status`, `last_node_check`). Those are owned by validator SSH checks.
    /// New rows start as `online` with `last_node_check = now` at registration time.
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_registered_node(
        &self,
        miner_id: &str,
        host: &str,
        port: u32,
        username: &str,
        gpu_category: &str,
        gpu_count: u32,
        hourly_rate_cents: u32,
    ) -> Result<bool> {
        // Compute node_id deterministically from host (validator-side, not trusting miner)
        let node_id = basilica_common::node_identity::NodeId::new(host)?
            .uuid
            .to_string();

        // Build ssh_endpoint in the standard format: user@host:port
        let ssh_endpoint = format!("{}@{}:{}", username, host, port);
        let relationship_id = format!("{}_{}", miner_id, node_id);

        // Check if this host is already used by any other node
        let existing_miner: Option<String> = sqlx::query_scalar(
            "SELECT miner_id FROM miner_nodes WHERE node_ip = ? AND id != ? LIMIT 1",
        )
        .bind(host)
        .bind(&relationship_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(other_miner) = existing_miner {
            anyhow::bail!("Host {} is already registered to {}", host, other_miner);
        }

        // Check if this node already exists for this miner
        let existing_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM miner_nodes WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(&node_id)
        .fetch_one(self.pool())
        .await?;

        if existing_count == 0 {
            // Insert new node
            sqlx::query(
                r#"
                INSERT INTO miner_nodes (
                    id, miner_id, node_id, ssh_endpoint, node_ip, gpu_count,
                    hourly_rate_cents, gpu_category, status, bid_active, last_node_check, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'online', 1, datetime('now'), datetime('now'))
                "#,
            )
            .bind(&relationship_id)
            .bind(miner_id)
            .bind(&node_id)
            .bind(&ssh_endpoint)
            .bind(host)
            .bind(gpu_count as i64)
            .bind(hourly_rate_cents as i64)
            .bind(gpu_category)
            .execute(self.pool())
            .await?;

            info!(
                miner_id = miner_id,
                node_id = node_id,
                ssh_endpoint = ssh_endpoint,
                gpu_category = gpu_category,
                gpu_count = gpu_count,
                hourly_rate_cents = hourly_rate_cents,
                "Registered new node via RegisterBid"
            );
            Ok(true)
        } else {
            // Update existing node
            sqlx::query(
                r#"
                UPDATE miner_nodes
                SET ssh_endpoint = ?,
                    node_ip = ?,
                    gpu_count = ?,
                    hourly_rate_cents = ?,
                    gpu_category = ?,
                    bid_active = 1
                WHERE miner_id = ? AND node_id = ?
                "#,
            )
            .bind(&ssh_endpoint)
            .bind(host)
            .bind(gpu_count as i64)
            .bind(hourly_rate_cents as i64)
            .bind(gpu_category)
            .bind(miner_id)
            .bind(&node_id)
            .execute(self.pool())
            .await?;

            info!(
                miner_id = miner_id,
                node_id = node_id,
                hourly_rate_cents = hourly_rate_cents,
                "Updated existing node via RegisterBid"
            );
            Ok(false)
        }
    }

    /// Update miner heartbeat timestamp for nodes (HealthCheck RPC).
    /// If node_ids is empty, updates all nodes for the miner.
    /// Returns the number of nodes updated.
    pub async fn update_nodes_health_check(
        &self,
        miner_id: &str,
        node_ids: &[String],
    ) -> Result<u32> {
        let result = if node_ids.is_empty() {
            // Update all nodes for this miner
            sqlx::query(
                r#"
                UPDATE miner_nodes
                SET last_miner_health_check = datetime('now')
                WHERE miner_id = ?
                "#,
            )
            .bind(miner_id)
            .execute(self.pool())
            .await?
        } else {
            // Build query with IN clause for specific node_ids
            // Using a transaction to update each node
            let mut count = 0u64;
            for node_id in node_ids {
                let r = sqlx::query(
                    r#"
                    UPDATE miner_nodes
                    SET last_miner_health_check = datetime('now')
                    WHERE miner_id = ? AND node_id = ?
                    "#,
                )
                .bind(miner_id)
                .bind(node_id)
                .execute(self.pool())
                .await?;
                count += r.rows_affected();
            }
            return Ok(count as u32);
        };

        Ok(result.rows_affected() as u32)
    }

    /// Update hourly rate for a specific node (UpdateBid RPC).
    /// Returns true if the node was found and updated.
    pub async fn update_node_hourly_rate(
        &self,
        miner_id: &str,
        node_id: &str,
        hourly_rate_cents: u32,
    ) -> Result<bool> {
        let result = sqlx::query(
            r#"
            UPDATE miner_nodes
            SET hourly_rate_cents = ?
            WHERE miner_id = ? AND node_id = ?
            "#,
        )
        .bind(hourly_rate_cents as i64)
        .bind(miner_id)
        .bind(node_id)
        .execute(self.pool())
        .await?;

        if result.rows_affected() > 0 {
            info!(
                miner_id = miner_id,
                node_id = node_id,
                hourly_rate_cents = hourly_rate_cents,
                "Updated node price via UpdateBid"
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove nodes from availability (RemoveBid RPC).
    /// If node_ids is empty, removes all nodes for the miner.
    /// Returns the number of nodes removed (marked offline).
    pub async fn remove_registered_nodes(
        &self,
        miner_id: &str,
        node_ids: &[String],
    ) -> Result<u32> {
        let result = if node_ids.is_empty() {
            // Mark all nodes for this miner as offline and bid-inactive
            sqlx::query(
                r#"
                UPDATE miner_nodes
                SET status = 'offline',
                    bid_active = 0
                WHERE miner_id = ?
                "#,
            )
            .bind(miner_id)
            .execute(self.pool())
            .await?
        } else {
            // Mark specific nodes as offline and bid-inactive
            let mut count = 0u64;
            for node_id in node_ids {
                let r = sqlx::query(
                    r#"
                    UPDATE miner_nodes
                    SET status = 'offline',
                        bid_active = 0
                    WHERE miner_id = ? AND node_id = ?
                    "#,
                )
                .bind(miner_id)
                .bind(node_id)
                .execute(self.pool())
                .await?;
                count += r.rows_affected();
            }
            return Ok(count as u32);
        };

        let removed = result.rows_affected() as u32;
        if removed > 0 {
            info!(
                miner_id = miner_id,
                nodes_removed = removed,
                "Removed nodes via RemoveBid"
            );
        }
        Ok(removed)
    }

    /// Check if a miner has any registered nodes (regardless of health check status).
    pub async fn miner_has_registered_nodes(&self, miner_id: &str) -> Result<bool> {
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM miner_nodes WHERE miner_id = ?")
            .bind(miner_id)
            .fetch_one(self.pool())
            .await?;
        Ok(count > 0)
    }

    /// Deactivate bids for nodes not included in the latest RegisterBid.
    /// Sets bid_active = false for nodes belonging to this miner that are
    /// NOT in the provided active_node_ids list.
    /// Active rentals are NOT affected — only future bid eligibility changes.
    pub async fn deactivate_missing_bids(
        &self,
        miner_id: &str,
        active_node_ids: &[String],
    ) -> Result<u32> {
        if active_node_ids.is_empty() {
            // No nodes in the request — deactivate all for this miner
            let result = sqlx::query(
                r#"
                UPDATE miner_nodes
                SET bid_active = 0
                WHERE miner_id = ?
                "#,
            )
            .bind(miner_id)
            .execute(self.pool())
            .await?;

            let deactivated = result.rows_affected() as u32;
            if deactivated > 0 {
                info!(
                    miner_id = miner_id,
                    deactivated = deactivated,
                    "Deactivated all bids for miner (empty node list)"
                );
            }
            return Ok(deactivated);
        }

        // Deactivate nodes NOT in the active list
        // Build a parameterized NOT IN query
        let placeholders: Vec<&str> = active_node_ids.iter().map(|_| "?").collect();
        let in_clause = placeholders.join(", ");
        let query = format!(
            r#"
            UPDATE miner_nodes
            SET bid_active = 0
            WHERE miner_id = ? AND node_id NOT IN ({})
            "#,
            in_clause
        );

        let mut q = sqlx::query(&query).bind(miner_id);
        for node_id in active_node_ids {
            q = q.bind(node_id);
        }
        let result = q.execute(self.pool()).await?;
        let count = result.rows_affected();

        if count > 0 {
            info!(
                miner_id = miner_id,
                deactivated = count,
                active_count = active_node_ids.len(),
                "Deactivated bids for nodes not in RegisterBid"
            );
        }

        Ok(count as u32)
    }
}

#[cfg(test)]
mod tests {
    use crate::persistence::SimplePersistence;
    use sqlx::Row;
    use std::collections::HashSet;

    async fn create_test_persistence() -> SimplePersistence {
        SimplePersistence::for_testing()
            .await
            .expect("failed to create test persistence")
    }

    async fn insert_test_miner(persistence: &SimplePersistence, miner_id: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO miners (id, hotkey, endpoint, updated_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(miner_id)
        .bind(format!("hotkey_{miner_id}"))
        .bind("127.0.0.1:9090")
        .bind(&now)
        .execute(persistence.pool())
        .await
        .expect("failed to insert miner");
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_test_node(
        persistence: &SimplePersistence,
        miner_id: &str,
        node_id: &str,
        node_ip: &str,
        status: &str,
        bid_active: i64,
        last_node_check: Option<&str>,
        last_miner_health_check: Option<&str>,
        active_rental_id: Option<&str>,
        hourly_rate_cents: i64,
    ) {
        let relationship_id = format!("{miner_id}_{node_id}");
        let ssh_endpoint = format!("root@{node_ip}:22");
        sqlx::query(
            "INSERT INTO miner_nodes (
                id, miner_id, node_id, ssh_endpoint, node_ip, gpu_count, hourly_rate_cents,
                status, bid_active, last_node_check, last_miner_health_check, active_rental_id, created_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&relationship_id)
        .bind(miner_id)
        .bind(node_id)
        .bind(&ssh_endpoint)
        .bind(node_ip)
        .bind(1i64)
        .bind(hourly_rate_cents)
        .bind(status)
        .bind(bid_active)
        .bind(last_node_check)
        .bind(last_miner_health_check)
        .bind(active_rental_id)
        .execute(persistence.pool())
        .await
        .expect("failed to insert node");
    }

    async fn insert_gpu_assignment(
        persistence: &SimplePersistence,
        miner_id: &str,
        node_id: &str,
        gpu_uuid: &str,
        gpu_index: i64,
        gpu_name: &str,
    ) {
        sqlx::query(
            "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, last_verified)
             VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(gpu_uuid)
        .bind(gpu_index)
        .bind(node_id)
        .bind(miner_id)
        .bind(gpu_name)
        .execute(persistence.pool())
        .await
        .expect("failed to insert gpu assignment");
    }

    #[tokio::test]
    async fn migration_has_split_node_and_miner_health_columns() {
        let persistence = create_test_persistence().await;

        let column_rows = sqlx::query("PRAGMA table_info('miner_nodes')")
            .fetch_all(persistence.pool())
            .await
            .expect("failed to read table_info");
        let columns: HashSet<String> = column_rows
            .into_iter()
            .map(|row| row.get::<String, _>("name"))
            .collect();

        assert!(columns.contains("last_node_check"));
        assert!(columns.contains("last_miner_health_check"));
        assert!(columns.contains("gpu_category"));
        assert!(!columns.contains("last_health_check"));

        let index_rows = sqlx::query("PRAGMA index_list('miner_nodes')")
            .fetch_all(persistence.pool())
            .await
            .expect("failed to read index_list");
        let indexes: HashSet<String> = index_rows
            .into_iter()
            .map(|row| row.get::<String, _>("name"))
            .collect();

        assert!(indexes.contains("idx_miner_nodes_node_check"));
        assert!(!indexes.contains("idx_miner_nodes_health_check"));
    }

    #[tokio::test]
    async fn update_nodes_health_check_only_updates_miner_heartbeat() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let node_id = "node_1";
        let original_node_check = "2020-01-01 00:00:00";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            node_id,
            "10.0.0.1",
            "offline",
            1,
            Some(original_node_check),
            None,
            None,
            1000,
        )
        .await;

        let updated = persistence
            .update_nodes_health_check(miner_id, &[node_id.to_string()])
            .await
            .expect("health check update should succeed");
        assert_eq!(updated, 1);

        let row = sqlx::query(
            "SELECT status, last_node_check, last_miner_health_check
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(node_id)
        .fetch_one(persistence.pool())
        .await
        .expect("failed to read node");

        let status: String = row.get("status");
        let last_node_check: Option<String> = row.get("last_node_check");
        let last_miner_health_check: Option<String> = row.get("last_miner_health_check");

        assert_eq!(status, "offline");
        assert_eq!(last_node_check.as_deref(), Some(original_node_check));
        let heartbeat = last_miner_health_check.expect("last_miner_health_check should be set");
        assert!(!heartbeat.contains('T'));
        chrono::NaiveDateTime::parse_from_str(&heartbeat, "%Y-%m-%d %H:%M:%S")
            .expect("heartbeat should use SQLite datetime format");
    }

    #[tokio::test]
    async fn get_known_nodes_for_miner_uses_bid_or_active_rental_only() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_bid_active",
            "10.0.0.2",
            "offline",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_rented",
            "10.0.0.3",
            "offline",
            0,
            Some("2000-01-01 00:00:00"),
            None,
            Some("rental_1"),
            1000,
        )
        .await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_filtered",
            "10.0.0.4",
            "online",
            0,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;

        let known_nodes = persistence
            .get_known_nodes_for_miner(1)
            .await
            .expect("query should succeed");

        let ids: HashSet<String> = known_nodes.into_iter().map(|entry| entry.0).collect();
        assert!(ids.contains("node_bid_active"));
        assert!(ids.contains("node_rented"));
        assert!(!ids.contains("node_filtered"));
    }

    #[tokio::test]
    async fn get_node_bid_candidates_does_not_filter_on_last_node_check_freshness() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let node_id = "node_stale_but_bid_active";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            node_id,
            "10.0.0.5",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1200,
        )
        .await;

        sqlx::query(
            "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, last_verified)
             VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind("gpu-stale-1")
        .bind(0i64)
        .bind(node_id)
        .bind(miner_id)
        .bind("NVIDIA A100")
        .execute(persistence.pool())
        .await
        .expect("failed to insert gpu assignment");

        let candidates = persistence
            .get_node_bid_candidates("A100", 1, 2000, 10)
            .await
            .expect("candidate query should succeed");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].node_id, node_id);
    }

    #[tokio::test]
    async fn get_available_nodes_requires_validated_gpu_assignments_and_bid_active() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";

        insert_test_miner(&persistence, miner_id).await;

        insert_test_node(
            &persistence,
            miner_id,
            "node_no_gpu_assignments",
            "10.0.1.1",
            "online",
            1,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;

        insert_test_node(
            &persistence,
            miner_id,
            "node_visible_single_gpu",
            "10.0.1.2",
            "online",
            1,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            1200,
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            "node_visible_single_gpu",
            "gpu-visible-1",
            0,
            "NVIDIA A100",
        )
        .await;

        insert_test_node(
            &persistence,
            miner_id,
            "node_bid_inactive",
            "10.0.1.3",
            "online",
            0,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            900,
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            "node_bid_inactive",
            "gpu-inactive-1",
            0,
            "NVIDIA A100",
        )
        .await;

        insert_test_node(
            &persistence,
            miner_id,
            "node_offline_with_gpu",
            "10.0.1.4",
            "offline",
            1,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            900,
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            "node_offline_with_gpu",
            "gpu-offline-1",
            0,
            "NVIDIA A100",
        )
        .await;

        insert_test_node(
            &persistence,
            miner_id,
            "node_visible_two_gpus",
            "10.0.1.5",
            "online",
            1,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            1800,
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            "node_visible_two_gpus",
            "gpu-visible-2a",
            0,
            "NVIDIA H100",
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            "node_visible_two_gpus",
            "gpu-visible-2b",
            1,
            "NVIDIA H100",
        )
        .await;

        let ids_default: HashSet<String> = persistence
            .get_available_nodes(None, None, None, None)
            .await
            .expect("query should succeed")
            .into_iter()
            .map(|n| n.node_id)
            .collect();
        assert!(ids_default.contains("node_visible_single_gpu"));
        assert!(ids_default.contains("node_visible_two_gpus"));
        assert!(!ids_default.contains("node_no_gpu_assignments"));
        assert!(!ids_default.contains("node_bid_inactive"));
        assert!(!ids_default.contains("node_offline_with_gpu"));

        let ids_min_zero: HashSet<String> = persistence
            .get_available_nodes(None, None, Some(0), None)
            .await
            .expect("query should succeed")
            .into_iter()
            .map(|n| n.node_id)
            .collect();
        assert_eq!(ids_min_zero, ids_default);

        let ids_min_two: HashSet<String> = persistence
            .get_available_nodes(None, None, Some(2), None)
            .await
            .expect("query should succeed")
            .into_iter()
            .map(|n| n.node_id)
            .collect();
        assert_eq!(ids_min_two.len(), 1);
        assert!(ids_min_two.contains("node_visible_two_gpus"));
    }

    #[tokio::test]
    async fn get_available_nodes_hides_node_after_gpu_assignment_cleanup() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let node_id = "node_cleanup_target";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            node_id,
            "10.0.2.1",
            "online",
            1,
            Some("2025-01-01 00:00:00"),
            None,
            None,
            1300,
        )
        .await;
        insert_gpu_assignment(
            &persistence,
            miner_id,
            node_id,
            "gpu-cleanup-1",
            0,
            "NVIDIA A100",
        )
        .await;

        let before: HashSet<String> = persistence
            .get_available_nodes(None, None, None, None)
            .await
            .expect("query should succeed")
            .into_iter()
            .map(|n| n.node_id)
            .collect();
        assert!(before.contains(node_id));

        let cleaned = persistence
            .cleanup_gpu_assignments(node_id, miner_id, None)
            .await
            .expect("cleanup should succeed");
        assert_eq!(cleaned, 1);

        let after: HashSet<String> = persistence
            .get_available_nodes(None, None, None, None)
            .await
            .expect("query should succeed")
            .into_iter()
            .map(|n| n.node_id)
            .collect();
        assert!(!after.contains(node_id));
    }

    #[tokio::test]
    async fn cleanup_failed_nodes_keeps_active_rental_nodes_safe() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_online_rented",
            "10.0.0.6",
            "online",
            0,
            Some("2000-01-01 00:00:00"),
            None,
            Some("rental_online"),
            1000,
        )
        .await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_offline_rented",
            "10.0.0.7",
            "offline",
            0,
            Some("2000-01-01 00:00:00"),
            None,
            Some("rental_offline"),
            1000,
        )
        .await;

        sqlx::query(
            "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, last_verified)
             VALUES (?, ?, ?, ?, ?, datetime('now', '-3 hours'))",
        )
        .bind("gpu-rented-1")
        .bind(0i64)
        .bind("node_offline_rented")
        .bind(miner_id)
        .bind("NVIDIA A100")
        .execute(persistence.pool())
        .await
        .expect("failed to insert gpu assignment");

        let removed = persistence
            .cleanup_failed_nodes_after_failures(2, None)
            .await
            .expect("cleanup should succeed");
        assert!(removed.is_empty());

        let online_status: String = sqlx::query_scalar(
            "SELECT status FROM miner_nodes WHERE miner_id = ? AND node_id = 'node_online_rented'",
        )
        .bind(miner_id)
        .fetch_one(persistence.pool())
        .await
        .expect("missing online rented node");
        assert_eq!(online_status, "online");

        let offline_exists: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM miner_nodes WHERE miner_id = ? AND node_id = 'node_offline_rented'",
        )
        .bind(miner_id)
        .fetch_one(persistence.pool())
        .await
        .expect("failed to count offline rented node");
        assert_eq!(offline_exists, 1);

        let rented_gpu_exists: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM gpu_uuid_assignments WHERE miner_id = ? AND node_id = 'node_offline_rented'",
        )
        .bind(miner_id)
        .fetch_one(persistence.pool())
        .await
        .expect("failed to count rented gpu assignments");
        assert_eq!(rented_gpu_exists, 1);
    }

    #[tokio::test]
    async fn upsert_registered_node_persists_bid_metadata() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let host = "10.0.1.10";

        insert_test_miner(&persistence, miner_id).await;

        let created = persistence
            .upsert_registered_node(miner_id, host, 22, "root", "A100", 4, 1200)
            .await
            .expect("upsert should succeed");
        assert!(created);

        let metadata = persistence
            .get_node_bid_metadata(
                miner_id,
                &basilica_common::node_identity::NodeId::new(host)
                    .expect("valid host")
                    .uuid
                    .to_string(),
            )
            .await
            .expect("metadata query should succeed")
            .expect("metadata should exist");
        assert_eq!(metadata.gpu_category, "A100");
        assert_eq!(metadata.gpu_count, 4);

        let initial_liveness_row = sqlx::query(
            "SELECT status, last_node_check, last_miner_health_check
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(
            basilica_common::node_identity::NodeId::new(host)
                .expect("valid host")
                .uuid
                .to_string(),
        )
        .fetch_one(persistence.pool())
        .await
        .expect("failed to read node liveness fields");
        let initial_status: String = initial_liveness_row.get("status");
        let initial_last_node_check: Option<String> = initial_liveness_row.get("last_node_check");
        let initial_last_miner_health_check: Option<String> =
            initial_liveness_row.get("last_miner_health_check");
        assert_eq!(initial_status, "online");
        assert!(initial_last_node_check.is_some());
        assert!(initial_last_miner_health_check.is_none());

        let updated = persistence
            .upsert_registered_node(miner_id, host, 22, "root", "H100", 8, 2200)
            .await
            .expect("upsert update should succeed");
        assert!(!updated);

        let metadata_after = persistence
            .get_node_bid_metadata(
                miner_id,
                &basilica_common::node_identity::NodeId::new(host)
                    .expect("valid host")
                    .uuid
                    .to_string(),
            )
            .await
            .expect("metadata query should succeed")
            .expect("metadata should exist");
        assert_eq!(metadata_after.gpu_category, "H100");
        assert_eq!(metadata_after.gpu_count, 8);

        let updated_liveness_row = sqlx::query(
            "SELECT status, last_node_check, last_miner_health_check
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(
            basilica_common::node_identity::NodeId::new(host)
                .expect("valid host")
                .uuid
                .to_string(),
        )
        .fetch_one(persistence.pool())
        .await
        .expect("failed to read node liveness fields");
        let updated_status: String = updated_liveness_row.get("status");
        let updated_last_node_check: Option<String> = updated_liveness_row.get("last_node_check");
        let updated_last_miner_health_check: Option<String> =
            updated_liveness_row.get("last_miner_health_check");
        assert_eq!(updated_status, initial_status);
        assert_eq!(updated_last_node_check, initial_last_node_check);
        assert_eq!(
            updated_last_miner_health_check,
            initial_last_miner_health_check
        );
    }

    #[tokio::test]
    async fn upsert_registered_node_new_nodes_start_online_with_node_check_timestamp() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let host = "10.0.1.12";
        let node_id = basilica_common::node_identity::NodeId::new(host)
            .expect("valid host")
            .uuid
            .to_string();

        insert_test_miner(&persistence, miner_id).await;

        let created = persistence
            .upsert_registered_node(miner_id, host, 22, "root", "A100", 4, 1200)
            .await
            .expect("upsert should succeed");
        assert!(created);

        let row = sqlx::query(
            "SELECT status, bid_active, last_node_check, last_miner_health_check
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(&node_id)
        .fetch_one(persistence.pool())
        .await
        .expect("failed to read inserted node");

        let status: String = row.get("status");
        let bid_active: i64 = row.get("bid_active");
        let last_node_check: Option<String> = row.get("last_node_check");
        let last_miner_health_check: Option<String> = row.get("last_miner_health_check");

        assert_eq!(status, "online");
        assert_eq!(bid_active, 1);
        assert!(last_node_check.is_some());
        assert!(last_miner_health_check.is_none());
    }

    #[tokio::test]
    async fn upsert_registered_node_preserves_existing_validator_liveness() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let host = "10.0.1.11";
        let existing_node_id = basilica_common::node_identity::NodeId::new(host)
            .expect("valid host")
            .uuid
            .to_string();
        let existing_last_node_check = "2001-01-01 00:00:00";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            &existing_node_id,
            host,
            "offline",
            0,
            Some(existing_last_node_check),
            Some("2001-01-01 00:05:00"),
            None,
            1000,
        )
        .await;

        let updated = persistence
            .upsert_registered_node(miner_id, host, 2222, "root", "H100", 8, 2200)
            .await
            .expect("upsert update should succeed");
        assert!(!updated);

        let row = sqlx::query(
            "SELECT status, bid_active, last_node_check, ssh_endpoint, hourly_rate_cents, gpu_category, gpu_count
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(&existing_node_id)
        .fetch_one(persistence.pool())
        .await
        .expect("failed to read updated node");

        let status: String = row.get("status");
        let bid_active: i64 = row.get("bid_active");
        let last_node_check: Option<String> = row.get("last_node_check");
        let ssh_endpoint: String = row.get("ssh_endpoint");
        let hourly_rate_cents: i64 = row.get("hourly_rate_cents");
        let gpu_category: Option<String> = row.get("gpu_category");
        let gpu_count: i64 = row.get("gpu_count");

        assert_eq!(status, "offline");
        assert_eq!(bid_active, 1);
        assert_eq!(last_node_check.as_deref(), Some(existing_last_node_check));
        assert_eq!(ssh_endpoint, "root@10.0.1.11:2222");
        assert_eq!(hourly_rate_cents, 2200);
        assert_eq!(gpu_category.as_deref(), Some("H100"));
        assert_eq!(gpu_count, 8);
    }

    #[tokio::test]
    async fn get_node_bid_metadata_returns_empty_category_for_legacy_rows() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let node_id = "legacy_node";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            node_id,
            "10.0.0.10",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;

        let metadata = persistence
            .get_node_bid_metadata(miner_id, node_id)
            .await
            .expect("metadata query should succeed")
            .expect("metadata should exist for existing node");
        assert!(metadata.gpu_category.is_empty());
        assert_eq!(metadata.gpu_count, 1);
    }

    #[tokio::test]
    async fn deactivate_missing_bids_selective_deactivation() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";
        let active_node = "active_node";
        let removed_node = "removed_node";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            active_node,
            "10.0.0.11",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;
        insert_test_node(
            &persistence,
            miner_id,
            removed_node,
            "10.0.0.12",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;

        let deactivated = persistence
            .deactivate_missing_bids(miner_id, &[active_node.to_string()])
            .await
            .expect("deactivation should succeed");
        assert_eq!(deactivated, 1);

        let active_bid: i64 = sqlx::query_scalar(
            "SELECT bid_active FROM miner_nodes WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(active_node)
        .fetch_one(persistence.pool())
        .await
        .expect("active node should exist");
        assert_eq!(active_bid, 1);

        let removed_bid: i64 = sqlx::query_scalar(
            "SELECT bid_active FROM miner_nodes WHERE miner_id = ? AND node_id = ?",
        )
        .bind(miner_id)
        .bind(removed_node)
        .fetch_one(persistence.pool())
        .await
        .expect("removed node should exist");
        assert_eq!(removed_bid, 0);
    }

    #[tokio::test]
    async fn deactivate_missing_bids_with_empty_list_deactivates_all() {
        let persistence = create_test_persistence().await;
        let miner_id = "miner_1";

        insert_test_miner(&persistence, miner_id).await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_a",
            "10.0.0.13",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;
        insert_test_node(
            &persistence,
            miner_id,
            "node_b",
            "10.0.0.14",
            "online",
            1,
            Some("2000-01-01 00:00:00"),
            None,
            None,
            1000,
        )
        .await;

        let deactivated = persistence
            .deactivate_missing_bids(miner_id, &[])
            .await
            .expect("deactivation should succeed");
        assert_eq!(deactivated, 2);

        let active_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM miner_nodes WHERE miner_id = ? AND bid_active = 1",
        )
        .bind(miner_id)
        .fetch_one(persistence.pool())
        .await
        .expect("count query should succeed");
        assert_eq!(active_count, 0);
    }
}
