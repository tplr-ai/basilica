//! Miner persistence operations
//!
//! This module contains all SQL operations related to miners table management.

use crate::api::types::{NodeRegistration, UpdateMinerRequest};
use crate::miner_prover::types::MinerInfo;
use crate::persistence::types::{MinerData, MinerHealthData, NodeHealthData, NodeMetricData};
use crate::persistence::SimplePersistence;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::Row;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

impl SimplePersistence {
    /// Check if a miner with the given UID exists
    pub async fn check_miner_by_uid(&self, miner_uid: &str) -> Result<Option<(String, String)>> {
        let query = "SELECT id, hotkey FROM miners WHERE id = ?";
        let result = sqlx::query(query)
            .bind(miner_uid)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner by uid: {}", e))?;

        Ok(result.map(|row| {
            let id: String = row.get("id");
            let hotkey: String = row.get("hotkey");
            (id, hotkey)
        }))
    }

    /// Check if a miner with the given hotkey exists
    pub async fn check_miner_by_hotkey(&self, hotkey: &str) -> Result<Option<String>> {
        let query = "SELECT id FROM miners WHERE hotkey = ?";
        let result = sqlx::query(query)
            .bind(hotkey)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner by hotkey: {}", e))?;

        Ok(result.map(|row| row.get("id")))
    }

    /// Get the hotkey for a miner by their ID
    pub async fn get_miner_hotkey_by_id(&self, miner_id: &str) -> Result<Option<String>> {
        let query = "SELECT hotkey FROM miners WHERE id = ?";
        let result = sqlx::query(query)
            .bind(miner_id)
            .fetch_optional(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get miner hotkey: {}", e))?;

        Ok(result.map(|row| row.get("hotkey")))
    }

    /// Create a new miner record
    pub async fn create_new_miner(
        &self,
        miner_uid: &str,
        hotkey: &str,
        miner_info: &MinerInfo,
    ) -> Result<()> {
        let insert_query = r#"
            INSERT INTO miners (
                id, hotkey, endpoint, verification_score, uptime_percentage,
                last_seen, registered_at, updated_at, node_info
            ) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'), ?)
        "#;

        sqlx::query(insert_query)
            .bind(miner_uid)
            .bind(hotkey)
            .bind(&miner_info.endpoint)
            .bind(miner_info.verification_score)
            .bind(100.0)
            .bind("{}")
            .execute(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to insert miner: {}", e))?;

        info!(
            "Created miner record: {} with hotkey {} and endpoint {}",
            miner_uid, hotkey, miner_info.endpoint
        );

        Ok(())
    }

    /// Update existing miner data
    pub async fn update_miner_data(&self, miner_id: &str, miner_info: &MinerInfo) -> Result<()> {
        let update_query = r#"
            UPDATE miners SET
                endpoint = ?, verification_score = ?,
                last_seen = datetime('now'), updated_at = datetime('now')
            WHERE id = ?
        "#;

        sqlx::query(update_query)
            .bind(&miner_info.endpoint)
            .bind(miner_info.verification_score)
            .bind(miner_id)
            .execute(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to update miner: {}", e))?;

        debug!("Updated miner record: {} with latest data", miner_id);
        Ok(())
    }

    /// Handle case where miner UID already exists
    pub async fn handle_recycled_miner_uid(
        &self,
        miner_uid: &str,
        new_hotkey: &str,
        existing_hotkey: &str,
        miner_info: &MinerInfo,
    ) -> Result<()> {
        if existing_hotkey != new_hotkey {
            info!(
                miner_uid = miner_uid,
                "Miner {} exists with old hotkey {}, updating to new hotkey {}",
                miner_uid,
                existing_hotkey,
                new_hotkey
            );

            let update_query = r#"
                UPDATE miners SET
                    hotkey = ?, endpoint = ?, verification_score = ?,
                    last_seen = datetime('now'), updated_at = datetime('now')
                WHERE id = ?
            "#;

            sqlx::query(update_query)
                .bind(new_hotkey)
                .bind(&miner_info.endpoint)
                .bind(miner_info.verification_score)
                .bind(miner_uid)
                .execute(self.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to update miner with new hotkey: {}", e))?;

            debug!("Updated miner {} with new hotkey and data", miner_uid);
        } else {
            self.update_miner_data(miner_uid, miner_info).await?;
        }

        Ok(())
    }

    /// Handle case where hotkey exists but with different ID (UID change)
    pub async fn handle_uid_change(
        &self,
        old_miner_id: &str,
        new_miner_id: &str,
        hotkey: &str,
        miner_info: &MinerInfo,
    ) -> Result<()> {
        info!(
            "Detected UID change for hotkey {}: {} -> {}",
            hotkey, old_miner_id, new_miner_id
        );

        if let Err(e) = self
            .migrate_miner_uid(old_miner_id, new_miner_id, miner_info)
            .await
        {
            error!(
                "Failed to migrate miner UID from {} to {}: {}",
                old_miner_id, new_miner_id, e
            );
            return Err(e);
        }

        Ok(())
    }

    /// Migrate miner UID when it changes in the network
    pub async fn migrate_miner_uid(
        &self,
        old_miner_uid: &str,
        new_miner_uid: &str,
        miner_info: &MinerInfo,
    ) -> Result<()> {
        info!(
            "Starting UID migration: {} -> {} for hotkey {}",
            old_miner_uid, new_miner_uid, miner_info.hotkey
        );

        let mut tx = self
            .pool()
            .begin()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to begin transaction: {}", e))?;

        debug!("Fetching old miner record: {}", old_miner_uid);
        let get_old_miner = "SELECT * FROM miners WHERE id = ?";
        let old_miner_row = sqlx::query(get_old_miner)
            .bind(old_miner_uid)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch old miner record: {}", e))?;

        if old_miner_row.is_none() {
            return Err(anyhow::anyhow!(
                "Old miner record not found: {}",
                old_miner_uid
            ));
        }

        let old_row = old_miner_row.unwrap();
        debug!("Found old miner record for migration");

        debug!(
            "Checking for existing miners with hotkey: {}",
            miner_info.hotkey
        );
        let check_hotkey = "SELECT id FROM miners WHERE hotkey = ?";
        let all_with_hotkey = sqlx::query(check_hotkey)
            .bind(miner_info.hotkey.to_string())
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check hotkey existence: {}", e))?;

        let existing_with_hotkey = all_with_hotkey.into_iter().find(|row| {
            let id: String = row.get("id");
            id != old_miner_uid
        });

        let should_create_new = if let Some(row) = existing_with_hotkey {
            let existing_id: String = row.get("id");
            debug!(
                "Found existing miner with hotkey {}: id={}",
                miner_info.hotkey, existing_id
            );
            if existing_id == new_miner_uid {
                debug!("New miner record already exists with correct ID");
                false
            } else {
                warn!(
                    "Cannot migrate: Another miner {} already exists with hotkey {} (trying to create {})",
                    existing_id, miner_info.hotkey, new_miner_uid
                );
                return Err(anyhow::anyhow!(
                    "Cannot migrate: Another miner {} already exists with hotkey {}",
                    existing_id,
                    miner_info.hotkey
                ));
            }
        } else {
            debug!(
                "No existing miner with hotkey {}, will create new record",
                miner_info.hotkey
            );
            true
        };

        let verification_score = old_row
            .try_get::<f64, _>("verification_score")
            .unwrap_or(0.0);
        let uptime_percentage = old_row
            .try_get::<f64, _>("uptime_percentage")
            .unwrap_or(100.0);
        let registered_at = old_row
            .try_get::<String, _>("registered_at")
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
        let node_info = old_row
            .try_get::<String, _>("node_info")
            .unwrap_or_else(|_| "{}".to_string());

        debug!("Fetching related node data");
        let get_nodes = "SELECT * FROM miner_nodes WHERE miner_id = ?";
        let nodes = sqlx::query(get_nodes)
            .bind(old_miner_uid)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch nodes: {}", e))?;

        debug!("Found {} nodes to migrate", nodes.len());

        debug!("Deleting old miner record: {}", old_miner_uid);
        let delete_old_miner = "DELETE FROM miners WHERE id = ?";
        sqlx::query(delete_old_miner)
            .bind(old_miner_uid)
            .execute(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete old miner record: {}", e))?;

        debug!("Deleted old miner record and related data");

        if should_create_new {
            debug!("Creating new miner record: {}", new_miner_uid);
            let insert_new_miner = r#"
                INSERT INTO miners (
                    id, hotkey, endpoint, verification_score, uptime_percentage,
                    last_seen, registered_at, updated_at, node_info
                ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?, datetime('now'), ?)
            "#;

            sqlx::query(insert_new_miner)
                .bind(new_miner_uid)
                .bind(miner_info.hotkey.to_string())
                .bind(&miner_info.endpoint)
                .bind(verification_score)
                .bind(uptime_percentage)
                .bind(registered_at)
                .bind(node_info)
                .execute(&mut *tx)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create new miner record: {}", e))?;

            debug!("Successfully created new miner record");
        }

        let mut node_count = 0;
        for node_row in nodes {
            let node_id: String = node_row.get("node_id");
            let ssh_endpoint: String = node_row.get("ssh_endpoint");
            let gpu_count: i32 = node_row.get("gpu_count");
            let status: String = node_row
                .try_get("status")
                .unwrap_or_else(|_| "unknown".to_string());

            let existing_check = sqlx::query(
                "SELECT COUNT(*) as count FROM miner_nodes WHERE ssh_endpoint = ? AND miner_id != ?"
            )
            .bind(&ssh_endpoint)
            .bind(new_miner_uid)
            .fetch_one(&mut *tx)
            .await?;

            let existing_count: i64 = existing_check.get("count");
            if existing_count > 0 {
                warn!(
                    "Skipping node {} during UID migration: ssh_endpoint {} already in use by another miner",
                    node_id, ssh_endpoint
                );
                continue;
            }

            let new_id = format!("{new_miner_uid}_{node_id}");

            let insert_node = r#"
                INSERT INTO miner_nodes (
                    id, miner_id, node_id, ssh_endpoint, gpu_count,
                    status, last_health_check,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, datetime('now'), datetime('now'))
            "#;

            sqlx::query(insert_node)
                .bind(&new_id)
                .bind(new_miner_uid)
                .bind(&node_id)
                .bind(&ssh_endpoint)
                .bind(gpu_count)
                .bind(&status)
                .execute(&mut *tx)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to recreate node relationship: {}", e))?;

            node_count += 1;
        }

        debug!("Recreated {} node relationships", node_count);

        debug!(
            "Migrating GPU UUID assignments from {} to {}",
            old_miner_uid, new_miner_uid
        );
        let update_gpu_assignments = r#"
            UPDATE gpu_uuid_assignments
            SET miner_id = ?
            WHERE miner_id = ?
        "#;

        let gpu_result = sqlx::query(update_gpu_assignments)
            .bind(new_miner_uid)
            .bind(old_miner_uid)
            .execute(&mut *tx)
            .await?;

        debug!(
            "Migrated {} GPU UUID assignments",
            gpu_result.rows_affected()
        );

        debug!("Committing transaction");
        tx.commit()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to commit transaction: {}", e))?;

        info!(
            "Successfully migrated miner UID: {} -> {}. Migrated {} nodes",
            old_miner_uid, new_miner_uid, node_count
        );

        Ok(())
    }

    /// Ensure miner exists in miners table
    ///
    /// This function handles three scenarios:
    /// 1. if UID already exists with same hotkey -> Update data
    /// 2. if UID already exists with different hotkey -> Update to new hotkey (recycled UID)
    /// 3. if UID doesn't exist but hotkey does -> on re-registration, migrate the UID
    /// 4. if neither UID nor hotkey exist -> Create new miner
    pub async fn ensure_miner_exists_with_info(&self, miner_info: &MinerInfo) -> Result<()> {
        let new_miner_uid = format!("miner_{}", miner_info.uid.as_u16());
        let hotkey = miner_info.hotkey.to_string();

        let existing_by_uid = self.check_miner_by_uid(&new_miner_uid).await?;

        if let Some((_, existing_hotkey)) = existing_by_uid {
            return self
                .handle_recycled_miner_uid(&new_miner_uid, &hotkey, &existing_hotkey, miner_info)
                .await;
        }

        let existing_by_hotkey = self.check_miner_by_hotkey(&hotkey).await?;

        if let Some(old_miner_uid) = existing_by_hotkey {
            return self
                .handle_uid_change(&old_miner_uid, &new_miner_uid, &hotkey, miner_info)
                .await;
        }

        self.create_new_miner(&new_miner_uid, &hotkey, miner_info)
            .await
    }

    /// Sync miners from metagraph to database
    pub async fn sync_miners_from_metagraph(&self, miners: &[MinerInfo]) -> Result<()> {
        info!("Syncing {} miners from metagraph to database", miners.len());

        for miner in miners {
            if let Err(e) = self.ensure_miner_exists_with_info(miner).await {
                warn!(
                    "Failed to sync miner {} to database: {}",
                    miner.uid.as_u16(),
                    e
                );
            } else {
                debug!(
                    "Successfully synced miner {} with endpoint {} to database",
                    miner.uid.as_u16(),
                    miner.endpoint
                );
            }
        }

        info!("Completed syncing miners from metagraph");
        Ok(())
    }

    /// Query recent verification logs for a miner's nodes
    pub async fn query_recent_miner_verification_logs(
        &self,
        miner_uid: u16,
        cutoff_time: &str,
    ) -> Result<Vec<sqlx::sqlite::SqliteRow>> {
        let query = r#"
            SELECT vl.*, me.miner_id, me.status
            FROM verification_logs vl
            INNER JOIN miner_nodes me ON vl.node_id = me.node_id
            WHERE me.miner_id = ?
                AND vl.timestamp >= ?
                AND me.status IN ('online', 'verified')
                AND EXISTS (
                    SELECT 1 FROM gpu_uuid_assignments ga
                    WHERE ga.node_id = vl.node_id
                    AND ga.miner_id = me.miner_id
                )
            ORDER BY vl.timestamp DESC
        "#;

        let miner_id = format!("miner_{miner_uid}");
        let rows = sqlx::query(query)
            .bind(&miner_id)
            .bind(cutoff_time)
            .fetch_all(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to query verification logs: {}", e))?;

        Ok(rows)
    }

    /// Get all unique miner IDs from recent validations
    pub async fn get_miners_with_recent_validations(
        &self,
        cutoff_time: &str,
    ) -> Result<Vec<String>> {
        let query = r#"
            SELECT DISTINCT me.miner_id
            FROM miner_nodes me
            JOIN verification_logs vl ON me.node_id = vl.node_id
            WHERE vl.timestamp >= ?
        "#;

        let rows = sqlx::query(query)
            .bind(cutoff_time)
            .fetch_all(self.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to query miners: {}", e))?;

        let miner_ids = rows.into_iter().map(|row| row.get("miner_id")).collect();
        Ok(miner_ids)
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
                last_seen, registered_at, node_info,
                (SELECT COUNT(*) FROM miner_nodes WHERE miner_id = miners.id) as node_count
             FROM miners
             ORDER BY registered_at DESC
             LIMIT ? OFFSET ?",
        )
        .bind(page_size as i64)
        .bind(offset as i64)
        .fetch_all(self.pool())
        .await?;

        let mut miners = Vec::new();
        for row in rows {
            let node_info_str: String = row.get("node_info");
            let node_count: i64 = row.get("node_count");
            let last_seen_str: String = row.get("last_seen");
            let registered_at_str: String = row.get("registered_at");

            miners.push(MinerData {
                miner_id: row.get("id"),
                hotkey: row.get("hotkey"),
                endpoint: row.get("endpoint"),
                node_count: node_count as u32,
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
                node_info: serde_json::from_str(&node_info_str)
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
        nodes: &[NodeRegistration],
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        let node_info = serde_json::to_string(&nodes)?;

        let mut tx = self.pool().begin().await?;

        // Validate that ssh_endpoint are not already registered
        for node in nodes {
            let existing =
                sqlx::query("SELECT COUNT(*) as count FROM miner_nodes WHERE ssh_endpoint = ?")
                    .bind(&node.ssh_endpoint)
                    .fetch_one(&mut *tx)
                    .await?;

            let count: i64 = existing.get("count");
            if count > 0 {
                return Err(anyhow::anyhow!(
                    "Node with ssh_endpoint {} is already registered",
                    node.ssh_endpoint
                ));
            }
        }

        sqlx::query(
            "INSERT INTO miners (id, hotkey, endpoint, last_seen, registered_at, updated_at, node_info)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(miner_id)
        .bind(hotkey)
        .bind(endpoint)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .bind(&node_info)
        .execute(&mut *tx)
        .await?;

        for node in nodes {
            let node_id = Uuid::new_v4().to_string();

            sqlx::query(
                "INSERT INTO miner_nodes (id, miner_id, node_id, ssh_endpoint, gpu_count, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&node_id)
            .bind(miner_id)
            .bind(&node.node_id)
            .bind(&node.ssh_endpoint)
            .bind(node.gpu_count as i64)
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
                last_seen, registered_at, node_info,
                (SELECT COUNT(*) FROM miner_nodes WHERE miner_id = miners.id) as node_count
             FROM miners
             WHERE id = ?",
        )
        .bind(miner_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(row) = row {
            let node_info_str: String = row.get("node_info");
            let node_count: i64 = row.get("node_count");
            let last_seen_str: String = row.get("last_seen");
            let registered_at_str: String = row.get("registered_at");

            Ok(Some(MinerData {
                miner_id: row.get("id"),
                hotkey: row.get("hotkey"),
                endpoint: row.get("endpoint"),
                node_count: node_count as u32,
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
                node_info: serde_json::from_str(&node_info_str)
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
        request: &UpdateMinerRequest,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();

        if let Some(endpoint) = &request.endpoint {
            let result = sqlx::query("UPDATE miners SET endpoint = ?, updated_at = ? WHERE id = ?")
                .bind(endpoint)
                .bind(&now)
                .bind(miner_id)
                .execute(self.pool())
                .await?;

            if result.rows_affected() == 0 {
                return Err(anyhow::anyhow!("Miner not found"));
            }
        }

        if let Some(nodes) = &request.nodes {
            // When updating nodes, we need to handle the miner_nodes table
            let mut tx = self.pool().begin().await?;

            // First, validate that new ssh_endpoints aren't already registered by other miners
            for node in nodes {
                let existing = sqlx::query(
                    "SELECT COUNT(*) as count FROM miner_nodes
                     WHERE ssh_endpoint = ? AND miner_id != ?",
                )
                .bind(&node.ssh_endpoint)
                .bind(miner_id)
                .fetch_one(&mut *tx)
                .await?;

                let count: i64 = existing.get("count");
                if count > 0 {
                    return Err(anyhow::anyhow!(
                        "Node with ssh_endpoint {} is already registered by another miner",
                        node.ssh_endpoint
                    ));
                }
            }

            // Delete existing nodes for this miner
            sqlx::query("DELETE FROM miner_nodes WHERE miner_id = ?")
                .bind(miner_id)
                .execute(&mut *tx)
                .await?;

            // Insert new nodes
            for node in nodes {
                let node_id = Uuid::new_v4().to_string();

                sqlx::query(
                    "INSERT INTO miner_nodes (id, miner_id, node_id, ssh_endpoint, gpu_count, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&node_id)
                .bind(miner_id)
                .bind(&node.node_id)
                .bind(&node.ssh_endpoint)
                .bind(node.gpu_count as i64)
                .bind(&now)
                .bind(&now)
                .execute(&mut *tx)
                .await?;
            }

            // Also update the node_info JSON in the miners table
            let node_info = serde_json::to_string(nodes)?;
            let result =
                sqlx::query("UPDATE miners SET node_info = ?, updated_at = ? WHERE id = ?")
                    .bind(&node_info)
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
            .execute(self.pool())
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
            "SELECT node_id, status, last_health_check, created_at
             FROM miner_nodes
             WHERE miner_id = ?",
        )
        .bind(miner_id)
        .fetch_all(self.pool())
        .await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let mut node_health = Vec::new();
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

            node_health.push(NodeHealthData {
                node_id: row.get("node_id"),
                status: row
                    .get::<Option<String>, _>("status")
                    .unwrap_or_else(|| "unknown".to_string()),
                last_seen,
            });
        }

        Ok(Some(MinerHealthData {
            last_health_check: latest_check,
            node_health,
        }))
    }

    /// Get all nodes with their GPU and rental data for metrics initialization
    /// This eliminates N+1 queries by fetching everything in a single query with joins
    pub async fn get_all_nodes_for_metrics(&self) -> Result<Vec<NodeMetricData>, anyhow::Error> {
        let query = r#"
            SELECT
                me.node_id,
                me.miner_id,
                gua.gpu_name,
                CASE WHEN r.id IS NOT NULL THEN 1 ELSE 0 END as has_active_rental
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments gua
                ON me.node_id = gua.node_id
                AND me.miner_id = gua.miner_id
            LEFT JOIN rentals r
                ON me.node_id = r.node_id
                AND me.miner_id = r.miner_id
                AND r.state = 'active'
            WHERE gua.gpu_name IS NOT NULL
            GROUP BY me.node_id, me.miner_id
        "#;

        let rows = sqlx::query(query).fetch_all(self.pool()).await?;

        let mut node_metrics = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
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

            node_metrics.push(NodeMetricData {
                node_id,
                miner_id,
                miner_uid,
                gpu_name,
                has_active_rental: has_active_rental != 0,
            });
        }

        Ok(node_metrics)
    }

    /// Helper function to parse datetime from various formats
    fn parse_datetime(s: &str) -> Result<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|_| {
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                    .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
            })
            .map_err(|e| anyhow::anyhow!("Failed to parse datetime '{}': {}", s, e))
    }

    /// Calculate node uptime multiplier based on verification history
    /// Returns tuple of (uptime_minutes, multiplier):
    /// - uptime_minutes: Total continuous uptime in minutes
    /// - multiplier: Value between 0.0 and 1.0 (0.0 = new node, 1.0 = 14+ days uptime)
    pub async fn calculate_node_uptime_multiplier(
        &self,
        miner_id: &str,
        node_id: &str,
    ) -> Result<(f64, f64)> {
        // Step 1: Get node registration time
        let node_query = r#"
            SELECT created_at
            FROM miner_nodes
            WHERE miner_id = ? AND node_id = ?
        "#;

        let node_row = sqlx::query(node_query)
            .bind(miner_id)
            .bind(node_id)
            .fetch_one(self.pool())
            .await?;

        let created_at_str: String = node_row.get("created_at");

        // Step 2: Get FULL validation logs since registration
        // IMPORTANT: Only consider full validations (last_binary_validation IS NOT NULL)
        let logs_query = r#"
            SELECT timestamp, success
            FROM verification_logs
            WHERE node_id = ?
                AND timestamp >= ?
                AND last_binary_validation IS NOT NULL
            ORDER BY timestamp ASC
        "#;

        let logs = sqlx::query(logs_query)
            .bind(node_id)
            .bind(&created_at_str)
            .fetch_all(self.pool())
            .await?;

        if logs.is_empty() {
            // No full validations yet
            debug!(
                miner_id = %miner_id,
                node_id = %node_id,
                "No full validation logs found, returning (0.0, 0.0)"
            );
            return Ok((0.0, 0.0));
        }

        // Step 3: Find start of current continuous success period
        // Only the current uninterrupted success period counts toward uptime
        let mut current_success_start: Option<DateTime<Utc>> = None;

        for log in logs {
            let timestamp_str: String = log.get("timestamp");
            let timestamp = Self::parse_datetime(&timestamp_str)?;
            let success: i32 = log.get("success");

            if success == 1 {
                // Successful validation
                if current_success_start.is_none() {
                    // Start tracking new success period
                    current_success_start = Some(timestamp);
                }
                // If already tracking, continue (no action needed)
            } else {
                // Failed validation - RESET the uptime period
                current_success_start = None;
            }
        }

        // Calculate uptime from current success period start to now
        let total_uptime_minutes = if let Some(start) = current_success_start {
            let now = Utc::now();
            (now - start).num_minutes() as f64
        } else {
            // No current success period (last validation was a failure)
            0.0
        };

        // Step 5: Calculate ramp-up multiplier (0-100% over 14 days)
        const FULL_WEIGHT_MINUTES: f64 = 20_160.0; // 14 days
        let multiplier = (total_uptime_minutes / FULL_WEIGHT_MINUTES).min(1.0);

        debug!(
            miner_id = %miner_id,
            node_id = %node_id,
            total_uptime_minutes = total_uptime_minutes,
            multiplier = multiplier,
            "Calculated node uptime multiplier"
        );

        Ok((total_uptime_minutes, multiplier))
    }
}
