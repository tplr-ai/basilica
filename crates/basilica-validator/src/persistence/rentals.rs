use chrono::{DateTime, Utc};
use sqlx::{QueryBuilder, Row};
use tracing::warn;
use uuid::Uuid;

use crate::persistence::types::RentalFilter;
use crate::persistence::ValidatorPersistence;
use crate::rental::{RentalInfo, RentalState};

use crate::persistence::entities::{Rental, RentalStatus};
use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    pub async fn create_rental(&self, rental: &Rental) -> Result<(), anyhow::Error> {
        let query = r#"
            INSERT INTO rentals (
                id, node_id, customer_public_key, docker_image, env_vars,
                gpu_requirements, ssh_access_info, cost_per_hour,
                status, created_at, updated_at, started_at, terminated_at,
                termination_reason, total_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let status_str = match rental.status {
            RentalStatus::Pending => "Pending",
            RentalStatus::Active => "Active",
            RentalStatus::Terminated => "Terminated",
            RentalStatus::Failed => "Failed",
        };

        sqlx::query(query)
            .bind(rental.id.to_string())
            .bind(&rental.node_id)
            .bind(&rental.customer_public_key)
            .bind(&rental.docker_image)
            .bind(
                rental
                    .env_vars
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap()),
            )
            .bind(serde_json::to_string(&rental.gpu_requirements)?)
            .bind(serde_json::to_string(&rental.ssh_access_info)?)
            .bind(rental.cost_per_hour)
            .bind(status_str)
            .bind(rental.created_at.to_rfc3339())
            .bind(rental.updated_at.to_rfc3339())
            .bind(rental.started_at.map(|dt| dt.to_rfc3339()))
            .bind(rental.terminated_at.map(|dt| dt.to_rfc3339()))
            .bind(&rental.termination_reason)
            .bind(rental.total_cost)
            .execute(&self.pool)
            .await?;

        tracing::info!(
            rental_id = %rental.id,
            node_id = %rental.node_id,
            status = ?rental.status,
            "Rental created"
        );

        Ok(())
    }

    /// Get rental by ID
    pub async fn get_rental(&self, rental_id: &Uuid) -> Result<Option<Rental>, anyhow::Error> {
        let row = sqlx::query("SELECT * FROM rentals WHERE id = ?")
            .bind(rental_id.to_string())
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            Ok(Some(self.row_to_rental(row)?))
        } else {
            Ok(None)
        }
    }

    /// Update rental record
    pub async fn update_rental(&self, rental: &Rental) -> Result<(), anyhow::Error> {
        let status_str = match rental.status {
            RentalStatus::Pending => "Pending",
            RentalStatus::Active => "Active",
            RentalStatus::Terminated => "Terminated",
            RentalStatus::Failed => "Failed",
        };

        let query = r#"
            UPDATE rentals SET
                status = ?, updated_at = ?, started_at = ?,
                terminated_at = ?, termination_reason = ?, total_cost = ?
            WHERE id = ?
        "#;

        sqlx::query(query)
            .bind(status_str)
            .bind(rental.updated_at.to_rfc3339())
            .bind(rental.started_at.map(|dt| dt.to_rfc3339()))
            .bind(rental.terminated_at.map(|dt| dt.to_rfc3339()))
            .bind(&rental.termination_reason)
            .bind(rental.total_cost)
            .bind(rental.id.to_string())
            .execute(&self.pool)
            .await?;

        tracing::info!(
            rental_id = %rental.id,
            status = ?rental.status,
            "Rental updated"
        );

        Ok(())
    }

    /// Check if a node has an active rental by reading the `active_rental_id`
    /// column on `miner_nodes`. This is the single source of truth for node
    /// availability — see the INVARIANT comment in `miner_nodes.rs`.
    pub async fn has_active_rental(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let active_rental: Option<Option<String>> = sqlx::query_scalar(
            "SELECT active_rental_id FROM miner_nodes WHERE node_id = ? AND miner_id = ? LIMIT 1",
        )
        .bind(node_id)
        .bind(miner_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(active_rental.flatten().is_some())
    }

    pub async fn get_last_rental_terminated_at(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<Option<DateTime<Utc>>, anyhow::Error> {
        let query = r#"
            SELECT MAX(terminated_at) as terminated_at
            FROM rentals
            WHERE node_id = ?
              AND miner_id = ?
              AND terminated_at IS NOT NULL
        "#;

        let row = sqlx::query(query)
            .bind(node_id)
            .bind(miner_id)
            .fetch_one(&self.pool)
            .await?;

        let terminated_at: Option<String> = row.get("terminated_at");
        if let Some(ts) = terminated_at {
            let parsed = DateTime::parse_from_rfc3339(&ts)?.with_timezone(&Utc);
            Ok(Some(parsed))
        } else {
            Ok(None)
        }
    }

    /// Helper function to parse rental state from string
    fn parse_rental_state(state_str: &str, rental_id: &str) -> RentalState {
        match state_str {
            "provisioning" => RentalState::Provisioning,
            "active" => RentalState::Active,
            "stopping" => RentalState::Stopping,
            "stopped" => RentalState::Stopped,
            "failed" => RentalState::Failed,
            unknown => {
                warn!(
                    "Unknown rental state '{}' for rental {}, defaulting to Failed",
                    unknown, rental_id
                );
                RentalState::Failed
            }
        }
    }

    /// Helper function to parse a rental row from the database
    fn parse_rental_row(
        &self,
        row: sqlx::sqlite::SqliteRow,
        node_details: crate::api::types::NodeDetails,
    ) -> Result<RentalInfo, anyhow::Error> {
        let state_str: String = row.get("state");
        let created_at_str: String = row.get("created_at");
        let updated_at_str: Option<String> = row.try_get("updated_at").ok();
        let container_spec_str: String = row.get("container_spec");
        let rental_id: String = row.get("id");
        let node_id: String = row.get("node_id");
        let metadata: String = row.get("metadata");

        let state = Self::parse_rental_state(&state_str, &rental_id);
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc);
        let updated_at = updated_at_str
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(created_at);

        Ok(RentalInfo {
            rental_id,
            validator_hotkey: row.get("validator_hotkey"),
            node_id,
            container_id: row.get("container_id"),
            ssh_session_id: row.get("ssh_session_id"),
            ssh_credentials: row.get("ssh_credentials"),
            state,
            created_at,
            updated_at,
            container_spec: serde_json::from_str(&container_spec_str)?,
            miner_id: row.get::<String, _>("miner_id"),
            node_details,
            metadata: serde_json::from_str(&metadata)?,
        })
    }

    /// Query rentals with flexible filtering criteria
    pub(crate) async fn query_rentals(
        &self,
        filter: RentalFilter,
    ) -> Result<Vec<RentalInfo>, anyhow::Error> {
        let mut builder = QueryBuilder::new("SELECT * FROM rentals");
        let mut has_where = false;

        if let Some(rental_id) = filter.rental_id {
            builder.push(" WHERE id = ");
            builder.push_bind(rental_id);
            has_where = true;
        }

        if let Some(validator_hotkey) = filter.validator_hotkey {
            builder.push(if has_where { " AND " } else { " WHERE " });
            builder.push("validator_hotkey = ");
            builder.push_bind(validator_hotkey);
            has_where = true;
        }

        if let Some(exclude_states) = filter.exclude_states {
            if !exclude_states.is_empty() {
                builder.push(if has_where { " AND " } else { " WHERE " });
                builder.push("state NOT IN (");
                for (i, state) in exclude_states.iter().enumerate() {
                    if i > 0 {
                        builder.push(", ");
                    }
                    builder.push_bind(match state {
                        RentalState::Provisioning => "provisioning",
                        RentalState::Active => "active",
                        RentalState::Restarting => "restarting",
                        RentalState::Stopping => "stopping",
                        RentalState::Stopped => "stopped",
                        RentalState::Failed => "failed",
                    });
                }
                builder.push(")");
            }
        }

        if filter.order_by_created_desc {
            builder.push(" ORDER BY created_at DESC");
        }

        let query = builder.build();
        let rows = query.fetch_all(&self.pool).await?;

        let mut rentals = Vec::new();
        for row in rows {
            let node_id: String = row.get("node_id");
            let miner_id: String = row.get("miner_id");

            let node_details = match self.get_node_details(&node_id, &miner_id).await {
                Ok(Some(details)) => details,
                _ => crate::api::types::NodeDetails {
                    id: node_id.clone(),
                    gpu_specs: vec![],
                    cpu_specs: crate::api::types::CpuSpec {
                        cores: 0,
                        model: "Unknown".to_string(),
                        memory_gb: 0,
                    },
                    location: None,
                    network_speed: None,
                    hourly_rate_cents: None,
                },
            };

            rentals.push(self.parse_rental_row(row, node_details)?);
        }

        Ok(rentals)
    }

    /// Helper function to convert database row to Rental
    fn row_to_rental(&self, row: sqlx::sqlite::SqliteRow) -> Result<Rental, anyhow::Error> {
        let id_str: String = row.get("id");
        let env_vars_str: Option<String> = row.get("env_vars");
        let gpu_requirements_str: String = row.get("gpu_requirements");
        let ssh_access_info_str: String = row.get("ssh_access_info");
        let status_str: String = row.get("status");
        let created_at_str: String = row.get("created_at");
        let updated_at_str: String = row.get("updated_at");
        let started_at_str: Option<String> = row.get("started_at");
        let terminated_at_str: Option<String> = row.get("terminated_at");

        let status = match status_str.as_str() {
            "Pending" => RentalStatus::Pending,
            "Active" => RentalStatus::Active,
            "Terminated" => RentalStatus::Terminated,
            "Failed" => RentalStatus::Failed,
            _ => return Err(anyhow::anyhow!("Invalid rental status: {}", status_str)),
        };

        Ok(Rental {
            id: Uuid::parse_str(&id_str)?,
            node_id: row.get("node_id"),
            customer_public_key: row.get("customer_public_key"),
            docker_image: row.get("docker_image"),
            env_vars: env_vars_str.map(|s| serde_json::from_str(&s)).transpose()?,
            gpu_requirements: serde_json::from_str(&gpu_requirements_str)?,
            ssh_access_info: serde_json::from_str(&ssh_access_info_str)?,
            cost_per_hour: row.get("cost_per_hour"),
            status,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.with_timezone(&Utc),
            started_at: started_at_str.map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .unwrap()
                    .with_timezone(&Utc)
            }),
            terminated_at: terminated_at_str.map(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .unwrap()
                    .with_timezone(&Utc)
            }),
            termination_reason: row.get("termination_reason"),
            total_cost: row.get("total_cost"),
        })
    }

    /// Get GPU count for a specific rental by querying gpu_uuid_assignments table.
    /// Returns the number of GPUs assigned to the node associated with the rental.
    pub async fn get_rental_gpu_count(&self, rental_id: &Uuid) -> Result<u32, anyhow::Error> {
        let query = r#"
            SELECT COUNT(*) as gpu_count
            FROM gpu_uuid_assignments AS gua
            JOIN rentals ON gua.node_id = rentals.node_id
            WHERE rentals.id = ?
        "#;

        let count: i64 = sqlx::query_scalar(query)
            .bind(rental_id.to_string())
            .fetch_one(&self.pool)
            .await?;

        tracing::debug!(
            rental_id = %rental_id,
            gpu_count = count,
            "Retrieved GPU count for rental"
        );

        Ok(count as u32)
    }

    /// Get GPU count for an active rental only.
    /// Returns the number of GPUs assigned to the node if the rental is in 'active' state.
    pub async fn get_active_rental_gpu_count(&self, rental_id: &str) -> Result<u32, anyhow::Error> {
        let query = r#"
            SELECT COUNT(*) as gpu_count
            FROM gpu_uuid_assignments AS gua
            JOIN rentals ON gua.node_id = rentals.node_id
            WHERE rentals.state = 'active'
                AND rentals.id = ?
        "#;

        let count: i64 = sqlx::query_scalar(query)
            .bind(rental_id)
            .fetch_one(&self.pool)
            .await?;

        tracing::debug!(
            rental_id = %rental_id,
            gpu_count = count,
            "Retrieved GPU count for active rental"
        );

        Ok(count as u32)
    }

    /// Get GPU count for active rentals by node ID.
    /// Returns the number of GPUs assigned to a specific node that has an active rental.
    pub async fn get_rental_gpu_count_by_node_id(
        &self,
        node_id: &str,
    ) -> Result<u32, anyhow::Error> {
        let query = r#"
            SELECT COUNT(*) as gpu_count
            FROM gpu_uuid_assignments AS gua
            JOIN rentals ON gua.node_id = rentals.node_id
            WHERE rentals.state = 'active'
                AND gua.node_id = ?
        "#;

        let count: i64 = sqlx::query_scalar(query)
            .bind(node_id)
            .fetch_one(&self.pool)
            .await?;

        tracing::debug!(
            node_id = %node_id,
            gpu_count = count,
            "Retrieved GPU count for active rental by node"
        );

        Ok(count as u32)
    }
}

#[async_trait::async_trait]
impl ValidatorPersistence for SimplePersistence {
    async fn save_rental(&self, rental: &RentalInfo) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO rentals (
                id, validator_hotkey, node_id, container_id, ssh_session_id,
                ssh_credentials, state, created_at, updated_at, container_spec, miner_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                container_id = excluded.container_id,
                ssh_session_id = excluded.ssh_session_id,
                ssh_credentials = excluded.ssh_credentials,
                miner_id = excluded.miner_id,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata",
        )
        .bind(&rental.rental_id)
        .bind(&rental.validator_hotkey)
        .bind(&rental.node_id)
        .bind(&rental.container_id)
        .bind(&rental.ssh_session_id)
        .bind(&rental.ssh_credentials)
        .bind(match &rental.state {
            RentalState::Provisioning => "provisioning",
            RentalState::Active => "active",
            RentalState::Restarting => "restarting",
            RentalState::Stopping => "stopping",
            RentalState::Stopped => "stopped",
            RentalState::Failed => "failed",
        })
        .bind(rental.created_at.to_rfc3339())
        .bind(rental.updated_at.to_rfc3339())
        .bind(serde_json::to_string(&rental.container_spec)?)
        .bind(&rental.miner_id)
        .bind(serde_json::to_string(&rental.metadata)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_rental(&self, rental_id: &str) -> anyhow::Result<Option<RentalInfo>> {
        let filter = RentalFilter {
            rental_id: Some(rental_id.to_string()),
            ..Default::default()
        };
        self.query_rentals(filter)
            .await
            .map(|mut rentals| rentals.pop())
    }

    async fn list_validator_rentals(
        &self,
        validator_hotkey: &str,
    ) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            validator_hotkey: Some(validator_hotkey.to_string()),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn query_non_terminated_rentals(&self) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            exclude_states: Some(vec![RentalState::Stopped, RentalState::Failed]),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn delete_rental(&self, rental_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM rentals WHERE id = ?")
            .bind(rental_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    async fn setup_test_db() -> SimplePersistence {
        SimplePersistence::for_testing()
            .await
            .expect("Failed to create in-memory database with migrations")
    }

    async fn insert_test_rental(
        persistence: &SimplePersistence,
        rental_id: &Uuid,
        node_id: &str,
        state: &str,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO rentals (
                id, validator_hotkey, node_id, container_id, ssh_session_id,
                ssh_credentials, state, created_at, updated_at, container_spec, miner_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(rental_id.to_string())
        .bind("test_validator_hotkey")
        .bind(node_id)
        .bind("test_container_id")
        .bind("test_ssh_session_id")
        .bind("test_ssh_credentials")
        .bind(state)
        .bind(&now)
        .bind(&now)
        .bind("{}")
        .bind("test_miner_id")
        .bind("{}")
        .execute(&persistence.pool)
        .await?;

        Ok(())
    }

    async fn insert_test_gpu_assignment(
        persistence: &SimplePersistence,
        gpu_uuid: &str,
        gpu_index: i32,
        node_id: &str,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO gpu_uuid_assignments (
                gpu_uuid, gpu_index, node_id, miner_id, gpu_name,
                gpu_memory_gb, last_verified, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(gpu_uuid)
        .bind(gpu_index)
        .bind(node_id)
        .bind("test_miner_id")
        .bind("Tesla V100")
        .bind(32.0)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .execute(&persistence.pool)
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_with_single_gpu() {
        let persistence = setup_test_db().await;
        let rental_id = Uuid::new_v4();
        let node_id = "node_single_gpu";

        insert_test_rental(&persistence, &rental_id, node_id, "active")
            .await
            .unwrap();

        insert_test_gpu_assignment(&persistence, "gpu_0", 0, node_id)
            .await
            .unwrap();

        let count = persistence
            .get_rental_gpu_count(&rental_id)
            .await
            .expect("Failed to get GPU count");

        assert_eq!(count, 1, "Expected 1 GPU for rental");
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_with_multiple_gpus() {
        let persistence = setup_test_db().await;
        let rental_id = Uuid::new_v4();
        let node_id = "node_multi_gpu";

        insert_test_rental(&persistence, &rental_id, node_id, "active")
            .await
            .unwrap();

        for i in 0..8 {
            insert_test_gpu_assignment(&persistence, &format!("gpu_{}", i), i, node_id)
                .await
                .unwrap();
        }

        let count = persistence
            .get_rental_gpu_count(&rental_id)
            .await
            .expect("Failed to get GPU count");

        assert_eq!(count, 8, "Expected 8 GPUs for rental");
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_with_zero_gpus() {
        let persistence = setup_test_db().await;
        let rental_id = Uuid::new_v4();
        let node_id = "node_no_gpu";

        insert_test_rental(&persistence, &rental_id, node_id, "active")
            .await
            .unwrap();

        let count = persistence
            .get_rental_gpu_count(&rental_id)
            .await
            .expect("Failed to get GPU count");

        assert_eq!(
            count, 0,
            "Expected 0 GPUs for rental with no GPU assignments"
        );
    }

    #[tokio::test]
    async fn test_get_active_rental_gpu_count_filters_by_state() {
        let persistence = setup_test_db().await;
        let active_rental_id = Uuid::new_v4();
        let stopped_rental_id = Uuid::new_v4();
        let active_node_id = "node_active";
        let stopped_node_id = "node_stopped";

        insert_test_rental(&persistence, &active_rental_id, active_node_id, "active")
            .await
            .unwrap();
        insert_test_rental(&persistence, &stopped_rental_id, stopped_node_id, "stopped")
            .await
            .unwrap();

        for i in 0..4 {
            insert_test_gpu_assignment(
                &persistence,
                &format!("gpu_active_{}", i),
                i,
                active_node_id,
            )
            .await
            .unwrap();

            insert_test_gpu_assignment(
                &persistence,
                &format!("gpu_stopped_{}", i),
                i,
                stopped_node_id,
            )
            .await
            .unwrap();
        }

        let active_count = persistence
            .get_active_rental_gpu_count(&active_rental_id.to_string())
            .await
            .expect("Failed to get GPU count for active rental");

        let stopped_count = persistence
            .get_active_rental_gpu_count(&stopped_rental_id.to_string())
            .await
            .expect("Failed to get GPU count for stopped rental");

        assert_eq!(active_count, 4, "Expected 4 GPUs for active rental");
        assert_eq!(
            stopped_count, 0,
            "Expected 0 GPUs for stopped rental (filtered out)"
        );
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_by_node_id() {
        let persistence = setup_test_db().await;
        let rental_id_1 = Uuid::new_v4();
        let rental_id_2 = Uuid::new_v4();
        let node_id_1 = "node_by_id_1";
        let node_id_2 = "node_by_id_2";

        insert_test_rental(&persistence, &rental_id_1, node_id_1, "active")
            .await
            .unwrap();
        insert_test_rental(&persistence, &rental_id_2, node_id_2, "active")
            .await
            .unwrap();

        for i in 0..2 {
            insert_test_gpu_assignment(&persistence, &format!("gpu_node1_{}", i), i, node_id_1)
                .await
                .unwrap();
        }

        for i in 0..6 {
            insert_test_gpu_assignment(&persistence, &format!("gpu_node2_{}", i), i, node_id_2)
                .await
                .unwrap();
        }

        let count_node1 = persistence
            .get_rental_gpu_count_by_node_id(node_id_1)
            .await
            .expect("Failed to get GPU count by node_id");

        let count_node2 = persistence
            .get_rental_gpu_count_by_node_id(node_id_2)
            .await
            .expect("Failed to get GPU count by node_id");

        assert_eq!(count_node1, 2, "Expected 2 GPUs for node_id_1");
        assert_eq!(count_node2, 6, "Expected 6 GPUs for node_id_2");
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_by_node_id_filters_inactive_rentals() {
        let persistence = setup_test_db().await;
        let active_rental_id = Uuid::new_v4();
        let stopped_rental_id = Uuid::new_v4();
        let active_node_id = "node_active_by_id";
        let stopped_node_id = "node_stopped_by_id";

        insert_test_rental(&persistence, &active_rental_id, active_node_id, "active")
            .await
            .unwrap();
        insert_test_rental(&persistence, &stopped_rental_id, stopped_node_id, "stopped")
            .await
            .unwrap();

        for i in 0..3 {
            insert_test_gpu_assignment(
                &persistence,
                &format!("gpu_active_byid_{}", i),
                i,
                active_node_id,
            )
            .await
            .unwrap();

            insert_test_gpu_assignment(
                &persistence,
                &format!("gpu_stopped_byid_{}", i),
                i,
                stopped_node_id,
            )
            .await
            .unwrap();
        }

        let active_count = persistence
            .get_rental_gpu_count_by_node_id(active_node_id)
            .await
            .expect("Failed to get GPU count by node_id");

        let stopped_count = persistence
            .get_rental_gpu_count_by_node_id(stopped_node_id)
            .await
            .expect("Failed to get GPU count by node_id");

        assert_eq!(active_count, 3, "Expected 3 GPUs for active node");
        assert_eq!(
            stopped_count, 0,
            "Expected 0 GPUs for stopped node (filtered out)"
        );
    }

    #[tokio::test]
    async fn test_get_rental_gpu_count_nonexistent_rental() {
        let persistence = setup_test_db().await;
        let nonexistent_rental_id = Uuid::new_v4();

        let count = persistence
            .get_rental_gpu_count(&nonexistent_rental_id)
            .await
            .expect("Failed to get GPU count");

        assert_eq!(
            count, 0,
            "Expected 0 GPUs for nonexistent rental (no matching rows)"
        );
    }
}
