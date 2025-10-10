use chrono::{DateTime, Utc};
use sqlx::{QueryBuilder, Row};
use tracing::warn;
use uuid::Uuid;

use crate::persistence::entities::{Rental, RentalStatus};
use crate::persistence::simple_persistence::SimplePersistence;
use crate::persistence::types::RentalFilter;
use crate::rental::{RentalInfo, RentalState};

impl SimplePersistence {
    pub async fn create_rental(&self, rental: &Rental) -> Result<(), anyhow::Error> {
        let query = r#"
            INSERT INTO rentals (
                id, node_id, customer_public_key, docker_image, env_vars,
                gpu_requirements, ssh_access_info, max_duration_hours, cost_per_hour,
                status, created_at, updated_at, started_at, terminated_at,
                termination_reason, total_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            .bind(rental.max_duration_hours as i64)
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

    /// Check if an node has an active rental
    pub async fn has_active_rental(
        &self,
        node_id: &str,
        miner_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let query = r#"
            SELECT COUNT(*) as count
            FROM rentals
            WHERE node_id = ?
                AND miner_id = ?
                AND state = 'active'
        "#;

        let row = sqlx::query(query)
            .bind(node_id)
            .bind(miner_id)
            .fetch_one(&self.pool)
            .await?;

        let count: i64 = row.get("count");
        Ok(count > 0)
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
        let container_spec_str: String = row.get("container_spec");
        let rental_id: String = row.get("id");
        let node_id: String = row.get("node_id");
        let metadata: String = row.get("metadata");

        let state = Self::parse_rental_state(&state_str, &rental_id);

        Ok(RentalInfo {
            rental_id,
            validator_hotkey: row.get("validator_hotkey"),
            node_id,
            container_id: row.get("container_id"),
            ssh_session_id: row.get("ssh_session_id"),
            ssh_credentials: row.get("ssh_credentials"),
            state,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc),
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
            max_duration_hours: row.get::<i64, _>("max_duration_hours") as u32,
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
}
