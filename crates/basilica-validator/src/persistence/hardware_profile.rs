use sqlx::Row;
use tracing::info;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    #[allow(clippy::too_many_arguments)]
    pub async fn store_node_hardware_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        cpu_model: Option<String>,
        cpu_cores: Option<i32>,
        ram_gb: Option<i32>,
        disk_gb: Option<i32>,
        full_hardware_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO node_hardware_profile
            (miner_uid, node_id, cpu_model, cpu_cores, ram_gb, disk_gb, full_hardware_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                cpu_model = excluded.cpu_model,
                cpu_cores = excluded.cpu_cores,
                ram_gb = excluded.ram_gb,
                disk_gb = excluded.disk_gb,
                full_hardware_json = excluded.full_hardware_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(cpu_model)
        .bind(cpu_cores)
        .bind(ram_gb)
        .bind(disk_gb)
        .bind(full_hardware_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "Stored hardware profile for node"
        );

        Ok(())
    }

    pub async fn get_node_hardware_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<
        Option<(
            String,
            Option<String>,
            Option<i32>,
            Option<i32>,
            Option<i32>,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT cpu_model, cpu_cores, ram_gb, disk_gb, full_hardware_json
            FROM node_hardware_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_hardware_json: String = row.get("full_hardware_json");
            let cpu_model: Option<String> = row.get("cpu_model");
            let cpu_cores: Option<i32> = row.get("cpu_cores");
            let ram_gb: Option<i32> = row.get("ram_gb");
            let disk_gb: Option<i32> = row.get("disk_gb");

            Ok(Some((
                full_hardware_json,
                cpu_model,
                cpu_cores,
                ram_gb,
                disk_gb,
            )))
        } else {
            Ok(None)
        }
    }
}
