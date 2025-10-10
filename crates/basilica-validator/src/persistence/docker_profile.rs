use sqlx::Row;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    #[allow(clippy::too_many_arguments)]
    pub async fn store_node_docker_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        service_active: bool,
        docker_version: Option<String>,
        images_pulled: Vec<String>,
        dind_supported: bool,
        validation_error: Option<String>,
        full_json: &str,
    ) -> Result<(), anyhow::Error> {
        let images_json = serde_json::to_string(&images_pulled)?;

        sqlx::query(
            r#"
            INSERT INTO node_docker_profile
            (miner_uid, node_id, service_active, docker_version, images_pulled,
             dind_supported, validation_error, full_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                service_active = excluded.service_active,
                docker_version = excluded.docker_version,
                images_pulled = excluded.images_pulled,
                dind_supported = excluded.dind_supported,
                validation_error = excluded.validation_error,
                full_json = excluded.full_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(service_active)
        .bind(docker_version)
        .bind(&images_json)
        .bind(dind_supported)
        .bind(validation_error)
        .bind(full_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_node_docker_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<
        Option<(
            String,
            bool,
            Option<String>,
            Vec<String>,
            bool,
            Option<String>,
        )>,
        anyhow::Error,
    > {
        let row = sqlx::query(
            r#"
            SELECT service_active, docker_version, images_pulled, dind_supported, validation_error, full_json
            FROM node_docker_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_json: String = row.get("full_json");
            let service_active: bool = row.get("service_active");
            let docker_version: Option<String> = row.get("docker_version");
            let images_pulled_json: String = row.get("images_pulled");
            let images_pulled: Vec<String> = serde_json::from_str(&images_pulled_json)?;
            let dind_supported: bool = row.get("dind_supported");
            let validation_error: Option<String> = row.get("validation_error");

            Ok(Some((
                full_json,
                service_active,
                docker_version,
                images_pulled,
                dind_supported,
                validation_error,
            )))
        } else {
            Ok(None)
        }
    }
}
