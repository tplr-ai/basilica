use sqlx::Row;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    pub async fn store_node_nat_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        profile: &crate::miner_prover::validation_nat::NatProfile,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO node_nat_profile
            (miner_uid, node_id, is_accessible, test_port, test_path, container_id,
             response_content, test_timestamp, full_json, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                is_accessible = excluded.is_accessible,
                test_port = excluded.test_port,
                test_path = excluded.test_path,
                container_id = excluded.container_id,
                response_content = excluded.response_content,
                test_timestamp = excluded.test_timestamp,
                full_json = excluded.full_json,
                error_message = excluded.error_message,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(profile.is_accessible)
        .bind(profile.test_port as i32)
        .bind(&profile.test_path)
        .bind(&profile.container_id)
        .bind(&profile.response_content)
        .bind(profile.test_timestamp.to_rfc3339())
        .bind(&profile.full_json)
        .bind(&profile.error_message)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_node_nat_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<crate::miner_prover::validation_nat::NatProfile>, anyhow::Error> {
        let row = sqlx::query(
            r#"
            SELECT is_accessible, test_port, test_path, container_id, response_content,
                   test_timestamp, full_json, error_message
            FROM node_nat_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let is_accessible: bool = row.get("is_accessible");
            let test_port: i32 = row.get("test_port");
            let test_path: String = row.get("test_path");
            let container_id: Option<String> = row.get("container_id");
            let response_content: Option<String> = row.get("response_content");
            let test_timestamp_str: String = row.get("test_timestamp");
            let full_json: String = row.get("full_json");
            let error_message: Option<String> = row.get("error_message");

            let test_timestamp = chrono::DateTime::parse_from_rfc3339(&test_timestamp_str)?
                .with_timezone(&chrono::Utc);

            Ok(Some(crate::miner_prover::validation_nat::NatProfile {
                is_accessible,
                test_port: test_port as u16,
                test_path,
                container_id,
                response_content,
                test_timestamp,
                full_json,
                error_message,
            }))
        } else {
            Ok(None)
        }
    }
}
