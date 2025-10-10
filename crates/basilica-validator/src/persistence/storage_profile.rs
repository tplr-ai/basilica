use sqlx::Row;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    pub async fn store_node_storage_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        profile: &crate::miner_prover::validation_storage::StorageProfile,
    ) -> Result<(), anyhow::Error> {
        let filesystem_details_json = serde_json::to_string(&profile.filesystem_details)?;

        sqlx::query(
            r#"
            INSERT INTO node_storage_profile
            (miner_uid, node_id, total_bytes, available_bytes,
             required_bytes, filesystem_details, collection_timestamp, full_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                total_bytes = excluded.total_bytes,
                available_bytes = excluded.available_bytes,
                required_bytes = excluded.required_bytes,
                filesystem_details = excluded.filesystem_details,
                collection_timestamp = excluded.collection_timestamp,
                full_json = excluded.full_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(profile.total_bytes as i64)
        .bind(profile.available_bytes as i64)
        .bind(profile.required_bytes as i64)
        .bind(&filesystem_details_json)
        .bind(profile.collection_timestamp.to_rfc3339())
        .bind(&profile.full_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_node_storage_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<crate::miner_prover::validation_storage::StorageProfile>, anyhow::Error>
    {
        let row = sqlx::query(
            r#"
            SELECT total_bytes, available_bytes, required_bytes,
                   filesystem_details, collection_timestamp, full_json
            FROM node_storage_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let total_bytes: i64 = row.get("total_bytes");
            let available_bytes: i64 = row.get("available_bytes");
            let required_bytes: i64 = row.get("required_bytes");
            let filesystem_details_json: String = row.get("filesystem_details");
            let collection_timestamp_str: String = row.get("collection_timestamp");
            let full_json: String = row.get("full_json");

            let filesystem_details = serde_json::from_str(&filesystem_details_json)?;
            let collection_timestamp =
                chrono::DateTime::parse_from_rfc3339(&collection_timestamp_str)?
                    .with_timezone(&chrono::Utc);

            Ok(Some(
                crate::miner_prover::validation_storage::StorageProfile {
                    total_bytes: total_bytes as u64,
                    available_bytes: available_bytes as u64,
                    required_bytes: required_bytes as u64,
                    filesystem_details,
                    collection_timestamp,
                    full_json,
                },
            ))
        } else {
            Ok(None)
        }
    }
}
