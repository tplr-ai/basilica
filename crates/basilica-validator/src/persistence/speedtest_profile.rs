use sqlx::Row;
use tracing::info;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    #[allow(clippy::too_many_arguments)]
    pub async fn store_node_speedtest_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
        download_mbps: Option<f64>,
        upload_mbps: Option<f64>,
        test_timestamp: &str,
        test_server: Option<String>,
        full_result_json: &str,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            r#"
            INSERT INTO node_speedtest_profile
            (miner_uid, node_id, download_mbps, upload_mbps, test_timestamp, test_server, full_result_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid, node_id) DO UPDATE SET
                download_mbps = excluded.download_mbps,
                upload_mbps = excluded.upload_mbps,
                test_timestamp = excluded.test_timestamp,
                test_server = excluded.test_server,
                full_result_json = excluded.full_result_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .bind(download_mbps)
        .bind(upload_mbps)
        .bind(test_timestamp)
        .bind(test_server)
        .bind(full_result_json)
        .execute(&self.pool)
        .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            download_mbps = download_mbps.unwrap_or(0.0),
            upload_mbps = upload_mbps.unwrap_or(0.0),
            "Stored speedtest profile for node"
        );

        Ok(())
    }

    pub async fn get_node_speedtest_profile(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<(String, Option<f64>, Option<f64>, String, Option<String>)>, anyhow::Error>
    {
        let row = sqlx::query(
            r#"
            SELECT download_mbps, upload_mbps, test_timestamp, test_server, full_result_json
            FROM node_speedtest_profile
            WHERE miner_uid = ? AND node_id = ?
            "#,
        )
        .bind(miner_uid as i32)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let full_result_json: String = row.get("full_result_json");
            let download_mbps: Option<f64> = row.get("download_mbps");
            let upload_mbps: Option<f64> = row.get("upload_mbps");
            let test_timestamp: String = row.get("test_timestamp");
            let test_server: Option<String> = row.get("test_server");

            Ok(Some((
                full_result_json,
                download_mbps,
                upload_mbps,
                test_timestamp,
                test_server,
            )))
        } else {
            Ok(None)
        }
    }
}
