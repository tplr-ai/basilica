use chrono::Utc;

use crate::persistence::simple_persistence::SimplePersistence;

impl SimplePersistence {
    /// Schedule verification for miner
    pub async fn schedule_verification(
        &self,
        miner_id: &str,
        verification_id: &str,
        verification_type: &str,
        node_id: Option<&str>,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO verification_requests (id, miner_id, verification_type, node_id, scheduled_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(verification_id)
        .bind(miner_id)
        .bind(verification_type)
        .bind(node_id)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
