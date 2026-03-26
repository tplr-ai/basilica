use crate::persistence::SimplePersistence;
use alloy_primitives::{Address, U256};
use chrono::Utc;
use collateral_contract::{Deposit, Reclaimed, Slashed};
use hex::ToHex;
use sqlx::Row;
use tracing::warn;

impl SimplePersistence {
    pub async fn get_last_scanned_block_number(&self) -> Result<u64, anyhow::Error> {
        let query = "SELECT last_scanned_block_number FROM collateral_scan_status WHERE id = 1";

        let row = sqlx::query(query).fetch_one(self.pool()).await?;

        let block_number: i64 = row.get(0);
        Ok(block_number as u64)
    }

    pub async fn update_last_scanned_block_number(
        &self,
        last_scanned_block: u64,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        let query =
            "UPDATE collateral_scan_status SET last_scanned_block_number = ?, updated_at = ? WHERE id = 1";

        sqlx::query(query)
            .bind(last_scanned_block as i64)
            .bind(now)
            .execute(self.pool())
            .await?;

        Ok(())
    }

    pub async fn get_collateral_status_id(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<(i64, U256)>, anyhow::Error> {
        let query = "SELECT id, collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?";

        let row = sqlx::query(query)
            .bind(hotkey)
            .bind(node_id)
            .fetch_optional(self.pool())
            .await?;

        if let Some(row) = row {
            let id: i64 = row.get(0);
            let collateral_str: String = row.get(1);
            let collateral = U256::from_str_radix(&collateral_str, 10)
                .map_err(|_| anyhow::anyhow!("Invalid collateral"))?;
            Ok(Some((id, collateral)))
        } else {
            Ok(None)
        }
    }

    async fn get_collateral_amount_internal(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hotkey)
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(row) = row {
            let collateral_str: String = row.get(0);
            let collateral = U256::from_str_radix(&collateral_str, 10)
                .map_err(|_| anyhow::anyhow!("Invalid collateral"))?;
            Ok(Some(collateral))
        } else {
            Ok(None)
        }
    }

    pub async fn get_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount_internal(hotkey, node_id).await
    }

    pub async fn get_alpha_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount_internal(hotkey, node_id).await
    }

    pub async fn handle_deposit(&self, deposit: &Deposit) -> Result<(), anyhow::Error> {
        match self
            .get_collateral_status_id(
                deposit.hotkey.encode_hex::<String>().as_str(),
                deposit.nodeId.encode_hex::<String>().as_str(),
            )
            .await?
        {
            Some((id, collateral)) => {
                let now = Utc::now().to_rfc3339();
                let query =
                    "UPDATE collateral_status SET collateral = ?, updated_at = ? WHERE id = ?";
                let new_collateral = collateral.saturating_add(deposit.alphaAmount);
                sqlx::query(query)
                    .bind(new_collateral.to_string())
                    .bind(now)
                    .bind(id)
                    .execute(self.pool())
                    .await?;
            }
            None => {
                let query = "INSERT INTO collateral_status (hotkey, node_id, miner, collateral, updated_at) VALUES (?, ?, ?, ?, ?)";
                sqlx::query(query)
                    .bind(deposit.hotkey.encode_hex::<String>())
                    .bind(deposit.nodeId.encode_hex::<String>())
                    .bind(format!(
                        "0x{}",
                        deposit.miner.as_slice().encode_hex::<String>()
                    ))
                    .bind(deposit.alphaAmount.to_string())
                    .bind(Utc::now().to_rfc3339())
                    .execute(self.pool())
                    .await?;
            }
        }

        Ok(())
    }

    pub async fn handle_reclaimed(&self, reclaimed: &Reclaimed) -> Result<(), anyhow::Error> {
        match self
            .get_collateral_status_id(
                reclaimed.hotkey.encode_hex::<String>().as_str(),
                reclaimed.nodeId.encode_hex::<String>().as_str(),
            )
            .await?
        {
            Some((id, collateral)) => {
                let now = Utc::now().to_rfc3339();
                let query =
                    "UPDATE collateral_status SET collateral = ?, updated_at = ? WHERE id = ?";
                let new_collateral = collateral.saturating_sub(reclaimed.alphaAmount);
                sqlx::query(query)
                    .bind(new_collateral.to_string())
                    .bind(now)
                    .bind(id)
                    .execute(self.pool())
                    .await?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("Collateral status not found")),
        }
    }

    pub async fn handle_slashed(&self, slashed: &Slashed) -> Result<(), anyhow::Error> {
        match self
            .get_collateral_status_id(
                slashed.hotkey.encode_hex::<String>().as_str(),
                slashed.nodeId.encode_hex::<String>().as_str(),
            )
            .await?
        {
            Some((id, collateral)) => {
                let now = Utc::now().to_rfc3339();
                let query = "UPDATE collateral_status SET collateral = ?, miner = ? , url = ? , url_content_md5_checksum = ?, updated_at = ? WHERE id = ?";
                if slashed.slashAlphaAmount != collateral {
                    warn!(
                        "Slashed amount {} does not match collateral {} in database",
                        slashed.slashAlphaAmount, collateral
                    );
                }
                let new_collateral = collateral.saturating_sub(slashed.slashAlphaAmount);

                sqlx::query(query)
                    .bind(new_collateral.to_string())
                    .bind(format!(
                        "0x{}",
                        Address::ZERO.as_slice().encode_hex::<String>()
                    ))
                    .bind(slashed.url.clone())
                    .bind(slashed.urlContentMd5Checksum.encode_hex::<String>())
                    .bind(now)
                    .bind(id)
                    .execute(self.pool())
                    .await?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("Collateral status not found")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::FixedBytes;
    use collateral_contract::config::CONTRACT_DEPLOYED_BLOCK_NUMBER;

    fn make_hotkey(byte: u8) -> [u8; 32] {
        [byte; 32]
    }
    fn make_node_id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn ev_deposit(hk: [u8; 32], ex: [u8; 16], amount: u64) -> Deposit {
        Deposit {
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner: Address::from_slice(&[0u8; 20]),
            amount: U256::ZERO,
            alphaHotkey: FixedBytes::from_slice(&[0u8; 32]),
            alphaAmount: U256::from(amount),
        }
    }
    fn ev_reclaimed(hk: [u8; 32], ex: [u8; 16], amount: u64) -> Reclaimed {
        Reclaimed {
            reclaimRequestId: U256::from(1u64),
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner: Address::from_slice(&[0u8; 20]),
            amount: U256::ZERO,
            alphaColdkey: FixedBytes::from_slice(&[0u8; 32]),
            alphaAmount: U256::from(amount),
        }
    }
    fn ev_slashed(hk: [u8; 32], ex: [u8; 16], amount: u64) -> Slashed {
        Slashed {
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner: Address::from_slice(&[0u8; 20]),
            slashAmount: U256::ZERO,
            slashAlphaAmount: U256::from(amount),
            url: String::new(),
            urlContentMd5Checksum: FixedBytes::from_slice(&[0u8; 16]),
        }
    }

    #[tokio::test]
    async fn test_tables_and_index_creation() {
        let _persistence = SimplePersistence::for_testing().await.expect("persistence");
    }

    #[tokio::test]
    async fn test_scan_block_number_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        // seed row
        sqlx::query("UPDATE collateral_scan_status SET last_scanned_block_number = 1 WHERE id = 1")
            .execute(persistence.pool())
            .await
            .unwrap();

        let n = persistence.get_last_scanned_block_number().await.unwrap();
        assert_eq!(n, 1);

        persistence
            .update_last_scanned_block_number(42)
            .await
            .unwrap();

        let n2: i64 =
            sqlx::query_scalar("SELECT last_scanned_block_number FROM collateral_scan_status")
                .fetch_one(persistence.pool())
                .await
                .unwrap();
        assert_eq!(n2 as u64, 42);
    }

    #[tokio::test]
    async fn test_handle_deposit_insert_and_update() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(1);
        let ex = make_node_id(2);

        // first deposit inserts
        let d1 = ev_deposit(hk, ex, 100);
        persistence.handle_deposit(&d1).await.unwrap();

        let coll1: String = sqlx::query_scalar(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(d1.hotkey.encode_hex::<String>())
        .bind(d1.nodeId.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();
        assert_eq!(coll1, "100");

        // second deposit updates
        let d2 = ev_deposit(hk, ex, 50);
        persistence.handle_deposit(&d2).await.unwrap();
        let coll2: String = sqlx::query_scalar(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(d1.hotkey.encode_hex::<String>())
        .bind(d1.nodeId.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();
        assert_eq!(coll2, "150");
    }

    #[tokio::test]
    async fn test_handle_reclaimed_and_slashed() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(9);
        let ex = make_node_id(7);

        // seed with deposit 200
        let d = ev_deposit(hk, ex, 200);
        persistence.handle_deposit(&d).await.unwrap();

        // reclaim 80
        let r = ev_reclaimed(hk, ex, 80);
        persistence.handle_reclaimed(&r).await.unwrap();
        let coll_after_reclaim: String = sqlx::query_scalar(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(d.hotkey.encode_hex::<String>())
        .bind(d.nodeId.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();
        assert_eq!(coll_after_reclaim, "120");

        // slash 20
        let s = ev_slashed(hk, ex, 20);
        persistence.handle_slashed(&s).await.unwrap();
        let coll_after_slash: String = sqlx::query_scalar(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(d.hotkey.encode_hex::<String>())
        .bind(d.nodeId.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();
        assert_eq!(coll_after_slash, "100");
    }

    #[tokio::test]
    async fn test_get_collateral_status_id_found() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(5);
        let ex = make_node_id(6);
        let d = ev_deposit(hk, ex, 999);
        persistence.handle_deposit(&d).await.unwrap();

        let result = persistence
            .get_collateral_status_id(
                &d.hotkey.encode_hex::<String>(),
                &d.nodeId.encode_hex::<String>(),
            )
            .await
            .unwrap();

        assert!(result.is_some());
        let (id, collateral) = result.unwrap();
        assert!(id > 0);
        assert_eq!(collateral, U256::from(999));
    }

    #[tokio::test]
    async fn test_get_collateral_status_id_not_found() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let result = persistence
            .get_collateral_status_id("nonexistent_hotkey", "nonexistent_node")
            .await
            .unwrap();

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_collateral_amount_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();
        let hk = make_hotkey(40);
        let ex = make_node_id(41);
        let d = ev_deposit(hk, ex, 321);
        persistence.handle_deposit(&d).await.unwrap();

        let amount = persistence
            .get_alpha_collateral_amount(
                &d.hotkey.encode_hex::<String>(),
                &d.nodeId.encode_hex::<String>(),
            )
            .await
            .unwrap();
        assert_eq!(amount, Some(U256::from(321u64)));
    }

    #[tokio::test]
    async fn test_handle_reclaimed_not_found() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(10);
        let ex = make_node_id(11);
        let r = ev_reclaimed(hk, ex, 50);

        let result = persistence.handle_reclaimed(&r).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Collateral status not found"));
    }

    #[tokio::test]
    async fn test_handle_slashed_not_found() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(12);
        let ex = make_node_id(13);
        let s = ev_slashed(hk, ex, 100);

        let result = persistence.handle_slashed(&s).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Collateral status not found"));
    }

    #[tokio::test]
    async fn test_handle_slashed_with_url_data() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(14);
        let ex = make_node_id(15);

        // Setup initial deposit
        let d = ev_deposit(hk, ex, 500);
        persistence.handle_deposit(&d).await.unwrap();

        // Create slashed event with URL data
        let mut s = ev_slashed(hk, ex, 500);
        s.url = "https://example.com/proof".to_string();
        s.urlContentMd5Checksum = FixedBytes::from_slice(&[
            0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56,
            0x78, 0x90,
        ]);

        persistence.handle_slashed(&s).await.unwrap();

        // Verify URL and checksum were stored
        let (url, checksum): (String, String) = sqlx::query_as(
            "SELECT url, url_content_md5_checksum FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(d.hotkey.encode_hex::<String>())
        .bind(d.nodeId.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(url, "https://example.com/proof");
        assert_eq!(checksum, "abcdef1234567890abcdef1234567890");
    }

    #[tokio::test]
    async fn test_update_last_scanned_block_number_large_values() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let large_block = u64::MAX - 1000;
        persistence
            .update_last_scanned_block_number(large_block)
            .await
            .unwrap();

        let retrieved = persistence.get_last_scanned_block_number().await.unwrap();
        assert_eq!(retrieved, large_block);
    }

    #[tokio::test]
    async fn test_handle_deposit_multiple_miners_same_node() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let ex = make_node_id(20);
        let hk1 = make_hotkey(21);
        let hk2 = make_hotkey(22);

        // Same node ID, different hotkeys
        let d1 = ev_deposit(hk1, ex, 100);
        let d2 = ev_deposit(hk2, ex, 200);

        persistence.handle_deposit(&d1).await.unwrap();
        persistence.handle_deposit(&d2).await.unwrap();

        // Verify both entries exist separately
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM collateral_status WHERE node_id = ?")
                .bind(ex.encode_hex::<String>())
                .fetch_one(persistence.pool())
                .await
                .unwrap();

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_handle_deposit_overflow_protection() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(25);
        let ex = make_node_id(26);

        // Create a deposit with a very large amount
        let mut d1 = ev_deposit(hk, ex, 0);
        d1.alphaAmount = U256::MAX - U256::from(1000u64);
        persistence.handle_deposit(&d1).await.unwrap();

        // Add another deposit that would overflow if not using saturating_add
        let mut d2 = ev_deposit(hk, ex, 0);
        d2.alphaAmount = U256::from(2000u64);
        persistence.handle_deposit(&d2).await.unwrap();

        // Verify the result is U256::MAX (saturating add)
        let result = persistence
            .get_collateral_status_id(
                &d1.hotkey.encode_hex::<String>(),
                &d1.nodeId.encode_hex::<String>(),
            )
            .await
            .unwrap();

        assert!(result.is_some());
        let (_, collateral) = result.unwrap();
        assert_eq!(collateral, U256::MAX);
    }

    #[tokio::test]
    async fn test_handle_reclaimed_underflow_protection() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(27);
        let ex = make_node_id(28);

        // Setup with small deposit
        let d = ev_deposit(hk, ex, 100);
        persistence.handle_deposit(&d).await.unwrap();

        // Try to reclaim more than available (should use saturating_sub)
        let r = ev_reclaimed(hk, ex, 200);
        persistence.handle_reclaimed(&r).await.unwrap();

        // Verify the result is 0 (saturating sub)
        let result = persistence
            .get_collateral_status_id(
                &d.hotkey.encode_hex::<String>(),
                &d.nodeId.encode_hex::<String>(),
            )
            .await
            .unwrap();

        assert!(result.is_some());
        let (_, collateral) = result.unwrap();
        assert_eq!(collateral, U256::ZERO);
    }

    #[tokio::test]
    async fn test_table_unique_constraint() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(30);
        let ex = make_node_id(31);

        // First deposit should succeed
        let d1 = ev_deposit(hk, ex, 100);
        persistence.handle_deposit(&d1).await.unwrap();

        // Second deposit with same hotkey and node should update, not duplicate
        let d2 = ev_deposit(hk, ex, 50);
        persistence.handle_deposit(&d2).await.unwrap();

        // Verify only one row exists
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hk.encode_hex::<String>())
        .bind(ex.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(count, 1);

        // Verify the collateral was updated to 150
        let collateral: String = sqlx::query_scalar(
            "SELECT collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hk.encode_hex::<String>())
        .bind(ex.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(collateral, "150");
    }

    #[tokio::test]
    async fn test_scan_status_table_initialization() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        // Verify the scan status table has the initial row
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_scan_status")
            .fetch_one(persistence.pool())
            .await
            .unwrap();

        assert_eq!(count, 1);

        // Verify the initial block number is set to CONTRACT_DEPLOYED_BLOCK_NUMBER
        let initial_block: i64 = sqlx::query_scalar(
            "SELECT last_scanned_block_number FROM collateral_scan_status WHERE id = 1",
        )
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(initial_block as u64, CONTRACT_DEPLOYED_BLOCK_NUMBER);
    }

    #[tokio::test]
    async fn test_timestamp_fields_updated() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(35);
        let ex = make_node_id(36);

        // Create deposit and verify updated_at is set
        let d = ev_deposit(hk, ex, 100);
        persistence.handle_deposit(&d).await.unwrap();

        let updated_at: String = sqlx::query_scalar(
            "SELECT updated_at FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hk.encode_hex::<String>())
        .bind(ex.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        // Verify it's a valid RFC3339 timestamp
        assert!(chrono::DateTime::parse_from_rfc3339(&updated_at).is_ok());

        // Update scan block number and verify timestamp
        let old_timestamp: String =
            sqlx::query_scalar("SELECT updated_at FROM collateral_scan_status WHERE id = 1")
                .fetch_one(persistence.pool())
                .await
                .unwrap();

        // Sleep briefly to ensure timestamp difference
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        persistence
            .update_last_scanned_block_number(12345)
            .await
            .unwrap();

        let new_timestamp: String =
            sqlx::query_scalar("SELECT updated_at FROM collateral_scan_status WHERE id = 1")
                .fetch_one(persistence.pool())
                .await
                .unwrap();

        assert_ne!(old_timestamp, new_timestamp);
        assert!(chrono::DateTime::parse_from_rfc3339(&new_timestamp).is_ok());
    }
}
