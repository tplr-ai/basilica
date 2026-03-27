use crate::persistence::SimplePersistence;
use alloy_primitives::{Address, U256};
use chrono::Utc;
use hex::ToHex;
use sqlx::Row;

pub(crate) fn parse_u256_decimal(value: &str, field: &str) -> Result<U256, anyhow::Error> {
    U256::from_str_radix(value, 10).map_err(|_| anyhow::anyhow!("Invalid {}", field))
}

pub(crate) fn address_to_string(address: Address) -> String {
    format!("0x{}", address.as_slice().encode_hex::<String>())
}

#[derive(Debug, Clone)]
pub struct CollateralNodeRecord {
    pub hotkey: String,
    pub node_id: String,
    pub tao_collateral: U256,
    pub alpha_collateral: U256,
}

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
        let query =
            "SELECT id, alpha_collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?";

        let row = sqlx::query(query)
            .bind(hotkey)
            .bind(node_id)
            .fetch_optional(self.pool())
            .await?;

        if let Some(row) = row {
            let id: i64 = row.get(0);
            let collateral_str: String = row.get(1);
            let collateral = parse_u256_decimal(&collateral_str, "alpha_collateral")?;
            Ok(Some((id, collateral)))
        } else {
            Ok(None)
        }
    }

    async fn get_collateral_amount_internal(
        &self,
        hotkey: &str,
        node_id: &str,
        column: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        let query = match column {
            "tao_collateral" => {
                "SELECT tao_collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?"
            }
            "alpha_collateral" => {
                "SELECT alpha_collateral FROM collateral_status WHERE hotkey = ? AND node_id = ?"
            }
            _ => return Err(anyhow::anyhow!("Unsupported collateral column: {}", column)),
        };

        let row = sqlx::query(query)
            .bind(hotkey)
            .bind(node_id)
            .fetch_optional(self.pool())
            .await?;

        if let Some(row) = row {
            let collateral_str: String = row.get(0);
            let collateral = parse_u256_decimal(&collateral_str, column)?;
            Ok(Some(collateral))
        } else {
            Ok(None)
        }
    }

    pub async fn get_tao_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount_internal(hotkey, node_id, "tao_collateral")
            .await
    }

    pub async fn get_alpha_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount_internal(hotkey, node_id, "alpha_collateral")
            .await
    }

    pub async fn get_all_collateral_nodes(
        &self,
    ) -> Result<Vec<CollateralNodeRecord>, anyhow::Error> {
        let rows = sqlx::query(
            "SELECT hotkey, node_id, tao_collateral, alpha_collateral FROM collateral_status",
        )
        .fetch_all(self.pool())
        .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let hotkey: String = row.get("hotkey");
            let node_id: String = row.get("node_id");
            let tao_collateral =
                parse_u256_decimal(&row.get::<String, _>("tao_collateral"), "tao_collateral")?;
            let alpha_collateral = parse_u256_decimal(
                &row.get::<String, _>("alpha_collateral"),
                "alpha_collateral",
            )?;
            records.push(CollateralNodeRecord {
                hotkey,
                node_id,
                tao_collateral,
                alpha_collateral,
            });
        }
        Ok(records)
    }

    /// Replace the entire collateral_status table with on-chain node data.
    /// Nodes present on-chain are upserted; DB rows not present on-chain are deleted.
    pub async fn sync_all_collateral_nodes(
        &self,
        nodes: &[collateral_contract::NodeCollateralInfo],
    ) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        let now = Utc::now().to_rfc3339();

        // Build a set of on-chain (hotkey, node_id) pairs
        let mut on_chain_keys: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::with_capacity(nodes.len());

        for node in nodes {
            let hotkey = format!("0x{}", hex::encode(node.miner_hotkey));
            let node_id = format!("0x{}", hex::encode(node.node_id));
            let miner = address_to_string(node.miner);

            on_chain_keys.insert((hotkey.clone(), node_id.clone()));

            // Upsert: INSERT OR REPLACE
            sqlx::query(
                "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?)
                 ON CONFLICT(hotkey, node_id) DO UPDATE SET
                   miner = excluded.miner,
                   tao_collateral = excluded.tao_collateral,
                   alpha_collateral = excluded.alpha_collateral,
                   updated_at = excluded.updated_at",
            )
            .bind(&hotkey)
            .bind(&node_id)
            .bind(&miner)
            .bind(node.tao_collateral.to_string())
            .bind(node.alpha_collateral.to_string())
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        // Delete DB rows that are no longer on-chain (zero-balance nodes)
        let db_nodes: Vec<(String, String)> =
            sqlx::query_as("SELECT hotkey, node_id FROM collateral_status")
                .fetch_all(&mut *tx)
                .await?;

        for (hotkey, node_id) in &db_nodes {
            if !on_chain_keys.contains(&(hotkey.clone(), node_id.clone())) {
                sqlx::query("DELETE FROM collateral_status WHERE hotkey = ? AND node_id = ?")
                    .bind(hotkey)
                    .bind(node_id)
                    .execute(&mut *tx)
                    .await?;
            }
        }

        tx.commit().await?;
        Ok(())
    }

    /// Sync the collateral_reclaims table with on-chain reclaim data.
    /// Reclaims present on-chain are upserted; DB rows not present on-chain are deleted.
    pub async fn sync_all_reclaims(
        &self,
        reclaims: &[collateral_contract::ReclaimInfo],
    ) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        let now = Utc::now().to_rfc3339();

        // Build a set of on-chain reclaim_request_ids
        let mut on_chain_ids: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(reclaims.len());

        for r in reclaims {
            let reclaim_request_id = r.reclaim_request_id.to_string();
            let hotkey = format!("0x{}", hex::encode(r.miner_hotkey));
            let node_id = format!("0x{}", hex::encode(r.node_id));
            let miner = address_to_string(r.miner);

            on_chain_ids.insert(reclaim_request_id.clone());

            // Upsert: INSERT OR UPDATE on primary key conflict
            sqlx::query(
                "INSERT INTO collateral_reclaims (reclaim_request_id, hotkey, node_id, miner, requested_tao_amount, requested_alpha_amount, deny_timeout, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(reclaim_request_id) DO UPDATE SET
                   hotkey = excluded.hotkey,
                   node_id = excluded.node_id,
                   miner = excluded.miner,
                   requested_tao_amount = excluded.requested_tao_amount,
                   requested_alpha_amount = excluded.requested_alpha_amount,
                   deny_timeout = excluded.deny_timeout,
                   updated_at = excluded.updated_at",
            )
            .bind(&reclaim_request_id)
            .bind(&hotkey)
            .bind(&node_id)
            .bind(&miner)
            .bind(r.amount.to_string())
            .bind(r.alpha_amount.to_string())
            .bind(r.deny_timeout.to_string())
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        // Delete DB rows that are no longer on-chain
        let db_ids: Vec<(String,)> =
            sqlx::query_as("SELECT reclaim_request_id FROM collateral_reclaims")
                .fetch_all(&mut *tx)
                .await?;

        for (id,) in &db_ids {
            if !on_chain_ids.contains(id) {
                sqlx::query("DELETE FROM collateral_reclaims WHERE reclaim_request_id = ?")
                    .bind(id)
                    .execute(&mut *tx)
                    .await?;
            }
        }

        tx.commit().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    async fn insert_collateral_row(
        persistence: &SimplePersistence,
        hotkey: &str,
        node_id: &str,
        miner: &str,
        tao: u64,
        alpha: u64,
    ) {
        sqlx::query(
            "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(hotkey)
        .bind(node_id)
        .bind(miner)
        .bind(U256::from(tao).to_string())
        .bind(U256::from(alpha).to_string())
        .bind(Utc::now().to_rfc3339())
        .execute(persistence.pool())
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_tables_and_index_creation() {
        let _persistence = SimplePersistence::for_testing().await.expect("persistence");
    }

    #[tokio::test]
    async fn test_scan_block_number_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

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
    async fn test_get_collateral_accessors_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hotkey = "0x0404040404040404040404040404040404040404040404040404040404040404";
        let node_id = "0x05050505050505050505050505050505";
        insert_collateral_row(
            &persistence,
            hotkey,
            node_id,
            "0x0606060606060606060606060606060606060606",
            123,
            456,
        )
        .await;

        let tao = persistence
            .get_tao_collateral_amount(hotkey, node_id)
            .await
            .unwrap();
        let alpha = persistence
            .get_alpha_collateral_amount(hotkey, node_id)
            .await
            .unwrap();

        assert_eq!(tao, Some(U256::from(123u64)));
        assert_eq!(alpha, Some(U256::from(456u64)));
    }

    #[tokio::test]
    async fn test_get_all_collateral_nodes() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let nodes = persistence.get_all_collateral_nodes().await.unwrap();
        assert!(nodes.is_empty());

        let hk1 = "0x1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e";
        let ex1 = "0x1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f";
        let hk2 = "0x20202020202020202020202020202020202020202020202020202020202020202020";
        let ex2 = "0x21212121212121212121212121212121";

        insert_collateral_row(
            &persistence,
            hk1,
            ex1,
            "0x0000000000000000000000000000000000000000",
            100,
            200,
        )
        .await;
        insert_collateral_row(
            &persistence,
            hk2,
            ex2,
            "0x0000000000000000000000000000000000000000",
            300,
            400,
        )
        .await;

        let nodes = persistence.get_all_collateral_nodes().await.unwrap();
        assert_eq!(nodes.len(), 2);

        let n1 = nodes.iter().find(|n| n.hotkey == hk1).unwrap();
        assert_eq!(n1.tao_collateral, U256::from(100u64));
        assert_eq!(n1.alpha_collateral, U256::from(200u64));

        let n2 = nodes.iter().find(|n| n.hotkey == hk2).unwrap();
        assert_eq!(n2.tao_collateral, U256::from(300u64));
        assert_eq!(n2.alpha_collateral, U256::from(400u64));
    }
}
