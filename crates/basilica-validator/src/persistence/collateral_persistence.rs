use crate::persistence::SimplePersistence;
use alloy_primitives::{Address, U256};
use chrono::Utc;
use collateral_contract::CollateralEventWithMeta;
use hex::ToHex;
use sqlx::{Row, Sqlite, Transaction};

#[derive(Debug, Clone)]
pub(crate) struct NodeCollateralState {
    pub(crate) id: i64,
    pub(crate) tao_collateral: U256,
    pub(crate) alpha_collateral: U256,
    pub(crate) pending_tao_reclaim: U256,
    pub(crate) pending_alpha_reclaim: U256,
    pub(crate) miner: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ReclaimRecord {
    pub(crate) hotkey: String,
    pub(crate) node_id: String,
    pub(crate) requested_tao_amount: U256,
    pub(crate) requested_alpha_amount: U256,
}

pub(crate) struct InsertReclaimParams<'a> {
    pub(crate) reclaim_request_id: &'a str,
    pub(crate) hotkey: &'a str,
    pub(crate) node_id: &'a str,
    pub(crate) miner: &'a str,
    pub(crate) requested_tao_amount: &'a U256,
    pub(crate) requested_alpha_amount: &'a U256,
    pub(crate) deny_timeout: &'a str,
}

pub(crate) fn parse_u256_decimal(value: &str, field: &str) -> Result<U256, anyhow::Error> {
    U256::from_str_radix(value, 10).map_err(|_| anyhow::anyhow!("Invalid {}", field))
}

pub(crate) fn address_to_string(address: Address) -> String {
    format!("0x{}", address.as_slice().encode_hex::<String>())
}

pub(crate) fn zero_address_string() -> String {
    address_to_string(Address::ZERO)
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

    pub(crate) async fn update_last_scanned_block_number_with_tx(
        &self,
        last_scanned_block: u64,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        let query =
            "UPDATE collateral_scan_status SET last_scanned_block_number = ?, updated_at = ? WHERE id = 1";

        sqlx::query(query)
            .bind(last_scanned_block as i64)
            .bind(now)
            .execute(&mut **tx)
            .await?;

        Ok(())
    }

    pub(crate) async fn insert_raw_event_with_tx(
        &self,
        block_number: u64,
        event_with_meta: &CollateralEventWithMeta,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<bool, anyhow::Error> {
        let event = &event_with_meta.event;
        let event_type = event.event_type();
        let hotkey = event.hotkey_hex();
        let node_id = event.node_id_hex();
        let event_data = event.to_json().to_string();

        let result = sqlx::query(
            "INSERT OR IGNORE INTO collateral_event_log \
             (event_type, block_number, tx_hash, log_index, hotkey, node_id, event_data) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(event_type)
        .bind(block_number as i64)
        .bind(&event_with_meta.tx_hash)
        .bind(event_with_meta.log_index as i64)
        .bind(hotkey)
        .bind(node_id)
        .bind(event_data)
        .execute(&mut **tx)
        .await?;

        Ok(result.rows_affected() > 0)
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

    pub async fn reconcile_collateral(
        &self,
        hotkey: &str,
        node_id: &str,
        tao_collateral: U256,
        alpha_collateral: U256,
    ) -> Result<(), anyhow::Error> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE collateral_status SET tao_collateral = ?, alpha_collateral = ?, updated_at = ? WHERE hotkey = ? AND node_id = ?",
        )
        .bind(tao_collateral.to_string())
        .bind(alpha_collateral.to_string())
        .bind(now)
        .bind(hotkey)
        .bind(node_id)
        .execute(self.pool())
        .await?;

        tracing::warn!(
            hotkey = hotkey,
            node_id = node_id,
            tao_collateral = %tao_collateral,
            alpha_collateral = %alpha_collateral,
            rows_affected = result.rows_affected(),
            "Reconciliation overwrote collateral values in DB"
        );

        Ok(())
    }

    pub(crate) async fn load_node_state_with_tx(
        &self,
        hotkey: &str,
        node_id: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<Option<NodeCollateralState>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT id, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, miner FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hotkey)
        .bind(node_id)
        .fetch_optional(&mut **tx)
        .await?;

        if let Some(row) = row {
            let id: i64 = row.get("id");
            let tao_collateral =
                parse_u256_decimal(&row.get::<String, _>("tao_collateral"), "tao_collateral")?;
            let alpha_collateral = parse_u256_decimal(
                &row.get::<String, _>("alpha_collateral"),
                "alpha_collateral",
            )?;
            let pending_tao_reclaim = parse_u256_decimal(
                &row.get::<String, _>("pending_tao_reclaim"),
                "pending_tao_reclaim",
            )?;
            let pending_alpha_reclaim = parse_u256_decimal(
                &row.get::<String, _>("pending_alpha_reclaim"),
                "pending_alpha_reclaim",
            )?;
            let miner: String = row.get("miner");

            Ok(Some(NodeCollateralState {
                id,
                tao_collateral,
                alpha_collateral,
                pending_tao_reclaim,
                pending_alpha_reclaim,
                miner,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) async fn ensure_node_state_with_tx(
        &self,
        hotkey: &str,
        node_id: &str,
        miner: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<NodeCollateralState, anyhow::Error> {
        if let Some(state) = self.load_node_state_with_tx(hotkey, node_id, tx).await? {
            return Ok(state);
        }

        let now = Utc::now().to_rfc3339();
        let query = "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

        sqlx::query(query)
            .bind(hotkey)
            .bind(node_id)
            .bind(miner)
            .bind(U256::ZERO.to_string())
            .bind(U256::ZERO.to_string())
            .bind(U256::ZERO.to_string())
            .bind(U256::ZERO.to_string())
            .bind(now)
            .execute(&mut **tx)
            .await?;

        self.load_node_state_with_tx(hotkey, node_id, tx)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Failed to load inserted collateral status"))
    }

    pub(crate) async fn save_node_state_with_tx(
        &self,
        state: &NodeCollateralState,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            "UPDATE collateral_status SET tao_collateral = ?, alpha_collateral = ?, pending_tao_reclaim = ?, pending_alpha_reclaim = ?, miner = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state.tao_collateral.to_string())
        .bind(state.alpha_collateral.to_string())
        .bind(state.pending_tao_reclaim.to_string())
        .bind(state.pending_alpha_reclaim.to_string())
        .bind(&state.miner)
        .bind(Utc::now().to_rfc3339())
        .bind(state.id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    pub(crate) async fn update_node_slash_evidence_with_tx(
        &self,
        id: i64,
        url: &str,
        url_sha256: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            "UPDATE collateral_status SET url = ?, url_content_sha256 = ?, updated_at = ? WHERE id = ?",
        )
        .bind(url)
        .bind(url_sha256)
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    pub(crate) async fn load_reclaim_record_with_tx(
        &self,
        reclaim_request_id: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<Option<ReclaimRecord>, anyhow::Error> {
        let row = sqlx::query(
            "SELECT hotkey, node_id, requested_tao_amount, requested_alpha_amount FROM collateral_reclaims WHERE reclaim_request_id = ?",
        )
        .bind(reclaim_request_id)
        .fetch_optional(&mut **tx)
        .await?;

        if let Some(row) = row {
            let hotkey: String = row.get("hotkey");
            let node_id: String = row.get("node_id");
            let requested_tao_amount = parse_u256_decimal(
                &row.get::<String, _>("requested_tao_amount"),
                "requested_tao_amount",
            )?;
            let requested_alpha_amount = parse_u256_decimal(
                &row.get::<String, _>("requested_alpha_amount"),
                "requested_alpha_amount",
            )?;

            Ok(Some(ReclaimRecord {
                hotkey,
                node_id,
                requested_tao_amount,
                requested_alpha_amount,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) async fn insert_reclaim_with_tx(
        &self,
        params: &InsertReclaimParams<'_>,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        sqlx::query(
            "INSERT INTO collateral_reclaims (reclaim_request_id, hotkey, node_id, miner, requested_tao_amount, requested_alpha_amount, deny_timeout, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(params.reclaim_request_id)
        .bind(params.hotkey)
        .bind(params.node_id)
        .bind(params.miner)
        .bind(params.requested_tao_amount.to_string())
        .bind(params.requested_alpha_amount.to_string())
        .bind(params.deny_timeout)
        .bind(Utc::now().to_rfc3339())
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    pub(crate) async fn delete_reclaim_record_with_tx(
        &self,
        reclaim_request_id: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        sqlx::query("DELETE FROM collateral_reclaims WHERE reclaim_request_id = ?")
            .bind(reclaim_request_id)
            .execute(&mut **tx)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use collateral_contract::config::CONTRACT_DEPLOYED_BLOCK_NUMBER;

    async fn insert_collateral_row(
        persistence: &SimplePersistence,
        hotkey: &str,
        node_id: &str,
        miner: &str,
        tao: u64,
        alpha: u64,
    ) {
        sqlx::query(
            "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(hotkey)
        .bind(node_id)
        .bind(miner)
        .bind(U256::from(tao).to_string())
        .bind(U256::from(alpha).to_string())
        .bind(U256::ZERO.to_string())
        .bind(U256::ZERO.to_string())
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
    async fn test_scan_status_table_initialization() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_scan_status")
            .fetch_one(persistence.pool())
            .await
            .unwrap();

        assert_eq!(count, 1);

        let initial_block: i64 = sqlx::query_scalar(
            "SELECT last_scanned_block_number FROM collateral_scan_status WHERE id = 1",
        )
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(initial_block as u64, CONTRACT_DEPLOYED_BLOCK_NUMBER);
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

    #[tokio::test]
    async fn test_reconcile_collateral() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hotkey = "0x2424242424242424242424242424242424242424242424242424242424242424";
        let node_id = "0x25252525252525252525252525252525";
        insert_collateral_row(
            &persistence,
            hotkey,
            node_id,
            "0x2626262626262626262626262626262626262626",
            100,
            200,
        )
        .await;

        persistence
            .reconcile_collateral(hotkey, node_id, U256::from(999u64), U256::from(888u64))
            .await
            .unwrap();

        let tao = persistence
            .get_tao_collateral_amount(hotkey, node_id)
            .await
            .unwrap();
        let alpha = persistence
            .get_alpha_collateral_amount(hotkey, node_id)
            .await
            .unwrap();

        assert_eq!(tao, Some(U256::from(999u64)));
        assert_eq!(alpha, Some(U256::from(888u64)));
    }
}
