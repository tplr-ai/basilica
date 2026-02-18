use crate::persistence::SimplePersistence;
use alloy_primitives::{Address, U256};
use chrono::Utc;
use collateral_contract::{
    CollateralEvent, Denied, Deposit, ReclaimProcessStarted, Reclaimed, Slashed,
};
use hex::ToHex;
use sqlx::{Row, Sqlite, Transaction};
use tracing::warn;

#[derive(Debug, Clone)]
struct NodeCollateralState {
    id: i64,
    tao_collateral: U256,
    alpha_collateral: U256,
    pending_tao_reclaim: U256,
    pending_alpha_reclaim: U256,
    miner: String,
}

#[derive(Debug, Clone)]
struct ReclaimRecord {
    hotkey: String,
    node_id: String,
    requested_tao_amount: U256,
    requested_alpha_amount: U256,
}

fn parse_u256_decimal(value: &str, field: &str) -> Result<U256, anyhow::Error> {
    U256::from_str_radix(value, 10).map_err(|_| anyhow::anyhow!("Invalid {}", field))
}

fn address_to_string(address: Address) -> String {
    format!("0x{}", address.as_slice().encode_hex::<String>())
}

fn zero_address_string() -> String {
    address_to_string(Address::ZERO)
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

    async fn update_last_scanned_block_number_with_tx(
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

    pub async fn apply_collateral_events_for_block(
        &self,
        block_number: u64,
        events: &[CollateralEvent],
    ) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;

        for event in events {
            self.apply_collateral_event(event, &mut tx).await?;
        }

        self.update_last_scanned_block_number_with_tx(block_number, &mut tx)
            .await?;
        tx.commit().await?;

        Ok(())
    }

    async fn apply_collateral_event(
        &self,
        event: &CollateralEvent,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        match event {
            CollateralEvent::Deposit(deposit) => self.handle_deposit_with_tx(deposit, tx).await,
            CollateralEvent::ReclaimProcessStarted(reclaim_started) => {
                self.handle_reclaim_process_started_with_tx(reclaim_started, tx)
                    .await
            }
            CollateralEvent::Denied(denied) => self.handle_denied_with_tx(denied, tx).await,
            CollateralEvent::Reclaimed(reclaimed) => {
                self.handle_reclaimed_with_tx(reclaimed, tx).await
            }
            CollateralEvent::Slashed(slashed) => self.handle_slashed_with_tx(slashed, tx).await,
        }
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

    pub async fn get_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        // TAO is retained for on-chain state sync and operator visibility.
        // Eligibility and slash policy use alpha collateral.
        self.get_collateral_amount_internal(hotkey, node_id, "tao_collateral")
            .await
    }

    pub async fn get_tao_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount(hotkey, node_id).await
    }

    pub async fn get_alpha_collateral_amount(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<U256>, anyhow::Error> {
        self.get_collateral_amount_internal(hotkey, node_id, "alpha_collateral")
            .await
    }

    pub async fn handle_deposit(&self, deposit: &Deposit) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        self.handle_deposit_with_tx(deposit, &mut tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn handle_reclaim_process_started(
        &self,
        reclaim_started: &ReclaimProcessStarted,
    ) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        self.handle_reclaim_process_started_with_tx(reclaim_started, &mut tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn handle_denied(&self, denied: &Denied) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        self.handle_denied_with_tx(denied, &mut tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn handle_reclaimed(&self, reclaimed: &Reclaimed) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        self.handle_reclaimed_with_tx(reclaimed, &mut tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn handle_slashed(&self, slashed: &Slashed) -> Result<(), anyhow::Error> {
        let mut tx = self.pool().begin().await?;
        self.handle_slashed_with_tx(slashed, &mut tx).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn load_node_state_with_tx(
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

    async fn ensure_node_state_with_tx(
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

    async fn load_reclaim_record_with_tx(
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

    async fn delete_reclaim_record_with_tx(
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

    async fn apply_ownership_rule_with_tx(
        &self,
        hotkey: &str,
        node_id: &str,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let Some(state) = self.load_node_state_with_tx(hotkey, node_id, tx).await? else {
            return Ok(());
        };

        if state.tao_collateral == U256::ZERO
            && state.alpha_collateral == U256::ZERO
            && state.pending_tao_reclaim == U256::ZERO
            && state.pending_alpha_reclaim == U256::ZERO
            && state.miner != zero_address_string()
        {
            sqlx::query("UPDATE collateral_status SET miner = ?, updated_at = ? WHERE id = ?")
                .bind(zero_address_string())
                .bind(Utc::now().to_rfc3339())
                .bind(state.id)
                .execute(&mut **tx)
                .await?;
        }

        Ok(())
    }

    async fn handle_deposit_with_tx(
        &self,
        deposit: &Deposit,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let hotkey = deposit.hotkey.encode_hex::<String>();
        let node_id = deposit.nodeId.encode_hex::<String>();
        let miner = address_to_string(deposit.miner);

        if let Some(mut state) = self.load_node_state_with_tx(&hotkey, &node_id, tx).await? {
            state.tao_collateral = state.tao_collateral.saturating_add(deposit.amount);
            state.alpha_collateral = state.alpha_collateral.saturating_add(deposit.alphaAmount);
            state.miner = miner;

            sqlx::query(
                "UPDATE collateral_status SET tao_collateral = ?, alpha_collateral = ?, miner = ?, updated_at = ? WHERE id = ?",
            )
            .bind(state.tao_collateral.to_string())
            .bind(state.alpha_collateral.to_string())
            .bind(state.miner)
            .bind(Utc::now().to_rfc3339())
            .bind(state.id)
            .execute(&mut **tx)
            .await?;
        } else {
            sqlx::query(
                "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(hotkey)
            .bind(node_id)
            .bind(miner)
            .bind(deposit.amount.to_string())
            .bind(deposit.alphaAmount.to_string())
            .bind(U256::ZERO.to_string())
            .bind(U256::ZERO.to_string())
            .bind(Utc::now().to_rfc3339())
            .execute(&mut **tx)
            .await?;
        }

        Ok(())
    }

    async fn handle_reclaim_process_started_with_tx(
        &self,
        reclaim_started: &ReclaimProcessStarted,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = reclaim_started.reclaimRequestId.to_string();

        if self
            .load_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?
            .is_some()
        {
            warn!(
                "Reclaim request {} already exists in persistence, skipping duplicate",
                reclaim_request_id
            );
            return Ok(());
        }

        let hotkey = reclaim_started.hotkey.encode_hex::<String>();
        let node_id = reclaim_started.nodeId.encode_hex::<String>();
        let miner = address_to_string(reclaim_started.miner);

        let mut state = self
            .ensure_node_state_with_tx(&hotkey, &node_id, &miner, tx)
            .await?;

        sqlx::query(
            "INSERT INTO collateral_reclaims (reclaim_request_id, hotkey, node_id, miner, requested_tao_amount, requested_alpha_amount, deny_timeout, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&reclaim_request_id)
        .bind(&hotkey)
        .bind(&node_id)
        .bind(&miner)
        .bind(reclaim_started.amount.to_string())
        .bind(reclaim_started.alphaAmount.to_string())
        .bind(reclaim_started.expirationTime.to_string())
        .bind(Utc::now().to_rfc3339())
        .execute(&mut **tx)
        .await?;

        state.pending_tao_reclaim = state
            .pending_tao_reclaim
            .saturating_add(reclaim_started.amount);
        state.pending_alpha_reclaim = state
            .pending_alpha_reclaim
            .saturating_add(reclaim_started.alphaAmount);
        state.miner = miner;

        sqlx::query(
            "UPDATE collateral_status SET pending_tao_reclaim = ?, pending_alpha_reclaim = ?, miner = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state.pending_tao_reclaim.to_string())
        .bind(state.pending_alpha_reclaim.to_string())
        .bind(state.miner)
        .bind(Utc::now().to_rfc3339())
        .bind(state.id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    async fn handle_denied_with_tx(
        &self,
        denied: &Denied,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = denied.reclaimRequestId.to_string();
        let reclaim = self
            .load_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Reclaim record {} not found while handling Denied event",
                    reclaim_request_id
                )
            })?;

        let mut state = self
            .load_node_state_with_tx(&reclaim.hotkey, &reclaim.node_id, tx)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collateral status not found"))?;

        state.pending_tao_reclaim = state
            .pending_tao_reclaim
            .saturating_sub(reclaim.requested_tao_amount);
        state.pending_alpha_reclaim = state
            .pending_alpha_reclaim
            .saturating_sub(reclaim.requested_alpha_amount);

        sqlx::query(
            "UPDATE collateral_status SET pending_tao_reclaim = ?, pending_alpha_reclaim = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state.pending_tao_reclaim.to_string())
        .bind(state.pending_alpha_reclaim.to_string())
        .bind(Utc::now().to_rfc3339())
        .bind(state.id)
        .execute(&mut **tx)
        .await?;

        self.delete_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?;
        self.apply_ownership_rule_with_tx(&reclaim.hotkey, &reclaim.node_id, tx)
            .await?;

        Ok(())
    }

    async fn handle_reclaimed_with_tx(
        &self,
        reclaimed: &Reclaimed,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = reclaimed.reclaimRequestId.to_string();
        let reclaim = self
            .load_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Reclaim record {} not found while handling Reclaimed event",
                    reclaim_request_id
                )
            })?;

        let event_hotkey = reclaimed.hotkey.encode_hex::<String>();
        let event_node_id = reclaimed.nodeId.encode_hex::<String>();
        if event_hotkey != reclaim.hotkey || event_node_id != reclaim.node_id {
            return Err(anyhow::anyhow!(
                "Reclaim event node mismatch for request {}",
                reclaim_request_id
            ));
        }

        let mut state = self
            .load_node_state_with_tx(&reclaim.hotkey, &reclaim.node_id, tx)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collateral status not found"))?;

        state.pending_tao_reclaim = state
            .pending_tao_reclaim
            .saturating_sub(reclaim.requested_tao_amount);
        state.pending_alpha_reclaim = state
            .pending_alpha_reclaim
            .saturating_sub(reclaim.requested_alpha_amount);

        state.tao_collateral = state.tao_collateral.saturating_sub(reclaimed.amount);
        state.alpha_collateral = state.alpha_collateral.saturating_sub(reclaimed.alphaAmount);

        sqlx::query(
            "UPDATE collateral_status SET tao_collateral = ?, alpha_collateral = ?, pending_tao_reclaim = ?, pending_alpha_reclaim = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state.tao_collateral.to_string())
        .bind(state.alpha_collateral.to_string())
        .bind(state.pending_tao_reclaim.to_string())
        .bind(state.pending_alpha_reclaim.to_string())
        .bind(Utc::now().to_rfc3339())
        .bind(state.id)
        .execute(&mut **tx)
        .await?;

        self.delete_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?;
        self.apply_ownership_rule_with_tx(&reclaim.hotkey, &reclaim.node_id, tx)
            .await?;

        Ok(())
    }

    async fn handle_slashed_with_tx(
        &self,
        slashed: &Slashed,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let hotkey = slashed.hotkey.encode_hex::<String>();
        let node_id = slashed.nodeId.encode_hex::<String>();

        let mut state = self
            .load_node_state_with_tx(&hotkey, &node_id, tx)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collateral status not found"))?;

        if slashed.slashAmount > state.tao_collateral {
            warn!(
                "Slashed TAO amount {} exceeds TAO collateral {} in database",
                slashed.slashAmount, state.tao_collateral
            );
        }
        if slashed.slashAlphaAmount > state.alpha_collateral {
            warn!(
                "Slashed alpha amount {} exceeds alpha collateral {} in database",
                slashed.slashAlphaAmount, state.alpha_collateral
            );
        }

        state.tao_collateral = state.tao_collateral.saturating_sub(slashed.slashAmount);
        state.alpha_collateral = state
            .alpha_collateral
            .saturating_sub(slashed.slashAlphaAmount);

        sqlx::query(
            "UPDATE collateral_status SET tao_collateral = ?, alpha_collateral = ?, url = ?, url_content_sha256 = ?, updated_at = ? WHERE id = ?",
        )
        .bind(state.tao_collateral.to_string())
        .bind(state.alpha_collateral.to_string())
        .bind(slashed.url.clone())
        .bind(slashed.urlContentSha256.encode_hex::<String>())
        .bind(Utc::now().to_rfc3339())
        .bind(state.id)
        .execute(&mut **tx)
        .await?;

        self.apply_ownership_rule_with_tx(&hotkey, &node_id, tx)
            .await?;

        Ok(())
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

    fn make_miner(byte: u8) -> Address {
        Address::from_slice(&[byte; 20])
    }

    fn ev_deposit(hk: [u8; 32], ex: [u8; 16], miner: Address, tao: u64, alpha: u64) -> Deposit {
        Deposit {
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner,
            amount: U256::from(tao),
            alphaHotkey: FixedBytes::from_slice(&[0u8; 32]),
            alphaAmount: U256::from(alpha),
        }
    }

    fn ev_reclaim_started(
        reclaim_request_id: u64,
        hk: [u8; 32],
        ex: [u8; 16],
        miner: Address,
        tao: u64,
        alpha: u64,
    ) -> ReclaimProcessStarted {
        ReclaimProcessStarted {
            reclaimRequestId: U256::from(reclaim_request_id),
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner,
            amount: U256::from(tao),
            alphaColdkey: FixedBytes::from_slice(&[0u8; 32]),
            alphaAmount: U256::from(alpha),
            expirationTime: 123456,
            url: "https://example.com/reclaim".to_string(),
            urlContentSha256: FixedBytes::from_slice(&[0x11; 32]),
        }
    }

    fn ev_denied(reclaim_request_id: u64) -> Denied {
        Denied {
            reclaimRequestId: U256::from(reclaim_request_id),
            url: "https://example.com/deny".to_string(),
            urlContentSha256: FixedBytes::from_slice(&[0x22; 32]),
        }
    }

    fn ev_reclaimed(
        reclaim_request_id: u64,
        hk: [u8; 32],
        ex: [u8; 16],
        tao: u64,
        alpha: u64,
    ) -> Reclaimed {
        Reclaimed {
            reclaimRequestId: U256::from(reclaim_request_id),
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner: make_miner(9),
            amount: U256::from(tao),
            alphaColdkey: FixedBytes::from_slice(&[0u8; 32]),
            alphaAmount: U256::from(alpha),
        }
    }

    fn ev_slashed(hk: [u8; 32], ex: [u8; 16], tao: u64, alpha: u64) -> Slashed {
        Slashed {
            hotkey: FixedBytes::from_slice(&hk),
            nodeId: FixedBytes::from_slice(&ex),
            miner: make_miner(7),
            slashAmount: U256::from(tao),
            slashAlphaAmount: U256::from(alpha),
            url: String::new(),
            urlContentSha256: FixedBytes::from_slice(&[0u8; 32]),
        }
    }

    async fn fetch_state(
        persistence: &SimplePersistence,
        hk: [u8; 32],
        ex: [u8; 16],
    ) -> (String, String, String, String, String) {
        sqlx::query_as(
            "SELECT tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, miner FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hk.encode_hex::<String>())
        .bind(ex.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap()
    }

    fn assert_event_supported(event: &CollateralEvent) {
        match event {
            CollateralEvent::Deposit(_) => {}
            CollateralEvent::ReclaimProcessStarted(_) => {}
            CollateralEvent::Denied(_) => {}
            CollateralEvent::Reclaimed(_) => {}
            CollateralEvent::Slashed(_) => {}
        }
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
    async fn test_handle_deposit_insert_and_update_dual_balances() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(1);
        let ex = make_node_id(2);
        let miner = make_miner(3);

        let d1 = ev_deposit(hk, ex, miner, 10, 100);
        persistence.handle_deposit(&d1).await.unwrap();

        let d2 = ev_deposit(hk, ex, miner, 5, 50);
        persistence.handle_deposit(&d2).await.unwrap();

        let (tao, alpha, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "15");
        assert_eq!(alpha, "150");
        assert_eq!(pending_tao, "0");
        assert_eq!(pending_alpha, "0");
        assert_eq!(stored_miner, address_to_string(miner));
    }

    #[tokio::test]
    async fn test_get_collateral_accessors_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(4);
        let ex = make_node_id(5);
        let d = ev_deposit(hk, ex, make_miner(6), 123, 456);
        persistence.handle_deposit(&d).await.unwrap();

        let hotkey_hex = d.hotkey.encode_hex::<String>();
        let node_hex = d.nodeId.encode_hex::<String>();

        let tao = persistence
            .get_tao_collateral_amount(&hotkey_hex, &node_hex)
            .await
            .unwrap();
        let alpha = persistence
            .get_alpha_collateral_amount(&hotkey_hex, &node_hex)
            .await
            .unwrap();

        assert_eq!(tao, Some(U256::from(123u64)));
        assert_eq!(alpha, Some(U256::from(456u64)));
    }

    #[tokio::test]
    async fn test_reclaim_lifecycle_start_and_deny() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(8);
        let ex = make_node_id(9);
        let miner = make_miner(10);

        persistence
            .handle_deposit(&ev_deposit(hk, ex, miner, 200, 300))
            .await
            .unwrap();

        persistence
            .handle_reclaim_process_started(&ev_reclaim_started(1, hk, ex, miner, 40, 60))
            .await
            .unwrap();

        let (tao, alpha, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "200");
        assert_eq!(alpha, "300");
        assert_eq!(pending_tao, "40");
        assert_eq!(pending_alpha, "60");
        assert_eq!(stored_miner, address_to_string(miner));

        persistence.handle_denied(&ev_denied(1)).await.unwrap();

        let (_, _, pending_tao_after, pending_alpha_after, stored_miner_after) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(pending_tao_after, "0");
        assert_eq!(pending_alpha_after, "0");
        assert_eq!(stored_miner_after, address_to_string(miner));

        let reclaim_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_reclaims")
            .fetch_one(persistence.pool())
            .await
            .unwrap();
        assert_eq!(reclaim_count, 0);
    }

    #[tokio::test]
    async fn test_partial_finalize_clears_pending_by_requested_amount() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(11);
        let ex = make_node_id(12);
        let miner = make_miner(13);

        persistence
            .handle_deposit(&ev_deposit(hk, ex, miner, 100, 100))
            .await
            .unwrap();
        persistence
            .handle_reclaim_process_started(&ev_reclaim_started(7, hk, ex, miner, 100, 100))
            .await
            .unwrap();

        persistence
            .handle_slashed(&ev_slashed(hk, ex, 60, 60))
            .await
            .unwrap();
        persistence
            .handle_reclaimed(&ev_reclaimed(7, hk, ex, 40, 40))
            .await
            .unwrap();

        let (tao, alpha, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "0");
        assert_eq!(alpha, "0");
        assert_eq!(pending_tao, "0");
        assert_eq!(pending_alpha, "0");
        assert_eq!(stored_miner, zero_address_string());
    }

    #[tokio::test]
    async fn test_full_slash_with_pending_reclaim_keeps_owner_until_pending_cleared() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(14);
        let ex = make_node_id(15);
        let miner = make_miner(16);

        persistence
            .handle_deposit(&ev_deposit(hk, ex, miner, 100, 100))
            .await
            .unwrap();
        persistence
            .handle_reclaim_process_started(&ev_reclaim_started(2, hk, ex, miner, 100, 100))
            .await
            .unwrap();
        persistence
            .handle_slashed(&ev_slashed(hk, ex, 100, 100))
            .await
            .unwrap();

        let (_, _, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(pending_tao, "100");
        assert_eq!(pending_alpha, "100");
        assert_eq!(stored_miner, address_to_string(miner));

        persistence.handle_denied(&ev_denied(2)).await.unwrap();

        let (_, _, pending_tao_after, pending_alpha_after, stored_miner_after) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(pending_tao_after, "0");
        assert_eq!(pending_alpha_after, "0");
        assert_eq!(stored_miner_after, zero_address_string());
    }

    #[tokio::test]
    async fn test_handle_reclaimed_not_found_without_reclaim_record() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(17);
        let ex = make_node_id(18);

        persistence
            .handle_deposit(&ev_deposit(hk, ex, make_miner(19), 10, 10))
            .await
            .unwrap();

        let result = persistence
            .handle_reclaimed(&ev_reclaimed(999, hk, ex, 1, 1))
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Reclaim record 999 not found"));
    }

    #[tokio::test]
    async fn test_handle_slashed_with_url_data() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(20);
        let ex = make_node_id(21);
        persistence
            .handle_deposit(&ev_deposit(hk, ex, make_miner(22), 500, 500))
            .await
            .unwrap();

        let mut slashed = ev_slashed(hk, ex, 50, 60);
        slashed.url = "https://example.com/proof".to_string();
        slashed.urlContentSha256 = FixedBytes::from_slice(&[
            0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56,
            0x78, 0x90, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x12,
            0x34, 0x56, 0x78, 0x90,
        ]);

        persistence.handle_slashed(&slashed).await.unwrap();

        let (url, checksum): (String, String) = sqlx::query_as(
            "SELECT url, url_content_sha256 FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hk.encode_hex::<String>())
        .bind(ex.encode_hex::<String>())
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(url, "https://example.com/proof");
        assert_eq!(
            checksum,
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        );
    }

    #[tokio::test]
    async fn test_apply_block_events_atomic_rollback() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let hk = make_hotkey(23);
        let ex = make_node_id(24);
        let miner = make_miner(25);

        let initial_block = persistence.get_last_scanned_block_number().await.unwrap();

        let events = vec![
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 7, 11)),
            CollateralEvent::Denied(ev_denied(404)),
        ];

        let result = persistence
            .apply_collateral_events_for_block(initial_block + 1, &events)
            .await;
        assert!(result.is_err());

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_status")
            .fetch_one(persistence.pool())
            .await
            .unwrap();
        assert_eq!(count, 0);

        let unchanged = persistence.get_last_scanned_block_number().await.unwrap();
        assert_eq!(unchanged, initial_block);
    }

    #[tokio::test]
    async fn test_event_coverage_guard() {
        let hk = make_hotkey(26);
        let ex = make_node_id(27);
        let miner = make_miner(28);

        let events = vec![
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 1, 2)),
            CollateralEvent::ReclaimProcessStarted(ev_reclaim_started(55, hk, ex, miner, 1, 2)),
            CollateralEvent::Denied(ev_denied(55)),
            CollateralEvent::Reclaimed(ev_reclaimed(55, hk, ex, 1, 2)),
            CollateralEvent::Slashed(ev_slashed(hk, ex, 1, 2)),
        ];

        for event in &events {
            assert_event_supported(event);
        }
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
}
