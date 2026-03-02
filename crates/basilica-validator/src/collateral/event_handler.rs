use crate::persistence::collateral_persistence::{
    address_to_string, zero_address_string, InsertReclaimParams, NodeCollateralState,
};
use crate::persistence::SimplePersistence;
use alloy_primitives::U256;
use collateral_contract::{
    CollateralEvent, CollateralEventWithMeta, Denied, Deposit, ReclaimProcessStarted, Reclaimed,
    Slashed,
};
use hex::ToHex;
use sqlx::{Sqlite, Transaction};
use std::sync::Arc;
use tracing::warn;

pub struct CollateralEventHandler {
    persistence: Arc<SimplePersistence>,
}

impl CollateralEventHandler {
    pub fn new(persistence: Arc<SimplePersistence>) -> Self {
        Self { persistence }
    }

    pub async fn apply_collateral_events_for_block(
        &self,
        block_number: u64,
        events: &[CollateralEventWithMeta],
    ) -> Result<(), anyhow::Error> {
        let mut tx = self.persistence.pool().begin().await?;

        for event_with_meta in events {
            let inserted = self
                .persistence
                .insert_raw_event_with_tx(block_number, event_with_meta, &mut tx)
                .await?;
            if inserted {
                self.apply_collateral_event(&event_with_meta.event, &mut tx)
                    .await?;
            }
        }

        self.persistence
            .update_last_scanned_block_number_with_tx(block_number, &mut tx)
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
            CollateralEvent::Deposit(deposit) => self.handle_deposit(deposit, tx).await,
            CollateralEvent::ReclaimProcessStarted(reclaim_started) => {
                self.handle_reclaim_process_started(reclaim_started, tx)
                    .await
            }
            CollateralEvent::Denied(denied) => self.handle_denied(denied, tx).await,
            CollateralEvent::Reclaimed(reclaimed) => self.handle_reclaimed(reclaimed, tx).await,
            CollateralEvent::Slashed(slashed) => self.handle_slashed(slashed, tx).await,
        }
    }

    async fn handle_deposit(
        &self,
        deposit: &Deposit,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let hotkey = format!("0x{}", deposit.hotkey.encode_hex::<String>());
        let node_id = format!("0x{}", deposit.nodeId.encode_hex::<String>());
        let miner = address_to_string(deposit.miner);

        let mut state = self
            .persistence
            .ensure_node_state_with_tx(&hotkey, &node_id, &miner, tx)
            .await?;

        state.tao_collateral = state.tao_collateral.saturating_add(deposit.amount);
        state.alpha_collateral = state.alpha_collateral.saturating_add(deposit.alphaAmount);
        state.miner = miner;

        self.persistence.save_node_state_with_tx(&state, tx).await?;

        Ok(())
    }

    async fn handle_reclaim_process_started(
        &self,
        reclaim_started: &ReclaimProcessStarted,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = reclaim_started.reclaimRequestId.to_string();

        if self
            .persistence
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

        let hotkey = format!("0x{}", reclaim_started.hotkey.encode_hex::<String>());
        let node_id = format!("0x{}", reclaim_started.nodeId.encode_hex::<String>());
        let miner = address_to_string(reclaim_started.miner);

        let mut state = self
            .persistence
            .ensure_node_state_with_tx(&hotkey, &node_id, &miner, tx)
            .await?;

        self.persistence
            .insert_reclaim_with_tx(
                &InsertReclaimParams {
                    reclaim_request_id: &reclaim_request_id,
                    hotkey: &hotkey,
                    node_id: &node_id,
                    miner: &miner,
                    requested_tao_amount: &reclaim_started.amount,
                    requested_alpha_amount: &reclaim_started.alphaAmount,
                    deny_timeout: &reclaim_started.expirationTime.to_string(),
                },
                tx,
            )
            .await?;

        state.pending_tao_reclaim = state
            .pending_tao_reclaim
            .saturating_add(reclaim_started.amount);
        state.pending_alpha_reclaim = state
            .pending_alpha_reclaim
            .saturating_add(reclaim_started.alphaAmount);
        state.miner = miner;

        self.persistence.save_node_state_with_tx(&state, tx).await?;

        Ok(())
    }

    async fn handle_denied(
        &self,
        denied: &Denied,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = denied.reclaimRequestId.to_string();
        let reclaim = self
            .persistence
            .load_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Reclaim record {} not found while handling Denied event",
                    reclaim_request_id
                )
            })?;

        let mut state = self
            .persistence
            .load_node_state_with_tx(&reclaim.hotkey, &reclaim.node_id, tx)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collateral status not found"))?;

        state.pending_tao_reclaim = state
            .pending_tao_reclaim
            .saturating_sub(reclaim.requested_tao_amount);
        state.pending_alpha_reclaim = state
            .pending_alpha_reclaim
            .saturating_sub(reclaim.requested_alpha_amount);

        apply_ownership_rule(&mut state);

        self.persistence.save_node_state_with_tx(&state, tx).await?;

        self.persistence
            .delete_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?;

        Ok(())
    }

    async fn handle_reclaimed(
        &self,
        reclaimed: &Reclaimed,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let reclaim_request_id = reclaimed.reclaimRequestId.to_string();
        let reclaim = self
            .persistence
            .load_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Reclaim record {} not found while handling Reclaimed event",
                    reclaim_request_id
                )
            })?;

        let event_hotkey = format!("0x{}", reclaimed.hotkey.encode_hex::<String>());
        let event_node_id = format!("0x{}", reclaimed.nodeId.encode_hex::<String>());
        if event_hotkey != reclaim.hotkey || event_node_id != reclaim.node_id {
            return Err(anyhow::anyhow!(
                "Reclaim event node mismatch for request {}",
                reclaim_request_id
            ));
        }

        let mut state = self
            .persistence
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

        apply_ownership_rule(&mut state);

        self.persistence.save_node_state_with_tx(&state, tx).await?;

        self.persistence
            .delete_reclaim_record_with_tx(&reclaim_request_id, tx)
            .await?;

        Ok(())
    }

    async fn handle_slashed(
        &self,
        slashed: &Slashed,
        tx: &mut Transaction<'_, Sqlite>,
    ) -> Result<(), anyhow::Error> {
        let hotkey = format!("0x{}", slashed.hotkey.encode_hex::<String>());
        let node_id = format!("0x{}", slashed.nodeId.encode_hex::<String>());

        let mut state = self
            .persistence
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

        apply_ownership_rule(&mut state);

        self.persistence.save_node_state_with_tx(&state, tx).await?;

        self.persistence
            .update_node_slash_evidence_with_tx(
                state.id,
                &slashed.url,
                &format!(
                    "0x{}",
                    slashed.urlContentSha256.as_slice().encode_hex::<String>()
                ),
                tx,
            )
            .await?;

        Ok(())
    }
}

fn apply_ownership_rule(state: &mut NodeCollateralState) {
    if state.tao_collateral == U256::ZERO
        && state.alpha_collateral == U256::ZERO
        && state.pending_tao_reclaim == U256::ZERO
        && state.pending_alpha_reclaim == U256::ZERO
        && state.miner != zero_address_string()
    {
        state.miner = zero_address_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::collateral_persistence::address_to_string;
    use alloy_primitives::{Address, FixedBytes};
    use collateral_contract::CollateralEventWithMeta;

    fn make_hotkey(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn make_node_id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn make_miner(byte: u8) -> Address {
        Address::from_slice(&[byte; 20])
    }

    fn wrap_event(event: CollateralEvent, log_index: u64) -> CollateralEventWithMeta {
        CollateralEventWithMeta {
            event,
            tx_hash: format!("0x{}", hex::encode([0xaa; 32])),
            log_index,
        }
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
        .bind(format!("0x{}", hk.encode_hex::<String>()))
        .bind(format!("0x{}", ex.encode_hex::<String>()))
        .fetch_one(persistence.pool())
        .await
        .unwrap()
    }

    async fn make_handler() -> (CollateralEventHandler, Arc<SimplePersistence>) {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let handler = CollateralEventHandler::new(persistence.clone());
        (handler, persistence)
    }

    async fn apply_event(
        handler: &CollateralEventHandler,
        event: CollateralEvent,
    ) -> Result<(), anyhow::Error> {
        let mut tx = handler.persistence.pool().begin().await?;
        handler.apply_collateral_event(&event, &mut tx).await?;
        tx.commit().await?;
        Ok(())
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
    async fn test_handle_deposit_insert_and_update_dual_balances() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(1);
        let ex = make_node_id(2);
        let miner = make_miner(3);

        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 10, 100)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 5, 50)),
        )
        .await
        .unwrap();

        let (tao, alpha, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "15");
        assert_eq!(alpha, "150");
        assert_eq!(pending_tao, "0");
        assert_eq!(pending_alpha, "0");
        assert_eq!(stored_miner, address_to_string(miner));
    }

    #[tokio::test]
    async fn test_reclaim_lifecycle_start_and_deny() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(8);
        let ex = make_node_id(9);
        let miner = make_miner(10);

        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 200, 300)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::ReclaimProcessStarted(ev_reclaim_started(1, hk, ex, miner, 40, 60)),
        )
        .await
        .unwrap();

        let (tao, alpha, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "200");
        assert_eq!(alpha, "300");
        assert_eq!(pending_tao, "40");
        assert_eq!(pending_alpha, "60");
        assert_eq!(stored_miner, address_to_string(miner));

        apply_event(&handler, CollateralEvent::Denied(ev_denied(1)))
            .await
            .unwrap();

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
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(11);
        let ex = make_node_id(12);
        let miner = make_miner(13);

        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 100, 100)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::ReclaimProcessStarted(ev_reclaim_started(7, hk, ex, miner, 100, 100)),
        )
        .await
        .unwrap();

        apply_event(
            &handler,
            CollateralEvent::Slashed(ev_slashed(hk, ex, 60, 60)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::Reclaimed(ev_reclaimed(7, hk, ex, 40, 40)),
        )
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
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(14);
        let ex = make_node_id(15);
        let miner = make_miner(16);

        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 100, 100)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::ReclaimProcessStarted(ev_reclaim_started(2, hk, ex, miner, 100, 100)),
        )
        .await
        .unwrap();
        apply_event(
            &handler,
            CollateralEvent::Slashed(ev_slashed(hk, ex, 100, 100)),
        )
        .await
        .unwrap();

        let (_, _, pending_tao, pending_alpha, stored_miner) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(pending_tao, "100");
        assert_eq!(pending_alpha, "100");
        assert_eq!(stored_miner, address_to_string(miner));

        apply_event(&handler, CollateralEvent::Denied(ev_denied(2)))
            .await
            .unwrap();

        let (_, _, pending_tao_after, pending_alpha_after, stored_miner_after) =
            fetch_state(&persistence, hk, ex).await;
        assert_eq!(pending_tao_after, "0");
        assert_eq!(pending_alpha_after, "0");
        assert_eq!(stored_miner_after, zero_address_string());
    }

    #[tokio::test]
    async fn test_handle_reclaimed_not_found_without_reclaim_record() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(17);
        let ex = make_node_id(18);

        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, make_miner(19), 10, 10)),
        )
        .await
        .unwrap();

        let result = apply_event(
            &handler,
            CollateralEvent::Reclaimed(ev_reclaimed(999, hk, ex, 1, 1)),
        )
        .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Reclaim record 999 not found"));

        drop(persistence);
    }

    #[tokio::test]
    async fn test_handle_slashed_with_url_data() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(20);
        let ex = make_node_id(21);
        apply_event(
            &handler,
            CollateralEvent::Deposit(ev_deposit(hk, ex, make_miner(22), 500, 500)),
        )
        .await
        .unwrap();

        let mut slashed = ev_slashed(hk, ex, 50, 60);
        slashed.url = "https://example.com/proof".to_string();
        slashed.urlContentSha256 = FixedBytes::from_slice(&[
            0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56,
            0x78, 0x90, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0x12,
            0x34, 0x56, 0x78, 0x90,
        ]);

        apply_event(&handler, CollateralEvent::Slashed(slashed))
            .await
            .unwrap();

        let (url, checksum): (String, String) = sqlx::query_as(
            "SELECT url, url_content_sha256 FROM collateral_status WHERE hotkey = ? AND node_id = ?",
        )
        .bind(format!("0x{}", hk.encode_hex::<String>()))
        .bind(format!("0x{}", ex.encode_hex::<String>()))
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(url, "https://example.com/proof");
        assert_eq!(
            checksum,
            "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        );
    }

    #[tokio::test]
    async fn test_apply_block_events_atomic_rollback() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(23);
        let ex = make_node_id(24);
        let miner = make_miner(25);

        let initial_block = persistence.get_last_scanned_block_number().await.unwrap();

        let events = vec![
            wrap_event(
                CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 7, 11)),
                0,
            ),
            wrap_event(CollateralEvent::Denied(ev_denied(404)), 1),
        ];

        let result = handler
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
    async fn test_event_log_written_on_block_apply() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(40);
        let ex = make_node_id(41);
        let miner = make_miner(42);

        let events = vec![wrap_event(
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 10, 100)),
            0,
        )];

        handler
            .apply_collateral_events_for_block(1000, &events)
            .await
            .unwrap();

        let row_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_event_log")
            .fetch_one(persistence.pool())
            .await
            .unwrap();
        assert_eq!(row_count, 1);

        let (event_type, block_number, hotkey, node_id): (
            String,
            i64,
            Option<String>,
            Option<String>,
        ) = sqlx::query_as(
            "SELECT event_type, block_number, hotkey, node_id FROM collateral_event_log LIMIT 1",
        )
        .fetch_one(persistence.pool())
        .await
        .unwrap();

        assert_eq!(event_type, "Deposit");
        assert_eq!(block_number, 1000);
        assert!(hotkey.unwrap().starts_with("0x"));
        assert!(node_id.unwrap().starts_with("0x"));
    }

    #[tokio::test]
    async fn test_event_log_deduplication() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(43);
        let ex = make_node_id(44);
        let miner = make_miner(45);

        let events = vec![wrap_event(
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 10, 100)),
            0,
        )];

        handler
            .apply_collateral_events_for_block(2000, &events)
            .await
            .unwrap();

        // Applying the same events again (same tx_hash + log_index) should not duplicate rows.
        // The block number advances but the INSERT OR IGNORE deduplicates on (tx_hash, log_index).
        handler
            .apply_collateral_events_for_block(2001, &events)
            .await
            .unwrap();

        let row_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collateral_event_log")
            .fetch_one(persistence.pool())
            .await
            .unwrap();
        assert_eq!(row_count, 1);

        let (tao, alpha, pending_tao, pending_alpha, _) = fetch_state(&persistence, hk, ex).await;
        assert_eq!(tao, "10");
        assert_eq!(alpha, "100");
        assert_eq!(pending_tao, "0");
        assert_eq!(pending_alpha, "0");
    }

    #[tokio::test]
    async fn test_event_log_json_content() {
        let (handler, persistence) = make_handler().await;

        let hk = make_hotkey(46);
        let ex = make_node_id(47);
        let miner = make_miner(48);

        let events = vec![wrap_event(
            CollateralEvent::Deposit(ev_deposit(hk, ex, miner, 77, 999)),
            0,
        )];

        handler
            .apply_collateral_events_for_block(3000, &events)
            .await
            .unwrap();

        let event_data: String =
            sqlx::query_scalar("SELECT event_data FROM collateral_event_log LIMIT 1")
                .fetch_one(persistence.pool())
                .await
                .unwrap();

        let json: serde_json::Value = serde_json::from_str(&event_data).unwrap();
        assert!(json.get("hotkey").is_some());
        assert!(json.get("nodeId").is_some());
        assert!(json.get("miner").is_some());
        assert!(
            json.get("amount").is_some(),
            "TAO amount field missing from event JSON"
        );
        assert!(json.get("alphaAmount").is_some());
        assert_eq!(json["amount"].as_str().unwrap(), "77");
        assert_eq!(json["alphaAmount"].as_str().unwrap(), "999");
    }
}
