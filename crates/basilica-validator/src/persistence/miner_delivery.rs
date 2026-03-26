use anyhow::Result;
use basilica_protocol::billing::MinerDelivery;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::sync::Arc;

use crate::persistence::SimplePersistence;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedDelivery {
    miner_hotkey: String,
    miner_uid: u32,
    gpu_category: String,
    node_id: String,
    total_hours: f64,
    revenue_usd: f64,
}

impl From<&MinerDelivery> for CachedDelivery {
    fn from(d: &MinerDelivery) -> Self {
        Self {
            miner_hotkey: d.miner_hotkey.clone(),
            miner_uid: d.miner_uid,
            gpu_category: d.gpu_category.clone(),
            node_id: d.node_id.clone(),
            total_hours: d.total_hours,
            revenue_usd: d.revenue_usd,
        }
    }
}

impl From<CachedDelivery> for MinerDelivery {
    fn from(c: CachedDelivery) -> Self {
        Self {
            miner_hotkey: c.miner_hotkey,
            miner_uid: c.miner_uid,
            gpu_category: c.gpu_category,
            node_id: c.node_id,
            total_hours: c.total_hours,
            revenue_usd: c.revenue_usd,
        }
    }
}

#[derive(Clone)]
pub struct MinerDeliveryRepository {
    persistence: Arc<SimplePersistence>,
}

impl MinerDeliveryRepository {
    pub fn new(persistence: Arc<SimplePersistence>) -> Self {
        Self { persistence }
    }

    pub async fn store_deliveries(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        deliveries: &[MinerDelivery],
    ) -> Result<()> {
        let cached: Vec<CachedDelivery> = deliveries.iter().map(CachedDelivery::from).collect();
        let json = serde_json::to_string(&cached)?;

        sqlx::query(
            r#"
            INSERT INTO miner_delivery_cache (period_start, period_end, deliveries, received_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(period_start, period_end)
            DO UPDATE SET deliveries = excluded.deliveries, received_at = excluded.received_at
            "#,
        )
        .bind(period_start.timestamp())
        .bind(period_end.timestamp())
        .bind(&json)
        .bind(Utc::now().timestamp())
        .execute(self.persistence.pool())
        .await?;

        Ok(())
    }

    pub async fn get_deliveries(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
        _miner_hotkeys: Option<Vec<String>>,
    ) -> Result<Vec<MinerDelivery>> {
        let rows = sqlx::query(
            "SELECT deliveries FROM miner_delivery_cache WHERE period_end >= ? AND period_start <= ?",
        )
        .bind(since.timestamp())
        .bind(until.timestamp())
        .fetch_all(self.persistence.pool())
        .await?;

        let mut result = Vec::new();
        for row in rows {
            let json: String = row.get("deliveries");
            let cached: Vec<CachedDelivery> = serde_json::from_str(&json)?;
            result.extend(cached.into_iter().map(MinerDelivery::from));
        }
        Ok(result)
    }

    pub async fn get_deliveries_for_window(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        _miner_hotkeys: Option<Vec<String>>,
    ) -> Result<Vec<MinerDelivery>> {
        let row = sqlx::query(
            "SELECT deliveries FROM miner_delivery_cache WHERE period_start = ? AND period_end = ?",
        )
        .bind(period_start.timestamp())
        .bind(period_end.timestamp())
        .fetch_optional(self.persistence.pool())
        .await?;

        match row {
            Some(row) => {
                let json: String = row.get("deliveries");
                let cached: Vec<CachedDelivery> = serde_json::from_str(&json)?;
                Ok(cached.into_iter().map(MinerDelivery::from).collect())
            }
            None => Ok(vec![]),
        }
    }
}
