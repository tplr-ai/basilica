CREATE TABLE IF NOT EXISTS miner_delivery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    miner_hotkey TEXT NOT NULL,
    miner_uid INTEGER NOT NULL,
    gpu_category TEXT NOT NULL,
    node_id TEXT NOT NULL DEFAULT '',
    period_start INTEGER NOT NULL,
    period_end INTEGER NOT NULL,
    total_hours REAL NOT NULL,
    revenue_usd REAL NOT NULL,
    received_at INTEGER NOT NULL,
    UNIQUE(miner_hotkey, node_id, gpu_category, period_start, period_end)
);

CREATE INDEX IF NOT EXISTS idx_delivery_cache_period
    ON miner_delivery_cache(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_delivery_cache_hotkey
    ON miner_delivery_cache(miner_hotkey);
CREATE INDEX IF NOT EXISTS idx_delivery_cache_node
    ON miner_delivery_cache(node_id);
