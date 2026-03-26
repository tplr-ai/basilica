DROP TABLE IF EXISTS miner_delivery_cache;

CREATE TABLE IF NOT EXISTS miner_delivery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start INTEGER NOT NULL,
    period_end INTEGER NOT NULL,
    deliveries TEXT NOT NULL DEFAULT '[]',
    received_at INTEGER NOT NULL,
    UNIQUE(period_start, period_end)
);

CREATE INDEX IF NOT EXISTS idx_delivery_cache_period
    ON miner_delivery_cache(period_start, period_end);
