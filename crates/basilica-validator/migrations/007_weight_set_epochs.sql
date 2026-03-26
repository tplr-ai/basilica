CREATE TABLE IF NOT EXISTS weight_set_epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    netuid INTEGER NOT NULL,
    period_start INTEGER NOT NULL,
    period_end INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_weight_set_epochs_netuid_status
    ON weight_set_epochs(netuid, status);
CREATE INDEX IF NOT EXISTS idx_weight_set_epochs_netuid_period_end
    ON weight_set_epochs(netuid, period_end);
