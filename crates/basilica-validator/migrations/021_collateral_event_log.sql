CREATE TABLE IF NOT EXISTS collateral_event_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,           -- 'Deposit', 'ReclaimProcessStarted', 'Denied', 'Reclaimed', 'Slashed'
  block_number INTEGER NOT NULL,
  tx_hash TEXT NOT NULL,
  log_index INTEGER NOT NULL,
  hotkey TEXT,                        -- common field, extracted for querying
  node_id TEXT,                       -- common field, extracted for querying
  event_data TEXT NOT NULL,           -- JSON blob with all decoded event fields
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(tx_hash, log_index)          -- natural dedup key
);

CREATE INDEX IF NOT EXISTS idx_event_log_block ON collateral_event_log(block_number);
CREATE INDEX IF NOT EXISTS idx_event_log_hotkey ON collateral_event_log(hotkey);
CREATE INDEX IF NOT EXISTS idx_event_log_node_id ON collateral_event_log(node_id);
CREATE INDEX IF NOT EXISTS idx_event_log_type ON collateral_event_log(event_type);
