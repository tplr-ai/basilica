-- Collateral grace periods table - tracks undercollateralized timestamps per hotkey/node
CREATE TABLE IF NOT EXISTS collateral_grace_periods (
  hotkey TEXT NOT NULL,
  node_id TEXT NOT NULL,
  undercollateralized_since TEXT NOT NULL,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (hotkey, node_id)
);
