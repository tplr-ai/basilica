CREATE INDEX IF NOT EXISTS idx_collateral_reclaims_hotkey_node
  ON collateral_reclaims(hotkey, node_id);
