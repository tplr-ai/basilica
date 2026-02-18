-- Switch collateral persistence to dual-collateral state with reclaim request tracking.
-- Collateral is not enabled yet, so we intentionally rebuild these tables instead of
-- preserving legacy rows.

ALTER TABLE collateral_status RENAME TO collateral_status_legacy;

CREATE TABLE IF NOT EXISTS collateral_status (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hotkey TEXT NOT NULL,
  node_id TEXT NOT NULL,
  miner TEXT NOT NULL,
  tao_collateral TEXT NOT NULL DEFAULT '0',
  alpha_collateral TEXT NOT NULL DEFAULT '0',
  pending_tao_reclaim TEXT NOT NULL DEFAULT '0',
  pending_alpha_reclaim TEXT NOT NULL DEFAULT '0',
  url TEXT,
  url_content_md5_checksum TEXT,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(hotkey, node_id)
);

CREATE TABLE IF NOT EXISTS collateral_reclaims (
  reclaim_request_id TEXT PRIMARY KEY,
  hotkey TEXT NOT NULL,
  node_id TEXT NOT NULL,
  miner TEXT NOT NULL,
  requested_tao_amount TEXT NOT NULL,
  requested_alpha_amount TEXT NOT NULL,
  deny_timeout TEXT NOT NULL,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE collateral_status_legacy;
