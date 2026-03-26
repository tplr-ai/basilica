-- Split validator node verification liveness from miner heartbeat liveness.
-- - last_node_check: validator SSH verification timestamp
-- - last_miner_health_check: miner process heartbeat timestamp

ALTER TABLE miner_nodes RENAME COLUMN last_health_check TO last_node_check;

ALTER TABLE miner_nodes ADD COLUMN last_miner_health_check TEXT;

DROP INDEX IF EXISTS idx_miner_nodes_health_check;

CREATE INDEX IF NOT EXISTS idx_miner_nodes_node_check ON miner_nodes(last_node_check);
