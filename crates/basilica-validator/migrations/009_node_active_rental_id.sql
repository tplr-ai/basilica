-- Replace node_reservations table with an inline active_rental_id column on miner_nodes.
-- When NULL the node is available; when set the node is claimed for a rental.
-- No TTL -- the column is cleared explicitly on every termination path.

ALTER TABLE miner_nodes ADD COLUMN active_rental_id TEXT DEFAULT NULL;

-- Backfill: set active_rental_id for nodes that currently have an active/provisioning rental.
UPDATE miner_nodes
SET active_rental_id = (
    SELECT r.id FROM rentals r
    WHERE r.node_id = miner_nodes.node_id
      AND r.miner_id = miner_nodes.miner_id
      AND r.state IN ('active', 'provisioning')
    LIMIT 1
)
WHERE EXISTS (
    SELECT 1 FROM rentals r
    WHERE r.node_id = miner_nodes.node_id
      AND r.miner_id = miner_nodes.miner_id
      AND r.state IN ('active', 'provisioning')
);

-- Index for fast availability checks (WHERE active_rental_id IS NULL).
CREATE INDEX IF NOT EXISTS idx_miner_nodes_active_rental ON miner_nodes(active_rental_id);

