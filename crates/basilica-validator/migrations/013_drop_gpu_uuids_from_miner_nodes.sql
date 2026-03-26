-- Drop gpu_uuids column from miner_nodes.
-- GPU UUIDs are tracked in gpu_uuid_assignments instead.
DROP INDEX IF EXISTS idx_nodes_gpu_uuids;

ALTER TABLE miner_nodes DROP COLUMN gpu_uuids;

-- Recreate executor_misbehaviour_log:
--   - Rename gpu_uuid → node_id
--   - Drop executor_id (was always identical to node_id)
--   - Use node_id in the PRIMARY KEY instead of executor_id
-- Table is assumed empty so we just drop and recreate.
DROP TABLE IF EXISTS executor_misbehaviour_log;

CREATE TABLE executor_misbehaviour_log (
    miner_uid INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    endpoint_executor TEXT NOT NULL,
    type_of_misbehaviour TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (miner_uid, node_id, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_misbehaviour_node_id
    ON executor_misbehaviour_log(node_id);
CREATE INDEX IF NOT EXISTS idx_misbehaviour_miner_node
    ON executor_misbehaviour_log(miner_uid, node_id);
CREATE INDEX IF NOT EXISTS idx_misbehaviour_recorded_at
    ON executor_misbehaviour_log(recorded_at);
