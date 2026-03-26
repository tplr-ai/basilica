-- Drop the redundant last_seen column from miners table.
-- updated_at is always set at the same time and serves the same purpose.
ALTER TABLE miners DROP COLUMN last_seen;

-- Drop dead columns: verification_score (always 0.0), uptime_percentage (always 100.0),
-- node_info (always '{}', redundant with miner_nodes), registered_at (just validator startup time).
ALTER TABLE miners DROP COLUMN verification_score;
ALTER TABLE miners DROP COLUMN uptime_percentage;
ALTER TABLE miners DROP COLUMN node_info;
ALTER TABLE miners DROP COLUMN registered_at;

-- Drop updated_at from miner_nodes: every read uses last_health_check (always set on INSERT),
-- so the last_health_check IS NULL fallback branches were dead code.
ALTER TABLE miner_nodes DROP COLUMN updated_at;
