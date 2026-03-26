-- Persist node GPU category for bid floor enforcement during UpdateBid.
-- Nullable for backward compatibility with legacy rows; validators can ask
-- miners to re-register if category is missing.
ALTER TABLE miner_nodes ADD COLUMN gpu_category TEXT;
