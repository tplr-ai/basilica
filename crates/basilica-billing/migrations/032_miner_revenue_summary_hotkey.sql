-- This migration adds miner_hotkey column to miner_revenue_summary table
-- and updates grouping to use (miner_hotkey, miner_uid) pairs for accurate
-- revenue attribution even when UIDs are recycled on the Bittensor network.

-- ===========================================================================
-- 1. Add miner_hotkey column
-- ===========================================================================

-- Add miner_hotkey column (NOT NULL since legacy data will be removed before migration)
ALTER TABLE billing.miner_revenue_summary
    ADD COLUMN miner_hotkey VARCHAR(128) NOT NULL;

COMMENT ON COLUMN billing.miner_revenue_summary.miner_hotkey IS
    'Bittensor miner hotkey for payment reconciliation - grouped with miner_uid to track unique hotkey-UID combinations';

-- ===========================================================================
-- 2. Update indexes
-- ===========================================================================

-- Add index for filtering by miner_hotkey
CREATE INDEX idx_miner_revenue_summary_miner_hotkey
    ON billing.miner_revenue_summary(miner_hotkey);

-- Drop old lookup index and create updated one with miner_hotkey
DROP INDEX IF EXISTS billing.idx_miner_revenue_summary_lookup;
CREATE INDEX idx_miner_revenue_summary_lookup ON billing.miner_revenue_summary(
    node_id, validator_id, miner_hotkey, miner_uid, period_start, period_end, computed_at DESC
);

-- ===========================================================================
-- 3. Update stored function to group by (miner_hotkey, miner_uid)
-- ===========================================================================

CREATE OR REPLACE FUNCTION billing.refresh_miner_revenue_summary(
    p_period_start TIMESTAMPTZ,
    p_period_end TIMESTAMPTZ,
    p_computation_version INTEGER DEFAULT 1
)
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER;
BEGIN
    -- Insert new summary rows (append-only, never update existing)
    -- Groups by (node_id, validator_id, miner_hotkey, miner_uid) to track
    -- revenue per unique hotkey-UID combination (handles UID recycling)
    INSERT INTO billing.miner_revenue_summary (
        node_id,
        validator_id,
        miner_hotkey,
        miner_uid,
        period_start,
        period_end,
        total_rentals,
        completed_rentals,
        failed_rentals,
        total_revenue,
        total_hours,
        avg_hourly_rate,
        avg_rental_duration_hours,
        computation_version
    )
    SELECT
        r.metadata->>'node_id' AS node_id,
        r.metadata->>'validator_id' AS validator_id,
        r.metadata->>'miner_hotkey' AS miner_hotkey,
        (r.metadata->>'miner_uid')::INTEGER AS miner_uid,
        p_period_start,
        p_period_end,
        COUNT(*) AS total_rentals,
        COUNT(*) FILTER (WHERE r.status = 'completed') AS completed_rentals,
        COUNT(*) FILTER (WHERE r.status = 'failed') AS failed_rentals,
        COALESCE(SUM(r.total_cost), 0) AS total_revenue,
        COALESCE(
            SUM(EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 3600),
            0
        ) AS total_hours,
        AVG(r.base_price_per_gpu * r.gpu_count) AS avg_hourly_rate,
        AVG(EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 3600) AS avg_rental_duration_hours,
        p_computation_version
    FROM billing.rentals r
    WHERE r.cloud_type = 'community'  -- Only community rentals (miners we need to pay)
      AND r.end_time IS NOT NULL
      AND r.end_time >= p_period_start
      AND r.end_time < p_period_end
      AND r.metadata->>'miner_hotkey' IS NOT NULL  -- Only include rentals with hotkey
    GROUP BY
        r.metadata->>'node_id',
        r.metadata->>'validator_id',
        r.metadata->>'miner_hotkey',
        (r.metadata->>'miner_uid')::INTEGER;

    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION billing.refresh_miner_revenue_summary IS
    'Refresh miner revenue summary for a given time period. Creates new snapshot rows (append-only). Groups by (node_id, validator_id, miner_hotkey, miner_uid) to track revenue per unique hotkey-UID combination. Only includes community cloud rentals with end_time IS NOT NULL and miner_hotkey present.';
