-- This migration updates the miner revenue summary function to use progressive
-- revenue recognition based on telemetry_processed billing events instead of
-- rental completion (end_time).
--
-- Key change: Revenue is now attributed by the telemetry event timestamp within
-- the requested window, not by rental end_time. This enables accurate revenue
-- calculation for ongoing rentals that span window boundaries.

-- ===========================================================================
-- Replace stored function for progressive revenue recognition
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
    --
    -- CHANGE: Revenue is now sourced from telemetry_processed billing events
    -- and attributed by event timestamp, not rental end_time. This enables
    -- progressive revenue recognition for ongoing rentals.
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
        -- Count distinct rentals that had telemetry in this window
        COUNT(DISTINCT r.rental_id) AS total_rentals,
        COUNT(DISTINCT r.rental_id) FILTER (WHERE r.status = 'completed') AS completed_rentals,
        COUNT(DISTINCT r.rental_id) FILTER (WHERE r.status = 'failed') AS failed_rentals,
        -- Sum incremental costs from telemetry events in this window
        COALESCE(SUM((be.event_data->>'incremental_cost')::DECIMAL), 0) AS total_revenue,
        -- Sum GPU hours from usage_metrics in telemetry events
        COALESCE(SUM((be.event_data->'usage_metrics'->>'gpu_hours')::DECIMAL), 0) AS total_hours,
        -- Average hourly rate from rental settings
        AVG(r.base_price_per_gpu * r.gpu_count) AS avg_hourly_rate,
        -- Average rental duration only for completed rentals (may be NULL for ongoing)
        AVG(
            EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 3600
        ) FILTER (WHERE r.end_time IS NOT NULL) AS avg_rental_duration_hours,
        p_computation_version
    FROM billing.billing_events be
    JOIN billing.rentals r ON r.rental_id = be.entity_id::uuid
    WHERE be.event_type = 'telemetry_processed'
      AND be.event_data->>'incremental_cost' IS NOT NULL
      AND (be.event_data->>'credits_deducted')::boolean = true
      -- Attribute revenue by telemetry event timestamp (not rental end_time)
      AND (be.event_data->>'timestamp')::timestamptz >= p_period_start
      AND (be.event_data->>'timestamp')::timestamptz <= p_period_end
      -- Only community rentals (miners we need to pay)
      AND r.cloud_type = 'community'
      -- Only include rentals with miner hotkey
      AND r.metadata->>'miner_hotkey' IS NOT NULL
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
    'Refresh miner revenue summary for a given time period using progressive revenue recognition.

Creates new snapshot rows (append-only). Groups by (node_id, validator_id, miner_hotkey, miner_uid)
to track revenue per unique hotkey-UID combination.

PROGRESSIVE RECOGNITION: Revenue is sourced from telemetry_processed billing events and attributed
by the event timestamp (not rental end_time). This enables accurate revenue calculation for ongoing
rentals that span window boundaries.

Only includes community cloud rentals with miner_hotkey present and credits_deducted = true.';
