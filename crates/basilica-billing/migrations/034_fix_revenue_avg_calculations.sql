-- Fix avg_rental_duration_hours and avg_hourly_rate calculations
--
-- Bug: The previous implementation computed AVG across all joined rows
-- (billing_events × rentals), which weighted the average by the number of
-- telemetry events per rental instead of treating each rental equally.
--
-- Fix: Use a CTE to first aggregate billing events by rental_id (1 row per
-- rental), then join to rentals. This ensures AVG calculations are computed
-- per-rental, not per-event.

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
    -- First aggregate billing events by rental (1 row per rental)
    -- This ensures AVG calculations below are per-rental, not per-event
    WITH billing_agg AS (
        SELECT
            be.entity_id::uuid AS rental_id,
            SUM((be.event_data->>'incremental_cost')::DECIMAL) AS incremental_revenue,
            SUM((be.event_data->'usage_metrics'->>'gpu_hours')::DECIMAL) AS gpu_hours
        FROM billing.billing_events be
        WHERE be.event_type = 'telemetry_processed'
          AND be.event_data->>'incremental_cost' IS NOT NULL
          AND (be.event_data->>'credits_deducted')::boolean = true
          AND (be.event_data->>'timestamp')::timestamptz >= p_period_start
          AND (be.event_data->>'timestamp')::timestamptz <= p_period_end
        GROUP BY be.entity_id::uuid
    )
    SELECT
        r.metadata->>'node_id' AS node_id,
        r.metadata->>'validator_id' AS validator_id,
        r.metadata->>'miner_hotkey' AS miner_hotkey,
        (r.metadata->>'miner_uid')::INTEGER AS miner_uid,
        p_period_start,
        p_period_end,
        -- Count rentals (now 1 row per rental due to CTE)
        COUNT(*) AS total_rentals,
        COUNT(*) FILTER (WHERE r.status = 'completed') AS completed_rentals,
        COUNT(*) FILTER (WHERE r.status = 'failed') AS failed_rentals,
        -- Sum revenue and hours from pre-aggregated billing events
        COALESCE(SUM(ba.incremental_revenue), 0) AS total_revenue,
        COALESCE(SUM(ba.gpu_hours), 0) AS total_hours,
        -- Now correctly averaged per-rental (not per-event)
        AVG(r.base_price_per_gpu * r.gpu_count) AS avg_hourly_rate,
        AVG(
            EXTRACT(EPOCH FROM (r.end_time - r.start_time)) / 3600
        ) FILTER (WHERE r.end_time IS NOT NULL) AS avg_rental_duration_hours,
        p_computation_version
    FROM billing_agg ba
    JOIN billing.rentals r ON r.rental_id = ba.rental_id
    WHERE r.cloud_type = 'community'
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

FIX (migration 034): avg_hourly_rate and avg_rental_duration_hours are now correctly computed
per-rental using a CTE, not weighted by number of billing events per rental.

Only includes community cloud rentals with miner_hotkey present and credits_deducted = true.';
