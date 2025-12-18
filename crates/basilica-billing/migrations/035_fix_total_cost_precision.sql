-- Fix total_cost precision to prevent cumulative rounding errors
--
-- Bug: The total_cost column was DECIMAL(10, 2) but CreditBalance uses 6 decimal
-- places internally. When incremental costs (~$0.0275) were written to the DB,
-- they rounded to $0.03, causing ~$0.0027 drift per telemetry event.
--
-- Over 1144 events, this caused ~$3.12 overcharge on a $31.20 rental (10% inflation).
--
-- Fix: Increase precision to DECIMAL(20, 8) to match internal precision and
-- align with other billing tables (usage_aggregation_facts, billing_summary_facts).

ALTER TABLE billing.rentals
ALTER COLUMN total_cost TYPE DECIMAL(20, 8);

COMMENT ON COLUMN billing.rentals.total_cost IS
    'Total accumulated cost for the rental (8 decimal places to match CreditBalance precision)';
