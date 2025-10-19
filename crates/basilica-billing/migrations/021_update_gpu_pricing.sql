-- Migration 021: Update GPU package pricing
-- Reduce pricing across all GPU packages
-- A100: $2.50 -> $1.80
-- H100: $3.50 -> $2.50
-- H200: $5.00 -> $3.75
-- B200: $15.00 -> $11.00
UPDATE
  billing.billing_packages
SET
  hourly_rate = 1.80,
  base_rate_per_hour = 1.80,
  updated_at = NOW()
WHERE
  package_id = 'a100';

UPDATE
  billing.billing_packages
SET
  hourly_rate = 2.50,
  base_rate_per_hour = 2.50,
  updated_at = NOW()
WHERE
  package_id = 'h100';

UPDATE
  billing.billing_packages
SET
  hourly_rate = 3.75,
  base_rate_per_hour = 3.75,
  updated_at = NOW()
WHERE
  package_id = 'h200';

UPDATE
  billing.billing_packages
SET
  hourly_rate = 11.00,
  base_rate_per_hour = 11.00,
  updated_at = NOW()
WHERE
  package_id = 'b200';

COMMENT ON TABLE billing.billing_packages IS 'GPU billing packages - pricing updated 2025-10-19 (migration 021)';
