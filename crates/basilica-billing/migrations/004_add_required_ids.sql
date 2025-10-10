-- Add required user_id and validator_id columns to usage_events table
-- These fields are CRITICAL for proper billing and auditing
-- Step 1: Add columns as nullable first (to allow existing rows)
ALTER TABLE billing.usage_events
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);
ALTER TABLE billing.usage_events
ADD COLUMN IF NOT EXISTS validator_id VARCHAR(255);
-- Step 2: Backfill existing rows with system default values
-- Use the system user for orphaned records that need proper attribution
UPDATE billing.usage_events
SET user_id = 'SYSTEM'
WHERE user_id IS NULL;
UPDATE billing.usage_events
SET validator_id = 'SYSTEM'
WHERE validator_id IS NULL;
-- Step 3: Add NOT NULL constraints
ALTER TABLE billing.usage_events
ALTER COLUMN user_id
SET NOT NULL;
ALTER TABLE billing.usage_events
ALTER COLUMN validator_id
SET NOT NULL;
-- Step 4: Add CHECK constraints to ensure non-empty trimmed strings
-- Drop if exists first to make idempotent
ALTER TABLE billing.usage_events DROP CONSTRAINT IF EXISTS check_user_id_not_empty;
ALTER TABLE billing.usage_events
ADD CONSTRAINT check_user_id_not_empty CHECK (TRIM(user_id) != '');
ALTER TABLE billing.usage_events DROP CONSTRAINT IF EXISTS check_validator_id_not_empty;
ALTER TABLE billing.usage_events
ADD CONSTRAINT check_validator_id_not_empty CHECK (TRIM(validator_id) != '');
-- Create indexes for the new columns for query performance
CREATE INDEX IF NOT EXISTS idx_usage_events_user_id ON billing.usage_events(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_validator_id ON billing.usage_events(validator_id);
-- Also add validator_id to active_rentals_facts with proper constraints
ALTER TABLE billing.active_rentals_facts
ADD COLUMN IF NOT EXISTS validator_id VARCHAR(255);
-- Backfill active_rentals_facts with system default
UPDATE billing.active_rentals_facts
SET validator_id = 'SYSTEM'
WHERE validator_id IS NULL;
ALTER TABLE billing.active_rentals_facts
ALTER COLUMN validator_id
SET NOT NULL;
ALTER TABLE billing.active_rentals_facts DROP CONSTRAINT IF EXISTS check_rental_validator_id_not_empty;
ALTER TABLE billing.active_rentals_facts
ADD CONSTRAINT check_rental_validator_id_not_empty CHECK (TRIM(validator_id) != '');
CREATE INDEX IF NOT EXISTS idx_active_rentals_validator_id ON billing.active_rentals_facts(validator_id);
-- Add comment explaining criticality
COMMENT ON COLUMN billing.usage_events.user_id IS 'REQUIRED: User who owns this rental - must never be empty';
COMMENT ON COLUMN billing.usage_events.validator_id IS 'REQUIRED: Validator managing this rental - must never be empty';
COMMENT ON COLUMN billing.usage_events.node_id IS 'REQUIRED: Node running the workload - must never be empty';
