-- Add port_mappings column to user_rentals table
ALTER TABLE user_rentals
ADD COLUMN IF NOT EXISTS port_mappings JSONB;

-- Add port_mappings column to terminated_user_rentals table
ALTER TABLE terminated_user_rentals
ADD COLUMN IF NOT EXISTS port_mappings JSONB;

-- Add index for JSONB queries on port_mappings if needed in the future
CREATE INDEX IF NOT EXISTS idx_user_rentals_port_mappings ON user_rentals USING GIN (port_mappings);
