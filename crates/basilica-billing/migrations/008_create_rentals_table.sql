-- Migration: Create rentals table for tracking GPU rentals
-- This table stores active and historical rental information
CREATE TABLE IF NOT EXISTS billing.rentals (
    rental_id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    node_id VARCHAR(255) NOT NULL,
    validator_id VARCHAR(255),
    package_id VARCHAR(100),
    resource_spec JSONB NOT NULL DEFAULT '{}',
    hourly_rate DECIMAL(10, 2) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    max_duration_hours INTEGER NOT NULL DEFAULT 24,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    total_cost DECIMAL(10, 2),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_rentals_user_id ON billing.rentals(user_id);
CREATE INDEX IF NOT EXISTS idx_rentals_node_id ON billing.rentals(node_id);
CREATE INDEX IF NOT EXISTS idx_rentals_status ON billing.rentals(status);
CREATE INDEX IF NOT EXISTS idx_rentals_start_time ON billing.rentals(start_time);
CREATE INDEX IF NOT EXISTS idx_rentals_end_time ON billing.rentals(end_time);
CREATE INDEX IF NOT EXISTS idx_rentals_active ON billing.rentals(user_id, status)
WHERE status IN ('active', 'pending');
-- Add comments
COMMENT ON TABLE billing.rentals IS 'Stores GPU rental records for billing purposes';
COMMENT ON COLUMN billing.rentals.rental_id IS 'Unique identifier for the rental';
COMMENT ON COLUMN billing.rentals.user_id IS 'User who initiated the rental';
COMMENT ON COLUMN billing.rentals.node_id IS 'Node providing the GPU resources';
COMMENT ON COLUMN billing.rentals.validator_id IS 'Validator overseeing the rental';
COMMENT ON COLUMN billing.rentals.package_id IS 'Billing package being used for this rental';
COMMENT ON COLUMN billing.rentals.resource_spec IS 'JSON specification of resources being rented';
COMMENT ON COLUMN billing.rentals.hourly_rate IS 'Credits charged per hour for this rental';
COMMENT ON COLUMN billing.rentals.status IS 'Current status: pending, active, completed, cancelled, failed';
COMMENT ON COLUMN billing.rentals.total_cost IS 'Final cost of the rental once completed';
COMMENT ON COLUMN billing.rentals.metadata IS 'Additional rental metadata as JSON';
