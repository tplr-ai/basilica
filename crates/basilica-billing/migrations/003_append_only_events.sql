-- Simple event store for usage events and audit trail
-- Usage events table
CREATE TABLE IF NOT EXISTS billing.usage_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rental_id UUID NOT NULL,
    node_id VARCHAR(128) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    -- 'telemetry', 'status_change', 'cost_update'
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    batch_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Billing events table for audit trail
CREATE TABLE IF NOT EXISTS billing.billing_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    -- 'credit_applied', 'rental_started', 'rental_ended', 'rule_applied'
    entity_type VARCHAR(50) NOT NULL,
    -- 'user', 'rental', 'credit', 'package'
    entity_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES billing.users(user_id),
    event_data JSONB NOT NULL,
    metadata JSONB,
    created_by VARCHAR(255),
    -- System component or user that created the event
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Processing batches table for tracking event processing
CREATE TABLE IF NOT EXISTS billing.processing_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_type VARCHAR(50) NOT NULL,
    -- 'usage_aggregation', 'billing_calculation'
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- 'pending', 'processing', 'completed', 'failed'
    events_count INTEGER NOT NULL DEFAULT 0,
    events_processed INTEGER NOT NULL DEFAULT 0,
    events_failed INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Telemetry buffer table for high-speed ingestion
CREATE UNLOGGED TABLE IF NOT EXISTS billing.telemetry_buffer (
    buffer_id BIGSERIAL PRIMARY KEY,
    rental_id UUID NOT NULL,
    node_id VARCHAR(128) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    cpu_percent DECIMAL(5, 2),
    memory_mb BIGINT,
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    gpu_metrics JSONB,
    custom_metrics JSONB,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Archive tables for old events
CREATE TABLE IF NOT EXISTS billing.usage_events_archive (LIKE billing.usage_events INCLUDING ALL);
CREATE TABLE IF NOT EXISTS billing.billing_events_archive (LIKE billing.billing_events INCLUDING ALL);
-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_usage_events_rental_id ON billing.usage_events(rental_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_node_id ON billing.usage_events(node_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_processed ON billing.usage_events(processed)
WHERE processed = false;
CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp ON billing.usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_events_type ON billing.usage_events(event_type);
CREATE INDEX IF NOT EXISTS idx_usage_events_batch ON billing.usage_events(batch_id)
WHERE batch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_billing_events_entity ON billing.billing_events(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_user_id ON billing.billing_events(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_type ON billing.billing_events(event_type);
CREATE INDEX IF NOT EXISTS idx_billing_events_created_at ON billing.billing_events(created_at);
CREATE INDEX IF NOT EXISTS idx_processing_batches_status ON billing.processing_batches(status);
CREATE INDEX IF NOT EXISTS idx_processing_batches_type ON billing.processing_batches(batch_type);
CREATE INDEX IF NOT EXISTS idx_telemetry_buffer_rental ON billing.telemetry_buffer(rental_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_buffer_timestamp ON billing.telemetry_buffer(timestamp);
