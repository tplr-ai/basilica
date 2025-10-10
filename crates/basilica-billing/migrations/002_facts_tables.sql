-- Simplified facts tables for billing analytics
-- Active rentals facts table
CREATE TABLE IF NOT EXISTS billing.active_rentals_facts (
    rental_id UUID PRIMARY KEY,
    node_id VARCHAR(128) NOT NULL,
    validator_id VARCHAR(128) NOT NULL,
    user_id UUID NOT NULL REFERENCES billing.users(user_id),
    rental_category VARCHAR(64) NOT NULL,
    -- 'gpu_compute', 'cpu_only', 'inference', 'training'
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    hourly_rate DECIMAL(10, 8) NOT NULL,
    resource_spec JSONB NOT NULL,
    gpu_count INTEGER DEFAULT 0,
    gpu_models TEXT [],
    cpu_cores INTEGER NOT NULL,
    memory_gb DECIMAL(10, 2) NOT NULL,
    status VARCHAR(32) NOT NULL,
    total_cost DECIMAL(20, 8),
    duration_hours DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Usage aggregation facts table (hourly aggregations)
CREATE TABLE IF NOT EXISTS billing.usage_aggregation_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rental_id UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES billing.users(user_id),
    aggregation_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    aggregation_type VARCHAR(20) NOT NULL,
    -- 'hourly', 'daily'
    cpu_usage_avg DECIMAL(5, 2),
    cpu_usage_max DECIMAL(5, 2),
    memory_usage_avg_gb DECIMAL(10, 2),
    memory_usage_max_gb DECIMAL(10, 2),
    gpu_usage_avg DECIMAL(5, 2),
    gpu_usage_max DECIMAL(5, 2),
    network_ingress_gb DECIMAL(10, 4),
    network_egress_gb DECIMAL(10, 4),
    disk_read_gb DECIMAL(10, 4),
    disk_write_gb DECIMAL(10, 4),
    disk_iops_avg INTEGER,
    disk_iops_max INTEGER,
    cost_for_period DECIMAL(10, 8),
    data_points_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Billing summary facts table (monthly aggregations)
CREATE TABLE IF NOT EXISTS billing.billing_summary_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES billing.users(user_id),
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    package_id VARCHAR(100) REFERENCES billing.billing_packages(package_id),
    total_rentals INTEGER NOT NULL DEFAULT 0,
    total_hours DECIMAL(10, 2) NOT NULL DEFAULT 0,
    total_cpu_hours DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_gpu_hours DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_memory_gb_hours DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_network_gb DECIMAL(15, 4) NOT NULL DEFAULT 0,
    total_disk_iops BIGINT NOT NULL DEFAULT 0,
    base_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
    discounts_applied DECIMAL(20, 8) NOT NULL DEFAULT 0,
    surcharges_applied DECIMAL(20, 8) NOT NULL DEFAULT 0,
    total_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
    credits_used DECIMAL(20, 8) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, year, month)
);
-- Indexes for facts tables
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_user_id ON billing.active_rentals_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_node_id ON billing.active_rentals_facts(node_id);
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_validator_id ON billing.active_rentals_facts(validator_id);
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_status ON billing.active_rentals_facts(status);
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_dates ON billing.active_rentals_facts(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_active_rentals_facts_category ON billing.active_rentals_facts(rental_category);
CREATE INDEX IF NOT EXISTS idx_usage_aggregation_facts_rental ON billing.usage_aggregation_facts(rental_id);
CREATE INDEX IF NOT EXISTS idx_usage_aggregation_facts_user ON billing.usage_aggregation_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_aggregation_facts_timestamp ON billing.usage_aggregation_facts(aggregation_timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_aggregation_facts_type ON billing.usage_aggregation_facts(aggregation_type);
CREATE INDEX IF NOT EXISTS idx_billing_summary_facts_user ON billing.billing_summary_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_summary_facts_period ON billing.billing_summary_facts(year, month);
CREATE INDEX IF NOT EXISTS idx_billing_summary_facts_package ON billing.billing_summary_facts(package_id);
