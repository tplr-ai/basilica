-- Add GPU, storage, billing, queue, and suspend fields to user_deployments table
ALTER TABLE user_deployments
ADD COLUMN IF NOT EXISTS gpu_count INTEGER,
ADD COLUMN IF NOT EXISTS gpu_models TEXT[],
ADD COLUMN IF NOT EXISTS min_cuda_version VARCHAR(20),
ADD COLUMN IF NOT EXISTS min_gpu_memory_gb INTEGER,
ADD COLUMN IF NOT EXISTS storage_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS storage_backend VARCHAR(10),
ADD COLUMN IF NOT EXISTS storage_bucket TEXT,
ADD COLUMN IF NOT EXISTS storage_region TEXT,
ADD COLUMN IF NOT EXISTS storage_endpoint TEXT,
ADD COLUMN IF NOT EXISTS storage_credentials_secret TEXT,
ADD COLUMN IF NOT EXISTS storage_sync_interval_ms BIGINT DEFAULT 1000,
ADD COLUMN IF NOT EXISTS storage_cache_size_mb INTEGER DEFAULT 1024,
ADD COLUMN IF NOT EXISTS storage_mount_path TEXT DEFAULT '/mnt/storage',
ADD COLUMN IF NOT EXISTS enable_billing BOOLEAN DEFAULT TRUE NOT NULL,
ADD COLUMN IF NOT EXISTS queue_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS suspended BOOLEAN DEFAULT FALSE NOT NULL,
ADD COLUMN IF NOT EXISTS priority VARCHAR(50);

-- Add constraints for GPU fields
ALTER TABLE user_deployments
ADD CONSTRAINT chk_gpu_count_positive CHECK (gpu_count IS NULL OR gpu_count > 0),
ADD CONSTRAINT chk_gpu_count_max CHECK (gpu_count IS NULL OR gpu_count <= 8),
ADD CONSTRAINT chk_gpu_memory_positive CHECK (min_gpu_memory_gb IS NULL OR min_gpu_memory_gb > 0),
ADD CONSTRAINT chk_storage_backend_valid CHECK (
    storage_backend IS NULL OR storage_backend IN ('r2', 's3', 'gcs')
),
ADD CONSTRAINT chk_storage_sync_interval CHECK (
    storage_sync_interval_ms >= 100 AND storage_sync_interval_ms <= 60000
),
ADD CONSTRAINT chk_storage_cache_size CHECK (
    storage_cache_size_mb >= 512 AND storage_cache_size_mb <= 16384
);

-- Index for GPU-based queries
CREATE INDEX IF NOT EXISTS idx_user_deployments_gpu_count ON user_deployments(gpu_count)
WHERE gpu_count IS NOT NULL;

-- Index for queue-based queries
CREATE INDEX IF NOT EXISTS idx_user_deployments_queue_name ON user_deployments(queue_name)
WHERE queue_name IS NOT NULL;

-- Index for suspended deployments
CREATE INDEX IF NOT EXISTS idx_user_deployments_suspended ON user_deployments(suspended)
WHERE suspended = TRUE;

-- Update existing constraint to allow 0 replicas for suspended deployments
ALTER TABLE user_deployments
DROP CONSTRAINT IF EXISTS chk_replicas_positive;

ALTER TABLE user_deployments
ADD CONSTRAINT chk_replicas_non_negative CHECK (replicas >= 0);

-- Comment explaining schema additions
COMMENT ON COLUMN user_deployments.gpu_count IS 'Number of GPUs required for this deployment';
COMMENT ON COLUMN user_deployments.gpu_models IS 'Array of acceptable GPU models (e.g., {A100, H100})';
COMMENT ON COLUMN user_deployments.min_cuda_version IS 'Minimum CUDA version required (e.g., 12.2)';
COMMENT ON COLUMN user_deployments.min_gpu_memory_gb IS 'Minimum GPU memory in GB';
COMMENT ON COLUMN user_deployments.storage_enabled IS 'Whether persistent storage is enabled';
COMMENT ON COLUMN user_deployments.storage_backend IS 'Storage backend type: r2, s3, or gcs';
COMMENT ON COLUMN user_deployments.storage_bucket IS 'Storage bucket name';
COMMENT ON COLUMN user_deployments.enable_billing IS 'Whether billing is enabled for this deployment';
COMMENT ON COLUMN user_deployments.queue_name IS 'Queue name for concurrency control';
COMMENT ON COLUMN user_deployments.suspended IS 'Whether deployment is suspended (replicas=0)';
COMMENT ON COLUMN user_deployments.priority IS 'Deployment priority (e.g., high, medium, low)';
