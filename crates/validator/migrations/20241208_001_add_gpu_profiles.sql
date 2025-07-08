-- Add miner GPU profiles table
CREATE TABLE IF NOT EXISTS miner_gpu_profiles (
    miner_uid INTEGER PRIMARY KEY,
    primary_gpu_model TEXT NOT NULL,
    gpu_counts_json TEXT NOT NULL,
    total_score REAL NOT NULL,
    verification_count INTEGER NOT NULL,
    last_updated TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_score CHECK (total_score >= 0.0 AND total_score <= 1.0),
    CONSTRAINT valid_count CHECK (verification_count >= 0)
);

-- Add emission metrics table
CREATE TABLE IF NOT EXISTS emission_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    burn_amount INTEGER NOT NULL,
    burn_percentage REAL NOT NULL,
    category_distributions_json TEXT NOT NULL,
    total_miners INTEGER NOT NULL,
    weight_set_block INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_burn_percentage CHECK (burn_percentage >= 0.0 AND burn_percentage <= 100.0),
    CONSTRAINT valid_total_miners CHECK (total_miners >= 0)
);

-- Add miner prover verification results table
CREATE TABLE IF NOT EXISTS miner_prover_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    miner_uid INTEGER NOT NULL,
    executor_id TEXT NOT NULL,
    gpu_model TEXT NOT NULL,
    gpu_count INTEGER NOT NULL,
    gpu_memory_gb INTEGER NOT NULL,
    attestation_valid INTEGER NOT NULL,
    verification_timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_gpu_count CHECK (gpu_count >= 0),
    CONSTRAINT valid_gpu_memory CHECK (gpu_memory_gb >= 0)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_model ON miner_gpu_profiles(primary_gpu_model);
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_score ON miner_gpu_profiles(total_score DESC);
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_updated ON miner_gpu_profiles(last_updated);
CREATE INDEX IF NOT EXISTS idx_emission_metrics_timestamp ON emission_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_emission_metrics_block ON emission_metrics(weight_set_block);
CREATE INDEX IF NOT EXISTS idx_prover_results_miner ON miner_prover_results(miner_uid);
CREATE INDEX IF NOT EXISTS idx_prover_results_timestamp ON miner_prover_results(verification_timestamp);