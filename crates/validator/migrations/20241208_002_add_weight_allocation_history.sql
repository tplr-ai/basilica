-- Add weight allocation history for auditing
CREATE TABLE IF NOT EXISTS weight_allocation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    miner_uid INTEGER NOT NULL,
    gpu_category TEXT NOT NULL,
    allocated_weight INTEGER NOT NULL,
    miner_score REAL NOT NULL,
    category_total_score REAL NOT NULL,
    weight_set_block INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- Foreign key to emission metrics
    emission_metrics_id INTEGER,
    FOREIGN KEY (emission_metrics_id) REFERENCES emission_metrics(id),
    
    -- Constraints
    CONSTRAINT valid_weight CHECK (allocated_weight >= 0),
    CONSTRAINT valid_scores CHECK (miner_score >= 0.0 AND category_total_score >= 0.0)
);

CREATE INDEX IF NOT EXISTS idx_weight_history_miner ON weight_allocation_history(miner_uid);
CREATE INDEX IF NOT EXISTS idx_weight_history_category ON weight_allocation_history(gpu_category);
CREATE INDEX IF NOT EXISTS idx_weight_history_block ON weight_allocation_history(weight_set_block);