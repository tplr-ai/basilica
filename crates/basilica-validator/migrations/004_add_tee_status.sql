-- Add TEE (Trusted Execution Environment) status tracking for nodes
-- Tracks TDX quote verification and GPU CC mode status

-- Node TEE status table - tracks TEE verification results per node
CREATE TABLE IF NOT EXISTS node_tee_status (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  
  -- TDX quote verification
  tdx_verified BOOLEAN NOT NULL DEFAULT 0,
  tdx_quote_valid BOOLEAN DEFAULT NULL,
  tdx_mrtd_matches BOOLEAN DEFAULT NULL,
  tdx_mrtd_hex TEXT,
  tdx_rtmr0_matches BOOLEAN DEFAULT NULL,
  tdx_rtmr1_matches BOOLEAN DEFAULT NULL,
  tdx_rtmr2_matches BOOLEAN DEFAULT NULL,
  tdx_rtmr3_matches BOOLEAN DEFAULT NULL,
  tdx_report_data_matches BOOLEAN DEFAULT NULL,
  
  -- GPU Confidential Computing mode
  gpu_cc_enabled BOOLEAN NOT NULL DEFAULT 0,
  gpu_cc_attestation_valid BOOLEAN DEFAULT NULL,
  gpu_cc_model TEXT,
  gpu_cc_uuid TEXT,
  gpu_cc_driver_version TEXT,
  gpu_cc_nonce_verified BOOLEAN DEFAULT NULL,
  
  -- Overall TEE status
  tee_verified BOOLEAN NOT NULL DEFAULT 0,
  
  -- Timestamps
  last_verification_at TEXT NOT NULL,
  verification_error TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  
  PRIMARY KEY (miner_uid, node_id)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_tee_status_miner ON node_tee_status(miner_uid);
CREATE INDEX IF NOT EXISTS idx_tee_status_verified ON node_tee_status(tee_verified);
CREATE INDEX IF NOT EXISTS idx_tee_status_tdx ON node_tee_status(tdx_verified);
CREATE INDEX IF NOT EXISTS idx_tee_status_gpu_cc ON node_tee_status(gpu_cc_enabled);
CREATE INDEX IF NOT EXISTS idx_tee_status_updated ON node_tee_status(updated_at);

-- Add TEE status columns to verification_logs for historical tracking
ALTER TABLE verification_logs ADD COLUMN tee_verified INTEGER DEFAULT NULL;
ALTER TABLE verification_logs ADD COLUMN tdx_verified INTEGER DEFAULT NULL;
ALTER TABLE verification_logs ADD COLUMN gpu_cc_verified INTEGER DEFAULT NULL;
ALTER TABLE verification_logs ADD COLUMN tee_error_message TEXT DEFAULT NULL;

