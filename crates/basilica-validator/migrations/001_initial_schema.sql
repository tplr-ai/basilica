-- Initial schema for validator database
-- Creates all base tables and indexes for the validator service
-- Miners table - core miner registry
CREATE TABLE IF NOT EXISTS miners (
  id TEXT PRIMARY KEY,
  hotkey TEXT NOT NULL UNIQUE,
  endpoint TEXT NOT NULL,
  verification_score REAL DEFAULT 0.0,
  uptime_percentage REAL DEFAULT 0.0,
  last_seen TEXT NOT NULL,
  registered_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  node_info TEXT NOT NULL DEFAULT '{}'
);

-- Miner nodes table - miner's executor nodes
CREATE TABLE IF NOT EXISTS miner_nodes (
  id TEXT PRIMARY KEY,
  miner_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  ssh_endpoint TEXT NOT NULL,
  gpu_count INTEGER NOT NULL,
  gpu_uuids TEXT,
  status TEXT DEFAULT 'unknown',
  last_health_check TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
);

-- Verification requests table - scheduled verification tasks
CREATE TABLE IF NOT EXISTS verification_requests (
  id TEXT PRIMARY KEY,
  miner_id TEXT NOT NULL,
  verification_type TEXT NOT NULL,
  node_id TEXT,
  status TEXT DEFAULT 'scheduled',
  scheduled_at TEXT NOT NULL,
  completed_at TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
);

-- Verification logs table - verification execution results
CREATE TABLE IF NOT EXISTS verification_logs (
  id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  validator_hotkey TEXT NOT NULL,
  verification_type TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  score REAL NOT NULL,
  success INTEGER NOT NULL,
  details TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  error_message TEXT,
  last_binary_validation TEXT,
  last_binary_validation_score REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Rentals table - GPU rental sessions
CREATE TABLE IF NOT EXISTS rentals (
  id TEXT PRIMARY KEY,
  validator_hotkey TEXT NOT NULL,
  node_id TEXT NOT NULL,
  miner_id TEXT NOT NULL DEFAULT '',
  container_id TEXT NOT NULL,
  ssh_session_id TEXT NOT NULL,
  ssh_credentials TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  container_spec TEXT NOT NULL,
  customer_public_key TEXT,
  docker_image TEXT,
  env_vars TEXT,
  gpu_requirements TEXT,
  ssh_access_info TEXT,
  cost_per_hour REAL,
  status TEXT,
  updated_at TEXT,
  started_at TEXT,
  terminated_at TEXT,
  termination_reason TEXT,
  total_cost REAL,
  metadata TEXT NOT NULL DEFAULT '{}'
);

-- Miner GPU profiles table - GPU scoring profiles
CREATE TABLE IF NOT EXISTS miner_gpu_profiles (
  miner_uid INTEGER PRIMARY KEY,
  gpu_counts_json TEXT NOT NULL,
  total_score REAL NOT NULL,
  verification_count INTEGER NOT NULL,
  last_updated TEXT NOT NULL,
  last_successful_validation TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT valid_score CHECK (
    total_score >= 0.0
    AND total_score <= 1.0
  ),
  CONSTRAINT valid_count CHECK (verification_count >= 0)
);

-- Emission metrics table - token emission tracking
CREATE TABLE IF NOT EXISTS emission_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  burn_amount INTEGER NOT NULL,
  burn_percentage REAL NOT NULL,
  category_distributions_json TEXT NOT NULL,
  total_miners INTEGER NOT NULL,
  weight_set_block INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT valid_burn_percentage CHECK (
    burn_percentage >= 0.0
    AND burn_percentage <= 100.0
  ),
  CONSTRAINT valid_total_miners CHECK (total_miners >= 0)
);

-- Miner prover results table - GPU attestation results
CREATE TABLE IF NOT EXISTS miner_prover_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  gpu_model TEXT NOT NULL,
  gpu_count INTEGER NOT NULL,
  gpu_memory_gb REAL NOT NULL,
  gpu_uuid TEXT,
  attestation_valid INTEGER NOT NULL,
  verification_timestamp TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT valid_gpu_count CHECK (gpu_count >= 0),
  CONSTRAINT valid_gpu_memory CHECK (gpu_memory_gb >= 0)
);

-- Node hardware profile table - CPU/RAM/disk specs
CREATE TABLE IF NOT EXISTS node_hardware_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  cpu_model TEXT,
  cpu_cores INTEGER,
  ram_gb INTEGER,
  disk_gb INTEGER,
  full_hardware_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Node speedtest profile table - network speed tests
CREATE TABLE IF NOT EXISTS node_speedtest_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  download_mbps REAL,
  upload_mbps REAL,
  test_timestamp TEXT NOT NULL,
  test_server TEXT,
  full_result_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Node network profile table - IP/geolocation data
CREATE TABLE IF NOT EXISTS node_network_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  ip_address TEXT,
  hostname TEXT,
  city TEXT,
  region TEXT,
  country TEXT,
  location TEXT,
  organization TEXT,
  postal_code TEXT,
  timezone TEXT,
  test_timestamp TEXT NOT NULL,
  full_result_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Node docker profile table - docker validation results
CREATE TABLE IF NOT EXISTS node_docker_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  service_active BOOLEAN NOT NULL,
  docker_version TEXT,
  images_pulled TEXT,
  dind_supported BOOLEAN DEFAULT 0,
  validation_error TEXT,
  full_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Node NAT profile table - NAT traversal tests
CREATE TABLE IF NOT EXISTS node_nat_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  is_accessible BOOLEAN NOT NULL,
  test_port INTEGER NOT NULL,
  test_path TEXT NOT NULL,
  container_id TEXT,
  response_content TEXT,
  test_timestamp TEXT NOT NULL,
  full_json TEXT NOT NULL,
  error_message TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Node storage profile table - disk space metrics
CREATE TABLE IF NOT EXISTS node_storage_profile (
  miner_uid INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  total_bytes INTEGER NOT NULL,
  available_bytes INTEGER NOT NULL,
  required_bytes INTEGER NOT NULL,
  filesystem_details TEXT NOT NULL,
  collection_timestamp TEXT NOT NULL,
  full_json TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (miner_uid, node_id)
);

-- Weight allocation history table - weight distribution history
CREATE TABLE IF NOT EXISTS weight_allocation_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  miner_uid INTEGER NOT NULL,
  gpu_category TEXT NOT NULL,
  allocated_weight INTEGER NOT NULL,
  miner_score REAL NOT NULL,
  category_total_score REAL NOT NULL,
  weight_set_block INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  emission_metrics_id INTEGER,
  FOREIGN KEY (emission_metrics_id) REFERENCES emission_metrics(id),
  CONSTRAINT valid_weight CHECK (allocated_weight >= 0),
  CONSTRAINT valid_scores CHECK (
    miner_score >= 0.0
    AND category_total_score >= 0.0
  )
);

-- GPU UUID assignments table - GPU UUID tracking
CREATE TABLE IF NOT EXISTS gpu_uuid_assignments (
  gpu_uuid TEXT PRIMARY KEY,
  gpu_index INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  miner_id TEXT NOT NULL,
  gpu_name TEXT,
  gpu_memory_gb REAL DEFAULT NULL,
  last_verified TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Collateral status table - tracks collateral status for hotkey/node pairs
CREATE TABLE IF NOT EXISTS collateral_status (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hotkey TEXT NOT NULL,
  node_id TEXT NOT NULL,
  miner TEXT NOT NULL,
  collateral TEXT NOT NULL,
  url TEXT,
  url_content_md5_checksum TEXT,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(hotkey, node_id)
);

-- Collateral scan status table - tracks last scanned blockchain block
CREATE TABLE IF NOT EXISTS collateral_scan_status (
  id INTEGER PRIMARY KEY,
  last_scanned_block_number INTEGER NOT NULL,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes for miner_gpu_profiles
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_score ON miner_gpu_profiles(total_score DESC);

CREATE INDEX IF NOT EXISTS idx_gpu_profiles_updated ON miner_gpu_profiles(last_updated);

-- Performance indexes for emission_metrics
CREATE INDEX IF NOT EXISTS idx_emission_metrics_timestamp ON emission_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_emission_metrics_block ON emission_metrics(weight_set_block);

-- Performance indexes for miner_prover_results
CREATE INDEX IF NOT EXISTS idx_prover_results_miner ON miner_prover_results(miner_uid);

CREATE INDEX IF NOT EXISTS idx_prover_results_timestamp ON miner_prover_results(verification_timestamp);

CREATE INDEX IF NOT EXISTS idx_prover_results_gpu_uuid ON miner_prover_results(gpu_uuid);

-- Performance indexes for weight_allocation_history
CREATE INDEX IF NOT EXISTS idx_weight_history_miner ON weight_allocation_history(miner_uid);

CREATE INDEX IF NOT EXISTS idx_weight_history_category ON weight_allocation_history(gpu_category);

CREATE INDEX IF NOT EXISTS idx_weight_history_block ON weight_allocation_history(weight_set_block);

-- Performance indexes for node profiles
CREATE INDEX IF NOT EXISTS idx_node_hardware_miner ON node_hardware_profile(miner_uid);

CREATE INDEX IF NOT EXISTS idx_node_hardware_updated ON node_hardware_profile(updated_at);

CREATE INDEX IF NOT EXISTS idx_node_speedtest_miner ON node_speedtest_profile(miner_uid);

CREATE INDEX IF NOT EXISTS idx_node_speedtest_timestamp ON node_speedtest_profile(test_timestamp);

CREATE INDEX IF NOT EXISTS idx_node_network_miner ON node_network_profile(miner_uid);

CREATE INDEX IF NOT EXISTS idx_node_network_timestamp ON node_network_profile(test_timestamp);

CREATE INDEX IF NOT EXISTS idx_node_network_country ON node_network_profile(country);

CREATE INDEX IF NOT EXISTS idx_node_docker_miner ON node_docker_profile(miner_uid);

CREATE INDEX IF NOT EXISTS idx_node_docker_updated ON node_docker_profile(updated_at);

-- Performance indexes for GPU assignments
CREATE INDEX IF NOT EXISTS idx_gpu_assignments_node ON gpu_uuid_assignments(node_id);

CREATE INDEX IF NOT EXISTS idx_gpu_assignments_miner ON gpu_uuid_assignments(miner_id);

CREATE INDEX IF NOT EXISTS idx_gpu_assignments_miner_node ON gpu_uuid_assignments(miner_id, node_id);

-- Performance indexes for miner_nodes
CREATE INDEX IF NOT EXISTS idx_miner_nodes_status ON miner_nodes(status);

CREATE INDEX IF NOT EXISTS idx_miner_nodes_health_check ON miner_nodes(last_health_check);

CREATE INDEX IF NOT EXISTS idx_nodes_gpu_uuids ON miner_nodes(gpu_uuids);
