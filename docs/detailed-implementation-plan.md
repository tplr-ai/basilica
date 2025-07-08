# Detailed Implementation Plan: Burn Emissions and GPU-Based Allocation

## Overview

This document provides a task-by-task implementation plan with specific code deliverables and comprehensive testing requirements. **No task can be marked complete without passing all unit tests.**

## Timeline: 3 Weeks (15 working days)

---

## Phase 1: Core Infrastructure (Days 1-5)

### Task 1.1: Configuration System Foundation
**Duration:** 2 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/config/emission.rs`** (New File)
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmissionConfig {
    /// Percentage of emissions to burn (0.0-100.0)
    pub burn_percentage: f64,
    
    /// UID to send burn weights to
    pub burn_uid: u16,
    
    /// GPU model allocation percentages (must sum to 100.0)
    pub gpu_allocations: HashMap<String, f64>,
    
    /// Blocks between weight setting
    pub weight_set_interval_blocks: u64,
    
    /// Minimum miners required per GPU category to receive allocation
    pub min_miners_per_category: u32,
}

impl EmissionConfig {
    pub fn validate(&self) -> Result<()> {
        // Implementation required
    }
    
    pub fn from_toml_file(path: &std::path::Path) -> Result<Self> {
        // Implementation required
    }
    
    pub fn merge_with_defaults(self) -> Self {
        // Implementation required
    }
}

impl Default for EmissionConfig {
    fn default() -> Self {
        // Implementation required
    }
}
```

// Add tests at the bottom of emission.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config_is_valid() {
        let config = EmissionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_burn_percentage_validation() {
        // Test valid ranges (0.0-100.0)
        // Test invalid ranges (negative, >100)
    }

    #[test]
    fn test_gpu_allocations_sum_to_100() {
        // Test valid allocation sums
        // Test invalid allocation sums
        // Test empty allocations
    }

    #[test]
    fn test_weight_interval_validation() {
        // Test valid intervals
        // Test zero interval (should fail)
    }

    #[test]
    fn test_config_serialization() {
        // Test TOML serialization/deserialization
        // Test JSON serialization/deserialization
    }

    #[test]
    fn test_config_from_toml_file() {
        // Test loading from valid TOML file
        // Test loading from invalid TOML file
        // Test loading from non-existent file
    }

    #[test]
    fn test_config_merge_with_defaults() {
        // Test partial config merging
        // Test complete config override
    }

    #[test]
    fn test_edge_cases() {
        // Test extreme values
        // Test unicode in GPU model names
        // Test very large numbers
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 8 test functions covering all validation scenarios
- ✅ Edge case handling (negative values, overflow, empty maps)
- ✅ File I/O error handling
- ✅ Serialization/deserialization round-trips
- ✅ Default configuration validation

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Documentation with examples
- [ ] Integration with existing config system
- [ ] Error messages are user-friendly

---

### Task 1.2: GPU Model Categorization System
**Duration:** 2 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/gpu/categorization.rs`** (New File)
```rust
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use common::identity::MinerUid;

#[derive(Debug, Clone, PartialEq)]
pub struct MinerGpuProfile {
    pub miner_uid: MinerUid,
    pub primary_gpu_model: String,
    pub gpu_counts: HashMap<String, u32>,
    pub total_score: f64,
    pub verification_count: u32,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuCategory {
    H100,
    H200,
    A100,
    RTX4090,
    RTX3090,
    Other(String),
}

pub struct GpuCategorizer;

impl GpuCategorizer {
    /// Normalize GPU model string to standard category
    pub fn normalize_gpu_model(gpu_model: &str) -> String {
        // Implementation required
    }
    
    /// Convert normalized model to category enum
    pub fn model_to_category(model: &str) -> GpuCategory {
        // Implementation required
    }
    
    /// Determine primary GPU model from validation results
    pub fn determine_primary_gpu_model(
        executor_validations: &[crate::bittensor_core::weight_setter::ExecutorValidationResult]
    ) -> String {
        // Implementation required
    }
    
    /// Calculate GPU model distribution for a miner
    pub fn calculate_gpu_distribution(
        executor_validations: &[crate::bittensor_core::weight_setter::ExecutorValidationResult]
    ) -> HashMap<String, u32> {
        // Implementation required
    }
}
```

// Add tests at the bottom of categorization.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_model_normalization() {
        // Test H100 variants
        assert_eq!(GpuCategorizer::normalize_gpu_model("NVIDIA H100 PCIe"), "H100");
        assert_eq!(GpuCategorizer::normalize_gpu_model("H100 SXM5"), "H100");
        assert_eq!(GpuCategorizer::normalize_gpu_model("h100"), "H100");
        
        // Test H200 variants
        assert_eq!(GpuCategorizer::normalize_gpu_model("NVIDIA H200"), "H200");
        assert_eq!(GpuCategorizer::normalize_gpu_model("H200 SXM"), "H200");
        
        // Test A100 variants
        assert_eq!(GpuCategorizer::normalize_gpu_model("A100 80GB"), "A100");
        assert_eq!(GpuCategorizer::normalize_gpu_model("Tesla A100"), "A100");
        
        // Test RTX variants
        assert_eq!(GpuCategorizer::normalize_gpu_model("GeForce RTX 4090"), "RTX4090");
        assert_eq!(GpuCategorizer::normalize_gpu_model("RTX 3090 Ti"), "RTX3090");
        
        // Test unknown models
        assert_eq!(GpuCategorizer::normalize_gpu_model("Unknown GPU"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model(""), "OTHER");
    }

    #[test]
    fn test_model_to_category_conversion() {
        // Test all known categories
        // Test case sensitivity
        // Test unknown models
    }

    #[test]
    fn test_primary_gpu_determination() {
        // Test single GPU type
        // Test multiple GPU types (should pick most common)
        // Test tie scenarios
        // Test empty validation results
        // Test all invalid validations
    }

    #[test]
    fn test_gpu_distribution_calculation() {
        // Test single GPU model
        // Test multiple GPU models
        // Test mixed valid/invalid validations
        // Test zero GPU counts
    }

    #[test]
    fn test_miner_gpu_profile_creation() {
        // Test complete profile creation
        // Test profile updates
        // Test timestamp handling
    }

    #[test]
    fn test_edge_cases() {
        // Test unicode GPU names
        // Test very long GPU names
        // Test special characters
        // Test null/empty strings
    }

    #[test]
    fn test_gpu_category_enum() {
        // Test enum variants
        // Test serialization
        // Test comparison
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 7 test functions with comprehensive GPU model coverage
- ✅ Edge cases for malformed/unknown GPU names
- ✅ Primary GPU determination logic verification
- ✅ Distribution calculation accuracy
- ✅ Profile creation and update scenarios

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Support for at least 10 different GPU model variants
- [ ] Consistent normalization across different naming conventions
- [ ] Performance benchmarks for large validation sets

---

### Task 1.3: Miner Prover Integration Enhancement
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/miner_prover/verification.rs`** (Modifications)
```rust
// Add GPU profile collection to verification engine
impl VerificationEngine {
    /// Verify executor with GPU profile collection
    pub async fn verify_executor_with_gpu_profile(
        &self,
        miner_info: &MinerInfo,
        executor_info: &ExecutorInfo,
    ) -> Result<VerificationResultWithGpu> {
        // Implementation required
    }
    
    /// Extract GPU information from attestation
    fn extract_gpu_info_from_attestation(
        &self,
        attestation: &AttestationResult,
    ) -> GpuExecutorInfo {
        // Implementation required
    }
}

#[derive(Debug, Clone)]
pub struct VerificationResultWithGpu {
    pub verification_result: VerificationResult,
    pub gpu_info: GpuExecutorInfo,
}

#[derive(Debug, Clone)]
pub struct GpuExecutorInfo {
    pub gpu_model: String,
    pub gpu_count: u32,
    pub gpu_memory_gb: u64,
    pub attestation_valid: bool,
}
```

**File: `crates/validator/src/miner_prover/scheduler.rs`** (Modifications)
```rust
// Add GPU profile tracking to scheduler
impl VerificationScheduler {
    /// Process verification results and update GPU profiles
    async fn process_verification_with_gpu_profile(
        &self,
        miner_uid: MinerUid,
        verification_results: Vec<VerificationResultWithGpu>,
    ) -> Result<()> {
        // Implementation required
    }
}
```

**File: `crates/integration-tests/tests/miner_prover_gpu_integration.rs`** (New File)
```rust
use integration_tests::*;

#[tokio::test]
async fn test_gpu_info_extraction_from_verification() {
    // Test GPU info extraction from attestation
    // Test multiple executor verification
    // Test invalid attestation handling
}

#[tokio::test]
async fn test_verification_result_with_gpu_profile() {
    // Test successful verification with GPU info
    // Test failed verification handling
    // Test missing GPU info scenarios
}

#[tokio::test]
async fn test_scheduler_gpu_profile_updates() {
    // Test profile updates from verification results
    // Test aggregation of multiple executor results
    // Test error handling in profile updates
}

#[tokio::test]
async fn test_concurrent_verification_with_gpu_tracking() {
    // Test concurrent verification of multiple miners
    // Test GPU profile consistency
    // Test resource usage
}

#[tokio::test]
async fn test_miner_prover_weight_setter_integration() {
    // Test data flow from prover to weight setter
    // Test GPU profile propagation
    // Test timing synchronization
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 5 test functions for miner prover integration
- ✅ GPU information extraction accuracy
- ✅ Concurrent operation safety
- ✅ Integration with existing verification flow

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Seamless integration with existing miner prover
- [ ] GPU data correctly propagated to weight setter
- [ ] Performance impact minimal

---

### Task 1.4: Integration with WeightSetter
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/bittensor_core/weight_setter.rs`** (Modifications)
```rust
// Add to existing WeightSetter struct
pub struct WeightSetter {
    // ... existing fields ...
    emission_config: EmissionConfig,
    gpu_scoring_engine: Arc<GpuScoringEngine>,
    miner_prover_integration: Arc<MinerProverIntegration>,
}

impl WeightSetter {
    /// Create WeightSetter with emission configuration
    pub fn new(
        config: BittensorConfig,
        bittensor_service: Arc<BittensorService>,
        storage: MemoryStorage,
        persistence: Arc<SimplePersistence>,
        min_score_threshold: f64,
        emission_config: EmissionConfig, // New parameter
    ) -> Result<Self> {
        // Implementation required
    }
    
    /// Update emission configuration at runtime
    pub async fn update_emission_config(&mut self, new_config: EmissionConfig) -> Result<()> {
        // Implementation required
    }
    
    /// Get current emission configuration
    pub fn get_emission_config(&self) -> &EmissionConfig {
        &self.emission_config
    }
    
    /// Process miner prover verification results
    pub async fn process_miner_prover_results(
        &self,
        miner_uid: MinerUid,
        verification_results: Vec<VerificationResultWithGpu>,
    ) -> Result<()> {
        // Implementation required
    }
}
```

**File: `crates/validator/src/bittensor_core/miner_prover_integration.rs`** (New File)
```rust
use crate::miner_prover::{MinerProver, VerificationResultWithGpu};
use crate::gpu::categorization::GpuCategorizer;
use common::identity::MinerUid;
use anyhow::Result;

/// Integrates miner prover results with weight setter
pub struct MinerProverIntegration {
    gpu_categorizer: GpuCategorizer,
}

impl MinerProverIntegration {
    pub fn new() -> Self {
        Self {
            gpu_categorizer: GpuCategorizer,
        }
    }
    
    /// Convert miner prover results to weight setter format
    pub fn convert_verification_results(
        &self,
        verification_results: Vec<VerificationResultWithGpu>,
    ) -> Vec<ExecutorValidationResult> {
        // Implementation required
    }
    
    /// Aggregate GPU information from multiple executors
    pub fn aggregate_gpu_info(
        &self,
        verification_results: &[VerificationResultWithGpu],
    ) -> HashMap<String, u32> {
        // Implementation required
    }
}
```

**File: `crates/integration-tests/tests/weight_setter_integration.rs`** (New File)
```rust
use integration_tests::*;

#[tokio::test]
async fn test_weight_setter_creation_with_emission_config() {
    // Test valid config initialization
    // Test invalid config rejection
}

#[tokio::test]
async fn test_emission_config_update() {
    // Test valid config updates
    // Test invalid config rejection
    // Test concurrent update handling
}

#[tokio::test]
async fn test_config_persistence() {
    // Test config survives restarts
    // Test config loading from storage
}

#[tokio::test]
async fn test_config_validation_integration() {
    // Test integration with validation system
    // Test error propagation
}

#[tokio::test]
async fn test_backward_compatibility() {
    // Test existing functionality still works
    // Test default config behavior
}

#[tokio::test]
async fn test_miner_prover_result_processing() {
    // Test conversion of prover results
    // Test GPU info aggregation
    // Test score updates
}

#[tokio::test]
async fn test_end_to_end_prover_to_weight_flow() {
    // Test complete flow from verification to weight setting
    // Test timing and synchronization
    // Test error handling
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 7 test functions for integration scenarios
- ✅ Configuration validation in context
- ✅ Backward compatibility verification
- ✅ Error handling and recovery
- ✅ Miner prover integration verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Existing WeightSetter functionality unaffected
- [ ] Clean integration with configuration system
- [ ] Miner prover results properly integrated
- [ ] Memory usage within acceptable limits

---

## Phase 2: Storage and Persistence (Days 6-8)

### Task 2.1: Database Schema and Migration
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/migrations/20241208_001_add_gpu_profiles.sql`** (New File)
```sql
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
```

**File: `crates/validator/migrations/20241208_002_add_weight_allocation_history.sql`** (New File)
```sql
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
```

**File: `crates/integration-tests/tests/database_migration_integration.rs`** (New File)
```rust
use integration_tests::*;
use sqlx::sqlite::SqlitePool;
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_gpu_profiles_table_creation() {
    // Test table creation succeeds
    // Test constraints are enforced
    // Test indexes are created
}

#[tokio::test]
async fn test_emission_metrics_table_creation() {
    // Test table creation succeeds
    // Test constraints are enforced
    // Test foreign keys work
}

#[tokio::test]
async fn test_miner_prover_results_table_creation() {
    // Test table creation succeeds
    // Test constraints are enforced
    // Test indexes are created
}

#[tokio::test]
async fn test_constraint_validation() {
    // Test score constraints (0.0-1.0)
    // Test burn percentage constraints (0.0-100.0)
    // Test non-negative integer constraints
}

#[tokio::test]
async fn test_index_performance() {
    // Test query performance with indexes
    // Test index usage in query plans
}

#[tokio::test]
async fn test_migration_rollback() {
    // Test migration can be rolled back
    // Test data integrity after rollback
}

#[tokio::test]
async fn test_migration_idempotency() {
    // Test migrations can be run multiple times
    // Test no data loss on re-run
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 7 test functions for database operations
- ✅ Constraint validation testing
- ✅ Performance benchmarking
- ✅ Migration safety verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Migration scripts run successfully
- [ ] Performance meets requirements (< 100ms queries)
- [ ] Rollback procedures tested and documented

---

### Task 2.2: GPU Profile Storage Implementation
**Duration:** 1.5 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/persistence/gpu_profile_repository.rs`** (New File)
```rust
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::SqlitePool;
use std::collections::HashMap;
use common::identity::MinerUid;
use crate::gpu::categorization::MinerGpuProfile;

#[async_trait]
pub trait GpuProfileRepository: Send + Sync {
    async fn store_profile(&self, profile: &MinerGpuProfile) -> Result<()>;
    async fn get_profile(&self, miner_uid: MinerUid) -> Result<Option<MinerGpuProfile>>;
    async fn get_all_profiles(&self, cutoff_hours: Option<u32>) -> Result<Vec<MinerGpuProfile>>;
    async fn get_profiles_by_gpu_model(&self, gpu_model: &str) -> Result<Vec<MinerGpuProfile>>;
    async fn delete_profile(&self, miner_uid: MinerUid) -> Result<()>;
    async fn cleanup_old_profiles(&self, older_than_hours: u32) -> Result<u64>;
    async fn get_profile_statistics(&self) -> Result<GpuProfileStatistics>;
    
    // New methods for miner prover integration
    async fn store_prover_results(&self, results: &[MinerProverResult]) -> Result<()>;
    async fn get_recent_prover_results(&self, miner_uid: MinerUid, hours: u32) -> Result<Vec<MinerProverResult>>;
}

pub struct SqliteGpuProfileRepository {
    pool: SqlitePool,
}

impl SqliteGpuProfileRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl GpuProfileRepository for SqliteGpuProfileRepository {
    async fn store_profile(&self, profile: &MinerGpuProfile) -> Result<()> {
        // Implementation required
    }
    
    async fn get_profile(&self, miner_uid: MinerUid) -> Result<Option<MinerGpuProfile>> {
        // Implementation required
    }
    
    async fn get_all_profiles(&self, cutoff_hours: Option<u32>) -> Result<Vec<MinerGpuProfile>> {
        // Implementation required
    }
    
    async fn get_profiles_by_gpu_model(&self, gpu_model: &str) -> Result<Vec<MinerGpuProfile>> {
        // Implementation required
    }
    
    async fn delete_profile(&self, miner_uid: MinerUid) -> Result<()> {
        // Implementation required
    }
    
    async fn cleanup_old_profiles(&self, older_than_hours: u32) -> Result<u64> {
        // Implementation required
    }
    
    async fn get_profile_statistics(&self) -> Result<GpuProfileStatistics> {
        // Implementation required
    }
    
    async fn store_prover_results(&self, results: &[MinerProverResult]) -> Result<()> {
        // Implementation required
    }
    
    async fn get_recent_prover_results(&self, miner_uid: MinerUid, hours: u32) -> Result<Vec<MinerProverResult>> {
        // Implementation required
    }
}

#[derive(Debug, Clone)]
pub struct GpuProfileStatistics {
    pub total_profiles: u64,
    pub gpu_model_distribution: HashMap<String, u64>,
    pub average_score_by_model: HashMap<String, f64>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MinerProverResult {
    pub miner_uid: MinerUid,
    pub executor_id: String,
    pub gpu_model: String,
    pub gpu_count: u32,
    pub gpu_memory_gb: u64,
    pub attestation_valid: bool,
    pub verification_timestamp: DateTime<Utc>,
}
```

// Add tests at the bottom of gpu_profile_repository.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve_profile() {
        // Test profile storage and retrieval
        // Test profile updates (upsert behavior)
        // Test JSON serialization/deserialization
    }

    #[tokio::test]
    async fn test_get_all_profiles() {
        // Test retrieving all profiles
        // Test cutoff time filtering
        // Test empty result handling
    }

    #[tokio::test]
    async fn test_profiles_by_gpu_model() {
        // Test filtering by GPU model
        // Test case sensitivity
        // Test non-existent models
    }

    #[tokio::test]
    async fn test_profile_deletion() {
        // Test successful deletion
        // Test deletion of non-existent profile
        // Test cascade effects
    }

    #[tokio::test]
    async fn test_cleanup_old_profiles() {
        // Test old profile cleanup
        // Test cleanup with no old profiles
        // Test return count accuracy
    }

    #[tokio::test]
    async fn test_profile_statistics() {
        // Test statistics calculation
        // Test with empty database
        // Test with various GPU models
    }

    #[tokio::test]
    async fn test_miner_prover_results_storage() {
        // Test storing prover results
        // Test retrieving recent results
        // Test time-based filtering
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        // Test concurrent profile updates
        // Test transaction handling
        // Test data consistency
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test database connection errors
        // Test invalid data handling
        // Test constraint violations
    }

    #[tokio::test]
    async fn test_performance() {
        // Test large dataset operations
        // Test query performance
        // Test batch operations
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 10 test functions covering all repository operations
- ✅ Concurrent operation testing
- ✅ Performance benchmarking with large datasets
- ✅ Error handling and recovery scenarios
- ✅ Miner prover integration testing

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Performance requirements met (< 50ms for single operations)
- [ ] Concurrent access safely handled
- [ ] Data consistency maintained
- [ ] Miner prover results properly stored

---

### Task 2.3: Emission Metrics Storage
**Duration:** 0.5 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/persistence/emission_metrics_repository.rs`** (New File)
```rust
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::SqlitePool;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EmissionMetrics {
    pub timestamp: DateTime<Utc>,
    pub burn_amount: u64,
    pub burn_percentage: f64,
    pub category_distributions: HashMap<String, CategoryAllocation>,
    pub total_miners: u32,
    pub weight_set_block: u64,
}

#[derive(Debug, Clone)]
pub struct CategoryAllocation {
    pub gpu_model: String,
    pub miner_count: u32,
    pub total_score: f64,
    pub weight_allocation: u64,
    pub allocation_percentage: f64,
}

#[async_trait]
pub trait EmissionMetricsRepository: Send + Sync {
    async fn store_metrics(&self, metrics: &EmissionMetrics) -> Result<u64>;
    async fn get_latest_metrics(&self) -> Result<Option<EmissionMetrics>>;
    async fn get_metrics_by_block_range(&self, start_block: u64, end_block: u64) -> Result<Vec<EmissionMetrics>>;
    async fn get_metrics_history(&self, days: u32) -> Result<Vec<EmissionMetrics>>;
    async fn cleanup_old_metrics(&self, older_than_days: u32) -> Result<u64>;
}

pub struct SqliteEmissionMetricsRepository {
    pool: SqlitePool,
}

impl SqliteEmissionMetricsRepository {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl EmissionMetricsRepository for SqliteEmissionMetricsRepository {
    // Implementation required for all methods
}
```

// Add tests at the bottom of emission_metrics_repository.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve_metrics() {
        // Test metrics storage and retrieval
        // Test complex data structures
        // Test JSON serialization handling
    }

    #[tokio::test]
    async fn test_block_range_queries() {
        // Test block range filtering
        // Test edge cases (empty ranges, invalid ranges)
        // Test large ranges
    }

    #[tokio::test]
    async fn test_metrics_history() {
        // Test time-based filtering
        // Test empty history
        // Test large history sets
    }

    #[tokio::test]
    async fn test_cleanup_operations() {
        // Test old metrics cleanup
        // Test cleanup accuracy
        // Test cleanup with no old data
    }

    #[tokio::test]
    async fn test_data_integrity() {
        // Test foreign key relationships
        // Test transaction handling
        // Test rollback scenarios
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 5 test functions for metrics operations
- ✅ Complex data structure handling
- ✅ Time-based filtering accuracy
- ✅ Data integrity verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Metrics accurately stored and retrieved
- [ ] Performance acceptable for historical queries
- [ ] Data integrity maintained

---

### Task 2.4: Cleanup Task Implementation
**Duration:** 0.5 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/persistence/cleanup_task.rs`** (New File)
```rust
use anyhow::Result;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{info, error};
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use crate::persistence::emission_metrics_repository::EmissionMetricsRepository;

pub struct CleanupTask {
    gpu_profile_repo: Arc<dyn GpuProfileRepository>,
    emission_metrics_repo: Arc<dyn EmissionMetricsRepository>,
    retention_hours: u32,
    interval_hours: u32,
}

impl CleanupTask {
    pub fn new(
        gpu_profile_repo: Arc<dyn GpuProfileRepository>,
        emission_metrics_repo: Arc<dyn EmissionMetricsRepository>,
        retention_hours: u32,
        interval_hours: u32,
    ) -> Self {
        Self {
            gpu_profile_repo,
            emission_metrics_repo,
            retention_hours,
            interval_hours,
        }
    }
    
    /// Start the cleanup task
    pub async fn start(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(self.interval_hours as u64 * 3600));
        
        loop {
            interval.tick().await;
            if let Err(e) = self.run_cleanup().await {
                error!("Cleanup task failed: {}", e);
            }
        }
    }
    
    /// Run cleanup operations
    async fn run_cleanup(&self) -> Result<()> {
        info!("Running cleanup task");
        
        // Cleanup old GPU profiles
        let profiles_deleted = self.gpu_profile_repo
            .cleanup_old_profiles(self.retention_hours)
            .await?;
        info!("Deleted {} old GPU profiles", profiles_deleted);
        
        // Cleanup old emission metrics
        let metrics_deleted = self.emission_metrics_repo
            .cleanup_old_metrics(self.retention_hours / 24) // Convert to days
            .await?;
        info!("Deleted {} old emission metrics", metrics_deleted);
        
        Ok(())
    }
}
```

// Add tests at the bottom of cleanup_task.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_cleanup_task_initialization() {
        // Test task creation
        // Test parameter validation
    }

    #[tokio::test]
    async fn test_cleanup_execution() {
        // Test cleanup runs successfully
        // Test old data is removed
        // Test recent data is preserved
    }

    #[tokio::test]
    async fn test_cleanup_error_handling() {
        // Test repository error handling
        // Test task continues after errors
        // Test error logging
    }

    #[tokio::test]
    async fn test_cleanup_timing() {
        // Test interval timing
        // Test first run timing
        // Test concurrent cleanup prevention
    }

    #[tokio::test]
    async fn test_cleanup_performance() {
        // Test cleanup with large datasets
        // Test memory usage during cleanup
        // Test database locking behavior
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 5 test functions for cleanup operations
- ✅ Error recovery testing
- ✅ Performance with large datasets
- ✅ Timing accuracy verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Cleanup runs reliably on schedule
- [ ] Performance impact minimal
- [ ] Error handling robust

---

## Phase 3: Weight Calculation Logic (Days 9-11)

### Task 3.1: Enhanced Miner Scoring System
**Duration:** 2 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/bittensor_core/gpu_scoring.rs`** (New File)
```rust
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use common::identity::MinerUid;
use crate::gpu::categorization::{GpuCategorizer, MinerGpuProfile};
use crate::persistence::gpu_profile_repository::{GpuProfileRepository, MinerProverResult};
use crate::miner_prover::VerificationResultWithGpu;

pub struct GpuScoringEngine {
    gpu_profile_repo: Box<dyn GpuProfileRepository>,
    ema_alpha: f64, // Exponential moving average factor
}

impl GpuScoringEngine {
    pub fn new(
        gpu_profile_repo: Box<dyn GpuProfileRepository>,
        ema_alpha: f64,
    ) -> Self {
        Self {
            gpu_profile_repo,
            ema_alpha,
        }
    }
    
    /// Update miner profile from miner prover validation results
    pub async fn update_miner_profile_from_prover_results(
        &self,
        miner_uid: MinerUid,
        prover_results: Vec<VerificationResultWithGpu>,
    ) -> Result<MinerGpuProfile> {
        // Convert prover results to repository format
        let miner_prover_results: Vec<MinerProverResult> = prover_results
            .iter()
            .map(|r| MinerProverResult {
                miner_uid,
                executor_id: r.verification_result.executor_id.clone(),
                gpu_model: r.gpu_info.gpu_model.clone(),
                gpu_count: r.gpu_info.gpu_count,
                gpu_memory_gb: r.gpu_info.gpu_memory_gb,
                attestation_valid: r.gpu_info.attestation_valid,
                verification_timestamp: r.verification_result.timestamp,
            })
            .collect();
        
        // Store prover results
        self.gpu_profile_repo.store_prover_results(&miner_prover_results).await?;
        
        // Calculate score and update profile
        self.update_miner_profile_from_validation(miner_uid, prover_results).await
    }
    
    /// Update miner profile from validation results
    async fn update_miner_profile_from_validation(
        &self,
        miner_uid: MinerUid,
        prover_results: Vec<VerificationResultWithGpu>,
    ) -> Result<MinerGpuProfile> {
        // Implementation required
    }
    
    /// Calculate verification score from prover results
    fn calculate_verification_score(
        &self,
        prover_results: &[VerificationResultWithGpu],
    ) -> f64 {
        // Implementation required
    }
    
    /// Apply EMA smoothing to score
    async fn apply_ema_smoothing(
        &self,
        miner_uid: MinerUid,
        gpu_model: &str,
        new_score: f64,
    ) -> Result<f64> {
        // Implementation required
    }
    
    /// Get all miners grouped by GPU category
    pub async fn get_miners_by_gpu_category(
        &self,
        cutoff_hours: u32,
    ) -> Result<HashMap<String, Vec<(MinerUid, f64)>>> {
        // Implementation required
    }
    
    /// Get category statistics
    pub async fn get_category_statistics(&self) -> Result<HashMap<String, CategoryStats>> {
        // Implementation required
    }
}

#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub miner_count: u32,
    pub average_score: f64,
    pub total_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutorValidationResult {
    pub executor_id: String,
    pub is_valid: bool,
    pub hardware_score: f64,
    pub gpu_count: usize,
    pub gpu_model: String,
    pub gpu_memory_gb: u64,
    pub network_bandwidth_mbps: f64,
    pub attestation_valid: bool,
    pub validation_timestamp: DateTime<Utc>,
}
```

// Add tests at the bottom of gpu_scoring.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verification_score_calculation() {
        // Test with valid attestations
        // Test with invalid attestations
        // Test with mixed results
        // Test with empty validations
        // Test hardware score weighting
    }

    #[tokio::test]
    async fn test_miner_profile_update_from_prover() {
        // Test new profile creation from prover results
        // Test existing profile update
        // Test GPU model changes
        // Test score history tracking
        // Test prover result storage
    }

    #[tokio::test]
    async fn test_ema_smoothing() {
        // Test initial score (no previous)
        // Test score smoothing with history
        // Test different alpha values
        // Test edge cases (zero scores)
    }

    #[tokio::test]
    async fn test_miners_by_gpu_category() {
        // Test category grouping
        // Test score sorting within categories
        // Test cutoff time filtering
        // Test empty categories
    }

    #[tokio::test]
    async fn test_category_statistics() {
        // Test statistics calculation
        // Test with various score distributions
        // Test with single miners
        // Test with empty categories
    }

    #[tokio::test]
    async fn test_gpu_model_transitions() {
        // Test miner changing GPU models
        // Test score preservation/reset
        // Test profile updates
    }

    #[tokio::test]
    async fn test_concurrent_updates() {
        // Test concurrent profile updates
        // Test score consistency
        // Test data race prevention
    }

    #[tokio::test]
    async fn test_prover_integration() {
        // Test prover result conversion
        // Test score calculation from prover data
        // Test attestation handling
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test repository errors
        // Test invalid validation data
        // Test network failures
    }

    #[tokio::test]
    async fn test_performance() {
        // Test with large validation sets
        // Test batch processing
        // Test memory usage
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 10 test functions covering scoring logic
- ✅ EMA smoothing mathematical verification
- ✅ Concurrent operation safety
- ✅ Performance with large datasets
- ✅ Miner prover integration verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Scoring logic mathematically verified
- [ ] Performance requirements met
- [ ] Thread safety ensured
- [ ] Miner prover results properly integrated

---

### Task 3.2: Burn and GPU Allocation Algorithm
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/bittensor_core/weight_allocation.rs`** (New File)
```rust
use anyhow::Result;
use std::collections::HashMap;
use bittensor::NormalizedWeight;
use common::identity::MinerUid;
use crate::config::emission::EmissionConfig;

pub struct WeightAllocationEngine {
    emission_config: EmissionConfig,
    min_score_threshold: f64,
}

impl WeightAllocationEngine {
    pub fn new(emission_config: EmissionConfig, min_score_threshold: f64) -> Self {
        Self {
            emission_config,
            min_score_threshold,
        }
    }
    
    /// Calculate weight distribution with burn and GPU allocation
    pub fn calculate_weight_distribution(
        &self,
        miners_by_category: HashMap<String, Vec<(MinerUid, f64)>>,
    ) -> Result<WeightDistribution> {
        // Implementation required
    }
    
    /// Calculate burn allocation
    fn calculate_burn_allocation(&self) -> Option<NormalizedWeight> {
        // Implementation required
    }
    
    /// Calculate category weight pools
    fn calculate_category_pools(&self, total_remaining_weight: u64) -> HashMap<String, u64> {
        // Implementation required
    }
    
    /// Distribute weight within a category
    fn distribute_category_weight(
        &self,
        category_miners: &[(MinerUid, f64)],
        category_weight_pool: u64,
    ) -> Vec<NormalizedWeight> {
        // Implementation required
    }
    
    /// Validate allocation results
    fn validate_allocation(&self, weights: &[NormalizedWeight]) -> Result<()> {
        // Implementation required
    }
}

#[derive(Debug, Clone)]
pub struct WeightDistribution {
    pub weights: Vec<NormalizedWeight>,
    pub burn_allocation: Option<BurnAllocation>,
    pub category_allocations: HashMap<String, CategoryAllocation>,
    pub total_weight: u64,
    pub miners_served: u32,
}

#[derive(Debug, Clone)]
pub struct BurnAllocation {
    pub uid: u16,
    pub weight: u16,
    pub percentage: f64,
}

#[derive(Debug, Clone)]
pub struct CategoryAllocation {
    pub gpu_model: String,
    pub miner_count: u32,
    pub total_score: f64,
    pub weight_pool: u64,
    pub allocation_percentage: f64,
}
```

// Add tests at the bottom of weight_allocation.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burn_allocation_calculation() {
        // Test various burn percentages (0%, 50%, 100%)
        // Test burn UID assignment
        // Test burn weight calculation
        // Test edge cases (burn > 100%)
    }

    #[test]
    fn test_category_pool_calculation() {
        // Test allocation percentage distribution
        // Test with different category counts
        // Test rounding behavior
        // Test edge cases (100% allocation)
    }

    #[test]
    fn test_within_category_distribution() {
        // Test proportional distribution by score
        // Test with equal scores
        // Test with single miner
        // Test with zero scores
        // Test weight truncation handling
    }

    #[test]
    fn test_complete_weight_distribution() {
        // Test full allocation flow
        // Test with burn enabled/disabled
        // Test with multiple categories
        // Test weight conservation
    }

    #[test]
    fn test_minimum_score_filtering() {
        // Test threshold enforcement
        // Test mixed score scenarios
        // Test all miners below threshold
    }

    #[test]
    fn test_allocation_validation() {
        // Test weight sum validation
        // Test UID uniqueness
        // Test weight bounds (0-65535)
    }

    #[test]
    fn test_edge_cases() {
        // Test empty miner categories
        // Test insufficient miners per category
        // Test allocation percentage errors
        // Test integer overflow prevention
    }

    #[test]
    fn test_mathematical_accuracy() {
        // Test proportional distribution accuracy
        // Test rounding error accumulation
        // Test large number handling
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 8 test functions for allocation algorithm
- ✅ Mathematical accuracy verification
- ✅ Edge case handling
- ✅ Weight conservation validation

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Mathematical accuracy verified
- [ ] Weight conservation guaranteed
- [ ] Performance acceptable for large datasets

---

## Phase 4: Integration and Enhanced WeightSetter (Days 12-13)

### Task 4.1: WeightSetter Integration
**Duration:** 1.5 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/bittensor_core/weight_setter.rs`** (Major Modifications)
```rust
// Enhanced WeightSetter with GPU allocation
impl WeightSetter {
    /// Set weights with burn and GPU-based allocation
    async fn set_weights_with_burn_and_gpu_allocation(&self) -> Result<()> {
        // Implementation required - integrate all components
    }
    
    /// Process miner prover validation result and update miner profile
    pub async fn process_prover_validation_with_gpu_profile(
        &self,
        miner_uid: MinerUid,
        prover_results: Vec<VerificationResultWithGpu>,
    ) -> Result<()> {
        // Implementation required
    }
    
    /// Update emission configuration
    pub async fn update_emission_config(&mut self, new_config: EmissionConfig) -> Result<()> {
        // Implementation required
    }
    
    /// Get allocation statistics
    pub async fn get_allocation_statistics(&self) -> Result<AllocationStatistics> {
        // Implementation required
    }
}

#[derive(Debug, Clone)]
pub struct AllocationStatistics {
    pub last_weight_set_block: u64,
    pub burn_statistics: Option<BurnStatistics>,
    pub category_statistics: HashMap<String, CategoryStatistics>,
    pub total_miners_served: u32,
    pub next_weight_set_block: u64,
}

#[derive(Debug, Clone)]
pub struct BurnStatistics {
    pub amount_burned: u64,
    pub percentage: f64,
    pub burn_uid: u16,
}

#[derive(Debug, Clone)]
pub struct CategoryStatistics {
    pub gpu_model: String,
    pub miners_count: u32,
    pub total_allocation: u64,
    pub average_score: f64,
    pub allocation_percentage: f64,
}
```

**File: `crates/integration-tests/tests/gpu_allocation_integration.rs`** (New File)
```rust
use integration_tests::*;

#[tokio::test]
async fn test_full_weight_setting_flow() {
    // Test complete end-to-end flow
    // Test prover -> profile update -> weight setting
    // Test multiple validation cycles
}

#[tokio::test]
async fn test_gpu_allocation_integration() {
    // Test GPU categorization integration
    // Test score calculation integration
    // Test weight allocation integration
}

#[tokio::test]
async fn test_burn_mechanism_integration() {
    // Test burn allocation with real weights
    // Test burn percentage changes
    // Test burn UID validation
}

#[tokio::test]
async fn test_configuration_updates() {
    // Test runtime configuration updates
    // Test configuration validation
    // Test invalid configuration handling
}

#[tokio::test]
async fn test_persistence_integration() {
    // Test database storage integration
    // Test data retrieval accuracy
    // Test transaction handling
}

#[tokio::test]
async fn test_miner_prover_integration() {
    // Test prover result processing
    // Test profile updates from prover data
    // Test scoring with prover results
}

#[tokio::test]
async fn test_error_handling() {
    // Test database connection failures
    // Test network failures
    // Test recovery mechanisms
}

#[tokio::test]
async fn test_performance_integration() {
    // Test with large miner populations
    // Test weight setting performance
    // Test memory usage
}

#[tokio::test]
async fn test_backward_compatibility() {
    // Test existing functionality preserved
    // Test migration from old system
    // Test configuration defaults
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 9 integration test functions
- ✅ End-to-end workflow verification
- ✅ Performance benchmarking
- ✅ Backward compatibility assurance
- ✅ Miner prover integration verification

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Full integration working correctly
- [ ] Performance requirements met
- [ ] Existing functionality preserved
- [ ] Miner prover data flow verified

---

### Task 4.2: Monitoring and Metrics
**Duration:** 0.5 days  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/src/metrics/emission_metrics.rs`** (New File)
```rust
use prometheus::{Counter, Gauge, Histogram, IntGauge, Registry};
use std::collections::HashMap;

pub struct EmissionMetrics {
    // Burn metrics
    pub burn_amount_total: Counter,
    pub burn_percentage_gauge: Gauge,
    
    // GPU allocation metrics
    pub gpu_category_miners: IntGauge,
    pub gpu_category_allocation: Gauge,
    pub gpu_category_scores: Histogram,
    
    // Weight setting metrics
    pub weight_setting_duration: Histogram,
    pub weight_setting_errors: Counter,
    pub miners_served_total: IntGauge,
    
    // Miner prover metrics
    pub prover_verifications_total: Counter,
    pub prover_verification_failures: Counter,
    pub prover_gpu_models_discovered: IntGauge,
    
    // Category-specific metrics
    category_metrics: HashMap<String, CategoryMetrics>,
}

struct CategoryMetrics {
    miner_count: IntGauge,
    total_score: Gauge,
    allocation_percentage: Gauge,
    average_score: Gauge,
}

impl EmissionMetrics {
    pub fn new(registry: &Registry) -> Self {
        // Implementation required
    }
    
    pub fn record_weight_setting(&self, distribution: &WeightDistribution) {
        // Implementation required
    }
    
    pub fn record_gpu_category_update(&self, gpu_model: &str, stats: &CategoryStats) {
        // Implementation required
    }
    
    pub fn record_burn_allocation(&self, burn_allocation: &Option<BurnAllocation>) {
        // Implementation required
    }
    
    pub fn record_prover_verification(&self, miner_uid: MinerUid, success: bool, gpu_model: &str) {
        // Implementation required
    }
}
```

// Add tests at the bottom of emission_metrics.rs file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        // Test metrics creation
        // Test registry integration
        // Test metric naming
    }

    #[test]
    fn test_weight_setting_recording() {
        // Test distribution metrics recording
        // Test metric value accuracy
        // Test multiple recordings
    }

    #[test]
    fn test_gpu_category_recording() {
        // Test category metrics recording
        // Test dynamic category creation
        // Test metric updates
    }

    #[test]
    fn test_burn_allocation_recording() {
        // Test burn metrics recording
        // Test with/without burn allocation
        // Test metric accuracy
    }

    #[test]
    fn test_prover_metrics_recording() {
        // Test prover verification metrics
        // Test success/failure tracking
        // Test GPU model discovery
    }
}
```

#### Unit Test Requirements (Must Pass 100%)
- ✅ 5 test functions for metrics system
- ✅ Metric accuracy verification
- ✅ Integration with Prometheus
- ✅ Performance impact assessment
- ✅ Miner prover metrics tracking

#### Completion Criteria
- [ ] All unit tests pass with 100% coverage
- [ ] Metrics accurately reflect system state
- [ ] Performance impact minimal
- [ ] Integration with existing metrics
- [ ] Miner prover activity tracked

---

## Phase 5: Testing and Deployment (Days 14-15)

### Task 5.1: End-to-End Integration Testing
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/integration-tests/tests/e2e_emission_allocation.rs`** (New File)
```rust
use integration_tests::*;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_complete_emission_allocation_flow() {
    // Setup: Create test environment with validators, miners, executors
    // Step 1: Start miner prover to discover miners
    // Step 2: Submit validation results for different GPU models
    // Step 3: Wait for profile updates
    // Step 4: Trigger weight setting
    // Step 5: Verify weight distribution matches configuration
    // Step 6: Verify burn allocation
    // Step 7: Verify category allocations
    // Step 8: Verify persistence
}

#[tokio::test]
async fn test_miner_prover_to_weight_setter_flow() {
    // Setup: Create test environment
    // Step 1: Configure miner prover
    // Step 2: Run miner discovery and verification
    // Step 3: Verify GPU profiles created
    // Step 4: Verify scores calculated correctly
    // Step 5: Verify weight allocation based on prover data
}

#[tokio::test]
async fn test_configuration_changes_during_operation() {
    // Setup: Running system with initial configuration
    // Step 1: Start miner prover with initial config
    // Step 2: Submit validations with initial config
    // Step 3: Change emission configuration
    // Step 4: Submit more validations
    // Step 5: Verify configuration changes take effect
    // Step 6: Verify no data corruption
}

#[tokio::test]
async fn test_gpu_model_transitions() {
    // Setup: Miners with specific GPU models
    // Step 1: Prover detects H100 miners
    // Step 2: Submit validations for H100 miners
    // Step 3: Same miners switch to H200
    // Step 4: Prover detects GPU change
    // Step 5: Submit validations for H200
    // Step 6: Verify profile updates
    // Step 7: Verify score preservation/reset logic
}

#[tokio::test]
async fn test_burn_percentage_changes() {
    // Test changing burn percentage from 0% to 50% to 100%
    // Verify weight distribution adjusts correctly
    // Verify burn UID receives correct allocation
}

#[tokio::test]
async fn test_large_scale_operations() {
    // Test with 1000+ miners across multiple GPU categories
    // Test miner prover performance with large scale
    // Verify performance remains acceptable
    // Verify memory usage within limits
    // Verify all miners receive appropriate allocations
}

#[tokio::test]
async fn test_error_recovery() {
    // Test miner prover connection failures
    // Test database connection failures
    // Test network interruptions
    // Test invalid configuration scenarios
    // Verify graceful recovery and retry logic
}

#[tokio::test]
async fn test_weight_setting_timing() {
    // Test 360-block interval timing
    // Test early/late weight setting prevention
    // Test block synchronization
    // Test miner prover scheduling alignment
}
```

**Update `crates/integration-tests/Cargo.toml`** to include new test files:
```toml
[[test]]
name = "miner_prover_gpu_integration"
path = "tests/miner_prover_gpu_integration.rs"

[[test]]
name = "weight_setter_integration"
path = "tests/weight_setter_integration.rs"

[[test]]
name = "database_migration_integration"
path = "tests/database_migration_integration.rs"

[[test]]
name = "gpu_allocation_integration"
path = "tests/gpu_allocation_integration.rs"

[[test]]
name = "e2e_emission_allocation"
path = "tests/e2e_emission_allocation.rs"
```

#### Test Requirements (Must Pass 100%)
- ✅ 8 comprehensive end-to-end tests
- ✅ Miner prover integration verification
- ✅ Large-scale performance verification
- ✅ Error recovery and resilience testing
- ✅ Real-world scenario simulation

#### Completion Criteria
- [ ] All e2e tests pass consistently
- [ ] Performance benchmarks met
- [ ] Error scenarios handled gracefully
- [ ] Real-world readiness verified
- [ ] Miner prover integration verified

---

### Task 5.2: Documentation and Deployment Preparation
**Duration:** 1 day  
**Status:** 🔴 Not Started

#### Code Deliverables

**File: `crates/validator/README_EMISSION_ALLOCATION.md`** (New File)
```markdown
# Emission Allocation System

## Overview
[Comprehensive documentation of the system]

## Architecture
### Miner Prover Integration
[Details on how miner prover feeds GPU data]

### GPU Scoring System
[Explanation of scoring methodology]

### Weight Allocation
[Details on burn and GPU-based allocation]

## Configuration
[Detailed configuration examples and explanations]

## Monitoring
[Metrics and monitoring guide]

## Troubleshooting
[Common issues and solutions]

## Migration Guide
[Step-by-step migration from existing system]
```

**File: `scripts/deploy_emission_allocation.sh`** (New File)
```bash
#!/bin/bash
# Deployment script with validation checks
# Database migration
# Configuration validation
# Service restart
# Health checks
```

**File: `scripts/rollback_emission_allocation.sh`** (New File)
```bash
#!/bin/bash
# Rollback script for emergency situations
# Database rollback
# Configuration restoration
# Service restart with old version
```

#### Deliverables
- [ ] Complete documentation
- [ ] Deployment automation
- [ ] Rollback procedures
- [ ] Health check scripts
- [ ] Configuration examples

#### Completion Criteria
- [ ] Documentation complete and reviewed
- [ ] Deployment scripts tested
- [ ] Rollback procedures verified
- [ ] All artifacts ready for production

---

## Testing Standards

### Code Coverage Requirements
- **Minimum Coverage:** 95% for all new code
- **Critical Path Coverage:** 100% for weight calculation and allocation logic
- **Edge Case Coverage:** 100% for all error conditions and boundary cases

### Test Categories Required

#### Unit Tests (Per Task)
- **Functional Tests:** Verify correct behavior under normal conditions
- **Edge Case Tests:** Verify behavior at boundaries and with invalid inputs
- **Error Handling Tests:** Verify graceful handling of all error conditions
- **Performance Tests:** Verify performance requirements are met
- **Concurrency Tests:** Verify thread safety and concurrent access handling

#### Integration Tests
- **Component Integration:** Verify components work together correctly
- **Database Integration:** Verify data persistence and retrieval accuracy
- **Configuration Integration:** Verify configuration changes propagate correctly
- **Miner Prover Integration:** Verify data flow from prover to weight setter

#### End-to-End Tests
- **Complete Workflow:** Verify entire system works from miner discovery to weight setting
- **Scale Testing:** Verify system handles expected load
- **Resilience Testing:** Verify system recovers from failures

### Performance Requirements
- **Weight Calculation:** < 30 seconds for 1000 miners
- **Database Operations:** < 50ms for single operations, < 5 seconds for batch operations
- **Memory Usage:** < 500MB additional memory usage
- **CPU Usage:** < 20% additional CPU usage during weight setting
- **Miner Prover:** < 10 minutes for full miner discovery cycle

### Quality Gates
Each task must pass all quality gates before being marked complete:

1. **Code Review:** All code reviewed by at least one other developer
2. **Unit Tests:** 100% of unit tests passing
3. **Integration Tests:** 100% of integration tests passing
4. **Performance Tests:** All performance requirements met
5. **Documentation:** Complete documentation for public APIs
6. **Security Review:** Security implications reviewed and addressed

## Risk Mitigation

### Rollback Strategy
- **Database Migrations:** All migrations must be reversible
- **Configuration Changes:** Old configuration format must remain supported
- **Code Deployment:** Feature flags to disable new functionality
- **Data Migration:** Backup and restore procedures tested

### Monitoring and Alerting
- **Real-time Metrics:** Monitor weight distribution, burn allocation, category performance
- **Error Tracking:** Alert on calculation failures, database errors, configuration issues
- **Performance Monitoring:** Track weight setting duration, memory usage, CPU usage
- **Business Metrics:** Monitor miner satisfaction, category distribution health
- **Miner Prover Health:** Track discovery success rate, verification failures

## Miner Prover Integration Points

### Data Flow
1. **Discovery:** Miner prover discovers active miners from metagraph
2. **Verification:** Prover verifies executors and collects GPU information
3. **Profile Update:** GPU profiles updated with verification results
4. **Scoring:** Scores calculated based on verification success and GPU capabilities
5. **Weight Setting:** Weights allocated based on GPU categories and scores

### Key Interfaces
- `VerificationResultWithGpu`: Extended verification result with GPU information
- `MinerProverIntegration`: Converts prover results to weight setter format
- `GpuScoringEngine`: Processes prover results into scores
- `GpuProfileRepository`: Stores prover verification results

This implementation plan ensures each component is thoroughly tested and validated before integration, maintaining high quality and system reliability throughout the development process.