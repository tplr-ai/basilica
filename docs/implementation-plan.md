# Implementation Plan: Burn Emissions and GPU-Based Allocation

## Overview

This document outlines the implementation plan for adding configurable burn emissions and GPU model-based allocation to the Basilica validator system.

## Timeline: 2-3 Weeks

### Phase 1: Core Infrastructure (Week 1)
**Goal**: Establish foundation for GPU categorization and burn mechanisms

#### Task 1.1: Configuration System (2 days)
- [ ] Create `EmissionConfig` struct with validation
- [ ] Add configuration loading from TOML files
- [ ] Implement runtime configuration updates
- [ ] Add configuration validation tests

**Files to modify:**
- `crates/validator/src/config.rs`
- `crates/validator/src/bittensor_core/weight_setter.rs`

**Deliverables:**
```rust
// New config structure
pub struct EmissionConfig {
    pub burn_percentage: f64,
    pub burn_uid: u16,
    pub gpu_allocations: HashMap<String, f64>,
    pub weight_set_interval_blocks: u64,
    pub min_miners_per_category: u32,
}

// Validation methods
impl EmissionConfig {
    pub fn validate(&self) -> Result<()>;
    pub fn from_toml(path: &Path) -> Result<Self>;
}
```

#### Task 1.2: GPU Model Categorization (3 days)
- [ ] Implement GPU model normalization
- [ ] Create `MinerGpuProfile` data structure
- [ ] Add primary GPU model determination logic
- [ ] Write tests for GPU categorization

**Files to modify:**
- `crates/validator/src/bittensor_core/weight_setter.rs`
- `crates/common/src/executor_identity/mod.rs` (if needed)

**Deliverables:**
```rust
// GPU categorization
pub fn normalize_gpu_model(gpu_model: &str) -> String;
pub fn determine_primary_gpu_model(validations: &[ExecutorValidationResult]) -> String;

// Miner profiles
pub struct MinerGpuProfile {
    pub miner_uid: MinerUid,
    pub primary_gpu_model: String,
    pub gpu_counts: HashMap<String, u32>,
    pub total_score: f64,
    pub last_updated: DateTime<Utc>,
}
```

### Phase 2: Storage and Persistence (Week 1-2)
**Goal**: Implement data persistence for GPU profiles and emission metrics

#### Task 2.1: Database Schema (1 day)
- [ ] Create migration scripts for new tables
- [ ] Add `miner_gpu_profiles` table
- [ ] Add `emission_metrics` table
- [ ] Create necessary indexes

**Files to create:**
- `crates/validator/migrations/add_gpu_profiles.sql`
- `crates/validator/migrations/add_emission_metrics.sql`

**Deliverables:**
```sql
-- New tables for GPU profiles and emission tracking
CREATE TABLE miner_gpu_profiles (...);
CREATE TABLE emission_metrics (...);
```

#### Task 2.2: Storage Implementation (2 days)
- [ ] Implement GPU profile storage/retrieval
- [ ] Add emission metrics persistence
- [ ] Create memory storage integration
- [ ] Write storage tests

**Files to modify:**
- `crates/validator/src/persistence/simple_persistence.rs`
- `crates/validator/src/bittensor_core/weight_setter.rs`

**Deliverables:**
```rust
// Storage methods
impl WeightSetter {
    async fn store_miner_gpu_profile(&self, profile: MinerGpuProfile) -> Result<()>;
    async fn get_all_miner_gpu_profiles(&self) -> Result<Vec<MinerGpuProfile>>;
    async fn store_emission_metrics(&self, metrics: EmissionMetrics) -> Result<()>;
}
```

### Phase 3: Weight Calculation Logic (Week 2)
**Goal**: Implement burn and GPU-based weight allocation

#### Task 3.1: Enhanced Scoring System (3 days)
- [ ] Modify verification result processing
- [ ] Implement category-specific score storage
- [ ] Preserve EMA smoothing within categories
- [ ] Add miner profile updates

**Files to modify:**
- `crates/validator/src/bittensor_core/weight_setter.rs`
- `crates/validator/src/miner_prover/verification.rs`

**Deliverables:**
```rust
// Enhanced scoring
impl WeightSetter {
    pub async fn update_miner_profile_from_validation(
        &self,
        miner_uid: MinerUid,
        executor_validations: Vec<ExecutorValidationResult>,
    ) -> Result<()>;
}
```

#### Task 3.2: Weight Distribution Algorithm (2 days)
- [ ] Implement burn weight calculation
- [ ] Create GPU category weight distribution
- [ ] Add proportional allocation within categories
- [ ] Implement safety checks and fallbacks

**Files to modify:**
- `crates/validator/src/bittensor_core/weight_setter.rs`

**Deliverables:**
```rust
// New weight setting method
impl WeightSetter {
    async fn set_weights_with_burn_and_gpu_allocation(&self) -> Result<()>;
}
```

### Phase 4: Integration and Testing (Week 2-3)
**Goal**: Integrate all components and ensure system stability

#### Task 4.1: System Integration (2 days)
- [ ] Update WeightSetter constructor
- [ ] Modify main weight setting loop
- [ ] Add configuration loading to startup
- [ ] Update verification result hooks

**Files to modify:**
- `crates/validator/src/main.rs`
- `crates/validator/src/bittensor_core/weight_setter.rs`
- `crates/validator/src/lib.rs`

#### Task 4.2: Monitoring and Metrics (2 days)
- [ ] Add emission allocation metrics
- [ ] Create category distribution logging
- [ ] Implement health checks
- [ ] Add performance monitoring

**Files to modify:**
- `crates/validator/src/metrics/business_metrics.rs`
- `crates/validator/src/bittensor_core/weight_setter.rs`

### Phase 5: Testing and Validation (Week 3)
**Goal**: Comprehensive testing and deployment preparation

#### Task 5.1: Unit Testing (2 days)
- [ ] GPU categorization tests
- [ ] Weight calculation tests
- [ ] Configuration validation tests
- [ ] Storage operation tests

**Files to create:**
- `crates/validator/src/bittensor_core/weight_setter_tests.rs`
- `crates/validator/tests/gpu_allocation_tests.rs`

#### Task 5.2: Integration Testing (2 days)
- [ ] End-to-end weight setting flow
- [ ] Configuration update scenarios
- [ ] Error handling and recovery
- [ ] Performance benchmarks

#### Task 5.3: Documentation and Deployment (1 day)
- [ ] Update configuration documentation
- [ ] Create deployment guide
- [ ] Prepare migration scripts
- [ ] Review and test deployment

## Implementation Details

### Key Files to Modify

1. **Core Weight Setting Logic**
   - `crates/validator/src/bittensor_core/weight_setter.rs`
   - Main implementation of burn and GPU allocation

2. **Configuration Management**
   - `crates/validator/src/config.rs`
   - `crates/validator/config.example.toml`

3. **Verification Integration**
   - `crates/validator/src/miner_prover/verification.rs`
   - Hook GPU profile updates to verification results

4. **Storage Layer**
   - `crates/validator/src/persistence/simple_persistence.rs`
   - Add GPU profile and metrics storage

5. **Main Application**
   - `crates/validator/src/main.rs`
   - Initialize new configuration and components

### Configuration Example

```toml
# config.toml additions

[emission]
# Burn 5% of emissions
burn_percentage = 5.0
burn_uid = 0

# Set weights every 360 blocks (~1.2 hours)
weight_set_interval_blocks = 360

# Require at least 1 miner per category
min_miners_per_category = 1

# GPU allocations (must sum to 100%)
[emission.gpu_allocations]
H100 = 40.0
H200 = 60.0
```

### Testing Strategy

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_model_normalization() {
        assert_eq!(normalize_gpu_model("NVIDIA H100 PCIe"), "H100");
        assert_eq!(normalize_gpu_model("H200 SXM"), "H200");
        assert_eq!(normalize_gpu_model("RTX 4090"), "RTX4090");
    }

    #[test]
    fn test_weight_calculation_with_burn() {
        // Test burn allocation and GPU distribution
    }

    #[test]
    fn test_config_validation() {
        // Test various configuration scenarios
    }
}
```

#### Integration Tests
```rust
#[tokio::test]
async fn test_end_to_end_weight_setting() {
    // Set up test environment
    // Submit verification results
    // Trigger weight setting
    // Verify correct allocation
}
```

### Migration Steps

1. **Database Migration**
   ```bash
   # Run migration scripts
   sqlx migrate run --database-url $DATABASE_URL
   ```

2. **Configuration Update**
   ```bash
   # Update config.toml with emission settings
   cp config.example.toml config.toml
   # Edit emission section
   ```

3. **Code Deployment**
   ```bash
   # Deploy new validator version
   cargo build --release
   # Restart validator with new config
   ```

### Risk Mitigation

#### Configuration Validation
- Strict validation prevents invalid configurations
- Fallback to default values for missing settings
- Runtime validation before applying changes

#### Gradual Rollout
- Start with 0% burn to test GPU allocation
- Gradually increase burn percentage
- Monitor weight distribution patterns

#### Monitoring
- Track category distribution metrics
- Alert on unexpected allocation patterns
- Log all configuration changes

#### Rollback Plan
- Keep previous weight setting logic as fallback
- Configuration flag to disable new features
- Database rollback scripts if needed

### Success Criteria

#### Functional Requirements
- [x] Configurable burn percentage (0-100%)
- [x] GPU-based allocation with configurable percentages
- [x] 360-block weight setting interval
- [x] Preserve existing verification quality
- [x] Maintain EMA smoothing stability

#### Performance Requirements
- Weight calculation completes within 30 seconds
- No degradation in verification throughput
- Database queries optimized for large miner populations

#### Quality Requirements
- 100% unit test coverage for new components
- Integration tests for all major workflows
- Configuration validation prevents invalid states

### Post-Deployment Monitoring

#### Metrics to Track
- Burn allocation accuracy
- GPU category distribution
- Miner satisfaction within categories
- Weight setting performance
- Configuration change frequency

#### Alerts
- Invalid configuration attempts
- Weight calculation failures
- Unusual allocation patterns
- Performance degradation

This implementation plan provides a structured approach to delivering the burn emissions and GPU-based allocation features while maintaining system stability and quality.