# Burn Emissions and GPU-Based Allocation Design

## Overview

This document describes the design for implementing configurable burn emissions and GPU model-based allocation in the Basilica validator system. The system will allow burning a percentage of emissions while distributing the remainder based on GPU model categories.

## Current System

The existing validator scoring system:
- Discovers executors through miners via gRPC
- Validates executors through SSH sessions with hardware attestation
- Calculates hardware scores based on GPU quality, CPU, memory, and network
- Aggregates scores per miner using exponential moving average (EMA)
- Sets weights on the Bittensor network every N blocks

## Proposed Changes

### 1. Burn Mechanism
- Configurable percentage of emissions to burn (0-100%)
- Designated burn UID to receive burn allocation
- Burn weight calculated as: `burn_percentage / 100 * total_weight`

### 2. GPU-Based Allocation
- Miners categorized by primary GPU model (H100, H200, etc.)
- Configurable allocation percentages per GPU category
- Remaining emissions (after burn) distributed proportionally within each category
- Default allocation: 40% to H100, 60% to H200

### 3. Enhanced Configuration
- 360-block interval for weight setting (configurable)
- Minimum miners per category requirement
- Validation that GPU allocations sum to 100%

## Architecture Design

### Configuration Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    
    /// Fallback allocation for unrecognized GPU models
    pub other_gpu_allocation: f64,
}

impl Default for EmissionConfig {
    fn default() -> Self {
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert("H100".to_string(), 40.0);
        gpu_allocations.insert("H200".to_string(), 60.0);
        
        Self {
            burn_percentage: 0.0,
            burn_uid: 0,
            gpu_allocations,
            weight_set_interval_blocks: 360,
            min_miners_per_category: 1,
            other_gpu_allocation: 0.0,
        }
    }
}
```

### GPU Model Categorization

```rust
#[derive(Debug, Clone)]
pub struct MinerGpuProfile {
    pub miner_uid: MinerUid,
    pub primary_gpu_model: String,
    pub gpu_counts: HashMap<String, u32>,
    pub total_score: f64,
    pub verification_count: u32,
    pub last_updated: DateTime<Utc>,
}

pub enum GpuCategory {
    H100,
    H200,
    A100,
    RTX4090,
    Other(String),
}
```

### Weight Calculation Flow

```
1. Verification Results → GPU Profile Update
   ├── Determine primary GPU model per miner
   ├── Calculate verification score
   └── Store in category-specific storage

2. Weight Setting (Every 360 blocks)
   ├── Calculate burn allocation
   ├── Group miners by GPU category
   ├── Distribute remaining weight by GPU allocations
   └── Submit normalized weights to chain

3. Weight Distribution Formula
   ├── Burn Weight = burn_percentage / 100 * MAX_WEIGHT
   ├── Category Weight = gpu_allocation / 100 * (MAX_WEIGHT - Burn Weight)
   └── Miner Weight = (score / category_total_score) * Category Weight
```

## Detailed Component Design

### 1. GPU Model Normalization

```rust
impl WeightSetter {
    fn normalize_gpu_model(gpu_model: &str) -> String {
        let model = gpu_model.to_uppercase();
        if model.contains("H100") {
            "H100".to_string()
        } else if model.contains("H200") {
            "H200".to_string()
        } else if model.contains("A100") {
            "A100".to_string()
        } else if model.contains("4090") {
            "RTX4090".to_string()
        } else if model.contains("3090") {
            "RTX3090".to_string()
        } else {
            "OTHER".to_string()
        }
    }
    
    fn determine_primary_gpu_model(
        executor_validations: &[ExecutorValidationResult]
    ) -> String {
        let mut gpu_counts: HashMap<String, u32> = HashMap::new();
        
        for validation in executor_validations.iter()
            .filter(|v| v.is_valid && v.attestation_valid) 
        {
            let normalized = Self::normalize_gpu_model(&validation.gpu_model);
            *gpu_counts.entry(normalized).or_insert(0) += validation.gpu_count as u32;
        }
        
        gpu_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(model, _)| model)
            .unwrap_or_else(|| "OTHER".to_string())
    }
}
```

### 2. Enhanced Miner Scoring

```rust
impl WeightSetter {
    pub async fn update_miner_profile_from_validation(
        &self,
        miner_uid: MinerUid,
        executor_validations: Vec<ExecutorValidationResult>,
    ) -> Result<()> {
        if executor_validations.is_empty() {
            return Ok(());
        }

        // Calculate verification score (preserve existing logic)
        let verification_score = self.calculate_verification_score(&executor_validations);
        
        // Determine GPU profile
        let primary_gpu_model = Self::determine_primary_gpu_model(&executor_validations);
        let mut gpu_counts = HashMap::new();
        
        for validation in &executor_validations {
            if validation.is_valid && validation.attestation_valid {
                let normalized = Self::normalize_gpu_model(&validation.gpu_model);
                *gpu_counts.entry(normalized).or_insert(0) += validation.gpu_count as u32;
            }
        }
        
        // Store/update GPU profile
        let profile = MinerGpuProfile {
            miner_uid,
            primary_gpu_model: primary_gpu_model.clone(),
            gpu_counts,
            total_score: verification_score,
            verification_count: executor_validations.len() as u32,
            last_updated: Utc::now(),
        };
        
        self.store_miner_gpu_profile(profile).await?;
        
        // Update category-specific score with EMA
        let category_key = format!("miner_score:{}:{}", primary_gpu_model, miner_uid.as_u16());
        let previous_score = self.storage.get_f64(&category_key).await
            .unwrap_or(None).unwrap_or(0.0);
        
        let smoothed_score = if previous_score > 0.0 {
            0.7 * previous_score + 0.3 * verification_score
        } else {
            verification_score
        };
        
        self.storage.set_f64(&category_key, smoothed_score).await?;
        
        info!(
            "Updated miner {} profile - GPU: {}, Score: {:.4}, Executors: {}",
            miner_uid.as_u16(), primary_gpu_model, smoothed_score, executor_validations.len()
        );
        
        Ok(())
    }
}
```

### 3. Weight Setting with Burn and GPU Allocation

```rust
impl WeightSetter {
    async fn set_weights_with_burn_and_gpu_allocation(&self) -> Result<()> {
        info!("Setting weights with burn and GPU allocation");
        
        let emission_config = &self.emission_config;
        
        // 1. Get current metagraph
        let metagraph = self.get_metagraph().await?;
        debug!("Retrieved metagraph with {} neurons", metagraph.hotkeys.len());
        
        // 2. Get all miner GPU profiles
        let gpu_profiles = self.get_all_miner_gpu_profiles().await?;
        info!("Found {} miner GPU profiles", gpu_profiles.len());
        
        // 3. Group miners by GPU category and filter by threshold
        let mut gpu_categories: HashMap<String, Vec<(MinerUid, f64)>> = HashMap::new();
        let mut total_qualifying_miners = 0;
        
        for profile in gpu_profiles {
            if profile.total_score >= self.min_score_threshold {
                // Check if this GPU model has an allocation
                if emission_config.gpu_allocations.contains_key(&profile.primary_gpu_model) {
                    let category_miners = gpu_categories
                        .entry(profile.primary_gpu_model.clone())
                        .or_insert_with(Vec::new);
                    
                    category_miners.push((profile.miner_uid, profile.total_score));
                    total_qualifying_miners += 1;
                }
            }
        }
        
        // 4. Log category statistics
        for (gpu_model, miners) in &gpu_categories {
            info!(
                "GPU Category {}: {} miners, allocation: {:.1}%",
                gpu_model,
                miners.len(),
                emission_config.gpu_allocations.get(gpu_model).unwrap_or(&0.0)
            );
        }
        
        if total_qualifying_miners == 0 {
            warn!("No qualifying miners found for weight setting");
            return Ok(());
        }
        
        // 5. Calculate weight distribution
        let mut final_weights: Vec<NormalizedWeight> = Vec::new();
        
        // Add burn weight if configured
        if emission_config.burn_percentage > 0.0 {
            let burn_weight = (emission_config.burn_percentage / 100.0 * u16::MAX as f64) as u16;
            final_weights.push(NormalizedWeight {
                uid: emission_config.burn_uid,
                weight: burn_weight,
            });
            info!("Burn allocation: {:.1}% to UID {}", emission_config.burn_percentage, emission_config.burn_uid);
        }
        
        // Calculate remaining weight for distribution
        let remaining_weight_ratio = (100.0 - emission_config.burn_percentage) / 100.0;
        let total_remaining_weight = (remaining_weight_ratio * u16::MAX as f64) as u64;
        
        // 6. Distribute remaining weight by GPU categories
        for (gpu_model, allocation_percentage) in &emission_config.gpu_allocations {
            if let Some(miners_in_category) = gpu_categories.get(gpu_model) {
                if miners_in_category.len() >= emission_config.min_miners_per_category as usize {
                    let category_weight_pool = (allocation_percentage / 100.0 * total_remaining_weight as f64) as u64;
                    
                    // Distribute proportionally within category based on scores
                    let total_category_score: f64 = miners_in_category.iter()
                        .map(|(_, score)| score).sum();
                    
                    if total_category_score > 0.0 {
                        for (miner_uid, score) in miners_in_category {
                            let weight_ratio = score / total_category_score;
                            let miner_weight = (weight_ratio * category_weight_pool as f64) as u16;
                            
                            if miner_weight > 0 {
                                final_weights.push(NormalizedWeight {
                                    uid: miner_uid.as_u16(),
                                    weight: miner_weight,
                                });
                            }
                        }
                        
                        info!(
                            "Distributed {:.1}% to {} {} miners (pool: {} weight units)",
                            allocation_percentage,
                            miners_in_category.len(),
                            gpu_model,
                            category_weight_pool
                        );
                    }
                } else {
                    warn!(
                        "Insufficient miners in {} category: {} < {}",
                        gpu_model,
                        miners_in_category.len(),
                        emission_config.min_miners_per_category
                    );
                }
            }
        }
        
        if final_weights.is_empty() {
            warn!("No weights calculated for submission");
            return Ok(());
        }
        
        info!(
            "Final weight distribution: {} recipients, total weight: {}",
            final_weights.len(),
            final_weights.iter().map(|w| w.weight as u64).sum::<u64>()
        );
        
        // 7. Submit weights to chain
        let version_key = self.get_version_key().await?;
        self.submit_weights_to_chain(final_weights.clone(), version_key).await?;
        
        // 8. Store submission metadata
        self.store_weight_submission_with_categories(&final_weights, &gpu_categories).await?;
        
        Ok(())
    }
}
```

## Storage and Persistence

### GPU Profile Storage

```rust
impl WeightSetter {
    async fn store_miner_gpu_profile(&self, profile: MinerGpuProfile) -> Result<()> {
        // Store in memory for fast access
        let profile_key = format!("gpu_profile:{}", profile.miner_uid.as_u16());
        let profile_json = serde_json::to_string(&profile)?;
        self.storage.set_string(&profile_key, &profile_json).await?;
        
        // Store in database for persistence
        let query = r#"
            INSERT OR REPLACE INTO miner_gpu_profiles 
            (miner_uid, primary_gpu_model, gpu_counts_json, total_score, verification_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        "#;
        
        let gpu_counts_json = serde_json::to_string(&profile.gpu_counts)?;
        
        sqlx::query(query)
            .bind(profile.miner_uid.as_u16())
            .bind(&profile.primary_gpu_model)
            .bind(&gpu_counts_json)
            .bind(profile.total_score)
            .bind(profile.verification_count)
            .bind(profile.last_updated.to_rfc3339())
            .execute(self.persistence.pool())
            .await?;
        
        Ok(())
    }
    
    async fn get_all_miner_gpu_profiles(&self) -> Result<Vec<MinerGpuProfile>> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(48);
        
        let query = r#"
            SELECT miner_uid, primary_gpu_model, gpu_counts_json, total_score, verification_count, last_updated
            FROM miner_gpu_profiles 
            WHERE last_updated >= ?
            ORDER BY total_score DESC
        "#;
        
        let rows = sqlx::query(query)
            .bind(cutoff_time.to_rfc3339())
            .fetch_all(self.persistence.pool())
            .await?;
        
        let mut profiles = Vec::new();
        for row in rows {
            let miner_uid = MinerUid::new(row.get::<i64, _>("miner_uid") as u16);
            let primary_gpu_model: String = row.get("primary_gpu_model");
            let gpu_counts_json: String = row.get("gpu_counts_json");
            let gpu_counts: HashMap<String, u32> = serde_json::from_str(&gpu_counts_json)?;
            let total_score: f64 = row.get("total_score");
            let verification_count: i64 = row.get("verification_count");
            let last_updated: String = row.get("last_updated");
            let last_updated = DateTime::parse_from_rfc3339(&last_updated)?.with_timezone(&Utc);
            
            profiles.push(MinerGpuProfile {
                miner_uid,
                primary_gpu_model,
                gpu_counts,
                total_score,
                verification_count: verification_count as u32,
                last_updated,
            });
        }
        
        Ok(profiles)
    }
}
```

## Configuration Management

### Config File Integration

```toml
# config.toml

[emission]
# Percentage of emissions to burn (0.0-100.0)
burn_percentage = 5.0

# UID to receive burn allocation
burn_uid = 0

# Blocks between weight setting
weight_set_interval_blocks = 360

# Minimum miners per GPU category
min_miners_per_category = 1

# GPU model allocations (must sum to 100.0)
[emission.gpu_allocations]
H100 = 40.0
H200 = 60.0

# Alternative allocation example:
# H100 = 30.0
# H200 = 50.0
# A100 = 15.0
# RTX4090 = 5.0
```

### Runtime Configuration Updates

```rust
impl WeightSetter {
    pub async fn update_emission_config(&mut self, new_config: EmissionConfig) -> Result<()> {
        // Validate configuration
        self.validate_emission_config(&new_config)?;
        
        // Update internal config
        self.emission_config = new_config.clone();
        
        // Store configuration for persistence
        let config_json = serde_json::to_string(&new_config)?;
        self.storage.set_string("emission_config", &config_json).await?;
        
        info!("Updated emission configuration: burn={:.1}%, GPU allocations={:?}",
              new_config.burn_percentage, new_config.gpu_allocations);
        
        Ok(())
    }
    
    fn validate_emission_config(&self, config: &EmissionConfig) -> Result<()> {
        if config.burn_percentage < 0.0 || config.burn_percentage > 100.0 {
            return Err(anyhow::anyhow!("Burn percentage must be 0-100, got: {}", config.burn_percentage));
        }
        
        let total_allocation: f64 = config.gpu_allocations.values().sum();
        if (total_allocation - 100.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("GPU allocations must sum to 100.0, got: {:.2}", total_allocation));
        }
        
        if config.weight_set_interval_blocks == 0 {
            return Err(anyhow::anyhow!("Weight set interval must be > 0"));
        }
        
        Ok(())
    }
}
```

## Monitoring and Analytics

### Enhanced Metrics

```rust
#[derive(Debug, Clone)]
pub struct EmissionMetrics {
    pub burn_amount: u64,
    pub burn_percentage: f64,
    pub gpu_category_distributions: HashMap<String, CategoryMetrics>,
    pub total_miners_served: u32,
    pub weight_set_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CategoryMetrics {
    pub gpu_model: String,
    pub miner_count: u32,
    pub total_score: f64,
    pub weight_allocation: u64,
    pub allocation_percentage: f64,
}

impl WeightSetter {
    async fn store_emission_metrics(&self, metrics: EmissionMetrics) -> Result<()> {
        let metrics_json = serde_json::to_string(&metrics)?;
        
        // Store in database for historical analysis
        let query = r#"
            INSERT INTO emission_metrics 
            (timestamp, burn_amount, burn_percentage, category_distributions_json, total_miners)
            VALUES (?, ?, ?, ?, ?)
        "#;
        
        sqlx::query(query)
            .bind(metrics.weight_set_timestamp.to_rfc3339())
            .bind(metrics.burn_amount as i64)
            .bind(metrics.burn_percentage)
            .bind(&serde_json::to_string(&metrics.gpu_category_distributions)?)
            .bind(metrics.total_miners_served)
            .execute(self.persistence.pool())
            .await?;
        
        Ok(())
    }
}
```

## Migration Strategy

### Database Schema Updates

```sql
-- Add miner GPU profiles table
CREATE TABLE IF NOT EXISTS miner_gpu_profiles (
    miner_uid INTEGER PRIMARY KEY,
    primary_gpu_model TEXT NOT NULL,
    gpu_counts_json TEXT NOT NULL,
    total_score REAL NOT NULL,
    verification_count INTEGER NOT NULL,
    last_updated TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Add emission metrics table
CREATE TABLE IF NOT EXISTS emission_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    burn_amount INTEGER NOT NULL,
    burn_percentage REAL NOT NULL,
    category_distributions_json TEXT NOT NULL,
    total_miners INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_model ON miner_gpu_profiles(primary_gpu_model);
CREATE INDEX IF NOT EXISTS idx_gpu_profiles_score ON miner_gpu_profiles(total_score DESC);
CREATE INDEX IF NOT EXISTS idx_emission_metrics_timestamp ON emission_metrics(timestamp);
```

## Security Considerations

### Input Validation
- Strict validation of GPU allocation percentages
- Bounds checking for burn percentage (0-100%)
- Sanitization of GPU model strings
- Verification of UID ranges

### Rate Limiting
- Prevent rapid configuration changes
- Minimum time between weight sets
- Maximum weight change thresholds

### Audit Trail
- Log all configuration changes
- Track weight distribution decisions
- Monitor for unusual allocation patterns

## Testing Strategy

### Unit Tests
- GPU model normalization
- Weight calculation logic
- Configuration validation
- Score aggregation

### Integration Tests
- End-to-end weight setting flow
- Database persistence
- Configuration updates
- Error handling

### Performance Tests
- Large miner population scenarios
- Weight calculation efficiency
- Database query performance

## Deployment Plan

See the accompanying implementation plan for detailed deployment steps and timeline.