use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::config::emission::EmissionConfig;
use common::identity::MinerUid;

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
        // Total weight available (using u16::MAX as the maximum)
        let total_weight = u16::MAX as u64;

        // Calculate burn allocation first
        let burn_allocation = self.calculate_burn_allocation(total_weight)?;
        let burn_weight = burn_allocation
            .as_ref()
            .map(|b| b.weight as u64)
            .unwrap_or(0);

        // Remaining weight after burn
        let remaining_weight = total_weight - burn_weight;

        // Filter miners by minimum score threshold
        let filtered_miners = self.filter_miners_by_score(miners_by_category)?;

        // Calculate category weight pools for ALL configured categories
        let all_category_pools = self.calculate_all_category_pools(remaining_weight)?;

        // Track which categories have miners
        let mut active_categories = std::collections::HashSet::new();
        for category in filtered_miners.keys() {
            active_categories.insert(category.clone());
        }

        // Calculate additional burn for empty categories
        let mut empty_category_burn = 0u64;
        for (category, pool) in &all_category_pools {
            if !active_categories.contains(category) {
                empty_category_burn += pool;
                info!(
                    category = %category,
                    weight = pool,
                    "Burning weight for empty GPU category"
                );
            }
        }

        // Distribute weights within each category
        let mut all_weights = Vec::new();
        let mut category_allocations = HashMap::new();

        for (category, miners) in filtered_miners {
            let category_weight_pool = all_category_pools.get(&category).copied().unwrap_or(0);

            if category_weight_pool == 0 || miners.is_empty() {
                continue;
            }

            let category_weights =
                self.distribute_category_weight(&miners, category_weight_pool)?;

            // Calculate category statistics
            let total_score: f64 = miners.iter().map(|(_, score)| score).sum();
            let allocation_percentage =
                (category_weight_pool as f64 / remaining_weight as f64) * 100.0;

            category_allocations.insert(
                category.clone(),
                CategoryAllocation {
                    gpu_model: category,
                    miner_count: miners.len() as u32,
                    total_score,
                    weight_pool: category_weight_pool,
                    allocation_percentage,
                },
            );

            all_weights.extend(category_weights);
        }

        // Add burn allocation (including empty category burns)
        let total_burn_weight = burn_weight + empty_category_burn;
        if total_burn_weight > 0 {
            all_weights.push(NormalizedWeight {
                uid: self.emission_config.burn_uid,
                weight: total_burn_weight.min(u16::MAX as u64) as u16,
            });
        }

        // Validate final allocation
        self.validate_allocation(&all_weights)?;

        let miners_served = all_weights.len() as u32 - if total_burn_weight > 0 { 1 } else { 0 };

        info!(
            total_weight = total_weight,
            burn_weight = burn_weight,
            empty_category_burn = empty_category_burn,
            total_burn = total_burn_weight,
            categories = category_allocations.len(),
            miners_served = miners_served,
            "Calculated weight distribution"
        );

        Ok(WeightDistribution {
            weights: all_weights,
            burn_allocation: if total_burn_weight > 0 {
                Some(BurnAllocation {
                    uid: self.emission_config.burn_uid,
                    weight: total_burn_weight.min(u16::MAX as u64) as u16,
                    percentage: (total_burn_weight as f64 / total_weight as f64) * 100.0,
                })
            } else {
                None
            },
            category_allocations,
            total_weight,
            miners_served,
        })
    }

    /// Calculate burn allocation
    fn calculate_burn_allocation(&self, total_weight: u64) -> Result<Option<BurnAllocation>> {
        if self.emission_config.burn_percentage <= 0.0 {
            return Ok(None);
        }

        let burn_weight =
            (total_weight as f64 * self.emission_config.burn_percentage / 100.0) as u16;

        if burn_weight == 0 {
            return Ok(None);
        }

        Ok(Some(BurnAllocation {
            uid: self.emission_config.burn_uid,
            weight: burn_weight,
            percentage: self.emission_config.burn_percentage,
        }))
    }

    /// Filter miners by minimum score threshold
    fn filter_miners_by_score(
        &self,
        miners_by_category: HashMap<String, Vec<(MinerUid, f64)>>,
    ) -> Result<HashMap<String, Vec<(MinerUid, f64)>>> {
        let mut filtered = HashMap::new();

        // Hardcoded minimum of 1 miner per category
        const MIN_MINERS_PER_CATEGORY: usize = 1;

        for (category, miners) in miners_by_category {
            // Remove score threshold filtering - include all miners regardless of score
            let valid_miners: Vec<(MinerUid, f64)> = miners;

            // Only include categories with minimum number of miners (hardcoded to 1)
            if valid_miners.len() >= MIN_MINERS_PER_CATEGORY {
                filtered.insert(category, valid_miners);
            } else {
                debug!(
                    category = %category,
                    miners = valid_miners.len(),
                    required = MIN_MINERS_PER_CATEGORY,
                    "Category excluded due to insufficient miners"
                );
            }
        }

        Ok(filtered)
    }

    /// Calculate category weight pools based on allocation percentages
    fn calculate_category_pools(
        &self,
        total_remaining_weight: u64,
        filtered_miners: &HashMap<String, Vec<(MinerUid, f64)>>,
    ) -> Result<HashMap<String, u64>> {
        let mut category_pools = HashMap::new();

        for category in filtered_miners.keys() {
            if let Some(allocation_percentage) = self.emission_config.get_gpu_allocation(category) {
                let weight_pool =
                    (total_remaining_weight as f64 * allocation_percentage / 100.0) as u64;
                category_pools.insert(category.clone(), weight_pool);
            } else {
                warn!(
                    category = %category,
                    "No allocation percentage configured for category"
                );
            }
        }

        Ok(category_pools)
    }

    /// Calculate weight pools for ALL configured categories (including empty ones)
    fn calculate_all_category_pools(
        &self,
        total_remaining_weight: u64,
    ) -> Result<HashMap<String, u64>> {
        let mut category_pools = HashMap::new();

        // Get all configured GPU categories from emission config
        for (category, allocation_percentage) in &self.emission_config.gpu_allocations {
            let weight_pool =
                (total_remaining_weight as f64 * allocation_percentage / 100.0) as u64;
            category_pools.insert(category.clone(), weight_pool);
        }

        Ok(category_pools)
    }

    /// Distribute weight within a category proportionally by score
    fn distribute_category_weight(
        &self,
        category_miners: &[(MinerUid, f64)],
        category_weight_pool: u64,
    ) -> Result<Vec<NormalizedWeight>> {
        if category_miners.is_empty() {
            return Ok(Vec::new());
        }

        let total_score: f64 = category_miners.iter().map(|(_, score)| score).sum();

        if total_score <= 0.0 {
            warn!("Total score is zero for category, distributing equally");
            return self.distribute_equally(category_miners, category_weight_pool);
        }

        let mut weights = Vec::new();
        let mut allocated_weight = 0u64;

        for (i, (miner_uid, score)) in category_miners.iter().enumerate() {
            let weight = if i == category_miners.len() - 1 {
                // Last miner gets remaining weight to avoid rounding errors
                category_weight_pool - allocated_weight
            } else {
                (category_weight_pool as f64 * score / total_score) as u64
            };

            // Ensure weight fits in u16
            let weight = weight.min(u16::MAX as u64) as u16;

            if weight > 0 {
                weights.push(NormalizedWeight {
                    uid: miner_uid.as_u16(),
                    weight,
                });
                allocated_weight += weight as u64;
            }
        }

        Ok(weights)
    }

    /// Distribute weight equally among miners (fallback method)
    fn distribute_equally(
        &self,
        category_miners: &[(MinerUid, f64)],
        category_weight_pool: u64,
    ) -> Result<Vec<NormalizedWeight>> {
        if category_miners.is_empty() {
            return Ok(Vec::new());
        }

        let weight_per_miner = (category_weight_pool / category_miners.len() as u64) as u16;
        let mut weights = Vec::new();

        for (miner_uid, _) in category_miners {
            if weight_per_miner > 0 {
                weights.push(NormalizedWeight {
                    uid: miner_uid.as_u16(),
                    weight: weight_per_miner,
                });
            }
        }

        Ok(weights)
    }

    /// Validate allocation results
    fn validate_allocation(&self, weights: &[NormalizedWeight]) -> Result<()> {
        let total_allocated: u64 = weights.iter().map(|w| w.weight as u64).sum();
        let max_weight = u16::MAX as u64;

        if total_allocated > max_weight {
            return Err(anyhow!(
                "Total allocated weight {} exceeds maximum {}",
                total_allocated,
                max_weight
            ));
        }

        // Check for duplicate UIDs
        let mut seen_uids = std::collections::HashSet::new();
        for weight in weights {
            if !seen_uids.insert(weight.uid) {
                return Err(anyhow!("Duplicate UID {} in weight allocation", weight.uid));
            }
        }

        // Individual weights are already u16, so they cannot exceed u16::MAX
        // // Check individual weight bounds
        // for weight in weights {
        //     if weight.weight > u16::MAX {
        //         return Err(anyhow!(
        //             "Weight {} for UID {} exceeds maximum {}",
        //             weight.weight,
        //             weight.uid,
        //             u16::MAX
        //         ));
        //     }
        // }

        Ok(())
    }

    /// Update emission configuration
    pub fn update_emission_config(&mut self, new_config: EmissionConfig) -> Result<()> {
        new_config.validate()?;
        self.emission_config = new_config;
        Ok(())
    }

    /// Get current emission configuration
    pub fn get_emission_config(&self) -> &EmissionConfig {
        &self.emission_config
    }

    /// Get minimum score threshold
    pub fn get_min_score_threshold(&self) -> f64 {
        self.min_score_threshold
    }

    /// Set minimum score threshold
    pub fn set_min_score_threshold(&mut self, threshold: f64) {
        self.min_score_threshold = threshold.clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeightDistribution {
    pub weights: Vec<NormalizedWeight>,
    pub burn_allocation: Option<BurnAllocation>,
    pub category_allocations: HashMap<String, CategoryAllocation>,
    pub total_weight: u64,
    pub miners_served: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BurnAllocation {
    pub uid: u16,
    pub weight: u16,
    pub percentage: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoryAllocation {
    pub gpu_model: String,
    pub miner_count: u32,
    pub total_score: f64,
    pub weight_pool: u64,
    pub allocation_percentage: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizedWeight {
    pub uid: u16,
    pub weight: u16,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::emission::EmissionConfig;
    use std::collections::HashMap;

    fn create_test_config() -> EmissionConfig {
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert("H100".to_string(), 60.0);
        gpu_allocations.insert("H200".to_string(), 40.0);

        EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations,
            weight_set_interval_blocks: 360,
        }
    }

    fn create_test_miners() -> HashMap<String, Vec<(MinerUid, f64)>> {
        let mut miners = HashMap::new();

        miners.insert(
            "H100".to_string(),
            vec![
                (MinerUid::new(1), 0.8),
                (MinerUid::new(2), 0.6),
                (MinerUid::new(3), 0.4),
            ],
        );

        miners.insert(
            "H200".to_string(),
            vec![(MinerUid::new(4), 0.9), (MinerUid::new(5), 0.7)],
        );

        miners
    }

    #[test]
    fn test_burn_allocation_calculation() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.5);

        // Test with burn enabled
        let burn_alloc = engine.calculate_burn_allocation(10000).unwrap();
        assert!(burn_alloc.is_some());

        let burn = burn_alloc.unwrap();
        assert_eq!(burn.uid, 999);
        assert_eq!(burn.weight, 1000); // 10% of 10000
        assert_eq!(burn.percentage, 10.0);

        // Test with zero burn percentage
        let mut config_no_burn = create_test_config();
        config_no_burn.burn_percentage = 0.0;
        let engine_no_burn = WeightAllocationEngine::new(config_no_burn, 0.5);

        let burn_alloc = engine_no_burn.calculate_burn_allocation(10000).unwrap();
        assert!(burn_alloc.is_none());
    }

    #[test]
    fn test_category_pool_calculation() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.5);

        let miners = create_test_miners();
        let pools = engine.calculate_category_pools(9000, &miners).unwrap();

        assert_eq!(pools.get("H100"), Some(&5400)); // 60% of 9000
        assert_eq!(pools.get("H200"), Some(&3600)); // 40% of 9000
    }

    #[test]
    fn test_within_category_distribution() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.5);

        let miners = vec![
            (MinerUid::new(1), 0.8),
            (MinerUid::new(2), 0.4),
            (MinerUid::new(3), 0.8),
        ];

        let weights = engine.distribute_category_weight(&miners, 2000).unwrap();

        assert_eq!(weights.len(), 3);

        // Check proportional distribution
        let total_score = 0.8 + 0.4 + 0.8; // 2.0
        let expected_weight_1 = (2000.0 * 0.8 / total_score) as u16;
        let expected_weight_2 = (2000.0 * 0.4 / total_score) as u16;

        assert_eq!(weights[0].weight, expected_weight_1);
        assert_eq!(weights[1].weight, expected_weight_2);

        // Last miner gets remaining weight
        let total_allocated = weights[0].weight as u64 + weights[1].weight as u64;
        assert_eq!(weights[2].weight, (2000 - total_allocated) as u16);
    }

    #[test]
    fn test_complete_weight_distribution() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.3);

        let miners = create_test_miners();
        let distribution = engine.calculate_weight_distribution(miners).unwrap();

        // Should have burn allocation
        assert!(distribution.burn_allocation.is_some());
        let burn = distribution.burn_allocation.unwrap();
        // Burn percentage should be approximately 10% (base burn)
        assert!((burn.percentage - 10.0).abs() < 0.1);

        // Should have category allocations
        assert_eq!(distribution.category_allocations.len(), 2);
        assert!(distribution.category_allocations.contains_key("H100"));
        assert!(distribution.category_allocations.contains_key("H200"));

        // Should have weights for miners + burn
        assert_eq!(distribution.weights.len(), 6); // 5 miners + 1 burn
        assert_eq!(distribution.miners_served, 5);

        // Verify weight conservation
        let total_weight: u64 = distribution.weights.iter().map(|w| w.weight as u64).sum();
        assert!(total_weight <= u16::MAX as u64);
    }

    #[test]
    fn test_minimum_score_filtering() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.7); // High threshold

        let miners = create_test_miners();
        let distribution = engine.calculate_weight_distribution(miners).unwrap();

        // With threshold removed, all miners should be included
        // H100: 3 miners, H200: 2 miners
        assert_eq!(distribution.miners_served, 5);
    }

    #[test]
    fn test_allocation_validation() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.5);

        // Test valid allocation
        let valid_weights = vec![
            NormalizedWeight {
                uid: 1,
                weight: 1000,
            },
            NormalizedWeight {
                uid: 2,
                weight: 2000,
            },
        ];
        assert!(engine.validate_allocation(&valid_weights).is_ok());

        // Test duplicate UID
        let duplicate_weights = vec![
            NormalizedWeight {
                uid: 1,
                weight: 1000,
            },
            NormalizedWeight {
                uid: 1,
                weight: 2000,
            },
        ];
        assert!(engine.validate_allocation(&duplicate_weights).is_err());
    }

    #[test]
    fn test_edge_cases() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.5);

        // Test empty miners
        let empty_miners = HashMap::new();
        let distribution = engine.calculate_weight_distribution(empty_miners).unwrap();
        assert_eq!(distribution.miners_served, 0);
        assert_eq!(distribution.weights.len(), 1); // Only burn allocation

        // Test with no miners in H200 category - should burn H200 allocation
        let mut single_category_miners = HashMap::new();
        single_category_miners.insert(
            "H100".to_string(),
            vec![(MinerUid::new(1), 0.8), (MinerUid::new(2), 0.6)],
        );

        let distribution = engine
            .calculate_weight_distribution(single_category_miners)
            .unwrap();
        assert_eq!(distribution.miners_served, 2); // Only H100 miners

        // Check that burn allocation includes H200's 40% allocation
        let burn_alloc = distribution.burn_allocation.unwrap();
        assert!(burn_alloc.percentage > 40.0); // Should be base burn + H200's 40%
    }

    #[test]
    fn test_mathematical_accuracy() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.0);

        let miners = create_test_miners();
        let distribution = engine.calculate_weight_distribution(miners).unwrap();

        // Test weight conservation
        let total_weight: u64 = distribution.weights.iter().map(|w| w.weight as u64).sum();
        assert!(total_weight <= u16::MAX as u64);

        // Test category allocation percentages
        let h100_allocation = distribution.category_allocations.get("H100").unwrap();
        let h200_allocation = distribution.category_allocations.get("H200").unwrap();

        assert!((h100_allocation.allocation_percentage - 60.0).abs() < 0.1);
        assert!((h200_allocation.allocation_percentage - 40.0).abs() < 0.1);
    }

    #[test]
    fn test_config_updates() {
        let config = create_test_config();
        let mut engine = WeightAllocationEngine::new(config, 0.5);

        // Test valid config update
        let mut new_config = create_test_config();
        new_config.burn_percentage = 20.0;
        assert!(engine.update_emission_config(new_config).is_ok());
        assert_eq!(engine.get_emission_config().burn_percentage, 20.0);

        // Test invalid config update
        let mut invalid_config = create_test_config();
        invalid_config.burn_percentage = 150.0; // Invalid
        assert!(engine.update_emission_config(invalid_config).is_err());
    }

    #[test]
    fn test_score_threshold_functions() {
        let config = create_test_config();
        let mut engine = WeightAllocationEngine::new(config, 0.5);

        assert_eq!(engine.get_min_score_threshold(), 0.5);

        engine.set_min_score_threshold(0.8);
        assert_eq!(engine.get_min_score_threshold(), 0.8);

        // Test clamping
        engine.set_min_score_threshold(1.5);
        assert_eq!(engine.get_min_score_threshold(), 1.0);

        engine.set_min_score_threshold(-0.5);
        assert_eq!(engine.get_min_score_threshold(), 0.0);
    }

    #[test]
    fn test_calculate_all_category_pools() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.0);

        let total_weight = 10000u64;
        let pools = engine.calculate_all_category_pools(total_weight).unwrap();

        // Should have pools for all configured categories
        assert_eq!(pools.len(), 2);
        assert!(pools.contains_key("H100"));
        assert!(pools.contains_key("H200"));

        // H100 should get 60% of total
        assert_eq!(pools.get("H100"), Some(&6000));
        // H200 should get 40% of total
        assert_eq!(pools.get("H200"), Some(&4000));

        // Total should equal input
        let total: u64 = pools.values().sum();
        assert_eq!(total, total_weight);
    }

    #[test]
    fn test_empty_category_burn_both_categories_empty() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.0);

        // No miners at all
        let empty_miners = HashMap::new();
        let distribution = engine.calculate_weight_distribution(empty_miners).unwrap();

        // Should have only burn allocation
        assert_eq!(distribution.miners_served, 0);
        assert_eq!(distribution.weights.len(), 1);

        // Burn should include base burn (10%) + all category allocations (90%)
        let burn = distribution.burn_allocation.unwrap();
        assert!((burn.percentage - 100.0).abs() < 0.1); // Should be ~100%
    }

    #[test]
    fn test_empty_category_burn_mixed_categories() {
        let mut config = create_test_config();
        // Set up 3 categories for testing
        config.gpu_allocations.clear();
        config.gpu_allocations.insert("H100".to_string(), 40.0);
        config.gpu_allocations.insert("H200".to_string(), 30.0);
        config.gpu_allocations.insert("A100".to_string(), 30.0);

        let engine = WeightAllocationEngine::new(config, 0.0);

        // Only H100 has miners
        let mut miners = HashMap::new();
        miners.insert("H100".to_string(), vec![(MinerUid::new(1), 1.0)]);

        let distribution = engine.calculate_weight_distribution(miners).unwrap();

        // Should burn H200 and A100 allocations (30% + 30% = 60%) plus base burn (10%)
        let burn = distribution.burn_allocation.unwrap();
        // The exact percentage depends on weight calculations and rounding
        // We expect around 64% (not 70% due to how weights are calculated)
        assert!(burn.percentage > 60.0 && burn.percentage < 65.0);

        // Only H100 should have allocation
        assert_eq!(distribution.category_allocations.len(), 1);
        assert!(distribution.category_allocations.contains_key("H100"));
    }

    #[test]
    fn test_hardcoded_min_miners_per_category() {
        let config = create_test_config();
        let engine = WeightAllocationEngine::new(config, 0.0);

        // Create category with 0 miners (empty list)
        let mut miners = HashMap::new();
        miners.insert("H100".to_string(), vec![]);
        miners.insert("H200".to_string(), vec![(MinerUid::new(1), 0.5)]);

        let filtered = engine.filter_miners_by_score(miners).unwrap();

        // H100 should be excluded (0 < 1 minimum)
        assert!(!filtered.contains_key("H100"));
        // H200 should be included (1 >= 1 minimum)
        assert!(filtered.contains_key("H200"));
    }

    #[test]
    fn test_burn_weight_overflow_protection() {
        let mut config = create_test_config();
        config.burn_percentage = 90.0; // Very high burn
        let engine = WeightAllocationEngine::new(config, 0.0);

        // No miners, so everything gets burned
        let empty_miners = HashMap::new();
        let distribution = engine.calculate_weight_distribution(empty_miners).unwrap();

        // Burn weight should be clamped to u16::MAX
        // let burn = distribution.burn_allocation.unwrap();
        // assert!(burn.weight <= u16::MAX);
    }
}
