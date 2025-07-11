use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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
}

impl EmissionConfig {
    /// Validate the emission configuration
    pub fn validate(&self) -> Result<()> {
        // Validate burn percentage
        if self.burn_percentage < 0.0 || self.burn_percentage > 100.0 {
            return Err(anyhow!(
                "Burn percentage must be between 0.0 and 100.0, got: {}",
                self.burn_percentage
            ));
        }

        // Validate weight set interval
        if self.weight_set_interval_blocks == 0 {
            return Err(anyhow!(
                "Weight set interval blocks must be greater than 0, got: {}",
                self.weight_set_interval_blocks
            ));
        }

        // Validate GPU allocations sum to 100.0
        if self.gpu_allocations.is_empty() {
            return Err(anyhow!("GPU allocations cannot be empty"));
        }

        let total_allocation: f64 = self.gpu_allocations.values().sum();
        if (total_allocation - 100.0).abs() > 0.01 {
            return Err(anyhow!(
                "GPU allocations must sum to 100.0, got: {:.2}",
                total_allocation
            ));
        }

        // Validate individual allocations are positive
        for (gpu_model, allocation) in &self.gpu_allocations {
            if *allocation < 0.0 {
                return Err(anyhow!(
                    "GPU allocation for {} must be non-negative, got: {}",
                    gpu_model,
                    allocation
                ));
            }
        }

        Ok(())
    }

    /// Load configuration from a TOML file
    pub fn from_toml_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {}: {}", path.display(), e))?;

        let config: Self =
            toml::from_str(&content).map_err(|e| anyhow!("Failed to parse TOML config: {}", e))?;

        config.validate()?;
        Ok(config)
    }

    /// Merge this config with defaults for missing fields
    pub fn merge_with_defaults(mut self) -> Self {
        let default_config = Self::default();

        // If GPU allocations is empty, use default
        if self.gpu_allocations.is_empty() {
            self.gpu_allocations = default_config.gpu_allocations;
        }

        // Ensure other fields have reasonable defaults if they're zero/invalid
        if self.weight_set_interval_blocks == 0 {
            self.weight_set_interval_blocks = default_config.weight_set_interval_blocks;
        }

        self
    }

    /// Create a configuration for testing with custom values
    pub fn for_testing() -> Self {
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert("H100".to_string(), 50.0);
        gpu_allocations.insert("H200".to_string(), 50.0);

        Self {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations,
            weight_set_interval_blocks: 360,
        }
    }

    /// Get the total number of configured GPU models
    pub fn gpu_model_count(&self) -> usize {
        self.gpu_allocations.len()
    }

    /// Check if a GPU model is configured
    pub fn has_gpu_model(&self, model: &str) -> bool {
        self.gpu_allocations.contains_key(model)
    }

    /// Get allocation percentage for a GPU model
    pub fn get_gpu_allocation(&self, model: &str) -> Option<f64> {
        self.gpu_allocations.get(model).copied()
    }

    /// Add or update a GPU allocation
    /// Note: This method doesn't validate the total sum. Call validate() separately.
    pub fn set_gpu_allocation(&mut self, model: String, percentage: f64) -> Result<()> {
        if percentage < 0.0 {
            return Err(anyhow!(
                "GPU allocation percentage cannot be negative: {}",
                percentage
            ));
        }

        self.gpu_allocations.insert(model, percentage);
        Ok(())
    }

    /// Remove a GPU allocation
    pub fn remove_gpu_allocation(&mut self, model: &str) -> Option<f64> {
        self.gpu_allocations.remove(model)
    }

    /// Get all GPU models sorted by allocation percentage (descending)
    pub fn gpu_models_by_allocation(&self) -> Vec<(String, f64)> {
        let mut models: Vec<(String, f64)> = self
            .gpu_allocations
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        models
    }
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
        }
    }
}
