use anyhow::{anyhow, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Default burn UID for Basilica validator emissions
pub const DEFAULT_BURN_UID: u16 = 204;

/// GPU allocation configuration with weight and minimum GPU count
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuAllocation {
    /// Weight percentage for this GPU model (0.0-100.0)
    pub weight: f64,
    /// Minimum number of GPUs required for incentives (default: 1)
    #[serde(default = "default_min_gpu_count")]
    pub min_gpu_count: u32,
    /// Minimum GPU VRAM in GB required for incentives (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_vram: Option<u64>,
}

fn default_min_gpu_count() -> u32 {
    1
}

fn default_min_miners_per_category() -> u32 {
    1
}

impl GpuAllocation {
    /// Create a new GPU allocation with default min_gpu_count
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            min_gpu_count: default_min_gpu_count(),
            min_gpu_vram: None,
        }
    }

    /// Create a new GPU allocation with specific min_gpu_count
    pub fn with_min_count(weight: f64, min_gpu_count: u32) -> Self {
        Self {
            weight,
            min_gpu_count,
            min_gpu_vram: None,
        }
    }

    /// Create a new GPU allocation with min_gpu_count and min_gpu_vram
    pub fn with_requirements(weight: f64, min_gpu_count: u32, min_gpu_vram: Option<u64>) -> Self {
        Self {
            weight,
            min_gpu_count,
            min_gpu_vram,
        }
    }
}

/// Custom deserializer for gpu_allocations that supports both formats:
/// - Legacy: "H100" = 8.0
/// - New: "H100" = { weight = 8.0, min_gpu_count = 4 }
fn deserialize_gpu_allocations<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, GpuAllocation>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum GpuAllocationEntry {
        Legacy(f64),
        Full(GpuAllocation),
    }

    let raw: HashMap<String, GpuAllocationEntry> = HashMap::deserialize(deserializer)?;
    let mut result = HashMap::new();

    for (key, value) in raw {
        let allocation = match value {
            GpuAllocationEntry::Legacy(weight) => GpuAllocation::new(weight),
            GpuAllocationEntry::Full(allocation) => allocation,
        };
        result.insert(key, allocation);
    }

    Ok(result)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmissionConfig {
    /// Percentage of emissions to burn (0.0-100.0)
    pub burn_percentage: f64,

    /// UID to send burn weights to
    pub burn_uid: u16,

    /// GPU model allocations with weights and minimum GPU counts
    #[serde(deserialize_with = "deserialize_gpu_allocations")]
    pub gpu_allocations: HashMap<String, GpuAllocation>,

    /// Minimum number of miners required per category for weight allocation
    #[serde(default = "default_min_miners_per_category")]
    pub min_miners_per_category: u32,

    /// Blocks between weight setting
    pub weight_set_interval_blocks: u64,

    /// Version key for weight setting operations
    /// This prevents replay attacks by incrementing with each weight set
    pub weight_version_key: u64,
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

        let total_allocation: f64 = self.gpu_allocations.values().map(|a| a.weight).sum();
        if (total_allocation - 100.0).abs() > 0.01 {
            return Err(anyhow!(
                "GPU allocations must sum to 100.0, got: {:.2}",
                total_allocation
            ));
        }

        // Validate individual allocations
        for (gpu_model, allocation) in &self.gpu_allocations {
            if allocation.weight < 0.0 {
                return Err(anyhow!(
                    "GPU allocation weight for {} must be non-negative, got: {}",
                    gpu_model,
                    allocation.weight
                ));
            }
            if allocation.min_gpu_count == 0 {
                return Err(anyhow!(
                    "GPU min_gpu_count for {} must be at least 1, got: {}",
                    gpu_model,
                    allocation.min_gpu_count
                ));
            }
            if let Some(vram) = allocation.min_gpu_vram {
                if vram == 0 {
                    return Err(anyhow!(
                        "GPU min_gpu_vram for {} must be greater than 0 if specified, got: {}",
                        gpu_model,
                        vram
                    ));
                }
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
        gpu_allocations.insert("A100".to_string(), GpuAllocation::new(8.0));
        gpu_allocations.insert("H100".to_string(), GpuAllocation::new(12.0));
        gpu_allocations.insert("B200".to_string(), GpuAllocation::new(80.0));

        Self {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
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

    /// Get allocation for a GPU model
    pub fn get_gpu_allocation(&self, model: &str) -> Option<&GpuAllocation> {
        self.gpu_allocations.get(model)
    }

    /// Get allocation weight percentage for a GPU model
    pub fn get_gpu_allocation_weight(&self, model: &str) -> Option<f64> {
        self.gpu_allocations.get(model).map(|a| a.weight)
    }

    /// Add or update a GPU allocation
    pub fn set_gpu_allocation(
        &mut self,
        model: String,
        weight: f64,
        min_gpu_count: u32,
    ) -> Result<()> {
        self.set_gpu_allocation_with_vram(model, weight, min_gpu_count, None)
    }

    /// Add or update a GPU allocation with VRAM requirement
    pub fn set_gpu_allocation_with_vram(
        &mut self,
        model: String,
        weight: f64,
        min_gpu_count: u32,
        min_gpu_vram: Option<u64>,
    ) -> Result<()> {
        if weight < 0.0 {
            return Err(anyhow!(
                "GPU allocation weight cannot be negative: {}",
                weight
            ));
        }
        if min_gpu_count == 0 {
            return Err(anyhow!("GPU min_gpu_count must be at least 1"));
        }
        if let Some(vram) = min_gpu_vram {
            if vram == 0 {
                return Err(anyhow!(
                    "GPU min_gpu_vram must be greater than 0 if specified"
                ));
            }
        }

        self.gpu_allocations.insert(
            model,
            GpuAllocation::with_requirements(weight, min_gpu_count, min_gpu_vram),
        );
        Ok(())
    }

    /// Remove a GPU allocation
    pub fn remove_gpu_allocation(&mut self, model: &str) -> Option<GpuAllocation> {
        self.gpu_allocations.remove(model)
    }

    /// Get all GPU models sorted by allocation weight (descending)
    pub fn gpu_models_by_allocation(&self) -> Vec<(String, GpuAllocation)> {
        let mut models: Vec<(String, GpuAllocation)> = self
            .gpu_allocations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        models.sort_by(|a, b| {
            b.1.weight
                .partial_cmp(&a.1.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        models
    }
}

impl Default for EmissionConfig {
    fn default() -> Self {
        let gpu_allocations = HashMap::new();

        Self {
            burn_percentage: 0.0,
            burn_uid: DEFAULT_BURN_UID,
            gpu_allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        }
    }
}
