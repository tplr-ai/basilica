use chrono::{DateTime, Utc};
use common::identity::MinerUid;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    Other(String),
}

pub struct GpuCategorizer;

impl GpuCategorizer {
    /// Normalize GPU model string to standard category
    pub fn normalize_gpu_model(gpu_model: &str) -> String {
        let model = gpu_model.to_uppercase();

        // Remove common prefixes and clean up
        let cleaned = model
            .replace("NVIDIA", "")
            .replace("GEFORCE", "")
            .replace("TESLA", "")
            .trim()
            .to_string();

        // Match against known patterns - only H100 and H200 for now
        if cleaned.contains("H100") {
            "H100".to_string()
        } else if cleaned.contains("H200") {
            "H200".to_string()
        } else {
            "OTHER".to_string()
        }
    }

    /// Convert normalized model to category enum
    pub fn model_to_category(model: &str) -> GpuCategory {
        match model.to_uppercase().as_str() {
            "H100" => GpuCategory::H100,
            "H200" => GpuCategory::H200,
            _ => GpuCategory::Other(model.to_string()),
        }
    }

    /// Determine primary GPU model from validation results
    /// NOTE: This function is deprecated. Use gpu_counts directly for multi-category scoring.
    pub fn determine_primary_gpu_model(
        executor_validations: &[ExecutorValidationResult],
    ) -> String {
        let mut gpu_counts: HashMap<String, u32> = HashMap::new();

        // Count GPUs by normalized model, weighted by GPU count
        for validation in executor_validations
            .iter()
            .filter(|v| v.is_valid && v.attestation_valid)
        {
            let normalized = Self::normalize_gpu_model(&validation.gpu_model);
            *gpu_counts.entry(normalized).or_insert(0) += validation.gpu_count as u32;
        }

        // Return the model with the highest count
        gpu_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(model, _)| model)
            .unwrap_or_else(|| "OTHER".to_string())
    }

    /// Calculate GPU model distribution for a miner
    pub fn calculate_gpu_distribution(
        executor_validations: &[ExecutorValidationResult],
    ) -> HashMap<String, u32> {
        let mut gpu_counts = HashMap::new();

        for validation in executor_validations
            .iter()
            .filter(|v| v.is_valid && v.attestation_valid)
        {
            let normalized = Self::normalize_gpu_model(&validation.gpu_model);
            *gpu_counts.entry(normalized).or_insert(0) += validation.gpu_count as u32;
        }

        gpu_counts
    }
}

impl MinerGpuProfile {
    /// Create a new GPU profile for a miner
    pub fn new(
        miner_uid: MinerUid,
        executor_validations: &[ExecutorValidationResult],
        total_score: f64,
    ) -> Self {
        let primary_gpu_model = GpuCategorizer::determine_primary_gpu_model(executor_validations);
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(executor_validations);
        let verification_count = executor_validations.len() as u32;

        Self {
            miner_uid,
            primary_gpu_model,
            gpu_counts,
            total_score,
            verification_count,
            last_updated: Utc::now(),
        }
    }

    /// Update the profile with new validation results
    pub fn update_with_validations(
        &mut self,
        executor_validations: &[ExecutorValidationResult],
        new_score: f64,
    ) {
        self.primary_gpu_model = GpuCategorizer::determine_primary_gpu_model(executor_validations);
        self.gpu_counts = GpuCategorizer::calculate_gpu_distribution(executor_validations);
        self.total_score = new_score;
        self.verification_count = executor_validations.len() as u32;
        self.last_updated = Utc::now();
    }

    /// Get the total number of GPUs across all models
    pub fn total_gpu_count(&self) -> u32 {
        self.gpu_counts.values().sum()
    }

    /// Check if this profile has any GPUs of a specific model
    pub fn has_gpu_model(&self, model: &str) -> bool {
        self.gpu_counts.contains_key(model)
    }

    /// Get the count of GPUs for a specific model
    pub fn get_gpu_count(&self, model: &str) -> u32 {
        self.gpu_counts.get(model).copied().unwrap_or(0)
    }

    /// Get GPU models sorted by count (descending)
    pub fn gpu_models_by_count(&self) -> Vec<(String, u32)> {
        let mut models: Vec<(String, u32)> = self
            .gpu_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        models.sort_by(|a, b| b.1.cmp(&a.1));
        models
    }
}

/// Executor validation result for GPU categorization
/// This is a simplified version focused on GPU information
#[derive(Debug, Clone)]
pub struct ExecutorValidationResult {
    pub executor_id: String,
    pub is_valid: bool,
    pub gpu_model: String,
    pub gpu_count: usize,
    pub gpu_memory_gb: u64,
    pub attestation_valid: bool,
    pub validation_timestamp: DateTime<Utc>,
}

impl ExecutorValidationResult {
    /// Create a new validation result for testing
    pub fn new_for_testing(
        executor_id: String,
        gpu_model: String,
        gpu_count: usize,
        is_valid: bool,
        attestation_valid: bool,
    ) -> Self {
        Self {
            executor_id,
            is_valid,
            gpu_model,
            gpu_count,
            gpu_memory_gb: 80, // Default 80GB
            attestation_valid,
            validation_timestamp: Utc::now(),
        }
    }
}
