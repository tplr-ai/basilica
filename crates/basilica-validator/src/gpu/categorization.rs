use basilica_common::identity::MinerUid;
use basilica_common::types::GpuCategory;
use chrono::{DateTime, Utc};
use sqlx::sqlite::SqliteRow;
use sqlx::Row;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MinerGpuProfile {
    pub miner_uid: MinerUid,
    pub gpu_counts: HashMap<String, u32>,
    pub total_score: f64,
    pub verification_count: u32,
    pub last_updated: DateTime<Utc>,
    pub last_successful_validation: Option<DateTime<Utc>>,
}

impl sqlx::FromRow<'_, SqliteRow> for MinerGpuProfile {
    fn from_row(row: &SqliteRow) -> Result<Self, sqlx::Error> {
        let miner_uid_val: i64 = row.get("miner_uid");
        let gpu_counts_json: String = row.get("gpu_counts_json");
        let total_score: f64 = row.get("total_score");
        let verification_count: i64 = row.get("verification_count");
        let last_updated_str: String = row.get("last_updated");
        let last_successful_validation_str: String = row.get("last_successful_validation");

        let gpu_counts: HashMap<String, u32> =
            serde_json::from_str(&gpu_counts_json).map_err(|e| sqlx::Error::ColumnDecode {
                index: "gpu_counts_json".to_string(),
                source: e.into(),
            })?;

        let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)
            .map_err(|e| sqlx::Error::ColumnDecode {
                index: "last_updated".to_string(),
                source: e.into(),
            })?
            .with_timezone(&Utc);

        let last_successful_validation = if last_successful_validation_str.is_empty() {
            None
        } else {
            Some(
                DateTime::parse_from_rfc3339(&last_successful_validation_str)
                    .map_err(|e| sqlx::Error::ColumnDecode {
                        index: "last_successful_validation".to_string(),
                        source: e.into(),
                    })?
                    .with_timezone(&Utc),
            )
        };

        Ok(Self {
            miner_uid: MinerUid::new(miner_uid_val as u16),
            gpu_counts,
            total_score,
            verification_count: verification_count as u32,
            last_updated,
            last_successful_validation,
        })
    }
}

pub struct GpuCategorizer;

impl GpuCategorizer {
    /// Convert model string to category enum
    pub fn model_to_category(model: &str) -> GpuCategory {
        // Use the FromStr implementation for consistency
        GpuCategory::from_str(model).unwrap()
    }

    /// Determine primary GPU model from validation results
    /// Calculate GPU model distribution for a miner
    pub fn calculate_gpu_distribution(
        node_validations: &[NodeValidationResult],
    ) -> HashMap<String, u32> {
        let mut gpu_counts = HashMap::new();
        let mut seen_nodes = std::collections::HashSet::new();

        // Count GPUs per unique node to avoid double-counting
        for validation in node_validations
            .iter()
            .filter(|v| v.is_valid && v.attestation_valid && v.gpu_count > 0)
        {
            // Only count each node once
            if seen_nodes.insert(&validation.node_id) {
                let category = GpuCategory::from_str(&validation.gpu_model).unwrap();
                let normalized = category.to_string();
                *gpu_counts.entry(normalized).or_insert(0) += validation.gpu_count as u32;
            }
        }

        gpu_counts
    }
}

impl MinerGpuProfile {
    /// Create a new GPU profile for a miner
    pub fn new(
        miner_uid: MinerUid,
        node_validations: &[NodeValidationResult],
        total_score: f64,
    ) -> Self {
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(node_validations);
        let verification_count = node_validations.len() as u32;

        Self {
            miner_uid,
            gpu_counts,
            total_score,
            verification_count,
            last_updated: Utc::now(),
            last_successful_validation: None,
        }
    }

    /// Update the profile with new validation results
    pub fn update_with_validations(
        &mut self,
        node_validations: &[NodeValidationResult],
        new_score: f64,
    ) {
        self.gpu_counts = GpuCategorizer::calculate_gpu_distribution(node_validations);
        self.total_score = new_score;
        self.verification_count = node_validations.len() as u32;
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

        models.sort_by_key(|m| std::cmp::Reverse(m.1));
        models
    }
}

/// Node validation result for GPU categorization
/// This is a simplified version focused on GPU information
#[derive(Debug, Clone)]
pub struct NodeValidationResult {
    pub node_id: String,
    pub is_valid: bool,
    pub gpu_model: String,
    pub gpu_count: usize,
    pub gpu_memory_gb: f64,
    pub attestation_valid: bool,
    pub validation_timestamp: DateTime<Utc>,
}

impl NodeValidationResult {
    /// Create a new validation result for testing
    pub fn new_for_testing(
        node_id: String,
        gpu_model: String,
        gpu_count: usize,
        is_valid: bool,
        attestation_valid: bool,
    ) -> Self {
        Self {
            node_id,
            is_valid,
            gpu_model,
            gpu_count,
            gpu_memory_gb: 80.0, // Default 80GB
            attestation_valid,
            validation_timestamp: Utc::now(),
        }
    }
}
