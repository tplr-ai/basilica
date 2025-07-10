//! Binary Validation Metrics
//!
//! Metrics collection for binary validation operations.

use std::sync::Arc;
use tokio::sync::Mutex;

/// Binary validation metrics collection
#[derive(Debug, Clone)]
pub struct BinaryValidationMetrics {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub average_execution_time_ms: f64,
    pub average_binary_score: f64,
    pub average_combined_score: f64,
    pub ssh_test_failures: u64,
    pub binary_upload_failures: u64,
    pub binary_execution_failures: u64,
    pub binary_parsing_failures: u64,
}

impl BinaryValidationMetrics {
    pub fn new() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            failed_validations: 0,
            average_execution_time_ms: 0.0,
            average_binary_score: 0.0,
            average_combined_score: 0.0,
            ssh_test_failures: 0,
            binary_upload_failures: 0,
            binary_execution_failures: 0,
            binary_parsing_failures: 0,
        }
    }

    pub fn record_validation_attempt(&mut self) {
        self.total_validations += 1;
    }

    pub fn record_successful_validation(&mut self, score: f64, execution_time_ms: u64) {
        self.successful_validations += 1;
        self.update_average_score(score);
        self.update_average_execution_time(execution_time_ms);
    }

    pub fn record_failed_validation(&mut self, failure_type: ValidationFailureType) {
        self.failed_validations += 1;
        match failure_type {
            ValidationFailureType::SshTest => self.ssh_test_failures += 1,
            ValidationFailureType::BinaryUpload => self.binary_upload_failures += 1,
            ValidationFailureType::BinaryExecution => self.binary_execution_failures += 1,
            ValidationFailureType::BinaryParsing => self.binary_parsing_failures += 1,
        }
    }

    fn update_average_score(&mut self, score: f64) {
        let count = self.successful_validations as f64;
        self.average_binary_score = ((self.average_binary_score * (count - 1.0)) + score) / count;
    }

    fn update_average_execution_time(&mut self, execution_time_ms: u64) {
        let count = self.successful_validations as f64;
        let time_ms = execution_time_ms as f64;
        self.average_execution_time_ms =
            ((self.average_execution_time_ms * (count - 1.0)) + time_ms) / count;
    }

    pub fn get_success_rate(&self) -> f64 {
        if self.total_validations == 0 {
            return 0.0;
        }
        self.successful_validations as f64 / self.total_validations as f64
    }
}

impl Default for BinaryValidationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum ValidationFailureType {
    SshTest,
    BinaryUpload,
    BinaryExecution,
    BinaryParsing,
}

/// Thread-safe binary validation metrics
pub type BinaryValidationMetricsContainer = Arc<Mutex<BinaryValidationMetrics>>;

/// Create a new metrics container
pub fn create_metrics_container() -> BinaryValidationMetricsContainer {
    Arc::new(Mutex::new(BinaryValidationMetrics::new()))
}
