//! System monitoring configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// System monitoring configuration
///
/// Handles all system monitoring related settings following the
/// Single Responsibility Principle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System info update interval
    pub update_interval: Duration,

    /// GPU monitoring enabled
    pub enable_gpu_monitoring: bool,

    /// Network monitoring enabled
    pub enable_network_monitoring: bool,

    /// Memory monitoring enabled
    pub enable_memory_monitoring: bool,

    /// CPU monitoring enabled
    pub enable_cpu_monitoring: bool,

    /// Maximum CPU usage percentage allowed
    pub max_cpu_usage: f32,

    /// Maximum memory usage percentage allowed
    pub max_memory_usage: f32,

    /// Maximum GPU memory usage percentage allowed
    pub max_gpu_memory_usage: f32,

    /// Minimum available disk space in GB
    pub min_disk_space_gb: u64,

    /// Enable metrics recording
    pub enable_metrics_recording: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(5),
            enable_gpu_monitoring: true,
            enable_network_monitoring: true,
            enable_memory_monitoring: true,
            enable_cpu_monitoring: true,
            max_cpu_usage: 90.0,
            max_memory_usage: 90.0,
            max_gpu_memory_usage: 90.0,
            min_disk_space_gb: 10,
            enable_metrics_recording: true,
        }
    }
}

/// System configuration validation trait
pub trait SystemConfigValidation {
    fn validate_usage_limits(&self) -> Result<(), String>;
    fn validate_monitoring_settings(&self) -> Result<(), String>;
    fn usage_warnings(&self) -> Vec<String>;
}

impl SystemConfigValidation for SystemConfig {
    fn validate_usage_limits(&self) -> Result<(), String> {
        if self.max_cpu_usage <= 0.0 || self.max_cpu_usage > 100.0 {
            return Err(format!(
                "CPU usage must be between 0 and 100, got: {}",
                self.max_cpu_usage
            ));
        }

        if self.max_memory_usage <= 0.0 || self.max_memory_usage > 100.0 {
            return Err(format!(
                "Memory usage must be between 0 and 100, got: {}",
                self.max_memory_usage
            ));
        }

        if self.max_gpu_memory_usage <= 0.0 || self.max_gpu_memory_usage > 100.0 {
            return Err(format!(
                "GPU memory usage must be between 0 and 100, got: {}",
                self.max_gpu_memory_usage
            ));
        }

        Ok(())
    }

    fn validate_monitoring_settings(&self) -> Result<(), String> {
        if self.update_interval.as_secs() == 0 {
            return Err("Update interval must be greater than 0".to_string());
        }

        if self.min_disk_space_gb == 0 {
            return Err("Minimum disk space must be greater than 0".to_string());
        }

        Ok(())
    }

    fn usage_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.max_cpu_usage > 95.0 {
            warnings.push("Very high CPU usage limit may affect system stability".to_string());
        }

        if self.max_memory_usage > 95.0 {
            warnings.push("Very high memory usage limit may affect system stability".to_string());
        }

        if self.max_gpu_memory_usage > 95.0 {
            warnings
                .push("Very high GPU memory usage limit may affect system stability".to_string());
        }

        warnings
    }
}
