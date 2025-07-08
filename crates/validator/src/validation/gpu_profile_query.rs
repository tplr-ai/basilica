//! GPU Profile Query Module
//!
//! Provides functionality for validators to query executor GPU profiles
//! via SSH and use the information to configure adaptive timeouts.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info};

use crate::ssh::ValidatorSshClient;
use common::ssh::SshConnectionDetails;

/// GPU profile data structure matching gpu-attestor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub devices: Vec<GpuDeviceProfile>,
    pub total_compute_power: f64,
    pub total_memory_bandwidth: f64,
    pub optimal_matrix_size: u32,
    pub performance_class: PerformanceClass,
    pub topology: SystemTopology,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceProfile {
    pub device_id: u32,
    pub model: String,
    pub compute_capability: (u32, u32),
    pub memory_gb: f64,
    pub bandwidth_gbps: f64,
    pub sm_count: u32,
    pub cores_per_sm: u32,
    pub tflops_fp32: f64,
    pub tflops_fp16: f64,
    pub tflops_tensor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PerformanceClass {
    DataCenter,
    Professional,
    Consumer,
    Entry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTopology {
    pub total_gpus: usize,
    pub nvlink_present: bool,
    pub peer_access_matrix: Vec<Vec<bool>>,
    pub numa_node_affinity: Vec<i32>,
}

/// GPU profile cache entry
#[derive(Debug, Clone)]
struct ProfileCacheEntry {
    profile: GpuProfile,
    timestamp: std::time::Instant,
}

/// GPU profile query client
pub struct GpuProfileQuery {
    ssh_client: ValidatorSshClient,
    cache: std::sync::Mutex<std::collections::HashMap<String, ProfileCacheEntry>>,
    cache_duration: Duration,
    gpu_attestor_path: String,
}

impl GpuProfileQuery {
    /// Create a new GPU profile query client
    pub fn new(ssh_client: ValidatorSshClient) -> Self {
        Self {
            ssh_client,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            cache_duration: Duration::from_secs(3600), // 1 hour cache
            gpu_attestor_path: "/tmp/gpu-attestor".to_string(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        ssh_client: ValidatorSshClient,
        cache_duration: Duration,
        gpu_attestor_path: String,
    ) -> Self {
        Self {
            ssh_client,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            cache_duration,
            gpu_attestor_path,
        }
    }

    /// Query GPU profile from executor
    pub async fn query_profile(
        &self,
        connection: &SshConnectionDetails,
        gpu_attestor_binary: &Path,
    ) -> Result<GpuProfile> {
        let cache_key = format!(
            "{}@{}:{}",
            connection.username, connection.host, connection.port
        );

        // Check cache first
        if let Some(profile) = self.get_cached_profile(&cache_key) {
            debug!("Using cached GPU profile for {}", cache_key);
            return Ok(profile);
        }

        info!("Querying GPU profile from executor at {}", connection.host);

        // Upload gpu-attestor binary if needed
        let remote_binary_path = format!("{}/gpu-attestor", self.gpu_attestor_path);

        // Check if binary exists and is executable
        let check_cmd = format!("test -x {remote_binary_path} && echo 'exists' || echo 'missing'");
        let check_result = self
            .ssh_client
            .execute_command(connection, &check_cmd, true)
            .await
            .context("Failed to check gpu-attestor binary")?;

        if check_result.trim() == "missing" {
            info!("Uploading gpu-attestor binary to executor");

            // Create directory if needed
            let mkdir_cmd = format!("mkdir -p {}", self.gpu_attestor_path);
            self.ssh_client
                .execute_command(connection, &mkdir_cmd, false)
                .await
                .context("Failed to create directory for gpu-attestor")?;

            // Upload binary
            self.ssh_client
                .upload_file(connection, gpu_attestor_binary, &remote_binary_path)
                .await
                .context("Failed to upload gpu-attestor binary")?;

            // Make it executable
            let chmod_cmd = format!("chmod +x {remote_binary_path}");
            self.ssh_client
                .execute_command(connection, &chmod_cmd, false)
                .await
                .context("Failed to make gpu-attestor executable")?;
        }

        // Execute GPU profile query
        let profile_cmd = format!("{remote_binary_path} --detect-gpus-json 2>/dev/null | tail -1");
        let profile_json = self
            .ssh_client
            .execute_command_with_retry(connection, &profile_cmd, true)
            .await
            .context("Failed to execute GPU profile query")?;

        // Parse the JSON output
        let profile: GpuProfile =
            serde_json::from_str(&profile_json).context("Failed to parse GPU profile JSON")?;

        // Validate the profile
        if profile.devices.is_empty() {
            return Err(anyhow::anyhow!("No GPUs detected on executor"));
        }

        info!(
            "Successfully queried GPU profile: {} GPUs, {:.1} TFLOPS, {:?} class",
            profile.devices.len(),
            profile.total_compute_power,
            profile.performance_class
        );

        // Cache the profile
        self.cache_profile(&cache_key, profile.clone());

        Ok(profile)
    }

    /// Get cached profile if available and not expired
    fn get_cached_profile(&self, cache_key: &str) -> Option<GpuProfile> {
        let cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get(cache_key) {
            if entry.timestamp.elapsed() < self.cache_duration {
                return Some(entry.profile.clone());
            }
        }
        None
    }

    /// Cache a GPU profile
    fn cache_profile(&self, cache_key: &str, profile: GpuProfile) {
        let mut cache = self.cache.lock().unwrap();
        cache.insert(
            cache_key.to_string(),
            ProfileCacheEntry {
                profile,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    /// Clear the profile cache
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Calculate GPU-aware timeouts based on profile
    pub fn calculate_timeouts(
        &self,
        profile: &GpuProfile,
        matrix_size: u32,
        network_latency_ms: u32,
    ) -> (u32, u32) {
        // Get GPU-specific performance estimate
        let best_gflops = self.estimate_matrix_multiply_gflops(profile, matrix_size);

        // Calculate operations: 2n³ for matrix multiply
        let operations = 2.0 * (matrix_size as f64).powi(3);
        let operations_giga = operations / 1e9; // Convert to giga-operations
        let base_time_seconds = operations_giga / best_gflops; // Time = work / rate
        let base_time_ms = (base_time_seconds * 1000.0) as u32;

        // Adjust for multi-GPU with parallel efficiency
        let gpu_count = profile.devices.len() as f32;
        let parallel_efficiency = if profile.topology.nvlink_present {
            0.85
        } else {
            0.75
        };
        let adjusted_time_ms = (base_time_ms as f32 / (gpu_count * parallel_efficiency)) as u32;

        // Add overhead for Merkle tree construction
        let merkle_overhead_ms = matrix_size / 100;

        // Apply safety factor based on performance class
        let safety_factor = match profile.performance_class {
            PerformanceClass::DataCenter => 1.5,
            PerformanceClass::Professional => 2.0,
            PerformanceClass::Consumer => 2.5,
            PerformanceClass::Entry => 3.0,
        };

        let computation_timeout_ms =
            ((adjusted_time_ms + merkle_overhead_ms) as f32 * safety_factor) as u32;
        let protocol_timeout_ms = computation_timeout_ms + (network_latency_ms * 2) + 100; // Network + overhead

        info!(
            "Calculated GPU-aware timeouts for {}×{} on {} {}(s): compute={}ms, protocol={}ms",
            matrix_size,
            matrix_size,
            gpu_count,
            profile
                .devices
                .first()
                .map(|d| &d.model)
                .unwrap_or(&"Unknown".to_string()),
            computation_timeout_ms,
            protocol_timeout_ms
        );

        (computation_timeout_ms, protocol_timeout_ms)
    }

    /// Estimate matrix multiplication GFLOPS based on GPU model
    fn estimate_matrix_multiply_gflops(&self, profile: &GpuProfile, matrix_size: u32) -> f64 {
        // Use the best GPU in the system
        let best_device = profile
            .devices
            .iter()
            .max_by(|a, b| a.tflops_fp32.partial_cmp(&b.tflops_fp32).unwrap())
            .unwrap_or(&profile.devices[0]);

        // Estimate based on model and size
        let efficiency = match matrix_size {
            0..=256 => 0.3, // Small matrices have poor efficiency
            257..=512 => 0.5,
            513..=1024 => 0.7,
            1025..=2048 => 0.8,
            _ => 0.85, // Large matrices approach peak performance
        };

        // Convert TFLOPS to GFLOPS and apply efficiency
        best_device.tflops_fp32 * 1000.0 * efficiency
    }
}

/// Helper to create default GPU profile for fallback
impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            devices: vec![GpuDeviceProfile {
                device_id: 0,
                model: "Unknown GPU".to_string(),
                compute_capability: (7, 0),
                memory_gb: 8.0,
                bandwidth_gbps: 500.0,
                sm_count: 40,
                cores_per_sm: 64,
                tflops_fp32: 10.0,
                tflops_fp16: 20.0,
                tflops_tensor: 40.0,
            }],
            total_compute_power: 10.0,
            total_memory_bandwidth: 500.0,
            optimal_matrix_size: 1024,
            performance_class: PerformanceClass::Entry,
            topology: SystemTopology {
                total_gpus: 1,
                nvlink_present: false,
                peer_access_matrix: vec![vec![false]],
                numa_node_affinity: vec![0],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_calculation() {
        let profile = GpuProfile {
            devices: vec![GpuDeviceProfile {
                device_id: 0,
                model: "NVIDIA H100 PCIe".to_string(),
                compute_capability: (9, 0),
                memory_gb: 80.0,
                bandwidth_gbps: 3000.0,
                sm_count: 114,
                cores_per_sm: 128,
                tflops_fp32: 67.0,
                tflops_fp16: 134.0,
                tflops_tensor: 1979.0,
            }],
            total_compute_power: 67.0,
            total_memory_bandwidth: 3000.0,
            optimal_matrix_size: 4096,
            performance_class: PerformanceClass::DataCenter,
            topology: SystemTopology {
                total_gpus: 1,
                nvlink_present: false,
                peer_access_matrix: vec![vec![false]],
                numa_node_affinity: vec![0],
            },
        };

        let query = GpuProfileQuery::new(ValidatorSshClient::new());
        let (compute_timeout, protocol_timeout) = query.calculate_timeouts(&profile, 1024, 50);

        // For H100 with 1024x1024 matrix:
        // Very fast computation with this GPU, expect low timeout
        assert!(compute_timeout > 10 && compute_timeout < 50);
        assert!(protocol_timeout > compute_timeout);
    }

    #[test]
    fn test_performance_scaling() {
        let profile = GpuProfile { performance_class: PerformanceClass::DataCenter, ..Default::default() };

        let query = GpuProfileQuery::new(ValidatorSshClient::new());

        // Test that larger matrices have longer timeouts
        let (timeout_256, _) = query.calculate_timeouts(&profile, 256, 50);
        let (timeout_512, _) = query.calculate_timeouts(&profile, 512, 50);
        let (timeout_1024, _) = query.calculate_timeouts(&profile, 1024, 50);

        assert!(timeout_512 > timeout_256);
        assert!(timeout_1024 > timeout_512);
    }
}
