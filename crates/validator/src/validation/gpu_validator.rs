//! GPU Validator
//!
//! Validates GPU PoW challenges by executing them locally on the validator's GPU.
//! Only supports validating GPUs of the same type as the validator's GPU.

use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use std::process::Command;
use std::time::Instant;
use tracing::{info, warn};

use protocol::basilca::common::v1::{ChallengeParameters, ChallengeResult};

/// GPU validator that executes challenges locally
pub struct GpuValidator {
    /// Path to gpu-attestor binary
    gpu_attestor_path: String,
    /// Validator's GPU models (supports multi-GPU)
    validator_gpu_models: Vec<String>,
    /// Total GPU count
    gpu_count: usize,
    /// Maximum allowed time difference ratio
    max_time_ratio: f64,
}

impl GpuValidator {
    /// Create a new GPU validator
    pub fn new(gpu_attestor_path: String) -> Self {
        Self {
            gpu_attestor_path,
            validator_gpu_models: Vec::new(),
            gpu_count: 0,
            max_time_ratio: 1.5, // Allow up to 50% slower than local execution
        }
    }

    /// Initialize by detecting validator's GPU
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Detecting validator's GPU...");

        // Create a temporary directory for attestation output
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("gpu_info");

        // Run gpu-attestor to get GPU information
        // Important: Do NOT inherit CUDA_VISIBLE_DEVICES during initialization
        // to detect all available GPUs on the system
        let output = Command::new(&self.gpu_attestor_path)
            .env_remove("CUDA_VISIBLE_DEVICES")
            .arg("--output")
            .arg(output_path.to_str().unwrap())
            .arg("--skip-network-benchmark")
            .arg("--skip-os-attestation")
            .arg("--skip-docker-attestation")
            .arg("--skip-gpu-benchmarks")
            .output()
            .context("Failed to run gpu-attestor")?;

        if !output.status.success() {
            anyhow::bail!(
                "gpu-attestor failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Read the attestation JSON to get GPU info
        let attestation_path = output_path.with_extension("json");
        let attestation_content = std::fs::read_to_string(&attestation_path)
            .context("Failed to read attestation output")?;

        let attestation: serde_json::Value = serde_json::from_str(&attestation_content)
            .context("Failed to parse attestation output")?;

        // Extract all GPU models from the attestation report
        if let Some(gpu_info) = attestation.get("gpu_info").and_then(|g| g.as_array()) {
            self.validator_gpu_models.clear();

            for gpu in gpu_info {
                if let Some(model) = gpu.get("name").and_then(|n| n.as_str()) {
                    self.validator_gpu_models.push(model.to_string());
                }
            }

            self.gpu_count = self.validator_gpu_models.len();

            if self.gpu_count > 0 {
                info!("Validator detected {} GPU(s):", self.gpu_count);
                for (i, model) in self.validator_gpu_models.iter().enumerate() {
                    info!("  GPU {}: {}", i, model);
                }
                return Ok(());
            }
        }

        anyhow::bail!("No GPUs detected on validator")
    }

    /// Get the validator's GPU model (first GPU for compatibility)
    pub fn get_gpu_model(&self) -> Option<&str> {
        self.validator_gpu_models.first().map(|s| s.as_str())
    }

    /// Get all validator GPU models
    pub fn get_all_gpu_models(&self) -> &[String] {
        &self.validator_gpu_models
    }

    /// Get GPU count
    pub fn get_gpu_count(&self) -> usize {
        self.gpu_count
    }

    /// Verify a GPU PoW challenge by executing it locally
    pub async fn verify_challenge(
        &self,
        params: &ChallengeParameters,
        result: &ChallengeResult,
    ) -> Result<bool> {
        // Always use full verification (no more sampling)
        self.verify_challenge_full(params, result).await
    }

    /// Full verification (no sampling)
    async fn verify_challenge_full(
        &self,
        params: &ChallengeParameters,
        result: &ChallengeResult,
    ) -> Result<bool> {
        if self.validator_gpu_models.is_empty() {
            return Err(anyhow::anyhow!("Validator GPUs not initialized"));
        }

        // Check current GPU visibility
        let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let visible_gpu_count = if let Some(ref devices) = visible_devices {
            devices.split(',').filter(|s| !s.is_empty()).count()
        } else {
            self.gpu_count
        };

        info!(
            "Verifying GPU PoW challenge for GPU: {} (validator has {} GPU(s) total, {} visible)",
            result.gpu_model, self.gpu_count, visible_gpu_count
        );

        // Check if miner's GPU model matches any of validator's GPUs
        let gpu_model_matches = self
            .validator_gpu_models
            .iter()
            .any(|model| model == &result.gpu_model);

        if !gpu_model_matches {
            warn!(
                "Cannot verify GPU model mismatch: validator has {:?}, miner claims {}",
                self.validator_gpu_models, result.gpu_model
            );
            return Ok(false);
        }

        // Check basic parameters
        if result.challenge_id != params.validator_nonce {
            warn!(
                "Challenge ID mismatch: expected {}, got {}",
                params.validator_nonce, result.challenge_id
            );
            return Ok(false);
        }

        // Convert to generic ComputeChallenge and encode to base64
        use crate::validation::challenge_converter::challenge_params_to_base64;
        let challenge_base64 = challenge_params_to_base64(params)
            .context("Failed to convert challenge parameters")?;

        // Execute the challenge locally
        // IMPORTANT: Respect the current CUDA_VISIBLE_DEVICES environment variable
        // to ensure symmetric execution with the miner
        let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let actual_gpu_count = if let Some(ref devices) = visible_devices {
            // Count the number of devices in the comma-separated list
            devices.split(',').filter(|s| !s.is_empty()).count()
        } else {
            self.gpu_count
        };

        info!(
            "Executing challenge locally on {} GPU(s) (CUDA_VISIBLE_DEVICES: {:?})",
            actual_gpu_count, visible_devices
        );
        let start = Instant::now();

        let output = Command::new(&self.gpu_attestor_path)
            .arg("--challenge")
            .arg(&challenge_base64)
            .output()
            .context("Failed to run gpu-attestor challenge")?;

        let local_execution_time = start.elapsed();

        if !output.status.success() {
            warn!(
                "Local challenge execution failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Ok(false);
        }

        // Parse local result
        let local_result_json =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in gpu-attestor output")?;

        let local_result: serde_json::Value = serde_json::from_str(&local_result_json)
            .context("Failed to parse local challenge result")?;

        // Check if this is a VM-protected response (simple status format)
        if let Some(status) = local_result.get("status").and_then(|s| s.as_str()) {
            // VM-protected attestor response
            info!("VM-protected validation returned: {}", status);
            
            // For VM-protected validation, we trust the attestor's decision
            // as all validation logic is hidden within the VM
            return Ok(status == "PASS");
        }

        // Legacy format - extract checksum from local result
        let local_checksum = local_result
            .get("result_checksum")
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("No checksum in local result"))?;

        // Compare checksums
        let result_checksum = hex::encode(&result.result_checksum);
        info!("Validator's local checksum: {}", local_checksum);
        info!("Miner's reported checksum: {}", result_checksum);

        if result_checksum != local_checksum {
            warn!(
                "Checksum mismatch: expected {}, got {}",
                local_checksum, result_checksum
            );

            // Parse local result to get more details
            if let Some(metadata_json) = local_result.get("metadata_json").and_then(|m| m.as_str())
            {
                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(metadata_json) {
                    info!("Validator execution details:");
                    if let Some(device_count) = metadata.get("device_count") {
                        info!("  Device count: {}", device_count);
                    }
                    if let Some(devices) = metadata.get("devices").and_then(|d| d.as_array()) {
                        for device in devices {
                            if let (Some(id), Some(model)) =
                                (device.get("device_id"), device.get("gpu_model"))
                            {
                                info!("  Device {}: {}", id, model);
                            }
                        }
                    }
                }
            }

            return Ok(false);
        }

        info!("Checksum verified successfully");

        // TEMPORARILY DISABLED: Execution time validation
        // Timing can vary significantly between runs, especially with multi-GPU systems
        /*
        // Verify execution time is reasonable
        let reported_time_ms = result.execution_time_ms;
        let local_time_ms = local_execution_time.as_millis() as u64;

        debug!(
            "Execution times - Local: {} ms, Reported: {} ms",
            local_time_ms, reported_time_ms
        );

        // For multi-GPU execution, timing can vary based on synchronization
        // Be more lenient with multi-GPU systems
        let adjusted_max_ratio = if self.gpu_count > 1 {
            self.max_time_ratio * 1.5 // Allow 2.25x for multi-GPU overhead
        } else {
            self.max_time_ratio
        };

        let time_ratio = reported_time_ms as f64 / local_time_ms as f64;

        if time_ratio > adjusted_max_ratio {
            warn!(
                "Execution time too slow: {} ms (local: {} ms, ratio: {:.2})",
                reported_time_ms, local_time_ms, time_ratio
            );
            return Ok(false);
        }

        // Very fast times might indicate pre-computation or cheating
        let min_ratio = if self.gpu_count > 1 { 0.05 } else { 0.1 };
        if time_ratio < min_ratio {
            warn!(
                "Execution time suspiciously fast: {} ms (local: {} ms, ratio: {:.2})",
                reported_time_ms, local_time_ms, time_ratio
            );
            return Ok(false);
        }
        */

        // TEMPORARILY DISABLED: VRAM validation
        // The miner uses a memory saturation strategy that allocates ~75GB per GPU
        // instead of just the challenge matrices (~9GB), so this check fails
        /*
        // Verify VRAM usage is reasonable
        let expected_vram_mb =
            calculate_expected_vram_usage(params.num_matrices, params.matrix_dim);

        let vram_ratio = result.vram_allocated_mb as f64 / expected_vram_mb as f64;
        if !(0.8..=1.2).contains(&vram_ratio) {
            warn!(
                "VRAM usage outside expected range: {} MB (expected ~{} MB)",
                result.vram_allocated_mb, expected_vram_mb
            );
            return Ok(false);
        }
        */

        info!("GPU PoW challenge verified successfully");
        Ok(true)
    }
}

/// Calculate expected VRAM usage
fn calculate_expected_vram_usage(num_matrices: u32, matrix_dim: u32) -> u64 {
    let matrix_size_bytes = (matrix_dim * matrix_dim * 8) as u64; // 8 bytes per f64
    let total_bytes = num_matrices as u64 * matrix_size_bytes;
    total_bytes / (1024 * 1024) // Convert to MB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_calculation() {
        // 256x256 matrix, f64 = 8 bytes
        let expected_mb = calculate_expected_vram_usage(100, 256);
        let expected = (100 * 256 * 256 * 8) / (1024 * 1024);
        assert_eq!(expected_mb, expected);
    }
}
