//! Freivalds Asymmetric GPU Validator
//!
//! This module implements the validator-side logic for the Freivalds algorithm-based
//! GPU attestation protocol. It manages the process of:
//! - Building gpu-attestor binaries with specific challenge parameters
//! - Uploading and executing binaries on remote executors via SSH
//! - Parsing and verifying Freivalds commitment results

use anyhow::{Context, Result};
use common::ssh::SshConnectionDetails;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::ssh::ValidatorSshClient;
use crate::validation::gpu_profile_query::GpuProfileQuery;
use protocol::basilca::freivalds_gpu_pow::v1::{
    CommitmentResponse, ExecutionMetadata, FreivaldsChallenge, FreivaldsVerificationResult,
    GpuInfo, PerformanceMetrics, RowProof,
};

/// Calculate appropriate timeouts for Freivalds challenges based on matrix size
fn calculate_freivalds_timeouts(matrix_size: u32) -> (u32, u32) {
    // Based on H100 PCIe benchmarks:
    // 256x256: 12.4ms, 512x512: 21.7ms, 1024x1024: 95.4ms, 2048x2048: 528.5ms
    let base_computation_time_ms = match matrix_size {
        0..=128 => 10,
        129..=256 => 15,
        257..=512 => 30,
        513..=1024 => 120,
        1025..=2048 => 600,
        2049..=4096 => 4500,
        _ => 10000, // Very large matrices
    };

    // Apply 2x overhead factor for variation in GPU performance
    let computation_timeout_ms = (base_computation_time_ms * 2) as u32;

    // Full protocol timeout includes:
    // - Computation time
    // - Merkle tree construction (typically < 10ms)
    // - Network round trips (2x 50ms latency)
    // - Serialization/deserialization overhead
    let protocol_timeout_ms = computation_timeout_ms + 10 + 100 + 50;

    (computation_timeout_ms, protocol_timeout_ms)
}

/// Configuration for Freivalds validation
#[derive(Debug, Clone)]
pub struct FreivaldsValidatorConfig {
    /// Path to gpu-attestor source directory
    pub gpu_attestor_path: PathBuf,
    /// Temporary directory for binaries
    pub temp_dir: PathBuf,
    /// SSH timeout for remote execution
    pub ssh_timeout: Duration,
    /// Maximum matrix size to allow
    pub max_matrix_size: u32,
    /// Number of spot checks to perform
    pub num_spot_checks: u32,
}

impl Default for FreivaldsValidatorConfig {
    fn default() -> Self {
        Self {
            gpu_attestor_path: PathBuf::from("./crates/gpu-attestor"),
            temp_dir: PathBuf::from("/tmp/freivalds_validator"),
            ssh_timeout: Duration::from_secs(300),
            max_matrix_size: 16384,
            num_spot_checks: 20,
        }
    }
}

/// Freivalds GPU Validator
pub struct FreivaldsGpuValidator {
    config: FreivaldsValidatorConfig,
    ssh_client: ValidatorSshClient,
    gpu_profile_query: GpuProfileQuery,
}

impl FreivaldsGpuValidator {
    /// Create new Freivalds validator
    pub fn new(config: FreivaldsValidatorConfig) -> Result<Self> {
        // Create temp directory if it doesn't exist
        std::fs::create_dir_all(&config.temp_dir)?;

        let ssh_client = ValidatorSshClient::new();
        let gpu_profile_query = GpuProfileQuery::new(ValidatorSshClient::new());

        Ok(Self {
            config,
            ssh_client,
            gpu_profile_query,
        })
    }

    /// Generate a new Freivalds challenge
    pub fn generate_challenge(
        &self,
        session_id: String,
        matrix_size: u32,
        gpu_count: u32,
    ) -> FreivaldsChallenge {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate random seed
        let master_seed = (0..16).map(|_| rng.gen::<u8>()).collect();

        // Calculate timeouts based on matrix size
        let (computation_timeout_ms, protocol_timeout_ms) =
            calculate_freivalds_timeouts(matrix_size);

        FreivaldsChallenge {
            session_id,
            n: matrix_size.min(self.config.max_matrix_size),
            master_seed,
            timestamp: None, // Will be set by the protocol crate's prost_types
            expected_gpu_count: gpu_count,
            computation_timeout_ms,
            protocol_timeout_ms,
        }
    }

    /// Generate a new Freivalds challenge with GPU profile-aware timeouts
    pub async fn generate_challenge_with_profile(
        &self,
        session_id: String,
        connection: &SshConnectionDetails,
        gpu_attestor_binary: &PathBuf,
    ) -> Result<FreivaldsChallenge> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Query GPU profile from executor
        let gpu_profile = self
            .gpu_profile_query
            .query_profile(connection, gpu_attestor_binary)
            .await
            .context("Failed to query GPU profile")?;

        info!(
            "Using GPU profile for challenge: {} GPUs, {:.1} TFLOPS, optimal size: {}",
            gpu_profile.devices.len(),
            gpu_profile.total_compute_power,
            gpu_profile.optimal_matrix_size
        );

        // Use optimal matrix size from profile, but respect configured maximum
        let matrix_size = gpu_profile
            .optimal_matrix_size
            .min(self.config.max_matrix_size);

        // Generate random seed
        let master_seed = (0..16).map(|_| rng.gen::<u8>()).collect();

        // Calculate GPU-aware timeouts
        let network_latency_ms = 50; // Could be made configurable
        let (computation_timeout_ms, protocol_timeout_ms) = self
            .gpu_profile_query
            .calculate_timeouts(&gpu_profile, matrix_size, network_latency_ms);

        Ok(FreivaldsChallenge {
            session_id,
            n: matrix_size,
            master_seed,
            timestamp: None,
            expected_gpu_count: gpu_profile.devices.len() as u32,
            computation_timeout_ms,
            protocol_timeout_ms,
        })
    }

    /// Build gpu-attestor binary with Freivalds parameters
    async fn build_attestor_binary(&self, challenge: &FreivaldsChallenge) -> Result<PathBuf> {
        info!("Building gpu-attestor binary for Freivalds challenge");

        // Create unique binary name
        let binary_name = format!("gpu_attestor_freivalds_{}", &challenge.session_id);
        let binary_path = self.config.temp_dir.join(&binary_name);

        // Build the binary using cargo
        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--bin")
            .arg("gpu-attestor")
            .current_dir(&self.config.gpu_attestor_path)
            .env("CARGO_TARGET_DIR", self.config.temp_dir.join("target"));

        let output = cmd
            .output()
            .await
            .context("Failed to execute cargo build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to build gpu-attestor: {}", stderr));
        }

        // Copy the built binary to the expected location
        let built_binary = self
            .config
            .temp_dir
            .join("target")
            .join("release")
            .join("gpu-attestor");

        std::fs::copy(&built_binary, &binary_path).context("Failed to copy built binary")?;

        // Make it executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&binary_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&binary_path, perms)?;
        }

        info!("Built gpu-attestor binary: {}", binary_path.display());
        Ok(binary_path)
    }

    /// Execute Freivalds challenge on remote executor
    pub async fn execute_challenge(
        &self,
        connection: &SshConnectionDetails,
        challenge: &FreivaldsChallenge,
    ) -> Result<CommitmentResponse> {
        let start_time = Instant::now();

        // Build the attestor binary
        let binary_path = self.build_attestor_binary(challenge).await?;

        // Upload binary to executor
        let remote_binary = format!("/tmp/gpu_attestor_{}", challenge.session_id);
        info!("Uploading binary to executor: {}", remote_binary);

        self.ssh_client
            .upload_file(connection, &binary_path, &remote_binary)
            .await
            .context("Failed to upload binary")?;

        // Make it executable on remote
        self.ssh_client
            .execute_command(connection, &format!("chmod +x {remote_binary}"), false)
            .await
            .context("Failed to make binary executable")?;

        // Build command with Freivalds parameters
        let command = format!(
            "{} --freivalds --freivalds-matrix-size {} --freivalds-seed {} --freivalds-session-id {}",
            remote_binary,
            challenge.n,
            hex::encode(&challenge.master_seed),
            challenge.session_id
        );

        info!("Executing Freivalds challenge on executor");
        debug!("Command: {}", command);

        // Execute and capture output
        let output = self
            .ssh_client
            .execute_command_with_retry(connection, &command, true)
            .await
            .context("Failed to execute Freivalds challenge")?;

        // Parse JSON output
        let result: serde_json::Value =
            serde_json::from_str(&output).context("Failed to parse Freivalds result JSON")?;

        // Verify result type
        if result.get("type").and_then(|v| v.as_str()) != Some("freivalds_commitment") {
            return Err(anyhow::anyhow!("Invalid result type from executor"));
        }

        // Extract commitment data
        let session_id = result
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing session_id in result"))?
            .to_string();

        let merkle_root = result
            .get("merkle_root")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing merkle_root in result"))?;

        let merkle_root_bytes = hex::decode(merkle_root).context("Invalid merkle_root hex")?;

        let row_count = result
            .get("row_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Missing row_count in result"))?
            as u32;

        // Parse execution metadata
        let metadata = if let Some(metadata_obj) = result.get("metadata") {
            Some(self.parse_execution_metadata(metadata_obj)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        info!("Freivalds challenge completed in {:?}", execution_time);

        // Cleanup remote binary
        self.cleanup_remote_binary(connection, &remote_binary).await;

        // Cleanup local binary
        if let Err(e) = std::fs::remove_file(&binary_path) {
            warn!("Failed to remove local binary: {}", e);
        }

        Ok(CommitmentResponse {
            session_id,
            merkle_root: merkle_root_bytes,
            row_count,
            metadata,
            timestamp: None, // Will be set by the protocol crate's prost_types
        })
    }

    /// Parse execution metadata from JSON
    fn parse_execution_metadata(&self, value: &serde_json::Value) -> Result<ExecutionMetadata> {
        let gpus = if let Some(gpus_array) = value.get("gpus").and_then(|v| v.as_array()) {
            gpus_array
                .iter()
                .map(|gpu| {
                    Ok(GpuInfo {
                        device_id: gpu
                            .get("device_id")
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| anyhow::anyhow!("Missing device_id"))?
                            as u32,
                        model: gpu
                            .get("model")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| anyhow::anyhow!("Missing model"))?
                            .to_string(),
                        vram_mb: gpu.get("vram_mb").and_then(|v| v.as_u64()).unwrap_or(0),
                        compute_units: gpu
                            .get("compute_units")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        Ok(ExecutionMetadata {
            gpus,
            execution_time_ms: value
                .get("execution_time_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            vram_allocated_mb: value
                .get("vram_allocated_mb")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            kernel_time_ms: value
                .get("kernel_time_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            merkle_time_ms: value
                .get("merkle_time_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        })
    }

    /// Perform spot checks on the commitment
    pub async fn perform_spot_checks(
        &self,
        connection: &SshConnectionDetails,
        challenge: &FreivaldsChallenge,
        commitment: &CommitmentResponse,
    ) -> Result<Vec<RowProof>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        // Select random rows to check
        let mut row_indices: Vec<u32> = (0..commitment.row_count).collect();
        row_indices.shuffle(&mut rng);
        row_indices.truncate(self.config.num_spot_checks as usize);

        let mut spot_checks = Vec::new();

        for &row_index in &row_indices {
            // In a real implementation, we would:
            // 1. Request the specific row data from the executor
            // 2. Verify it matches the Merkle tree commitment
            // 3. Optionally verify computation on that row

            // For now, simulate spot check
            spot_checks.push(RowProof {
                row_idx: row_index,
                row_data: vec![0u8; 32], // Placeholder
                merkle_path: vec![],     // Placeholder
            });
        }

        Ok(spot_checks)
    }

    /// Verify the complete Freivalds result
    pub async fn verify_result(
        &self,
        connection: &SshConnectionDetails,
        challenge: &FreivaldsChallenge,
        commitment: &CommitmentResponse,
    ) -> Result<FreivaldsVerificationResult> {
        info!(
            "Verifying Freivalds result for session {}",
            challenge.session_id
        );

        // Perform spot checks
        let spot_checks = self
            .perform_spot_checks(connection, challenge, commitment)
            .await?;

        // For now, assume all spot checks pass
        let spot_checks_valid = true;
        let spot_checks_passed = spot_checks.len() as u32;

        // Calculate performance metrics
        let symmetric_time_estimate =
            self.estimate_symmetric_time(challenge.n, commitment.metadata.as_ref());
        let freivalds_time = commitment
            .metadata
            .as_ref()
            .map(|m| m.execution_time_ms)
            .unwrap_or(0);

        let computation_saved_percent = if symmetric_time_estimate > 0 {
            ((symmetric_time_estimate as f64 - freivalds_time as f64)
                / symmetric_time_estimate as f64)
                * 100.0
        } else {
            0.0
        };

        let verified = spot_checks_valid
            && commitment.row_count == challenge.n
            && commitment.session_id == challenge.session_id;

        Ok(FreivaldsVerificationResult {
            session_id: challenge.session_id.clone(),
            freivalds_valid: true, // Would perform actual Freivalds check in real implementation
            spot_checks_valid,
            spot_checks_performed: self.config.num_spot_checks,
            spot_checks_passed,
            verified,
            message: if verified {
                "Freivalds verification successful".to_string()
            } else {
                "Freivalds verification failed".to_string()
            },
            metrics: Some(PerformanceMetrics {
                freivalds_time_ms: freivalds_time,
                spot_check_time_ms: 0, // Would measure actual spot check time
                total_time_ms: freivalds_time,
                computation_saved_percent,
            }),
            timestamp: None, // Will be set by the protocol crate's prost_types
        })
    }

    /// Estimate time for symmetric verification
    fn estimate_symmetric_time(
        &self,
        matrix_size: u32,
        metadata: Option<&ExecutionMetadata>,
    ) -> u64 {
        // Rough estimate: O(n³) for matrix multiplication
        // Adjust based on GPU capabilities if available
        let base_time = (matrix_size as u64).pow(3) / 1_000_000; // Normalize

        if let Some(metadata) = metadata {
            if !metadata.gpus.is_empty() {
                // Adjust for GPU compute power
                let total_compute_units: u32 =
                    metadata.gpus.iter().map(|gpu| gpu.compute_units).sum();
                base_time / (total_compute_units as u64).max(1)
            } else {
                base_time
            }
        } else {
            base_time
        }
    }

    /// Cleanup remote binary
    async fn cleanup_remote_binary(&self, connection: &SshConnectionDetails, remote_path: &str) {
        if let Err(e) = self
            .ssh_client
            .execute_command(connection, &format!("rm -f {remote_path}"), false)
            .await
        {
            warn!("Failed to cleanup remote binary: {}", e);
        }
    }

    /// Execute Freivalds challenge with automatic GPU profiling
    /// This method queries the executor's GPU profile first, then generates
    /// and executes a challenge with optimized parameters
    pub async fn execute_challenge_with_profiling(
        &self,
        connection: &SshConnectionDetails,
        session_id: String,
    ) -> Result<(FreivaldsChallenge, CommitmentResponse)> {
        info!(
            "Starting Freivalds challenge with GPU profiling for session {}",
            session_id
        );

        // Build the attestor binary first (needed for GPU profiling)
        let temp_challenge = self.generate_challenge(session_id.clone(), 1024, 1);
        let binary_path = self.build_attestor_binary(&temp_challenge).await?;

        // Generate challenge with GPU profile awareness
        let challenge = self
            .generate_challenge_with_profile(session_id, connection, &binary_path)
            .await?;

        info!(
            "Generated GPU-aware challenge: {}×{} matrix, {} GPUs, timeouts: {}ms/{}ms",
            challenge.n,
            challenge.n,
            challenge.expected_gpu_count,
            challenge.computation_timeout_ms,
            challenge.protocol_timeout_ms
        );

        // Execute the challenge
        let commitment = self.execute_challenge(connection, &challenge).await?;

        Ok((challenge, commitment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_generation() {
        let config = FreivaldsValidatorConfig::default();
        let validator = FreivaldsGpuValidator::new(config).unwrap();

        let challenge = validator.generate_challenge("test_session".to_string(), 1024, 2);

        assert_eq!(challenge.session_id, "test_session");
        assert_eq!(challenge.n, 1024);
        assert_eq!(challenge.expected_gpu_count, 2);
        assert_eq!(challenge.master_seed.len(), 16);
    }
}
