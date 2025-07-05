//! Generic GPU computation validator with VM-protected attestation
//!
//! This validator deploys a secure attestation binary and receives
//! PASS/FAIL results. All validation logic is hidden in the VM on the
//! executor's machine.

use anyhow::{Context, Result};
use common::ssh::SshConnectionDetails;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use crate::ssh::ValidatorSshClient;

/// Generic computational challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeChallenge {
    /// Unique session identifier
    pub session_id: String,
    /// Problem size parameter
    pub problem_size: u32,
    /// Random seed data
    pub seed_data: Vec<u8>,
    /// Challenge timestamp
    pub timestamp: Option<String>,
    /// Expected resource count
    pub resource_count: u32,
    /// Computation timeout in milliseconds
    pub computation_timeout_ms: u32,
    /// Protocol timeout in milliseconds
    pub protocol_timeout_ms: u32,
}

/// Attestation result from secure validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Whether attestation passed
    pub verified: bool,
    /// Status message
    pub message: String,
    /// Cryptographic proof (if verified)
    pub proof: Option<Vec<u8>>,
}

/// Configuration for VM-based secure validator
#[derive(Debug, Clone)]
pub struct SecureValidatorConfig {
    /// Path to secure attestor binary
    pub attestor_binary_path: PathBuf,
    /// SSH timeout for remote execution
    pub ssh_timeout: Duration,
    /// Maximum allowed execution time
    pub max_execution_time: Duration,
}

impl Default for SecureValidatorConfig {
    fn default() -> Self {
        Self {
            attestor_binary_path: PathBuf::from("./target/release/gpu-attestor"),
            ssh_timeout: Duration::from_secs(300),
            max_execution_time: Duration::from_secs(120),
        }
    }
}

/// Secure computational validator using VM protection
pub struct SecureValidator {
    config: SecureValidatorConfig,
    ssh_client: ValidatorSshClient,
}

impl SecureValidator {
    /// Create new secure validator
    pub fn new(config: SecureValidatorConfig) -> Result<Self> {
        if !config.attestor_binary_path.exists() {
            return Err(anyhow::anyhow!(
                "Secure attestor binary not found at: {:?}",
                config.attestor_binary_path
            ));
        }

        Ok(Self {
            config,
            ssh_client: ValidatorSshClient::new(),
        })
    }

    /// Validate a compute resource using VM-protected logic
    pub async fn validate_compute_resource(
        &self,
        connection: &SshConnectionDetails,
        problem_size: u32,
        expected_resource_count: u32,
    ) -> Result<bool> {
        let start_time = Instant::now();

        // Generate challenge
        let challenge = self.generate_challenge(problem_size, expected_resource_count)?;

        info!(
            "Starting secure validation for resource {}:{}, session: {}",
            connection.host, connection.port, challenge.session_id
        );

        // Deploy secure attestor binary
        self.deploy_binary(connection).await?;

        // Execute with challenge
        let result = self.execute_attestation(connection, &challenge).await?;

        let elapsed = start_time.elapsed();
        info!(
            "Validation completed in {:?}: verified={}, message={}",
            elapsed, result.verified, result.message
        );

        // Optionally verify proof
        if let Some(proof) = &result.proof {
            debug!("Received proof: {} bytes", proof.len());
            // Could verify proof signature here
        }

        Ok(result.verified)
    }

    /// Generate a new challenge
    fn generate_challenge(
        &self,
        problem_size: u32,
        expected_resource_count: u32,
    ) -> Result<ComputeChallenge> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate random session ID
        let session_id = format!("secure_session_{}", rng.gen::<u64>());

        // Generate random seed data
        let mut seed_data = vec![0u8; 16];
        rng.fill(&mut seed_data[..]);

        // Calculate timeouts based on problem size
        let (computation_timeout_ms, protocol_timeout_ms) = calculate_timeouts(problem_size);

        Ok(ComputeChallenge {
            session_id,
            problem_size,
            seed_data,
            timestamp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    .to_string(),
            ),
            resource_count: expected_resource_count,
            computation_timeout_ms,
            protocol_timeout_ms,
        })
    }

    /// Deploy secure attestor binary to resource
    async fn deploy_binary(&self, connection: &SshConnectionDetails) -> Result<()> {
        info!("Deploying secure attestor binary to resource");

        // Read binary
        let binary_data = std::fs::read(&self.config.attestor_binary_path)
            .context("Failed to read secure attestor binary")?;

        // Write binary data to temporary file
        let temp_path = std::env::temp_dir().join("secure-attestor-binary");
        std::fs::write(&temp_path, &binary_data).context("Failed to write binary to temp file")?;

        // Upload to resource
        self.ssh_client
            .upload_file(connection, &temp_path, "/tmp/secure-attestor")
            .await
            .context("Failed to upload binary")?;

        // Set executable permissions
        self.ssh_client
            .execute_command(connection, "chmod +x /tmp/secure-attestor", false)
            .await
            .context("Failed to set executable permissions")?;

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);

        debug!("Binary deployed successfully");
        Ok(())
    }

    /// Execute attestation on resource
    async fn execute_attestation(
        &self,
        connection: &SshConnectionDetails,
        challenge: &ComputeChallenge,
    ) -> Result<AttestationResult> {
        // Serialize challenge to JSON
        let challenge_json =
            serde_json::to_string(challenge).context("Failed to serialize challenge")?;

        // Build command
        let command = format!("/tmp/secure-attestor --challenge '{challenge_json}'");

        debug!("Executing command: {}", command);

        // Execute with timeout
        let output = tokio::time::timeout(
            self.config.max_execution_time,
            self.ssh_client.execute_command(connection, &command, true),
        )
        .await
        .context("Execution timeout")?
        .context("SSH execution failed")?;

        // Parse result from stdout
        let result: AttestationResult =
            serde_json::from_str(&output).context("Failed to parse attestation result")?;

        Ok(result)
    }
}

/// Calculate appropriate timeouts based on problem size
fn calculate_timeouts(problem_size: u32) -> (u32, u32) {
    let base_computation_time_ms = match problem_size {
        0..=128 => 10,
        129..=256 => 15,
        257..=512 => 30,
        513..=1024 => 120,
        1025..=2048 => 600,
        2049..=4096 => 4500,
        _ => 10000,
    };

    let computation_timeout_ms = (base_computation_time_ms * 2) as u32;
    let protocol_timeout_ms = computation_timeout_ms + 200;

    (computation_timeout_ms, protocol_timeout_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_challenge() {
        let config = SecureValidatorConfig::default();
        let validator = SecureValidator {
            config,
            ssh_client: ValidatorSshClient::new(),
        };

        let challenge = validator.generate_challenge(1024, 2).unwrap();

        assert_eq!(challenge.problem_size, 1024);
        assert_eq!(challenge.resource_count, 2);
        assert_eq!(challenge.seed_data.len(), 16);
        assert!(challenge.session_id.starts_with("secure_session_"));
    }

    #[test]
    fn test_calculate_timeouts() {
        let test_cases = vec![
            (256, 30, 230),
            (512, 60, 260),
            (1024, 240, 440),
            (2048, 1200, 1400),
        ];

        for (size, expected_comp, expected_proto) in test_cases {
            let (comp, proto) = calculate_timeouts(size);
            assert_eq!(comp, expected_comp, "Computation timeout for size {size}");
            assert_eq!(proto, expected_proto, "Protocol timeout for size {size}");
        }
    }

    #[test]
    fn test_attestation_result_serialization() {
        let result = AttestationResult {
            verified: true,
            message: "All checks passed".to_string(),
            proof: Some(vec![0x01, 0x02, 0x03]),
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: AttestationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.verified, result.verified);
        assert_eq!(parsed.message, result.message);
        assert_eq!(parsed.proof, result.proof);
    }
}
