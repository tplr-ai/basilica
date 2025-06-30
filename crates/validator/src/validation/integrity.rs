//! Binary integrity verification for gpu-attestor
//!
//! Provides SHA-256 checksum validation and binary verification capabilities
//! to ensure the integrity of the gpu-attestor binary before and after upload.

use super::types::{ValidationError, ValidationResult};
use crate::ssh::ValidatorSshClient;
use common::ssh::SshConnectionDetails;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, error, info, warn};

/// Binary integrity checker with checksum validation
#[derive(Debug, Clone)]
pub struct BinaryIntegrityChecker {
    /// Expected checksums mapping: binary_version -> sha256_hash
    expected_checksums: HashMap<String, String>,
    /// Cache of computed checksums to avoid recomputation
    checksum_cache: HashMap<String, (String, SystemTime)>,
    /// Cache TTL for computed checksums
    cache_ttl: Duration,
}

impl BinaryIntegrityChecker {
    /// Create a new binary integrity checker
    pub fn new() -> Self {
        Self {
            expected_checksums: HashMap::new(),
            checksum_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Add an expected checksum for a binary version
    pub fn add_expected_checksum(&mut self, version: String, checksum: String) {
        self.expected_checksums.insert(version, checksum);
    }

    /// Calculate SHA-256 checksum of a local binary file
    pub async fn calculate_file_checksum(&mut self, file_path: &Path) -> ValidationResult<String> {
        info!("Calculating SHA-256 checksum for: {}", file_path.display());

        if !file_path.exists() {
            return Err(ValidationError::BinaryNotFound(file_path.to_path_buf()));
        }

        // Check cache first
        let file_path_str = file_path.to_string_lossy().to_string();
        if let Some((cached_checksum, computed_at)) = self.checksum_cache.get(&file_path_str) {
            if computed_at.elapsed().unwrap_or(Duration::MAX) < self.cache_ttl {
                debug!("Using cached checksum for {}", file_path.display());
                return Ok(cached_checksum.clone());
            }
        }

        // Read file and compute checksum
        let file_contents = fs::read(file_path)
            .await
            .map_err(ValidationError::IoError)?;

        let mut hasher = Sha256::new();
        hasher.update(&file_contents);
        let checksum = hex::encode(hasher.finalize());

        // Update cache
        self.checksum_cache
            .insert(file_path_str, (checksum.clone(), SystemTime::now()));

        debug!("Computed SHA-256: {} for {}", checksum, file_path.display());
        Ok(checksum)
    }

    /// Verify local binary integrity against expected checksum
    pub async fn verify_local_binary(
        &mut self,
        file_path: &Path,
        expected_version: &str,
    ) -> ValidationResult<bool> {
        info!(
            "Verifying local binary integrity for: {}",
            file_path.display()
        );

        let computed_checksum = self.calculate_file_checksum(file_path).await?;

        if let Some(expected_checksum) = self.expected_checksums.get(expected_version) {
            let is_valid = computed_checksum == *expected_checksum;
            if is_valid {
                info!(
                    "Binary integrity verification passed for version: {}",
                    expected_version
                );
            } else {
                warn!(
                    "Binary integrity verification failed. Expected: {}, Got: {}",
                    expected_checksum, computed_checksum
                );
            }
            Ok(is_valid)
        } else {
            warn!(
                "No expected checksum found for version: {}",
                expected_version
            );
            Ok(false)
        }
    }

    /// Verify remote binary integrity via SSH
    pub async fn verify_remote_binary(
        &self,
        ssh_client: &ValidatorSshClient,
        ssh_details: &SshConnectionDetails,
        remote_path: &str,
    ) -> ValidationResult<String> {
        info!("Verifying remote binary integrity at: {}", remote_path);

        let checksum_command = format!("sha256sum {remote_path} | cut -d' ' -f1");

        match ssh_client
            .execute_command(ssh_details, &checksum_command, true)
            .await
        {
            Ok(output) => {
                let checksum = output.trim().to_string();
                debug!("Remote binary checksum: {}", checksum);

                if checksum.len() == 64 && checksum.chars().all(|c| c.is_ascii_hexdigit()) {
                    Ok(checksum)
                } else {
                    Err(ValidationError::AttestationValidationFailed(format!(
                        "Invalid remote checksum format: {checksum}"
                    )))
                }
            }
            Err(e) => {
                error!("Failed to compute remote binary checksum: {}", e);
                Err(ValidationError::ExecutionFailed(format!(
                    "Remote checksum calculation failed: {e}"
                )))
            }
        }
    }

    /// Compare local and remote binary checksums
    pub async fn compare_checksums(
        &mut self,
        local_path: &Path,
        ssh_client: &ValidatorSshClient,
        ssh_details: &SshConnectionDetails,
        remote_path: &str,
    ) -> ValidationResult<bool> {
        info!("Comparing local and remote binary checksums");

        let local_checksum = self.calculate_file_checksum(local_path).await?;
        let remote_checksum = self
            .verify_remote_binary(ssh_client, ssh_details, remote_path)
            .await?;

        let is_match = local_checksum == remote_checksum;

        if is_match {
            info!("Binary checksums match - upload integrity verified");
        } else {
            warn!(
                "Binary checksum mismatch! Local: {}, Remote: {}",
                local_checksum, remote_checksum
            );
        }

        Ok(is_match)
    }
}

impl Default for BinaryIntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_calculate_file_checksum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test content").unwrap();

        let mut checker = BinaryIntegrityChecker::new();
        let checksum = checker
            .calculate_file_checksum(temp_file.path())
            .await
            .unwrap();

        // Verify checksum is 64-character hex string
        assert_eq!(checksum.len(), 64);
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn test_verify_local_binary() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test content").unwrap();

        let mut checker = BinaryIntegrityChecker::new();

        // Calculate expected checksum
        let expected_checksum = checker
            .calculate_file_checksum(temp_file.path())
            .await
            .unwrap();
        checker.add_expected_checksum("test_version".to_string(), expected_checksum);

        // Verify binary
        let is_valid = checker
            .verify_local_binary(temp_file.path(), "test_version")
            .await
            .unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_checksum_cache() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test content").unwrap();

        let mut checker = BinaryIntegrityChecker::new();

        // First calculation
        let checksum1 = checker
            .calculate_file_checksum(temp_file.path())
            .await
            .unwrap();

        // Second calculation should use cache
        let checksum2 = checker
            .calculate_file_checksum(temp_file.path())
            .await
            .unwrap();

        assert_eq!(checksum1, checksum2);
        assert_eq!(checker.checksum_cache.len(), 1);
    }
}
