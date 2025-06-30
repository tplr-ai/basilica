//! Hardware Validator Implementation
//!
//! Main orchestrator for executor hardware validation.
//! Implements the complete validation flow while being decoupled from SSH details.

use super::{
    attestor::GpuAttestorIntegration, integrity::BinaryIntegrityChecker,
    key_manager::EphemeralKeyManager, signature_verifier::P256SignatureVerifier, types::*,
};
use crate::bittensor_core::WeightSetter;
use crate::persistence::{entities::VerificationLog, SimplePersistence};
use crate::ssh::{ExecutorSshDetails, ValidatorSshClient};
use common::identity::{ExecutorId, MinerUid};
use common::ssh::SshConnectionConfig;
use sqlx::Row;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, error, info, warn};

/// Hardware validator with enhanced cryptographic verification
pub struct HardwareValidator {
    ssh_client: ValidatorSshClient,
    persistence: Arc<SimplePersistence>,
    weight_setter: Option<Arc<WeightSetter>>,
    config: ValidationConfig,
    integrity_checker: BinaryIntegrityChecker,
    #[allow(dead_code)]
    key_manager: EphemeralKeyManager,
    signature_verifier: P256SignatureVerifier,
    attestor_integration: GpuAttestorIntegration,
}

impl HardwareValidator {
    /// Create a new hardware validator
    pub async fn new(
        config: ValidationConfig,
        persistence: Arc<SimplePersistence>,
    ) -> ValidationResult<Self> {
        Self::with_weight_setter(config, persistence, None).await
    }

    /// Create a new hardware validator with weight setter for scoring integration
    pub async fn with_weight_setter(
        config: ValidationConfig,
        persistence: Arc<SimplePersistence>,
        weight_setter: Option<Arc<WeightSetter>>,
    ) -> ValidationResult<Self> {
        // Create SSH client with appropriate configuration
        let ssh_config = SshConnectionConfig {
            execution_timeout: config.execution_timeout,
            max_transfer_size: config.max_transfer_size,
            cleanup_remote_files: config.cleanup_remote_files,
            ..Default::default()
        };
        let ssh_client = ValidatorSshClient::with_config(ssh_config);

        // Initialize integrity checker
        let integrity_checker = BinaryIntegrityChecker::new();

        // Initialize key manager
        let key_rotation_config = KeyRotationConfig::default();
        let mut key_manager = EphemeralKeyManager::new(key_rotation_config);
        key_manager
            .initialize()
            .await
            .map_err(|e| ValidationError::ConfigError(e.to_string()))?;

        // Initialize signature verifier
        let signature_verifier = P256SignatureVerifier::new();

        // Initialize gpu-attestor integration
        let attestor_integration = GpuAttestorIntegration::new(
            config
                .gpu_attestor_binary_path
                .to_string_lossy()
                .to_string(),
        );

        Ok(Self {
            ssh_client,
            persistence,
            weight_setter,
            config,
            integrity_checker,
            key_manager,
            signature_verifier,
            attestor_integration,
        })
    }

    /// Validate executor hardware
    ///
    /// This implements the complete hardware validation protocol:
    /// 1. SSH Connection: Test connection
    /// 2. Binary Transfer: Upload gpu-attestor binary via SCP
    /// 3. Remote Execution: Execute gpu-attestor on executor
    /// 4. Attestation Receipt: Download attestation files
    /// 5. Signature Validation: Verify P256 ECDSA signature
    /// 6. Database Storage: Store verified specifications
    pub async fn validate_executor(
        &mut self,
        executor_details: &ExecutorSshDetails,
    ) -> ValidationResult<AttestationResult> {
        info!(
            "Starting hardware validation for executor: {}",
            executor_details.executor_id()
        );

        match self.execute_validation_protocol(executor_details).await {
            Ok(result) => {
                info!(
                    "Hardware validation completed successfully for executor: {}",
                    executor_details.executor_id()
                );
                Ok(result)
            }
            Err(e) => {
                error!(
                    "Hardware validation failed for executor {}: {}",
                    executor_details.executor_id(),
                    e
                );

                let start_time = SystemTime::now();
                let validation_duration = start_time.elapsed().unwrap_or_default();

                Ok(AttestationResult {
                    executor_id: executor_details.executor_id().clone(),
                    validated_at: SystemTime::now(),
                    is_valid: false,
                    hardware_specs: None,
                    signature: None,
                    error_message: Some(e.to_string()),
                    validation_duration,
                })
            }
        }
    }

    /// Validate executor by ID (lookup SSH details from persistence)
    pub async fn validate_executor_by_id(
        &mut self,
        executor_id: &ExecutorId,
    ) -> ValidationResult<AttestationResult> {
        info!("Looking up SSH details for executor: {}", executor_id);

        // Look up executor SSH details from the miner_executors table
        let ssh_details = self.lookup_executor_ssh_details(executor_id).await?;

        match ssh_details {
            Some(details) => {
                info!("Found SSH details for executor: {}", executor_id);
                self.validate_executor(&details).await
            }
            None => {
                warn!("No SSH details found for executor: {}", executor_id);
                Err(ValidationError::ConfigError(format!(
                    "No SSH configuration found for executor: {executor_id}"
                )))
            }
        }
    }

    /// Validate executor with connection details
    pub async fn validate_executor_with_connection(
        &mut self,
        executor_id: ExecutorId,
        host: String,
        username: String,
        private_key_path: PathBuf,
        port: Option<u16>,
    ) -> ValidationResult<AttestationResult> {
        let executor_details =
            ExecutorSshDetails::new(executor_id, host, username, port, private_key_path, None);

        self.validate_executor(&executor_details).await
    }

    /// Internal implementation of the SSH-based validation protocol
    ///
    /// ## Security Design Rationale
    ///
    /// This implementation uses SSH-based validation instead of gRPC for several critical security reasons:
    ///
    /// 1. **Validator-Controlled Binary**: The validator uploads its own gpu-attestor binary with
    ///    an embedded public key. This prevents executors from using tampered binaries that could
    ///    produce fake attestations.
    ///
    /// 2. **Direct Hardware Access**: SSH ensures the attestation runs directly on the actual hardware,
    ///    making it much harder to intercept or spoof the attestation process compared to a
    ///    gRPC-based approach where responses could be proxied or modified.
    ///
    /// 3. **Ephemeral Execution**: The binary runs in a temporary directory and is cleaned up after
    ///    execution, preventing persistent tampering or analysis of the attestation binary.
    ///
    /// 4. **Tamper Resistance**: The entire attestation process (hardware detection, benchmarking,
    ///    signing) happens atomically on the remote machine, preventing man-in-the-middle attacks
    ///    that could occur with a multi-step gRPC protocol.
    ///
    /// 5. **Replay Protection**: Each attestation includes a validator-provided nonce, ensuring
    ///    that old attestations cannot be replayed.
    ///
    /// The trade-off is increased complexity and SSH access requirements, but this provides
    /// significantly stronger security guarantees for hardware attestation in a potentially
    /// adversarial environment.
    ///
    /// ## Protocol Steps:
    /// 1. Test SSH connection
    /// 2. Upload validator's gpu-attestor binary
    /// 3. Execute binary with nonce
    /// 4. Download attestation files
    /// 5. Verify signature and nonce
    /// 6. Store results
    /// 7. Cleanup remote files
    async fn execute_validation_protocol(
        &mut self,
        executor_details: &ExecutorSshDetails,
    ) -> ValidationResult<AttestationResult> {
        let start_time = SystemTime::now();

        // Step 1: Test SSH connection
        info!("Step 1: Testing SSH connection");
        self.ssh_client
            .test_connection(executor_details.connection())
            .await?;

        // Step 2: Upload gpu-attestor binary
        info!("Step 2: Uploading gpu-attestor binary");
        let remote_binary_path = format!("{}/gpu-attestor", self.config.remote_work_dir);
        self.upload_gpu_attestor_binary(executor_details, &remote_binary_path)
            .await?;

        // Step 3: Execute gpu-attestor remotely
        info!("Step 3: Executing gpu-attestor remotely");
        let (attestation_files, nonce) = self
            .execute_gpu_attestor(executor_details, &remote_binary_path)
            .await?;

        // Step 4: Download attestation files
        info!("Step 4: Downloading attestation files");
        let local_attestation_files = self
            .download_attestation_files(executor_details, &attestation_files)
            .await?;

        // Step 5: Validate attestation
        info!("Step 5: Validating attestation");
        let (is_valid, hardware_specs, signature) = self
            .validate_attestation(&local_attestation_files, &nonce)
            .await?;

        // Step 6: Store validation result in database
        if is_valid {
            info!("Step 6: Storing validation result in database");

            // Create verification log from attestation result
            let verification_log = VerificationLog::new(
                executor_details.executor_id().to_string(),
                "validator_hotkey".to_string(), // TODO: Get from config
                "attestation".to_string(),
                0.0, // Score will be calculated later by scoring service
                true,
                serde_json::to_value(hardware_specs.as_ref().unwrap())
                    .unwrap_or(serde_json::Value::Null),
                start_time.elapsed().unwrap_or_default().as_millis() as i64,
                None,
            );

            // Store in database
            if let Err(e) = self
                .persistence
                .create_verification_log(&verification_log)
                .await
            {
                warn!("Failed to store verification log in database: {}", e);
                // Don't fail the validation, just log the error
            } else {
                info!(
                    "Successfully stored verification log for executor: {}",
                    executor_details.executor_id()
                );

                // Process validation result for scoring if weight setter is available
                if let Some(weight_setter) = &self.weight_setter {
                    if let Some(miner_uid) = self
                        .get_miner_uid_for_executor(executor_details.executor_id())
                        .await?
                    {
                        if let Err(e) = weight_setter
                            .process_executor_validation(
                                miner_uid,
                                executor_details.executor_id().clone(),
                                &verification_log,
                            )
                            .await
                        {
                            warn!("Failed to process validation for scoring: {}", e);
                        } else {
                            info!(
                                "Successfully processed validation for miner {} scoring",
                                miner_uid.as_u16()
                            );
                        }
                    }
                }
            }

            // Also store executor details
            self.store_executor_details(executor_details, hardware_specs.as_ref().unwrap())
                .await?;
        } else {
            info!("Step 6: Storing failed validation result in database");

            // Store failed validation
            let verification_log = VerificationLog::new(
                executor_details.executor_id().to_string(),
                "validator_hotkey".to_string(), // TODO: Get from config
                "attestation".to_string(),
                0.0, // Score for failed validation
                false,
                serde_json::Value::Null,
                start_time.elapsed().unwrap_or_default().as_millis() as i64,
                None, // Error message is in AttestationResult.error_message
            );

            // Store in database
            if let Err(e) = self
                .persistence
                .create_verification_log(&verification_log)
                .await
            {
                warn!("Failed to store failed verification log in database: {}", e);
            } else {
                info!(
                    "Successfully stored failed verification log for executor: {}",
                    executor_details.executor_id()
                );
            }
        }

        // Step 7: Cleanup remote files
        info!("Step 7: Cleaning up remote files");
        let mut cleanup_files = vec![remote_binary_path];
        cleanup_files.extend(attestation_files);
        self.ssh_client
            .cleanup_remote_files(executor_details.connection(), &cleanup_files)
            .await?;

        let validation_duration = start_time.elapsed().unwrap_or_default();

        Ok(AttestationResult {
            executor_id: executor_details.executor_id().clone(),
            validated_at: SystemTime::now(),
            is_valid,
            hardware_specs,
            signature,
            error_message: None,
            validation_duration,
        })
    }

    /// Upload gpu-attestor binary to executor with integrity verification
    async fn upload_gpu_attestor_binary(
        &mut self,
        executor_details: &ExecutorSshDetails,
        remote_path: &str,
    ) -> ValidationResult<()> {
        info!("Uploading gpu-attestor binary with integrity verification");

        // Verify binary integrity before upload
        let binary_checksum = self
            .integrity_checker
            .calculate_file_checksum(&self.config.gpu_attestor_binary_path)
            .await?;

        info!("Local binary checksum: {}", binary_checksum);

        // Create remote work directory
        let mkdir_cmd = format!("mkdir -p {}", self.config.remote_work_dir);
        self.ssh_client
            .execute_command(executor_details.connection(), &mkdir_cmd, false)
            .await?;

        // Upload binary
        self.ssh_client
            .upload_file(
                executor_details.connection(),
                &self.config.gpu_attestor_binary_path,
                remote_path,
            )
            .await?;

        // Verify upload integrity
        let upload_integrity_verified = self
            .integrity_checker
            .compare_checksums(
                &self.config.gpu_attestor_binary_path,
                &self.ssh_client,
                executor_details.connection(),
                remote_path,
            )
            .await?;

        if !upload_integrity_verified {
            return Err(ValidationError::IntegrityCheckFailed(
                "Binary upload integrity verification failed".to_string(),
            ));
        }

        // Make binary executable
        let chmod_cmd = format!("chmod +x {remote_path}");
        self.ssh_client
            .execute_command(executor_details.connection(), &chmod_cmd, false)
            .await?;

        info!("Binary upload and integrity verification completed successfully");
        Ok(())
    }

    /// Execute gpu-attestor on remote executor
    async fn execute_gpu_attestor(
        &self,
        executor_details: &ExecutorSshDetails,
        binary_path: &str,
    ) -> ValidationResult<(Vec<String>, String)> {
        let output_dir = &self.config.remote_work_dir;
        let attestation_base_path = format!("{output_dir}/attestation");
        let attestation_file = format!("{output_dir}/attestation.json");
        let signature_file = format!("{output_dir}/attestation.sig");
        let pubkey_file = format!("{output_dir}/attestation.pub");

        // Generate a nonce for replay protection
        let nonce = uuid::Uuid::new_v4().to_string();

        // Execute gpu-attestor with full attestation capabilities
        // SSH-based execution ensures this runs on the actual hardware, preventing spoofing
        // The nonce prevents replay attacks by ensuring each attestation is unique
        // Note: gpu-attestor automatically creates .json, .sig, and .pub files
        let cmd = format!(
            "cd {} && {} --executor-id {} --output {} --nonce {}",
            output_dir,
            binary_path,
            executor_details.executor_id(),
            attestation_base_path,
            nonce
        );

        info!("Executing gpu-attestor: {}", cmd);
        self.ssh_client
            .execute_command_with_retry(executor_details.connection(), &cmd, false)
            .await?;

        // Verify files were created
        let verify_cmd = format!(
            "test -f {attestation_file} && test -f {signature_file} && test -f {pubkey_file} && echo 'files_exist'"
        );

        let output = self
            .ssh_client
            .execute_command(executor_details.connection(), &verify_cmd, true)
            .await?;

        if !output.trim().contains("files_exist") {
            return Err(ValidationError::ConfigError(
                "Attestation files were not created".to_string(),
            ));
        }

        Ok((vec![attestation_file, signature_file, pubkey_file], nonce))
    }

    /// Download attestation files from executor
    async fn download_attestation_files(
        &self,
        executor_details: &ExecutorSshDetails,
        remote_files: &[String],
    ) -> ValidationResult<Vec<PathBuf>> {
        let mut local_files = Vec::new();

        for remote_file in remote_files {
            let filename = Path::new(remote_file).file_name().ok_or_else(|| {
                ValidationError::ConfigError("Invalid remote file path".to_string())
            })?;

            let local_file = std::env::temp_dir().join(format!(
                "{}_{}_{}",
                executor_details.executor_id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                filename.to_string_lossy()
            ));

            self.ssh_client
                .download_file(executor_details.connection(), remote_file, &local_file)
                .await?;
            local_files.push(local_file);
        }

        Ok(local_files)
    }

    /// Validate attestation files and signature with P256 ECDSA verification
    async fn validate_attestation(
        &mut self,
        local_files: &[PathBuf],
        expected_nonce: &str,
    ) -> ValidationResult<(bool, Option<HardwareSpecs>, Option<String>)> {
        info!("Validating attestation files with cryptographic verification");

        if local_files.len() < 3 {
            return Err(ValidationError::AttestationValidationFailed(
                "Missing attestation files".to_string(),
            ));
        }

        let mut attestation_file = None;
        let mut signature_file = None;
        let mut pubkey_file = None;

        // Identify files by extension
        for file in local_files {
            let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if filename.contains("attestation.json") {
                attestation_file = Some(file);
            } else if filename.contains("attestation.sig") {
                signature_file = Some(file);
            } else if filename.contains("attestation.pub") {
                pubkey_file = Some(file);
            }
        }

        let attestation_path = attestation_file.ok_or_else(|| {
            ValidationError::AttestationValidationFailed("attestation.json not found".to_string())
        })?;

        let signature_path = signature_file.ok_or_else(|| {
            ValidationError::AttestationValidationFailed("attestation.sig not found".to_string())
        })?;

        let pubkey_path = pubkey_file.ok_or_else(|| {
            ValidationError::AttestationValidationFailed("attestation.pub not found".to_string())
        })?;

        // Read files
        let attestation_content = fs::read_to_string(attestation_path).await.map_err(|e| {
            ValidationError::AttestationValidationFailed(format!("Failed to read attestation: {e}"))
        })?;

        let signature_bytes = fs::read(signature_path).await.map_err(|e| {
            ValidationError::AttestationValidationFailed(format!("Failed to read signature: {e}"))
        })?;
        let signature_content = hex::encode(signature_bytes);

        // Read public key in PEM format (gpu-attestor now saves as PEM)
        let pubkey_content = fs::read_to_string(pubkey_path).await.map_err(|e| {
            ValidationError::AttestationValidationFailed(format!("Failed to read public key: {e}"))
        })?;

        // Verify P256 ECDSA signature
        info!("Performing P256 ECDSA signature verification");
        let signature_result = self
            .signature_verifier
            .verify_attestation_signature(&attestation_content, &signature_content, &pubkey_content)
            .map_err(|e| ValidationError::SignatureVerificationFailed(e.to_string()))?;

        if !signature_result.is_valid {
            warn!(
                "P256 ECDSA signature verification failed: {:?}",
                signature_result.error_message
            );

            // Cleanup local files
            for file in local_files {
                if let Err(e) = fs::remove_file(file).await {
                    warn!("Failed to cleanup local file {}: {}", file.display(), e);
                }
            }

            return Ok((false, None, None));
        }

        info!("P256 ECDSA signature verification successful");

        // Parse attestation using gpu-attestor integration
        let temp_dir = std::env::temp_dir();
        let temp_attestation_path = temp_dir.join("temp_attestation.json");
        fs::copy(attestation_path, &temp_attestation_path)
            .await
            .map_err(ValidationError::IoError)?;

        let attestation_report = self
            .attestor_integration
            .parse_attestation_file(temp_attestation_path.to_str().unwrap())
            .await
            .map_err(|e| ValidationError::AttestationValidationFailed(e.to_string()))?;

        // Verify the nonce matches what we sent
        if attestation_report.validator_nonce.as_ref() != Some(&expected_nonce.to_string()) {
            warn!(
                "Nonce mismatch - expected: {}, got: {:?}",
                expected_nonce, attestation_report.validator_nonce
            );

            // Cleanup files
            for file in local_files {
                if let Err(e) = fs::remove_file(file).await {
                    warn!("Failed to cleanup local file {}: {}", file.display(), e);
                }
            }
            if let Err(e) = fs::remove_file(&temp_attestation_path).await {
                warn!("Failed to cleanup temp file: {}", e);
            }

            return Err(ValidationError::AttestationValidationFailed(
                "Nonce mismatch - possible replay attack".to_string(),
            ));
        }

        // Validate attestation structure
        let structure_valid = self
            .attestor_integration
            .validate_attestation_structure(&attestation_report)
            .map_err(|e| ValidationError::AttestationValidationFailed(e.to_string()))?;

        if !structure_valid {
            warn!("Attestation report structure validation failed");

            // Cleanup files
            for file in local_files {
                if let Err(e) = fs::remove_file(file).await {
                    warn!("Failed to cleanup local file {}: {}", file.display(), e);
                }
            }
            if let Err(e) = fs::remove_file(&temp_attestation_path).await {
                warn!("Failed to cleanup temp file: {}", e);
            }

            return Ok((false, None, None));
        }

        // Convert to internal hardware specs format
        let hardware_specs = self
            .attestor_integration
            .convert_to_hardware_specs(&attestation_report)
            .map_err(|e| ValidationError::AttestationValidationFailed(e.to_string()))?;

        // Cleanup local files
        for file in local_files {
            if let Err(e) = fs::remove_file(file).await {
                warn!("Failed to cleanup local file {}: {}", file.display(), e);
            }
        }
        if let Err(e) = fs::remove_file(&temp_attestation_path).await {
            warn!("Failed to cleanup temp file: {}", e);
        }

        info!("Complete attestation validation successful");
        Ok((true, Some(hardware_specs), Some(signature_content)))
    }

    /// Store executor details in persistence
    async fn store_executor_details(
        &self,
        executor_details: &ExecutorSshDetails,
        hardware_specs: &HardwareSpecs,
    ) -> ValidationResult<()> {
        info!(
            "Storing executor details for {} with {} GPUs, {} GB RAM",
            executor_details.executor_id(),
            hardware_specs.gpu.len(),
            hardware_specs.memory.total_mb / 1024
        );

        // Check if executor already exists in miner_executors table
        let existing_executor = sqlx::query("SELECT id FROM miner_executors WHERE executor_id = ?")
            .bind(executor_details.executor_id().to_string())
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        let now = chrono::Utc::now().to_rfc3339();
        let gpu_specs_json = serde_json::to_string(&hardware_specs.gpu)
            .map_err(ValidationError::SerializationError)?;
        let cpu_specs_json = serde_json::to_string(&hardware_specs.cpu)
            .map_err(ValidationError::SerializationError)?;
        let location = format!(
            "{}:{}",
            executor_details.connection().host,
            executor_details.connection().port
        );

        if existing_executor.is_some() {
            // Update existing executor
            sqlx::query(
                "UPDATE miner_executors SET 
                 gpu_count = ?, gpu_specs = ?, cpu_specs = ?, 
                 location = ?, status = 'validated', last_health_check = ?, updated_at = ?
                 WHERE executor_id = ?",
            )
            .bind(hardware_specs.gpu.len() as i64)
            .bind(&gpu_specs_json)
            .bind(&cpu_specs_json)
            .bind(&location)
            .bind(&now)
            .bind(&now)
            .bind(executor_details.executor_id().to_string())
            .execute(self.persistence.pool())
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

            info!(
                "Updated existing executor details for: {}",
                executor_details.executor_id()
            );
        } else {
            // Insert new executor (associate with unknown miner for now)
            let executor_uuid = uuid::Uuid::new_v4().to_string();
            let miner_id = "unknown_miner"; // Will be updated when miner registers

            sqlx::query(
                "INSERT INTO miner_executors 
                 (id, miner_id, executor_id, grpc_address, gpu_count, gpu_specs, cpu_specs, 
                  location, status, last_health_check, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'validated', ?, ?, ?)",
            )
            .bind(&executor_uuid)
            .bind(miner_id)
            .bind(executor_details.executor_id().to_string())
            .bind(&location) // Use SSH host:port as fallback grpc address
            .bind(hardware_specs.gpu.len() as i64)
            .bind(&gpu_specs_json)
            .bind(&cpu_specs_json)
            .bind(&location)
            .bind(&now)
            .bind(&now)
            .bind(&now)
            .execute(self.persistence.pool())
            .await
            .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

            info!(
                "Inserted new executor details for: {}",
                executor_details.executor_id()
            );
        }

        Ok(())
    }

    /// Get validation history for an executor
    pub async fn get_validation_history(
        &self,
        executor_id: &ExecutorId,
    ) -> ValidationResult<Vec<AttestationResult>> {
        info!(
            "Retrieving validation history for executor: {}",
            executor_id
        );

        let rows = sqlx::query(
            "SELECT executor_id, validator_hotkey, verification_type, timestamp, score, 
                    success, details, duration_ms, error_message, created_at
             FROM verification_logs 
             WHERE executor_id = ? 
             ORDER BY timestamp DESC 
             LIMIT 100",
        )
        .bind(executor_id.to_string())
        .fetch_all(self.persistence.pool())
        .await
        .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        let mut history = Vec::new();
        for row in rows {
            let timestamp_str: String = row.get("timestamp");
            let validated_at = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                .map_err(|e| ValidationError::ParseError(e.to_string()))?
                .into();

            let success: i64 = row.get("success");
            let is_valid = success != 0;

            let details_str: String = row.get("details");
            let hardware_specs = if is_valid && !details_str.is_empty() {
                serde_json::from_str::<HardwareSpecs>(&details_str).ok()
            } else {
                None
            };

            let duration_ms: i64 = row.get("duration_ms");
            let validation_duration = Duration::from_millis(duration_ms as u64);

            let error_message: Option<String> = row.get("error_message");

            let attestation_result = AttestationResult {
                executor_id: executor_id.clone(),
                validated_at,
                is_valid,
                hardware_specs,
                signature: None, // Signature not stored in verification_logs currently
                error_message,
                validation_duration,
            };

            history.push(attestation_result);
        }

        info!(
            "Retrieved {} validation records for executor: {}",
            history.len(),
            executor_id
        );
        Ok(history)
    }

    /// Check if executor has valid recent attestation
    pub async fn has_valid_attestation(
        &self,
        executor_id: &ExecutorId,
        max_age: std::time::Duration,
    ) -> ValidationResult<bool> {
        info!(
            "Checking for valid recent attestation for executor: {} (max age: {:?})",
            executor_id, max_age
        );

        let cutoff_time = SystemTime::now() - max_age;
        let cutoff_rfc3339 = chrono::DateTime::<chrono::Utc>::from(cutoff_time).to_rfc3339();

        let row = sqlx::query(
            "SELECT COUNT(*) as count FROM verification_logs 
             WHERE executor_id = ? 
               AND success = 1 
               AND timestamp >= ? 
             ORDER BY timestamp DESC 
             LIMIT 1",
        )
        .bind(executor_id.to_string())
        .bind(&cutoff_rfc3339)
        .fetch_one(self.persistence.pool())
        .await
        .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        let count: i64 = row.get("count");
        let has_valid = count > 0;

        if has_valid {
            info!(
                "Found valid recent attestation for executor: {}",
                executor_id
            );
        } else {
            info!(
                "No valid recent attestation found for executor: {}",
                executor_id
            );
        }

        Ok(has_valid)
    }

    /// Get miner UID for an executor from the database
    async fn get_miner_uid_for_executor(
        &self,
        executor_id: &ExecutorId,
    ) -> ValidationResult<Option<MinerUid>> {
        let query = r#"
            SELECT miner_id FROM miner_executors WHERE executor_id = ?
        "#;

        let row = sqlx::query(query)
            .bind(executor_id.to_string())
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| {
                ValidationError::DatabaseError(format!("Failed to query miner for executor: {e}"))
            })?;

        if let Some(row) = row {
            let miner_id: String = row.get("miner_id");
            // Parse miner UID from format "miner_123"
            if let Some(uid_str) = miner_id.strip_prefix("miner_") {
                if let Ok(uid) = uid_str.parse::<u16>() {
                    return Ok(Some(MinerUid::new(uid)));
                }
            }
        }

        Ok(None)
    }

    /// Lookup SSH details for an executor from persistence
    async fn lookup_executor_ssh_details(
        &self,
        executor_id: &ExecutorId,
    ) -> ValidationResult<Option<ExecutorSshDetails>> {
        debug!("Looking up SSH details for executor: {}", executor_id);

        // Query miner_executors table for executor information
        let row = sqlx::query(
            "SELECT me.executor_id, me.grpc_address, me.location 
             FROM miner_executors me 
             WHERE me.executor_id = ? 
             LIMIT 1",
        )
        .bind(executor_id.to_string())
        .fetch_optional(self.persistence.pool())
        .await
        .map_err(|e| ValidationError::DatabaseError(e.to_string()))?;

        if let Some(row) = row {
            let grpc_address: String = row.get("grpc_address");
            let location: Option<String> = row.get("location");

            // Parse host and port from grpc_address or location
            let (host, port) = if let Some(ref location_str) = location {
                Self::parse_host_port(location_str)
            } else {
                Self::parse_host_port(&grpc_address)
            };

            // Use default SSH configuration for now
            // In production, SSH details should be stored in a dedicated table
            let ssh_details = ExecutorSshDetails::new(
                executor_id.clone(),
                host,
                "ubuntu".to_string(), // Default username - should be configurable
                Some(port.unwrap_or(22)),
                self.config.default_ssh_key_path.clone().unwrap_or_else(|| {
                    std::env::home_dir()
                        .unwrap_or_else(|| "/root".into())
                        .join(".ssh/id_rsa")
                }),
                Some(Duration::from_secs(30)),
            );

            debug!(
                "Found SSH details for executor: {} -> {}:{}",
                executor_id,
                ssh_details.connection().host,
                ssh_details.connection().port
            );
            Ok(Some(ssh_details))
        } else {
            debug!("No SSH details found for executor: {}", executor_id);
            Ok(None)
        }
    }

    /// Parse host and port from address string
    fn parse_host_port(address: &str) -> (String, Option<u16>) {
        if let Some(colon_pos) = address.rfind(':') {
            let host = address[..colon_pos].to_string();
            let port_str = &address[colon_pos + 1..];

            if let Ok(port) = port_str.parse::<u16>() {
                (host, Some(port))
            } else {
                (address.to_string(), None)
            }
        } else {
            (address.to_string(), None)
        }
    }
}
