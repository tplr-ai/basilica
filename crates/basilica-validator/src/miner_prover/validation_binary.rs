//! Binary Validation Module
//!
//! Handles the execution and parsing of validator binary outputs for hardware attestation.

use super::types::{
    BinaryCpuInfo, BinaryMemoryInfo, BinaryNetworkInfo, CompressedMatrix, CpuPowResult, GpuInfo,
    NodeResult, SmUtilizationStats, StoragePowResult, ValidatorBinaryOutput,
};
use anyhow::{Context, Result};
use basilica_common::ssh::SshConnectionDetails;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Import simplified process management utilities
use crate::os_process::{ProcessGroup, ProcessTerminator, ProcessUtils};

/// Request payload for validation server
#[derive(Debug, Clone, Serialize)]
struct ValidationRequest {
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key: String,
    executor_path: String,
    timeout: u64,
}

/// Response from validation server job submission
#[derive(Debug, Clone, Deserialize)]
struct JobSubmissionResponse {
    job_id: String,
}

/// Job status response from validation server
#[derive(Debug, Clone, Deserialize)]
pub struct JobStatusResponse {
    #[allow(dead_code)]
    #[serde(default)]
    job_id: String,
    status: JobStatus,
    #[serde(default)]
    error: Option<String>,
}

/// Job status enum matching server implementation
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    StartingExecutor,
    Generated,
    Challenged,
    Verifying,
    Succeeded,
    Failed,
    Cancelled,
}

/// Validation server lifecycle manager
pub struct ValidationServerManager {
    config: crate::config::ValidationServerConfig,
    binary_path: PathBuf,
    process: Arc<RwLock<Option<Child>>>,
    health_check_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl ValidationServerManager {
    /// Create a new validation server manager
    pub fn new(binary_path: PathBuf, config: crate::config::ValidationServerConfig) -> Self {
        Self {
            config,
            binary_path,
            process: Arc::new(RwLock::new(None)),
            health_check_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the validation server
    pub async fn start(&self) -> Result<()> {
        let mut process_guard = self.process.write().await;

        // Check if already running
        if let Some(child) = process_guard.as_mut() {
            if let Ok(None) = child.try_wait() {
                info!("Validation server already running");
                return Ok(());
            }
        }

        info!(
            bind_address = %self.config.bind_address,
            remote_concurrency = self.config.remote_concurrency,
            verify_concurrency = self.config.verify_concurrency,
            queue_capacity = self.config.queue_capacity,
            "Starting validation server"
        );

        let child = Self::spawn_server(&self.binary_path, &self.config)
            .await
            .map_err(|e| {
                error!("Failed to start validation server: {}", e);
                e
            })?;

        *process_guard = Some(child);

        // Wait a moment for the server to start
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Verify server is healthy
        if !self.is_healthy().await {
            error!("Validation server failed to start properly");
            self.stop_internal(&mut process_guard).await?;
            return Err(anyhow::anyhow!(
                "Validation server failed to start properly"
            ));
        }

        info!("Validation server started successfully");

        // Start health monitoring
        self.start_health_monitoring().await;

        Ok(())
    }

    /// Stop the validation server
    pub async fn stop(&self) -> Result<()> {
        let mut process_guard = self.process.write().await;
        self.stop_internal(&mut process_guard).await
    }

    /// Internal stop implementation
    async fn stop_internal(&self, process_guard: &mut Option<Child>) -> Result<()> {
        // Stop health monitoring
        if let Some(handle) = self.health_check_handle.write().await.take() {
            handle.abort();
        }

        if let Some(mut child) = process_guard.take() {
            info!("Stopping validation server");

            if let Some(pid) = child.id() {
                // Graceful termination with timeout
                if let Err(e) =
                    ProcessTerminator::terminate(pid as i32, Duration::from_secs(5)).await
                {
                    error!("Failed to terminate validation server: {}", e);
                    // Force kill as last resort
                    let _ = child.kill().await;
                } else {
                    info!("Validation server stopped successfully");
                }
            }
        }

        // Clean up any zombie processes
        ProcessUtils::reap_zombies();

        Ok(())
    }

    /// Check if the validation server is healthy
    pub async fn is_healthy(&self) -> bool {
        Self::health_check(&self.config.bind_address, None).await
    }

    /// Perform health check with optional client reuse
    async fn health_check(bind_address: &str, client: Option<&Client>) -> bool {
        let client_owned;
        let client = match client {
            Some(c) => c,
            None => {
                client_owned = Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .unwrap_or_else(|_| Client::new());
                &client_owned
            }
        };

        let health_url = format!("http://{}/healthz", bind_address);

        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => true,
            Ok(response) => {
                warn!(
                    "Validation server health check returned status: {}",
                    response.status()
                );
                false
            }
            Err(e) => {
                debug!("Validation server health check failed: {}", e);
                false
            }
        }
    }

    /// Start periodic health monitoring
    async fn start_health_monitoring(&self) {
        let process = Arc::clone(&self.process);
        let config = self.config.clone();
        let binary_path = self.binary_path.clone();
        let health_interval = Duration::from_secs(self.config.health_check_interval_secs);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_interval);
            let client = Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_else(|_| Client::new());

            loop {
                interval.tick().await;

                let is_healthy = Self::health_check(&config.bind_address, Some(&client)).await;

                if !is_healthy {
                    warn!("Validation server health check failed, attempting restart");

                    // Stop the current process
                    if let Some(child) = process.write().await.take() {
                        if let Some(pid) = child.id() {
                            let _ =
                                ProcessTerminator::terminate(pid as i32, Duration::from_secs(2))
                                    .await;
                        }
                    }

                    // Restart the server
                    match Self::spawn_server(&binary_path, &config).await {
                        Ok(child) => {
                            *process.write().await = Some(child);
                            info!("Validation server restarted successfully");
                        }
                        Err(e) => {
                            error!("Failed to restart validation server: {}", e);
                        }
                    }
                }
            }
        });

        *self.health_check_handle.write().await = Some(handle);
    }

    /// Helper to spawn server process with standard configuration
    async fn spawn_server(
        binary_path: &Path,
        config: &crate::config::ValidationServerConfig,
    ) -> Result<Child> {
        let mut command = Command::new(binary_path);

        // Build server command arguments
        command
            .arg("serve")
            .arg("--bind")
            .arg(&config.bind_address)
            .arg("--remote-concurrency")
            .arg(config.remote_concurrency.to_string())
            .arg("--verify-concurrency")
            .arg(config.verify_concurrency.to_string())
            .arg("--queue-capacity")
            .arg(config.queue_capacity.to_string());

        // Pass threshold overrides as environment variables if configured
        if let Some(cpu_threshold) = config.max_cpu_ms_per_iteration {
            command.env(
                "VERITAS_MAX_CPU_MS_PER_ITERATION",
                cpu_threshold.to_string(),
            );
        }
        if let Some(storage_threshold) = config.max_storage_duration_ms {
            command.env(
                "VERITAS_MAX_STORAGE_DURATION_MS",
                storage_threshold.to_string(),
            );
        }

        // Configure for process group isolation
        ProcessGroup::configure_command(&mut command);

        command
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());

        command
            .spawn()
            .context("Failed to spawn validation server process")
    }
}

impl Drop for ValidationServerManager {
    fn drop(&mut self) {
        // Best effort cleanup on drop
        let process = Arc::clone(&self.process);
        tokio::spawn(async move {
            if let Some(child) = process.write().await.take() {
                if let Some(pid) = child.id() {
                    let _ = ProcessTerminator::terminate(pid as i32, Duration::from_secs(2)).await;
                }
            }
        });
    }
}

/// HTTP client for validation server API
pub struct ValidationServerClient {
    client: Client,
    base_url: String,
    poll_interval_ms: u64,
    max_poll_attempts: usize,
}

impl ValidationServerClient {
    /// Create a new validation server client
    pub fn new(server_address: &str, poll_interval_ms: u64, max_poll_attempts: usize) -> Self {
        // Use a short timeout for status/health checks
        // Job submission will use a separate client with appropriate timeout
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            base_url: format!("http://{}", server_address),
            poll_interval_ms,
            max_poll_attempts,
        }
    }

    /// Submit a validation job to the server
    pub async fn submit_job(
        &self,
        ssh_details: &SshConnectionDetails,
        executor_path: &str,
        timeout_secs: u64,
    ) -> Result<String> {
        let request = ValidationRequest {
            ssh_host: ssh_details.host.clone(),
            ssh_port: ssh_details.port,
            ssh_user: ssh_details.username.clone(),
            ssh_key: ssh_details.private_key_path.to_string_lossy().to_string(),
            executor_path: executor_path.to_string(),
            timeout: timeout_secs,
        };

        // Create a client with timeout matching the validation request timeout
        // Add buffer for network overhead and server processing
        let submit_timeout = Duration::from_secs(timeout_secs.saturating_add(60));
        let submit_client = Client::builder()
            .timeout(submit_timeout)
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| Client::new());

        debug!(
            "[EVAL_FLOW] Submitting validation job with timeout: {} seconds",
            submit_timeout.as_secs()
        );

        let response = submit_client
            .post(format!("{}/validate", self.base_url))
            .json(&request)
            .send()
            .await
            .context("Failed to submit validation job")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Failed to submit job: {} - {}",
                status,
                error_text
            ));
        }

        let submission: JobSubmissionResponse = response
            .json()
            .await
            .context("Failed to parse job submission response")?;

        info!(
            job_id = submission.job_id,
            "[EVAL_FLOW] Submitted validation job: {}", submission.job_id
        );
        Ok(submission.job_id)
    }

    /// Poll job status until completion and retrieve results
    pub async fn poll_job_status(
        &self,
        job_id: &str,
        initial_timeout: Option<Duration>,
    ) -> Result<Vec<u8>> {
        let mut poll_interval = Duration::from_millis(self.poll_interval_ms);
        let start_time = tokio::time::Instant::now();
        let timeout_duration =
            Duration::from_millis(self.poll_interval_ms * self.max_poll_attempts as u64 * 2);

        info!(
            job_id = job_id,
            "[EVAL_FLOW] Starting to poll job {} status every {}ms, timeout: {}s",
            job_id,
            self.poll_interval_ms,
            timeout_duration.as_secs()
        );

        // initial jitter of delay before polling
        tokio::time::sleep(initial_timeout.unwrap_or_else(|| Duration::from_secs(1))).await;

        let mut consecutive_404s = 0;
        const MAX_404_BEFORE_RESULT_CHECK: u32 = 3;

        loop {
            let elapsed = start_time.elapsed();

            debug!(
                job_id = job_id,
                "[EVAL_FLOW] Polling job {} (elapsed: {}s)",
                job_id,
                elapsed.as_secs()
            );

            if elapsed > timeout_duration {
                error!(
                    job_id = job_id,
                    "[EVAL_FLOW] Job {} polling timed out after {} seconds, trying to fetch results anyway",
                    job_id,
                    elapsed.as_secs()
                );

                return self.get_job_result(job_id).await;
            }

            let response = match self
                .client
                .get(format!("{}/jobs/{}", self.base_url, job_id))
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    // On network errors, retry with backoff
                    warn!(
                        job_id = job_id,
                        "[EVAL_FLOW] Failed to poll job {} status: {}, retrying...", job_id, e
                    );
                    tokio::time::sleep(poll_interval).await;
                    continue;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());

                if status == reqwest::StatusCode::NOT_FOUND {
                    consecutive_404s += 1;
                    warn!(
                        job_id = job_id,
                        consecutive_404s = consecutive_404s,
                        "[EVAL_FLOW] Job {} status returned 404 (attempt {}/{}). Job may have completed and been cleaned up.",
                        job_id, consecutive_404s, MAX_404_BEFORE_RESULT_CHECK
                    );

                    if consecutive_404s >= MAX_404_BEFORE_RESULT_CHECK {
                        info!(
                            job_id = job_id,
                            "[EVAL_FLOW] Received {} consecutive 404s, attempting to fetch results directly",
                            consecutive_404s
                        );

                        match self.get_job_result(job_id).await {
                            Ok(result) => {
                                info!(
                                    job_id = job_id,
                                    "[EVAL_FLOW] Successfully retrieved results despite 404 on status check"
                                );
                                return Ok(result);
                            }
                            Err(e) => {
                                error!(
                                    job_id = job_id,
                                    "[EVAL_FLOW] Failed to retrieve results after 404s: {}. Job may not exist.",
                                    e
                                );
                                return Err(anyhow::anyhow!(
                                    "Failed to get job status (404) and results unavailable: {}",
                                    e
                                ));
                            }
                        }
                    }

                    tokio::time::sleep(poll_interval).await;
                    continue;
                }

                return Err(anyhow::anyhow!(
                    "Failed to get job status: {} - {}",
                    status,
                    error_text
                ));
            }

            consecutive_404s = 0;

            let response_text = response
                .text()
                .await
                .context("Failed to read job status response body")?;

            debug!(
                job_id = job_id,
                "Raw job status response: {}", response_text
            );

            // Try to parse the response
            let status: JobStatusResponse = match serde_json::from_str(&response_text) {
                Ok(s) => s,
                Err(e) => {
                    error!(
                        job_id = job_id,
                        "[EVAL_FLOW] Failed to parse job status response. Raw response: '{}', Error: {}",
                        response_text, e
                    );
                    return Err(anyhow::anyhow!(
                        "Failed to parse job status response. Raw response: '{}', Error: {}",
                        response_text,
                        e
                    ));
                }
            };

            debug!(
                job_id = job_id,
                "Job {} status: {:?}", job_id, status.status
            );

            match status.status {
                JobStatus::Succeeded => {
                    info!(job_id = job_id, "Job {} succeeded", job_id);
                    return self.get_job_result(job_id).await;
                }
                JobStatus::Failed => {
                    if status.error.is_none() {
                        warn!(
                            job_id = job_id,
                            "[EVAL_FLOW] Job {} marked as failed but error is null - treating as succeeded (server bug workaround)",
                            job_id
                        );
                        return self.get_job_result(job_id).await;
                    }

                    error!(job_id = job_id, "Job {} failed: {:?}", job_id, status.error);
                    return Err(anyhow::anyhow!(
                        "Job failed: {}",
                        status.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
                JobStatus::Cancelled => {
                    error!(job_id = job_id, "Job {} was cancelled", job_id);
                    return Err(anyhow::anyhow!("Job {} was cancelled", job_id));
                }
                _ => {
                    tokio::time::sleep(poll_interval).await;

                    // Implement exponential backoff up to 5 seconds
                    if poll_interval < Duration::from_secs(5) {
                        poll_interval = (poll_interval * 2).min(Duration::from_secs(5));
                    }
                }
            }
        }
    }

    /// Retrieve job results
    pub async fn get_job_result(&self, job_id: &str) -> Result<Vec<u8>> {
        let response = self
            .client
            .get(format!("{}/jobs/{}/result", self.base_url, job_id))
            .send()
            .await
            .context("Failed to retrieve job result")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Failed to get job result: {} - {}",
                status,
                error_text
            ));
        }

        let result_bytes = response
            .bytes()
            .await
            .context("Failed to read job result")?;

        Ok(result_bytes.to_vec())
    }
}

/// Binary validation node for running and parsing validator binaries
pub struct BinaryValidator {
    server_manager: Option<Arc<ValidationServerManager>>,
    server_client: Option<Arc<ValidationServerClient>>,
}

impl BinaryValidator {
    /// Create a new binary validator
    pub fn new(_ssh_client: Arc<crate::ssh::ValidatorSshClient>) -> Self {
        Self {
            server_manager: None,
            server_client: None,
        }
    }

    /// Initialize server mode (always on)
    pub async fn initialize_server_mode(
        &mut self,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<()> {
        if self.server_manager.is_some() && self.server_client.is_some() {
            info!("Validation server already initialized");
            return Ok(());
        }

        if let Some(manager) = &self.server_manager {
            let _ = manager.stop().await;
            self.server_manager = None;
        }
        self.server_client = None;

        info!("Initializing validation server");

        // Create server manager
        let server_manager = Arc::new(ValidationServerManager::new(
            binary_config.validator_binary_path.clone(),
            binary_config.server_mode.clone(),
        ));

        // Start the server
        server_manager.start().await?;

        // Create client
        let server_client = Arc::new(ValidationServerClient::new(
            &binary_config.server_mode.bind_address,
            binary_config.server_mode.job_poll_interval_ms,
            binary_config.server_mode.max_poll_attempts,
        ));

        self.server_manager = Some(server_manager);
        self.server_client = Some(server_client);

        info!("Validation server initialized successfully");
        Ok(())
    }

    /// Execute validator binary via server
    pub async fn execute(
        &self,
        ssh_details: &SshConnectionDetails,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<Vec<u8>> {
        let client = self
            .server_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Validation server client not initialized"))?;

        self.execute_with_retry(client, ssh_details, binary_config)
            .await
    }

    /// Ensure server is ready to accept jobs
    async fn ensure_server_ready(
        &self,
        config: &crate::config::ValidationServerConfig,
    ) -> Result<()> {
        let server_manager = self
            .server_manager
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Server manager not initialized"))?;

        let timeout = Duration::from_secs(config.server_ready_timeout_secs);
        let check_interval = Duration::from_millis(config.server_ready_check_interval_ms);
        let start_time = tokio::time::Instant::now();

        info!("[EVAL_FLOW] Ensuring validation server is ready...");

        loop {
            // Check if server is healthy
            if server_manager.is_healthy().await {
                info!("[EVAL_FLOW] Validation server is ready");
                return Ok(());
            }

            // Check if we've exceeded timeout
            if start_time.elapsed() >= timeout {
                error!(
                    "[EVAL_FLOW] Server failed to become ready within {} seconds",
                    config.server_ready_timeout_secs
                );
                return Err(anyhow::anyhow!(
                    "Validation server failed to become ready within timeout"
                ));
            }

            debug!(
                "[EVAL_FLOW] Server not ready, retrying in {}ms...",
                check_interval.as_millis()
            );

            // Wait before next check
            tokio::time::sleep(check_interval).await;
        }
    }

    /// Execute validation with retry logic for submission failures
    pub async fn execute_with_retry(
        &self,
        client: &ValidationServerClient,
        ssh_details: &SshConnectionDetails,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<Vec<u8>> {
        let max_attempts = binary_config.server_mode.max_workflow_retry_attempts;
        let base_delay_ms = binary_config.server_mode.workflow_retry_base_delay_ms;

        for attempt in 1..=max_attempts {
            info!(
                ssh_host = %ssh_details.host,
                attempt = attempt,
                max_attempts = max_attempts,
                "[EVAL_FLOW] Starting validation workflow attempt {}/{}",
                attempt, max_attempts
            );

            match self
                .execute_with_client(client, ssh_details, binary_config)
                .await
            {
                Ok(result) => {
                    if attempt > 1 {
                        info!(
                            ssh_host = %ssh_details.host,
                            attempt = attempt,
                            "[EVAL_FLOW] Validation workflow succeeded on retry attempt {}",
                            attempt
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    let is_404_related = error_msg.contains("404")
                        || error_msg.contains("Not Found")
                        || error_msg.contains("Failed to get job status");

                    if !is_404_related || attempt >= max_attempts {
                        error!(
                            ssh_host = %ssh_details.host,
                            attempt = attempt,
                            error = %e,
                            is_404_related = is_404_related,
                            "[EVAL_FLOW] Validation workflow failed on attempt {}: {}",
                            attempt, e
                        );
                        return Err(e);
                    }

                    let delay_ms = base_delay_ms * (2_u64.pow(attempt - 1));
                    warn!(
                        ssh_host = %ssh_details.host,
                        attempt = attempt,
                        next_delay_ms = delay_ms,
                        error = %e,
                        "[EVAL_FLOW] Validation workflow failed with 404-related error on attempt {}. Retrying in {}ms...",
                        attempt, delay_ms
                    );

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }

        Err(anyhow::anyhow!(
            "Validation workflow exhausted all {} retry attempts",
            max_attempts
        ))
    }

    pub async fn execute_with_client(
        &self,
        client: &ValidationServerClient,
        ssh_details: &SshConnectionDetails,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<Vec<u8>> {
        info!(
            ssh_host = %ssh_details.host,
            ssh_port = ssh_details.port,
            "[EVAL_FLOW] Executing validator via server mode"
        );

        // Ensure server is ready before submitting job
        self.ensure_server_ready(&binary_config.server_mode).await?;

        // Submit job
        let job_id = client
            .submit_job(
                ssh_details,
                &binary_config.executor_binary_path.to_string_lossy(),
                binary_config.execution_timeout_secs,
            )
            .await?;

        info!(
            job_id = job_id,
            "[EVAL_FLOW] Validation job submitted: {}", job_id
        );

        // Poll for completion and retrieve results
        let result_bytes = client
            .poll_job_status(&job_id, Some(Duration::from_secs(4 * 60)))
            .await?;

        info!(
            "[EVAL_FLOW] Successfully retrieved job {} results ({} bytes)",
            job_id,
            result_bytes.len()
        );

        Ok(result_bytes)
    }

    /// Parse validator binary output
    pub fn parse_validator_binary_output(
        &self,
        node_id: &str,
        miner_uid: u16,
        output: &[u8],
    ) -> Result<ValidatorBinaryOutput> {
        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[EVAL_FLOW] Parsing validator binary output ({} bytes)",
            output.len()
        );

        if output.is_empty() {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] Validator binary output is empty - this indicates a capture problem"
            );
            return Err(anyhow::anyhow!(
                "Validator binary produced no output - output capture failed"
            ));
        }

        let output_str = String::from_utf8_lossy(output);

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[EVAL_FLOW] Parsing validator binary output ({} bytes)",
            output.len()
        );
        debug!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[EVAL_FLOW] Raw output: {}",
            output_str
        );

        // Validate output contains some expected content
        if !output_str.contains("validator_binary")
            && !output_str.contains("success")
            && !output_str.contains("{")
        {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] Validator binary output does not appear to contain expected content"
            );
            return Err(anyhow::anyhow!(
                "Validator binary output does not contain expected validator_binary logs or JSON. Output: {}",
                output_str.chars().take(500).collect::<String>()
            ));
        }

        // Extract JSON from mixed log/JSON output
        let json_str = match self.extract_json_from_output(&output_str) {
            Ok(json) => json,
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[EVAL_FLOW] Failed to extract JSON from validator output: {}",
                    e
                );
                error!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[EVAL_FLOW] Raw output for debugging: {}",
                    output_str.chars().take(1000).collect::<String>()
                );
                return Err(e.context("Failed to extract JSON from validator binary output"));
            }
        };

        // Parse raw JSON and convert to expected format
        let parsed_output = self.parse_and_convert_validator_output(&json_str)?;

        debug!(miner_uid = miner_uid, node_id = node_id,
            "[EVAL_FLOW] Successfully parsed binary output - success: {}, execution_time: {}ms, validation_score: {:.3}",
            parsed_output.success, parsed_output.execution_time_ms, parsed_output.validation_score);

        if let Some(ref node_result) = parsed_output.node_result {
            debug!(miner_uid = miner_uid, node_id = node_id,
                "[EVAL_FLOW] Node hardware details - CPU cores: {}, Memory: {:.1}GB, Network interfaces: {}",
                node_result.cpu_info.cores, node_result.memory_info.total_gb,
                  node_result.network_info.interfaces.len());

            if !node_result.gpu_name.is_empty() {
                info!(miner_uid = miner_uid, node_id = node_id,
                    "[EVAL_FLOW] GPU Details: {} (UUID: {}), SMs: {}/{}, Memory bandwidth: {:.1} GB/s Memory: {:.1} GB",
                    node_result.gpu_name, node_result.gpu_uuid,
                    node_result.active_sms, node_result.total_sms,
                    node_result.memory_bandwidth_gbps,
                    node_result.gpu_infos.iter().map(|g| g.gpu_memory_gb).sum::<f64>()
                );
            } else {
                warn!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[EVAL_FLOW] No GPU information found in node result"
                );
            }

            info!(miner_uid = miner_uid, node_id = node_id,
                "[EVAL_FLOW] Binary validation metrics - Matrix computation: {:.2}ms, SM utilization: max={:.1}%, avg={:.1}%",
                node_result.computation_time_ns as f64 / 1_000_000.0,
                  node_result.sm_utilization.max_utilization,
                  node_result.sm_utilization.avg_utilization);
        } else {
            warn!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] No node result found in binary output"
            );
        }

        if let Some(ref error_msg) = parsed_output.error_message {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] Binary validation error message: {}",
                error_msg
            );
        }
        if !parsed_output.failure_reasons.is_empty() {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] Binary validation failure reasons: {:?}",
                parsed_output.failure_reasons
            );
        }

        // Validate structure
        if parsed_output.success && parsed_output.node_result.is_none() {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                "[EVAL_FLOW] Validator binary reported success but no node result provided"
            );
            return Err(anyhow::anyhow!(
                "Validator binary reported success but no node result provided"
            ));
        }

        Ok(parsed_output)
    }

    /// Extract JSON object from mixed log/JSON output
    fn extract_json_from_output(&self, output: &str) -> Result<String> {
        info!(
            "[EVAL_FLOW] Extracting JSON from validator binary output ({} bytes)",
            output.len()
        );

        if output.trim().is_empty() {
            error!("[EVAL_FLOW] Validator binary output is empty");
            return Err(anyhow::anyhow!("Validator binary produced no output"));
        }

        // Strategy 1: Find the last valid JSON object by scanning backwards for complete JSON blocks
        // This handles the case where JSON appears after log messages
        let mut candidates = Vec::new();
        let mut brace_count = 0;
        let mut current_start = None;
        let chars: Vec<char> = output.chars().collect();

        // Scan through entire output to find all potential JSON objects
        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                '{' => {
                    if brace_count == 0 {
                        current_start = Some(i);
                    }
                    brace_count += 1;
                }
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        if let Some(start) = current_start {
                            let json_candidate: String = chars[start..=i].iter().collect();
                            candidates.push((start, json_candidate));
                        }
                        current_start = None;
                    }
                }
                _ => {}
            }
        }

        debug!(
            "[EVAL_FLOW] Found {} potential JSON candidates",
            candidates.len()
        );

        // Test candidates in reverse order (last one first, as it's most likely the final JSON output)
        for (start_pos, candidate) in candidates.into_iter().rev() {
            let trimmed = candidate.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(parsed) => {
                    // Additional validation: ensure this looks like validator output
                    if self.is_valid_validator_output(&parsed) {
                        info!("[EVAL_FLOW] Successfully extracted valid JSON object ({} bytes) at position {}",
                              trimmed.len(), start_pos);
                        debug!("[EVAL_FLOW] Extracted JSON: {}", trimmed);
                        return Ok(trimmed.to_string());
                    } else {
                        debug!("[EVAL_FLOW] JSON candidate at position {} failed validator output validation", start_pos);
                    }
                }
                Err(e) => {
                    debug!(
                        "[EVAL_FLOW] JSON candidate at position {} failed parsing: {}",
                        start_pos, e
                    );
                }
            }
        }

        // Strategy 2: Look for JSON on lines that start with '{' (working backwards)
        let lines: Vec<&str> = output.lines().collect();
        for (line_num, line) in lines.iter().enumerate().rev() {
            let trimmed = line.trim();
            if trimmed.starts_with('{') && trimmed.len() > 10 {
                // Try parsing just this line first
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if self.is_valid_validator_output(&parsed) {
                        info!(
                            "[EVAL_FLOW] Found valid JSON on single line {} ({} bytes)",
                            line_num + 1,
                            trimmed.len()
                        );
                        return Ok(trimmed.to_string());
                    }
                }

                // Try parsing from this line to end of output
                let remaining_lines: Vec<&str> = lines[line_num..].to_vec();
                let multi_line_candidate = remaining_lines.join("\n");
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&multi_line_candidate)
                {
                    if self.is_valid_validator_output(&parsed) {
                        info!("[EVAL_FLOW] Found valid multi-line JSON starting at line {} ({} bytes)",
                              line_num + 1, multi_line_candidate.len());
                        return Ok(multi_line_candidate);
                    }
                }
            }
        }

        // Strategy 3: Look for JSON at the very end of output (common case)
        let output_suffix = output.trim_end();
        if let Some(last_brace) = output_suffix.rfind('}') {
            if let Some(first_brace) = output_suffix[..=last_brace].rfind('{') {
                let final_candidate = &output_suffix[first_brace..=last_brace];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(final_candidate) {
                    if self.is_valid_validator_output(&parsed) {
                        info!(
                            "[EVAL_FLOW] Found valid JSON at end of output ({} bytes)",
                            final_candidate.len()
                        );
                        return Ok(final_candidate.to_string());
                    }
                }
            }
        }

        // Log detailed failure information for debugging
        error!("[EVAL_FLOW] Failed to extract valid JSON from validator binary output");
        error!("[EVAL_FLOW] Output length: {} bytes", output.len());
        error!("[EVAL_FLOW] Output lines: {}", lines.len());
        error!(
            "[EVAL_FLOW] First 200 chars: {:?}",
            output.chars().take(200).collect::<String>()
        );
        error!(
            "[EVAL_FLOW] Last 200 chars: {:?}",
            output
                .chars()
                .rev()
                .take(200)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        );

        Err(anyhow::anyhow!(
            "Failed to extract valid JSON from validator binary output. Output contains {} lines and {} bytes. \
             Expected JSON output from validator binary with 'success', 'gpu_results', or 'execution_time_ms' fields.",
            lines.len(), output.len()
        ))
    }

    /// Validate that a parsed JSON object looks like valid validator output
    fn is_valid_validator_output(&self, parsed: &serde_json::Value) -> bool {
        // Check for expected top-level fields that indicate this is validator output
        let has_success = parsed.get("success").is_some();
        let has_gpu_results = parsed.get("gpu_results").is_some();
        let has_execution_time = parsed.get("execution_time_ms").is_some();
        let has_matrix_size = parsed.get("matrix_size").is_some();

        // Must have at least 2 of these key fields to be considered valid validator output
        let field_count = [
            has_success,
            has_gpu_results,
            has_execution_time,
            has_matrix_size,
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        let is_valid = field_count >= 2;

        if !is_valid {
            debug!("[EVAL_FLOW] JSON validation failed - has_success: {}, has_gpu_results: {}, has_execution_time: {}, has_matrix_size: {}",
                   has_success, has_gpu_results, has_execution_time, has_matrix_size);
        }

        is_valid
    }

    /// Parse and convert raw validator binary JSON to expected format
    fn parse_and_convert_validator_output(&self, json_str: &str) -> Result<ValidatorBinaryOutput> {
        info!("[EVAL_FLOW] Converting raw validator binary JSON to expected format");

        // Parse raw JSON into a generic Value first
        let raw_json: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            error!("[EVAL_FLOW] Failed to parse raw JSON: {}", e);
            anyhow::anyhow!("Failed to parse raw JSON: {}", e)
        })?;

        // Extract basic fields
        let success = raw_json
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let execution_time_ms = raw_json
            .get("execution_time_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        info!(
            "[EVAL_FLOW] Raw JSON parsing - success: {}, execution_time_ms: {}",
            success, execution_time_ms
        );

        let failure_reasons = raw_json
            .get("failure_reasons")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        let has_failures = !failure_reasons.is_empty();
        let effective_success = success && !has_failures;

        // Calculate validation score based on the results
        let validation_score = if effective_success { 1.0 } else { 0.0 };

        // Convert GPU results to node result if available
        let node_result = if effective_success {
            self.convert_gpu_results_to_node_result(&raw_json)?
        } else {
            None
        };

        // Extract error message if present
        let error_message = raw_json
            .get("error_message")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Extract GPU count from the original validator-binary data
        let gpu_count = raw_json
            .get("gpu_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        info!(
            "[EVAL_FLOW] Converted to ValidatorBinaryOutput - success: {}, failures: {}, has_node_result: {}, gpu_count: {}",
            effective_success,
            failure_reasons.len(),
            node_result.is_some(),
            gpu_count
        );

        Ok(ValidatorBinaryOutput {
            success: effective_success,
            node_result,
            error_message,
            failure_reasons,
            execution_time_ms,
            validation_score,
            gpu_count,
        })
    }

    /// Convert GPU results to NodeResult format
    pub fn convert_gpu_results_to_node_result(
        &self,
        raw_json: &serde_json::Value,
    ) -> Result<Option<NodeResult>> {
        let gpu_results = raw_json
            .get("gpu_results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("No gpu_results found in output"))?;

        if gpu_results.is_empty() {
            return Ok(None);
        }

        // Extract all GPU information
        let mut gpu_infos = Vec::new();
        for (index, gpu_result) in gpu_results.iter().enumerate() {
            let gpu_name = gpu_result
                .get("gpu_name")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown GPU")
                .to_string();

            let gpu_uuid = gpu_result
                .get("gpu_uuid")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown UUID")
                .to_string();

            // FIXME: gpu_memory_gb should be reported properly by the validator binary itself
            let mut gpu_memory_gb = gpu_result
                .get("gpu_memory_gb")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            if gpu_memory_gb == 0.0 {
                let parsed = crate::persistence::miner_nodes::extract_gpu_memory_gb(&gpu_name);
                if parsed > 0 {
                    warn!(
                        gpu_name = %gpu_name,
                        parsed_memory_gb = parsed,
                        "Binary reported gpu_memory_gb=0.0, falling back to GPU name parse"
                    );
                    gpu_memory_gb = parsed as f64;
                }
            }

            let computation_time_ns = gpu_result
                .get("computation_time_ns")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            // Get metrics object
            let metrics = gpu_result.get("metrics");

            let memory_bandwidth_gbps = metrics
                .and_then(|m| m.get("memory_bandwidth_gbps"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            let anti_debug_passed = metrics
                .and_then(|m| m.get("anti_debug_passed"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // SM utilization
            let sm_utilization =
                if let Some(sm_util) = metrics.and_then(|m| m.get("sm_utilization")) {
                    let min_util = sm_util.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let max_util = sm_util.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let avg_util = sm_util.get("avg").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    SmUtilizationStats {
                        min_utilization: min_util,
                        max_utilization: max_util,
                        avg_utilization: avg_util,
                        per_sm_stats: vec![],
                    }
                } else {
                    SmUtilizationStats {
                        min_utilization: 0.0,
                        max_utilization: 0.0,
                        avg_utilization: 0.0,
                        per_sm_stats: vec![],
                    }
                };

            let active_sms = metrics
                .and_then(|m| m.get("sm_utilization"))
                .and_then(|v| v.get("active_sms"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let total_sms = metrics
                .and_then(|m| m.get("sm_utilization"))
                .and_then(|v| v.get("total_sms"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            gpu_infos.push(GpuInfo {
                index: index as u32,
                gpu_name,
                gpu_uuid,
                gpu_memory_gb,
                computation_time_ns,
                memory_bandwidth_gbps,
                sm_utilization,
                active_sms,
                total_sms,
                anti_debug_passed,
            });
        }

        // Use the first GPU for primary information (backwards compatibility)
        let primary_gpu = &gpu_results[0];
        let primary_metrics = primary_gpu.get("metrics");

        let gpu_name = primary_gpu
            .get("gpu_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown GPU")
            .to_string();

        let gpu_uuid = primary_gpu
            .get("gpu_uuid")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown UUID")
            .to_string();

        let computation_time_ns = primary_gpu
            .get("computation_time_ns")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let memory_bandwidth_gbps = primary_metrics
            .and_then(|m| m.get("memory_bandwidth_gbps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let anti_debug_passed = primary_metrics
            .and_then(|m| m.get("anti_debug_passed"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sm_utilization = gpu_infos[0].sm_utilization.clone();
        let active_sms = gpu_infos[0].active_sms;
        let total_sms = gpu_infos[0].total_sms;

        let timing_fingerprint = raw_json
            .get("timing_fingerprint")
            .and_then(|v| v.as_str())
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        let cpu_pow = raw_json.get("cpu_pow").map(|v| CpuPowResult {
            valid: v.get("valid").and_then(|v| v.as_bool()).unwrap_or(false),
            cpu_model: v
                .get("cpu_model")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            iterations: v.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0),
            chunk_size: v.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(0),
            duration_ms: v.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0),
        });

        let storage_pow = raw_json.get("storage_pow").map(|v| StoragePowResult {
            valid: v.get("valid").and_then(|v| v.as_bool()).unwrap_or(false),
            file_id: v
                .get("file_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            file_size_bytes: v
                .get("file_size_bytes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            block_size: v.get("block_size").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            samples: v.get("samples").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            duration_ms: v.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0),
        });

        let node_result = NodeResult {
            gpu_name,
            gpu_uuid,
            gpu_infos,
            cpu_info: BinaryCpuInfo {
                model: "Unknown".to_string(),
                cores: 0,
                threads: 0,
                frequency_mhz: 0,
            },
            memory_info: BinaryMemoryInfo {
                total_gb: 0.0,
                available_gb: 0.0,
            },
            network_info: BinaryNetworkInfo { interfaces: vec![] },
            cpu_pow,
            storage_pow,
            matrix_c: CompressedMatrix {
                rows: 0,
                cols: 0,
                data: vec![],
            },
            computation_time_ns,
            checksum: [0u8; 32],
            sm_utilization,
            active_sms,
            total_sms,
            memory_bandwidth_gbps,
            anti_debug_passed,
            timing_fingerprint,
        };

        info!(
            "[EVAL_FLOW] Converted GPU results to NodeResult - GPU: {}, bandwidth: {:.1} GB/s, SMs: {}/{}",
            node_result.gpu_name, node_result.memory_bandwidth_gbps,
            node_result.active_sms, node_result.total_sms
        );

        Ok(Some(node_result))
    }

    /// Execute binary validation using validator-binary
    pub async fn execute_binary_validation(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<ValidatorBinaryOutput> {
        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            ssh_host = %ssh_details.host,
            ssh_port = ssh_details.port,
            "[EVAL_FLOW] Starting binary validation process"
        );

        // Execute validator-binary
        let execution_start = std::time::Instant::now();
        let binary_output = self.execute(ssh_details, binary_config).await?;
        let execution_duration = execution_start.elapsed();

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            ssh_host = %ssh_details.host,
            ssh_port = ssh_details.port,
            execution_duration = ?execution_duration,
            "[EVAL_FLOW] Validator binary executed"
        );

        // Parse and validate output
        let validation_result =
            self.parse_validator_binary_output(node_id, miner_uid, &binary_output)?;

        // Strict binary outcome: score is 1.0 if success, else 0.0
        let validation_score = if validation_result.success { 1.0 } else { 0.0 };

        Ok(ValidatorBinaryOutput {
            success: validation_result.success,
            node_result: validation_result.node_result,
            error_message: validation_result.error_message,
            failure_reasons: validation_result.failure_reasons,
            execution_time_ms: execution_duration.as_millis() as u64,
            validation_score,
            gpu_count: validation_result.gpu_count,
        })
    }

    /// Shutdown server if running
    pub async fn shutdown(&self) -> Result<()> {
        if let Some(manager) = &self.server_manager {
            manager.stop().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_validator() -> BinaryValidator {
        let mock_ssh_client = Arc::new(crate::ssh::ValidatorSshClient::new());
        BinaryValidator::new(mock_ssh_client)
    }

    #[test]
    fn test_parse_real_validator_binary_output() {
        let validator = create_test_validator();

        // Real output from your validator binary execution
        let real_output = r#"{
  "execution_time_ms": 680536,
  "gpu_count": 1,
  "gpu_results": [
    {
      "computation_time_ns": 214282408766,
      "gpu_index": 0,
      "gpu_name": "NVIDIA B200",
      "gpu_uuid": "GPU-12345678901234567890123456789abc",
      "gpu_memory_gb": 80.0,
      "merkle_root": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
      "metrics": {
        "anti_debug_passed": true,
        "memory_bandwidth_gbps": 0.7563359043671317,
        "sm_utilization": {
          "active_sms": 148,
          "avg": 0.5703122615814209,
          "max": 1.0011287927627563,
          "min": 0.0,
          "total_sms": 148
        }
      }
    }
  ],
  "matrix_size": 82176,
  "random_seed": "0xfb9e0f67d3814c10",
  "success": true,
  "timing_fingerprint": "0x1a99231c86c",
  "total_execution_time_ns": 676971022243
}"#;

        let result = validator.parse_validator_binary_output("", 0, real_output.as_bytes());
        assert!(
            result.is_ok(),
            "Failed to parse real validator output: {:?}",
            result.err()
        );

        let parsed = result.unwrap();
        assert!(parsed.success);
        assert_eq!(parsed.execution_time_ms, 680536);
        assert_eq!(parsed.gpu_count, 1);
        assert!(parsed.validation_score > 0.0);

        let node_result = parsed.node_result.expect("Should have node result");
        assert_eq!(node_result.gpu_name, "NVIDIA B200");
        assert_eq!(node_result.gpu_uuid, "GPU-12345678901234567890123456789abc");
        assert_eq!(node_result.computation_time_ns, 214282408766);
        assert_eq!(node_result.active_sms, 148);
        assert_eq!(node_result.total_sms, 148);
        assert!(node_result.anti_debug_passed);
        assert!((node_result.memory_bandwidth_gbps - 0.7563359043671317).abs() < 0.0001);
        assert!((node_result.sm_utilization.avg_utilization - 0.5703122615814209).abs() < 0.0001);
        assert!((node_result.sm_utilization.max_utilization - 1.0011287927627563).abs() < 0.0001);
        assert_eq!(node_result.sm_utilization.min_utilization, 0.0);
        assert_eq!(node_result.gpu_infos.len(), 1);
    }

    #[test]
    fn test_extract_json_from_mixed_output() {
        let validator = create_test_validator();

        // Test with logs mixed with JSON (common real scenario)
        let mixed_output = r#"
[INFO] Starting validator binary
[DEBUG] Connecting to SSH host
[INFO] Uploading node binary
[DEBUG] Running GPU validation
{
  "execution_time_ms": 680536,
  "gpu_count": 1,
  "gpu_results": [
    {
      "computation_time_ns": 214282408766,
      "gpu_name": "NVIDIA B200",
      "metrics": {
        "anti_debug_passed": true,
        "memory_bandwidth_gbps": 0.7563359043671317,
        "sm_utilization": {
          "avg": 0.5703122615814209
        }
      }
    }
  ],
  "success": true
}
[INFO] Validation complete
"#;

        let result = validator.extract_json_from_output(mixed_output);
        assert!(result.is_ok(), "Failed to extract JSON: {:?}", result.err());

        let json_str = result.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["execution_time_ms"], 680536);
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["gpu_count"], 1);
    }

    #[test]
    fn test_is_valid_validator_output() {
        let validator = create_test_validator();

        // Valid output with required fields
        let valid_json: serde_json::Value = serde_json::from_str(
            r#"{
            "success": true,
            "execution_time_ms": 1000,
            "gpu_results": [],
            "matrix_size": 1024
        }"#,
        )
        .unwrap();

        assert!(validator.is_valid_validator_output(&valid_json));

        // Invalid output missing required fields
        let invalid_json: serde_json::Value = serde_json::from_str(
            r#"{
            "some_other_field": "value"
        }"#,
        )
        .unwrap();

        assert!(!validator.is_valid_validator_output(&invalid_json));

        // Partially valid (only 1 required field)
        let partial_json: serde_json::Value = serde_json::from_str(
            r#"{
            "success": true
        }"#,
        )
        .unwrap();

        assert!(!validator.is_valid_validator_output(&partial_json));
    }
}
