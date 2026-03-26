//! Common utilities for deployment templates
//!
//! Shared functions used by vLLM, SGLang, and other inference server templates.
//! This module consolidates retry logic, environment parsing, and deployment status handling.

use crate::error::{CliError, DeployError};
use crate::output::print_info;
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{CreateDeploymentRequest, DeploymentResponse};
use basilica_sdk::BasilicaClient;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Maximum retries for transient failures
pub const MAX_RETRIES: u32 = 3;

/// Initial retry delay in milliseconds
pub const INITIAL_RETRY_DELAY_MS: u64 = 500;

/// Result of waiting for deployment to become ready
pub enum WaitResult {
    Ready(Box<DeploymentResponse>),
    Failed(String),
    Timeout,
}

/// Parse KEY=VALUE environment variable strings into a HashMap
pub fn parse_env_vars(env: &[String]) -> Result<HashMap<String, String>, DeployError> {
    let mut map = HashMap::new();

    for entry in env {
        let mut parts = entry.splitn(2, '=');
        let key = parts.next().ok_or_else(|| DeployError::Validation {
            message: format!("Invalid env var format: '{}'", entry),
        })?;
        let value = parts.next().ok_or_else(|| DeployError::Validation {
            message: format!("Invalid env var format: '{}'. Use KEY=VALUE", entry),
        })?;

        map.insert(key.to_string(), value.to_string());
    }

    Ok(map)
}

/// Create deployment with exponential backoff retry
pub async fn create_with_retry(
    client: &BasilicaClient,
    request: CreateDeploymentRequest,
) -> Result<DeploymentResponse, CliError> {
    use rand::Rng;

    let mut last_error = None;
    let mut delay = Duration::from_millis(INITIAL_RETRY_DELAY_MS);

    for attempt in 0..MAX_RETRIES {
        match client.create_deployment(request.clone()).await {
            Ok(response) => return Ok(response),
            Err(e) if is_quota_exceeded(&e) => {
                return Err(CliError::Deploy(DeployError::QuotaExceeded {
                    message: extract_quota_message(&e),
                }));
            }
            Err(e) if e.is_retryable() => {
                last_error = Some(e);
                if attempt < MAX_RETRIES - 1 {
                    let jitter_factor = rand::thread_rng().gen_range(0.75..1.25);
                    let jittered_delay = delay.mul_f64(jitter_factor);
                    tokio::time::sleep(jittered_delay).await;
                    delay *= 2;
                }
            }
            Err(e) => return Err(CliError::Api(e)),
        }
    }

    Err(CliError::Api(last_error.unwrap()))
}

/// Check if API error indicates quota exceeded
pub fn is_quota_exceeded(error: &basilica_sdk::error::ApiError) -> bool {
    match error {
        basilica_sdk::error::ApiError::QuotaExceeded { .. } => true,
        basilica_sdk::error::ApiError::ApiResponse { status, message } => {
            *status == 403
                || *status == 429
                || message.to_lowercase().contains("quota")
                || message.to_lowercase().contains("limit exceeded")
        }
        _ => false,
    }
}

/// Extract quota message from API error
pub fn extract_quota_message(error: &basilica_sdk::error::ApiError) -> String {
    match error {
        basilica_sdk::error::ApiError::QuotaExceeded { message } => message.clone(),
        basilica_sdk::error::ApiError::ApiResponse { message, .. } => message.clone(),
        _ => error.to_string(),
    }
}

/// Wait for deployment to become ready with status updates
///
/// # Arguments
/// * `client` - The Basilica API client
/// * `name` - The deployment name to monitor
/// * `timeout_secs` - Maximum time to wait in seconds
/// * `service_name` - Display name for the service (e.g., "vLLM", "SGLang")
pub async fn wait_for_ready(
    client: &BasilicaClient,
    name: &str,
    timeout_secs: u32,
    service_name: &str,
) -> Result<WaitResult, CliError> {
    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs as u64);
    let mut last_phase: Option<String> = None;
    let mut spinner = create_spinner(&format!("Waiting for {} summons...", service_name));

    loop {
        if start.elapsed() > timeout {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Timeout);
        }

        let status = client.get_deployment(name).await.map_err(CliError::Api)?;

        match status.phase.as_ref() {
            Some(phase) if !phase.trim().is_empty() => {
                if last_phase.as_ref() != Some(phase) {
                    complete_spinner_and_clear(spinner);

                    let phase_trimmed = phase.trim();
                    let phase_msg = format_phase_message(phase_trimmed, service_name);
                    print_info(&phase_msg);
                    spinner = create_spinner(&format!(
                        "Phase: {} ({}/{})",
                        phase_trimmed, status.replicas.ready, status.replicas.desired
                    ));
                    last_phase = Some(phase.clone());
                }
            }
            _ => {
                if last_phase.is_none() {
                    complete_spinner_and_clear(spinner);
                    let state_msg = format!(
                        "{} state: {} ({}/{})",
                        service_name, status.state, status.replicas.ready, status.replicas.desired
                    );
                    print_info(&state_msg);
                    spinner = create_spinner(&format!(
                        "State: {} ({}/{})",
                        status.state, status.replicas.ready, status.replicas.desired
                    ));
                    last_phase = Some(String::new());
                }
            }
        }

        if status.state == "Active" && status.replicas.ready >= status.replicas.desired {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Ready(Box::new(status)));
        }

        if status.state == "Failed" || status.phase.as_deref() == Some("failed") {
            complete_spinner_and_clear(spinner);
            let reason = status
                .message
                .unwrap_or_else(|| "Unknown error".to_string());
            return Ok(WaitResult::Failed(reason));
        }

        if status.state == "Terminating" || status.phase.as_deref() == Some("terminating") {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Failed(
                "Summons is being terminated".to_string(),
            ));
        }

        let sleep_duration = match status.phase.as_deref() {
            Some("scheduling") | Some("pulling") => Duration::from_secs(10),
            Some("storage_sync") => Duration::from_secs(3),
            _ => Duration::from_secs(5),
        };

        tokio::time::sleep(sleep_duration).await;
    }
}

/// Format human-readable phase message for a specific service
pub fn format_phase_message(phase: &str, service_name: &str) -> String {
    match phase {
        "pending" => "Summons created, waiting for scheduler...".to_string(),
        "scheduling" => "Finding suitable GPU node...".to_string(),
        "pulling" => format!("Pulling {} container image...", service_name),
        "initializing" => "Running init containers...".to_string(),
        "storage_sync" => "Syncing model cache storage...".to_string(),
        "starting" => format!("Starting {} server (loading model)...", service_name),
        "health_check" => "Running health checks...".to_string(),
        "ready" => format!("{} server ready!", service_name),
        _ => format!("Phase: {}", phase),
    }
}
