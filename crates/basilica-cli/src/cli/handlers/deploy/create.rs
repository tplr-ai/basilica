//! Deployment creation with phase tracking

use crate::cli::commands::DeployCommand;
use crate::cli::commands::{SpreadModeArg, TopologySpreadOptions};
use crate::error::{CliError, DeployError};
use crate::output::{print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use crate::source::{Framework, SourcePackager, SourceType};
use basilica_sdk::types::{
    CreateDeploymentRequest, DeploymentResponse, GpuRequirementsSpec, HealthCheckConfig,
    PersistentStorageSpec, ProbeConfig, ResourceRequirements, SpreadMode, StorageBackend,
    StorageSpec, TopologySpreadConfig, WebSocketConfig,
};
use basilica_sdk::BasilicaClient;
use std::time::{Duration, Instant};

/// Maximum retries for transient failures
const MAX_RETRIES: u32 = 3;

/// Initial retry delay
const INITIAL_RETRY_DELAY_MS: u64 = 500;

/// Create a new deployment
pub async fn handle_create(
    client: &BasilicaClient,
    source: &str,
    cmd: DeployCommand,
) -> Result<(), CliError> {
    // 1. Parse source and determine type
    let packager = SourcePackager::new(source).map_err(DeployError::from)?;

    // 2. Build deployment name (RFC 1123 compliant)
    let name = cmd
        .naming
        .name
        .clone()
        .unwrap_or_else(|| super::helpers::generate_deployment_name(source));

    // 3. Build image and command based on source type
    let (image, command, args) = match packager.source_type() {
        SourceType::DockerImage(img) => (img.clone(), None, None),
        SourceType::PythonFile { .. } | SourceType::InlineCode(_) => {
            let img = cmd
                .naming
                .image
                .clone()
                .unwrap_or_else(|| "python:3.11-slim".to_string());
            match packager.build_command(&cmd.networking.pip) {
                Some((cmd_parts, args_parts)) => (img, Some(cmd_parts), Some(args_parts)),
                None => (img, None, None),
            }
        }
    };

    // 4. Parse environment variables
    let env = super::helpers::parse_env_vars(&cmd.networking.env)?;

    // 5. Build resource requirements with request/limit distinction
    let resources = build_resources(&cmd);

    // 6. Build storage spec (with proper bucket handling)
    let storage = if cmd.storage.storage {
        Some(build_storage_spec(&cmd.storage))
    } else {
        None
    };

    // 7. Parse ports (primary port for K8s) - must be before health check
    let primary_port = super::helpers::parse_primary_port(&cmd.networking.port)?;

    // 8. Build health check config (uses primary_port for probe port default)
    let health_check = build_health_check(&cmd.health, &packager, primary_port);

    // 9. Build topology spread config (if specified)
    let topology_spread = build_topology_spread(&cmd.topology_spread);

    // 10. Build websocket config (if enabled)
    let websocket = build_websocket_config(&cmd.websocket)?;

    // 11. Build public flag using is_public() helper
    let is_public = cmd.networking.is_public();

    // 12. Create request
    let request = CreateDeploymentRequest {
        instance_name: name.clone(),
        image,
        replicas: cmd.naming.replicas,
        port: primary_port as u32,
        command,
        args,
        env: Some(env),
        resources: Some(resources),
        ttl_seconds: cmd.lifecycle.ttl,
        public: is_public,
        storage,
        health_check,
        enable_billing: true,
        queue_name: None,
        suspended: false,
        priority: None,
        topology_spread,
        websocket,
        public_metadata: cmd.networking.public_metadata,
    };

    // 13. Show progress spinner
    let spinner = create_spinner(&format!("Creating summons '{}'...", name));

    // 14. Create deployment with retry
    let response = create_with_retry(client, request.clone()).await?;

    complete_spinner_and_clear(spinner);

    // Use the instance_name returned by API (may differ from user-provided name)
    let actual_name = response.instance_name.clone();

    // 15. Wait for ready if not detached
    if !cmd.lifecycle.detach {
        let result = wait_for_ready_with_phases(
            client,
            &actual_name,
            cmd.lifecycle.timeout,
            cmd.show_phases,
        )
        .await?;

        match result {
            WaitResult::Ready(deployment) => {
                if cmd.json {
                    crate::output::json_output(&deployment)?;
                } else {
                    super::helpers::print_deployment_success(&deployment);
                    // Display share token for private deployments
                    if !is_public {
                        match (&deployment.share_token, &deployment.share_url) {
                            (Some(token), Some(share_url)) => {
                                super::helpers::print_share_token_info(token, share_url);
                            }
                            _ => {
                                // Token generation may have failed silently
                                crate::output::print_warning(
                                    "Share token was not generated. Use 'deploy share-token regenerate' to create one."
                                );
                            }
                        }
                    }
                }
            }
            WaitResult::Failed(reason) => {
                // Fetch events to help diagnose the failure
                fetch_and_print_events(client, &actual_name).await;
                return Err(CliError::Deploy(DeployError::DeploymentFailed {
                    name: actual_name,
                    reason,
                }));
            }
            WaitResult::Timeout => {
                // Fetch events to help diagnose the timeout
                fetch_and_print_events(client, &actual_name).await;
                return Err(CliError::Deploy(DeployError::Timeout {
                    name: actual_name,
                    timeout_secs: cmd.lifecycle.timeout,
                }));
            }
        }
    } else if cmd.json {
        crate::output::json_output(&response)?;
    } else {
        print_success(&format!(
            "Summons '{}' created (detached mode)",
            actual_name
        ));
        println!("  Check status: basilica summon status {}", actual_name);
    }

    Ok(())
}

/// Create deployment with exponential backoff retry and jitter
async fn create_with_retry(
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
                // Quota exceeded is not retryable - fail immediately with specific error
                return Err(CliError::Deploy(DeployError::QuotaExceeded {
                    message: extract_quota_message(&e),
                }));
            }
            Err(e) if e.is_retryable() => {
                last_error = Some(e);
                if attempt < MAX_RETRIES - 1 {
                    // Add jitter: delay +/- 25%
                    let jitter_factor = rand::thread_rng().gen_range(0.75..1.25);
                    let jittered_delay = delay.mul_f64(jitter_factor);
                    tokio::time::sleep(jittered_delay).await;
                    delay *= 2; // Exponential backoff
                }
            }
            Err(e) => return Err(CliError::Api(e)),
        }
    }

    Err(CliError::Api(last_error.unwrap()))
}

/// Check if API error indicates quota exceeded
fn is_quota_exceeded(error: &basilica_sdk::error::ApiError) -> bool {
    match error {
        basilica_sdk::error::ApiError::QuotaExceeded { .. } => true,
        basilica_sdk::error::ApiError::ApiResponse { status, message } => {
            *status == 403
                || *status == 429
                || message.to_lowercase().contains("quota")
                || message.to_lowercase().contains("limit exceeded")
                || message.to_lowercase().contains("resource quota")
        }
        _ => false,
    }
}

/// Extract quota-specific message from API error
fn extract_quota_message(error: &basilica_sdk::error::ApiError) -> String {
    match error {
        basilica_sdk::error::ApiError::QuotaExceeded { message } => message.clone(),
        basilica_sdk::error::ApiError::ApiResponse { message, .. } => message.clone(),
        _ => error.to_string(),
    }
}

/// Result of waiting for deployment
enum WaitResult {
    Ready(Box<DeploymentResponse>),
    Failed(String),
    Timeout,
}

/// Wait for deployment with phase-aware progress display
async fn wait_for_ready_with_phases(
    client: &BasilicaClient,
    name: &str,
    timeout_secs: u32,
    verbose: bool,
) -> Result<WaitResult, CliError> {
    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs as u64);
    let mut last_phase: Option<String> = None;
    let mut spinner = create_spinner("Waiting for summons...");

    loop {
        if start.elapsed() > timeout {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Timeout);
        }

        let status = client.get_deployment(name).await.map_err(CliError::Api)?;

        // Update phase display
        if let Some(ref phase) = status.phase {
            if last_phase.as_ref() != Some(phase) {
                complete_spinner_and_clear(spinner);

                if verbose {
                    let phase_msg = format_phase_message(phase, &status);
                    print_info(&phase_msg);
                }

                spinner = create_spinner(&format!(
                    "Phase: {} ({}/{})",
                    phase, status.replicas.ready, status.replicas.desired
                ));

                last_phase = Some(phase.clone());
            }

            // Show progress for storage sync
            if verbose && phase == "storage_sync" {
                if let Some(ref progress) = status.progress {
                    let pct = progress.percentage.unwrap_or(0.0);
                    print_info(&format!("  Storage sync: {:.1}%", pct));
                }
            }
        }

        // Check terminal states
        if status.state == "Active" && status.replicas.ready >= status.replicas.desired {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Ready(Box::new(status)));
        }

        if status.state == "Failed" {
            complete_spinner_and_clear(spinner);
            let reason = status
                .message
                .unwrap_or_else(|| "Unknown error".to_string());
            return Ok(WaitResult::Failed(reason));
        }

        // Handle Terminating phase - summons is being deleted while we wait
        if status.state == "Terminating" || status.phase.as_deref() == Some("terminating") {
            complete_spinner_and_clear(spinner);
            return Ok(WaitResult::Failed(
                "Summons is being terminated - it may have been deleted externally".to_string(),
            ));
        }

        // Dynamic sleep based on phase
        let sleep_duration = match status.phase.as_deref() {
            Some("scheduling") | Some("pulling") => Duration::from_secs(10),
            Some("storage_sync") => Duration::from_secs(3),
            _ => Duration::from_secs(5),
        };

        tokio::time::sleep(sleep_duration).await;
    }
}

/// Format human-readable phase message
fn format_phase_message(phase: &str, status: &DeploymentResponse) -> String {
    match phase {
        "pending" => "Summons created, waiting for scheduler...".to_string(),
        "scheduling" => "Finding suitable node for summons...".to_string(),
        "pulling" => "Pulling container image...".to_string(),
        "initializing" => "Running init containers...".to_string(),
        "storage_sync" => "Syncing storage volume...".to_string(),
        "starting" => "Starting application container...".to_string(),
        "health_check" => "Running health checks...".to_string(),
        "ready" => format!(
            "Summons ready! {}/{} replicas running",
            status.replicas.ready, status.replicas.desired
        ),
        "degraded" => format!(
            "Summons degraded: {}/{} replicas ready",
            status.replicas.ready, status.replicas.desired
        ),
        "failed" => "Summons failed".to_string(),
        "terminating" => "Summons is being terminated...".to_string(),
        _ => format!("Phase: {}", phase),
    }
}

/// Fetch and print summons events for debugging failures
async fn fetch_and_print_events(client: &BasilicaClient, name: &str) {
    match client.get_deployment_events(name, Some(10)).await {
        Ok(response) if !response.events.is_empty() => {
            eprintln!("\nRecent events for summons '{}':", name);
            for event in response.events.iter() {
                let event_type = match event.event_type.as_str() {
                    "Warning" => "\x1b[33mWarning\x1b[0m",
                    "Normal" => "Normal",
                    _ => &event.event_type,
                };
                let count_str = event
                    .count
                    .map(|c| format!(" (x{})", c))
                    .unwrap_or_default();
                eprintln!(
                    "  [{}] {}: {}{}",
                    event_type, event.reason, event.message, count_str
                );
            }
        }
        Ok(_) => {} // No events found
        Err(e) => {
            tracing::debug!("Failed to fetch deployment events: {}", e);
        }
    }
}

/// Build resource requirements with request/limit distinction
fn build_resources(cmd: &DeployCommand) -> ResourceRequirements {
    let gpus = cmd.gpu.gpu.map(|count| GpuRequirementsSpec {
        count,
        model: cmd.gpu.gpu_model.clone(),
        min_cuda_version: cmd.gpu.cuda_version.clone(),
        min_gpu_memory_gb: cmd.gpu.gpu_memory_gb,
        interconnect: cmd.gpu.interconnect.clone(),
        geo: cmd.gpu.region.clone(),
        spot: if cmd.gpu.spot {
            Some(true)
        } else if cmd.gpu.exclude_spot {
            Some(false)
        } else {
            None
        },
        infiniband: None,
    });

    // Use explicit requests or default to limits
    let cpu_request = cmd
        .resources
        .cpu_request
        .clone()
        .unwrap_or_else(|| cmd.resources.cpu.clone());
    let memory_request = cmd
        .resources
        .memory_request
        .clone()
        .unwrap_or_else(|| cmd.resources.memory.clone());

    ResourceRequirements {
        cpu: cmd.resources.cpu.clone(),
        memory: cmd.resources.memory.clone(),
        cpu_request: Some(cpu_request),
        memory_request: Some(memory_request),
        gpus,
    }
}

/// Build storage spec with proper bucket handling
fn build_storage_spec(storage: &crate::cli::commands::StorageOptions) -> StorageSpec {
    StorageSpec {
        persistent: Some(PersistentStorageSpec {
            enabled: true,
            backend: StorageBackend::R2,
            // Leave bucket empty - API will assign default user bucket
            // based on user_id from authentication context
            bucket: String::new(),
            region: Some("auto".to_string()),
            endpoint: None,
            // Secret "basilica-r2-credentials" must be provisioned by operator in user namespace
            credentials_secret: Some("basilica-r2-credentials".to_string()),
            sync_interval_ms: storage.storage_sync_ms,
            cache_size_mb: storage.storage_cache_mb,
            mount_path: storage.storage_path.clone(),
        }),
    }
}

/// Build topology spread config from CLI options
fn build_topology_spread(options: &TopologySpreadOptions) -> Option<TopologySpreadConfig> {
    // Determine effective mode
    let mode = if options.unique_nodes {
        SpreadMode::UniqueNodes
    } else {
        match options.spread_mode {
            Some(SpreadModeArg::Preferred) => SpreadMode::Preferred,
            Some(SpreadModeArg::Required) => SpreadMode::Required,
            Some(SpreadModeArg::UniqueNodes) => SpreadMode::UniqueNodes,
            None => return None, // No topology spread specified, use API default
        }
    };

    Some(TopologySpreadConfig {
        mode,
        max_skew: options.max_skew,
        topology_key: options.topology_key.clone(),
    })
}

/// Build websocket config from CLI options
fn build_websocket_config(
    options: &crate::cli::commands::WebSocketOptions,
) -> Result<Option<WebSocketConfig>, CliError> {
    if !options.websocket {
        return Ok(None);
    }

    let timeout = options.ws_idle_timeout;
    if !(60..=3600).contains(&timeout) {
        return Err(CliError::Deploy(DeployError::Validation {
            message: format!(
                "--ws-idle-timeout must be between 60 and 3600 seconds, got {}",
                timeout
            ),
        }));
    }

    Ok(Some(WebSocketConfig {
        enabled: true,
        idle_timeout_seconds: timeout,
    }))
}

/// Build health check configuration with startup probe support
fn build_health_check(
    health: &crate::cli::commands::HealthCheckOptions,
    packager: &SourcePackager,
    primary_port: u16,
) -> Option<HealthCheckConfig> {
    // Determine probe port (explicit or default to primary container port)
    let probe_port = health.health_port.unwrap_or(primary_port);

    // Use explicit paths if provided
    let liveness_path = health
        .liveness_path
        .clone()
        .or_else(|| health.health_path.clone());
    let readiness_path = health
        .readiness_path
        .clone()
        .or_else(|| health.health_path.clone());
    let startup_path = health
        .startup_path
        .clone()
        .or_else(|| health.health_path.clone());

    // Auto-configure for known frameworks if no explicit config
    let (auto_liveness, auto_readiness, auto_startup) =
        if liveness_path.is_none() && readiness_path.is_none() {
            auto_health_check_for_framework(packager)
        } else {
            (None, None, None)
        };

    let final_liveness = liveness_path.or(auto_liveness);
    let final_readiness = readiness_path.or(auto_readiness);
    let final_startup = startup_path.or(auto_startup);

    if final_liveness.is_none() && final_readiness.is_none() && final_startup.is_none() {
        return None;
    }

    Some(HealthCheckConfig {
        liveness: final_liveness.map(|path| ProbeConfig {
            path,
            port: Some(probe_port),
            initial_delay_seconds: health.health_initial_delay,
            period_seconds: health.health_period,
            timeout_seconds: health.health_timeout,
            failure_threshold: health.health_failure_threshold,
        }),
        readiness: final_readiness.map(|path| ProbeConfig {
            path,
            port: Some(probe_port),
            initial_delay_seconds: health.health_initial_delay,
            period_seconds: health.health_period,
            timeout_seconds: health.health_timeout,
            failure_threshold: health.health_failure_threshold,
        }),
        startup: final_startup.map(|path| ProbeConfig {
            path,
            port: Some(probe_port),
            // Startup probes have no initial delay - they run immediately
            initial_delay_seconds: 0,
            period_seconds: health.health_period,
            timeout_seconds: health.health_timeout,
            // Higher threshold for slow-starting apps (ML models, etc.)
            failure_threshold: health.startup_failure_threshold,
        }),
    })
}

/// Auto-configure health checks for known frameworks
/// Returns (liveness, readiness, startup) paths
fn auto_health_check_for_framework(
    packager: &SourcePackager,
) -> (Option<String>, Option<String>, Option<String>) {
    match packager.detect_framework() {
        Framework::FastApi => {
            // FastAPI auto-generates /docs (Swagger UI) which serves as a weak fallback.
            // For production, users should define explicit /health endpoints.
            (None, Some("/docs".to_string()), Some("/docs".to_string()))
        }
        Framework::Flask => {
            // Flask typically has root endpoint
            (None, Some("/".to_string()), Some("/".to_string()))
        }
        Framework::Streamlit => {
            // Streamlit has health endpoint
            (
                None,
                Some("/_stcore/health".to_string()),
                Some("/_stcore/health".to_string()),
            )
        }
        _ => (None, None, None),
    }
}
