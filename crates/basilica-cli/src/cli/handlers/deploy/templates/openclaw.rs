//! OpenClaw deployment template
//!
//! Provides a pre-configured deployment for OpenClaw.
//! Handles backend URL wiring, environment configuration, and UI token extraction.

use crate::cli::commands::{OpenclawOptions, OpenclawProvider, TemplateCommonOptions};
use crate::error::{CliError, DeployError};
use crate::output::print_success;
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{
    CreateDeploymentRequest, HealthCheckConfig, PersistentStorageSpec, ProbeConfig,
    ResourceRequirements, StorageBackend, StorageSpec, WebSocketConfig,
};
use basilica_sdk::BasilicaClient;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::env as std_env;
use std::time::{Duration, Instant};

use super::common::{create_with_retry, parse_env_vars, wait_for_ready, WaitResult};

/// Default OpenClaw Docker image
const OPENCLAW_IMAGE: &str = "ghcr.io/one-covenant/basilica-openclaw:latest";

/// Default port for OpenClaw gateway
const OPENCLAW_PORT: u32 = 18789;

/// Handle OpenClaw deployment
pub async fn handle_openclaw_deploy(
    client: &BasilicaClient,
    common: TemplateCommonOptions,
    openclaw: OpenclawOptions,
) -> Result<(), CliError> {
    let name = common
        .name
        .clone()
        .unwrap_or_else(|| format!("openclaw-{}", &uuid::Uuid::new_v4().to_string()[..8]));

    let mut env = parse_env_vars(&common.env)?;
    let preset = provider_preset(openclaw.provider);

    let backend_url = openclaw
        .backend_url
        .clone()
        .unwrap_or_else(|| preset.base_url.to_string());
    let backend_url = normalize_backend_url(&backend_url);
    inject_backend_env(&mut env, &backend_url);

    let provider_id = openclaw
        .provider_id
        .clone()
        .unwrap_or_else(|| preset.provider_id.to_string());
    let provider_api = openclaw
        .provider_api
        .clone()
        .unwrap_or_else(|| preset.provider_api.to_string());
    let context_window = openclaw.context_window.unwrap_or(preset.context_window);
    let max_tokens = openclaw.max_tokens.unwrap_or(preset.max_tokens);

    let model_id = match openclaw
        .model_id
        .clone()
        .or_else(|| preset.model_id.map(str::to_string))
    {
        Some(id) => id,
        None => {
            if provider_id != "openai" {
                return Err(CliError::Deploy(DeployError::Validation {
                    message: "Model ID is required for this provider. Use --model-id.".to_string(),
                }));
            }
            detect_model_id(&backend_url, &env).await?
        }
    };

    env.insert("OPENCLAW_MODEL_ID".to_string(), model_id);
    env.insert("OPENCLAW_PROVIDER_ID".to_string(), provider_id);
    env.insert("OPENCLAW_PROVIDER_API".to_string(), provider_api);
    env.insert(
        "OPENCLAW_CONTEXT_WINDOW".to_string(),
        context_window.to_string(),
    );
    env.insert("OPENCLAW_MAX_TOKENS".to_string(), max_tokens.to_string());

    // Pass through provider API keys from local env if not explicitly set via --env.
    if !env.contains_key("OPENAI_API_KEY") {
        if let Ok(val) = std_env::var("OPENAI_API_KEY") {
            if !val.trim().is_empty() {
                env.insert("OPENAI_API_KEY".to_string(), val);
            }
        }
    }
    if !env.contains_key("ANTHROPIC_API_KEY") {
        if let Ok(val) = std_env::var("ANTHROPIC_API_KEY") {
            if !val.trim().is_empty() {
                env.insert("ANTHROPIC_API_KEY".to_string(), val);
            }
        }
    }

    match env.get("OPENCLAW_PROVIDER_ID").map(|s| s.as_str()) {
        Some("anthropic") => {
            if !env.contains_key("ANTHROPIC_API_KEY") {
                return Err(CliError::Deploy(DeployError::Validation {
                    message: "ANTHROPIC_API_KEY is required for provider=anthropic. Set it in your environment or pass --env ANTHROPIC_API_KEY=...".to_string(),
                }));
            }
        }
        _ => {
            if !env.contains_key("OPENAI_API_KEY") {
                return Err(CliError::Deploy(DeployError::Validation {
                    message: "OPENAI_API_KEY is required for provider=openai. Set it in your environment or pass --env OPENAI_API_KEY=...".to_string(),
                }));
            }
        }
    }

    let resources = build_openclaw_resources(&common);

    let storage = if common.no_storage {
        None
    } else {
        Some(build_openclaw_storage())
    };

    let health_check = Some(build_openclaw_health_check());

    let request = CreateDeploymentRequest {
        instance_name: name.clone(),
        image: OPENCLAW_IMAGE.to_string(),
        replicas: 1,
        port: OPENCLAW_PORT,
        command: Some(vec!["/usr/local/bin/basilica-entrypoint.sh".to_string()]),
        args: None,
        env: Some(env),
        resources: Some(resources),
        ttl_seconds: common.ttl,
        public: true,
        storage,
        health_check,
        enable_billing: true,
        queue_name: None,
        suspended: false,
        priority: None,
        topology_spread: None,
        websocket: Some(WebSocketConfig::default()),
        public_metadata: false,
    };

    let spinner = create_spinner(&format!("Creating OpenClaw summons '{}'...", name));
    let response = create_with_retry(client, request).await?;
    complete_spinner_and_clear(spinner);

    let actual_name = response.instance_name.clone();

    if !common.detach {
        let result = wait_for_ready(client, &actual_name, common.timeout, "OpenClaw").await?;

        match result {
            WaitResult::Ready(deployment) => {
                if common.json {
                    crate::output::json_output(&deployment)?;
                } else {
                    let token = wait_for_gateway_token(client, &actual_name).await?;
                    wait_for_public_url_ready(&deployment.url, 90).await?;
                    print_openclaw_success(&deployment, &actual_name, &token);
                }
            }
            WaitResult::Failed(reason) => {
                return Err(CliError::Deploy(DeployError::DeploymentFailed {
                    name: actual_name,
                    reason,
                }));
            }
            WaitResult::Timeout => {
                return Err(CliError::Deploy(DeployError::Timeout {
                    name: actual_name,
                    timeout_secs: common.timeout,
                }));
            }
        }
    } else if common.json {
        crate::output::json_output(&response)?;
    } else {
        print_success(&format!(
            "OpenClaw summons '{}' created (detached mode)",
            actual_name
        ));
        println!("  Check status: basilica summon status {}", actual_name);
    }

    Ok(())
}

fn inject_backend_env(env: &mut HashMap<String, String>, backend_url: &str) {
    env.entry("OPENCLAW_BASE_URL".to_string())
        .or_insert_with(|| backend_url.to_string());
    env.entry("OPENAI_BASE_URL".to_string())
        .or_insert_with(|| backend_url.to_string());
}

fn normalize_backend_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{}/v1", trimmed)
    }
}

struct ProviderPreset {
    base_url: &'static str,
    provider_id: &'static str,
    provider_api: &'static str,
    model_id: Option<&'static str>,
    context_window: u32,
    max_tokens: u32,
}

fn provider_preset(provider: OpenclawProvider) -> ProviderPreset {
    match provider {
        OpenclawProvider::Openai => ProviderPreset {
            base_url: "https://api.openai.com/v1",
            provider_id: "openai",
            provider_api: "openai-responses",
            model_id: Some("gpt-5.2-pro"),
            context_window: 400_000,
            max_tokens: 128_000,
        },
        OpenclawProvider::Anthropic => ProviderPreset {
            base_url: "https://api.anthropic.com/v1",
            provider_id: "anthropic",
            provider_api: "anthropic-messages",
            model_id: Some("claude-opus-4-1-20250805"),
            context_window: 200_000,
            max_tokens: 32_000,
        },
    }
}

async fn detect_model_id(
    backend_url: &str,
    env: &HashMap<String, String>,
) -> Result<String, CliError> {
    let models_url = format!("{}/models", backend_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let mut request = client.get(&models_url);

    if let Some(key) = env.get("OPENAI_API_KEY") {
        request = request.bearer_auth(key);
    }

    let response = request.send().await.map_err(|e| {
        CliError::Deploy(DeployError::Validation {
            message: format!("Failed to fetch models from {}: {}", models_url, e),
        })
    })?;

    if !response.status().is_success() {
        return Err(CliError::Deploy(DeployError::Validation {
            message: format!(
                "Failed to fetch models from {}: HTTP {}. Provide --model-id.",
                models_url,
                response.status()
            ),
        }));
    }

    let body: Value = response.json().await.map_err(|e| {
        CliError::Deploy(DeployError::Validation {
            message: format!("Failed to parse models response: {}", e),
        })
    })?;

    if let Some(id) = body
        .get("data")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.iter().find_map(|item| item.get("id")))
        .and_then(|id| id.as_str())
    {
        return Ok(id.to_string());
    }

    if let Some(id) = body
        .get("models")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.iter().find_map(|item| item.get("id")))
        .and_then(|id| id.as_str())
    {
        return Ok(id.to_string());
    }

    Err(CliError::Deploy(DeployError::Validation {
        message: "No model ID found in /v1/models response. Provide --model-id.".to_string(),
    }))
}

fn build_openclaw_resources(common: &TemplateCommonOptions) -> ResourceRequirements {
    ResourceRequirements {
        cpu: "2".to_string(),
        memory: common.memory.clone(),
        cpu_request: Some("1".to_string()),
        memory_request: Some("4Gi".to_string()),
        gpus: None,
    }
}

fn build_openclaw_storage() -> StorageSpec {
    StorageSpec {
        persistent: Some(PersistentStorageSpec {
            enabled: true,
            backend: StorageBackend::R2,
            bucket: String::new(),
            region: Some("auto".to_string()),
            endpoint: None,
            credentials_secret: Some("basilica-r2-credentials".to_string()),
            sync_interval_ms: 1000,
            cache_size_mb: 2048,
            mount_path: "/data".to_string(),
        }),
    }
}

fn build_openclaw_health_check() -> HealthCheckConfig {
    let liveness = ProbeConfig {
        path: "/".to_string(),
        port: Some(OPENCLAW_PORT as u16),
        initial_delay_seconds: 10,
        period_seconds: 10,
        timeout_seconds: 5,
        failure_threshold: 3,
    };

    let readiness = ProbeConfig {
        path: "/".to_string(),
        port: Some(OPENCLAW_PORT as u16),
        initial_delay_seconds: 5,
        period_seconds: 5,
        timeout_seconds: 5,
        failure_threshold: 2,
    };

    // 5s period * 36 failures = 180s (3 min) max startup window.
    // Covers: FUSE wait (60s) + config write + gateway start + proxy start.
    // Liveness/readiness probes are suspended until startup probe passes.
    let startup = ProbeConfig {
        path: "/".to_string(),
        port: Some(OPENCLAW_PORT as u16),
        initial_delay_seconds: 5,
        period_seconds: 5,
        timeout_seconds: 5,
        failure_threshold: 36,
    };

    HealthCheckConfig {
        liveness: Some(liveness),
        readiness: Some(readiness),
        startup: Some(startup),
    }
}

fn print_openclaw_success(
    deployment: &basilica_sdk::types::DeploymentResponse,
    name: &str,
    token: &str,
) {
    print_success(&format!("OpenClaw summons '{}' is ready!", name));
    println!("  URL: {}", deployment.url);
    println!(
        "  Control UI: {}/chat?session=main&token={}",
        deployment.url, token
    );
}

async fn wait_for_public_url_ready(url: &str, timeout_secs: u64) -> Result<(), CliError> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let client = reqwest::Client::new();
    let url = url.trim_end_matches('/');
    let probe_url = format!("{}/", url);

    while Instant::now() < deadline {
        match client.get(&probe_url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
    }

    Err(CliError::Deploy(DeployError::Timeout {
        name: url.to_string(),
        timeout_secs: timeout_secs as u32,
    }))
}

async fn wait_for_gateway_token(client: &BasilicaClient, name: &str) -> Result<String, CliError> {
    let deadline = Instant::now() + Duration::from_secs(120);
    let re = Regex::new(r"(?:CLAWDBOT|OPENCLAW)_GATEWAY_TOKEN=([a-fA-F0-9]{64})")
        .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))?;

    let mut last_error: Option<String> = None;

    while Instant::now() < deadline {
        match client.get_deployment_logs(name, false, Some(200)).await {
            Ok(response) => match response.text().await {
                Ok(body) => {
                    if let Some(caps) = re.captures(&body) {
                        let token = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
                        if !token.is_empty() {
                            return Ok(token.to_string());
                        }
                    }
                }
                Err(err) => {
                    last_error = Some(format!("Failed to read logs response: {}", err));
                }
            },
            Err(err) => {
                last_error = Some(format!("Failed to fetch logs: {}", err));
            }
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    let reason = match last_error {
        Some(err) => format!(
            "Gateway token not found in logs within timeout. Last error: {}",
            err
        ),
        None => "Gateway token not found in logs within timeout.".to_string(),
    };

    Err(CliError::Deploy(DeployError::DeploymentFailed {
        name: name.to_string(),
        reason,
    }))
}
