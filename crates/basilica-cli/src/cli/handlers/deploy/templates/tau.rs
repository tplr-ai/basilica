//! Tau deployment template
//!
//! Provides a pre-configured deployment for the Tau Telegram agent.

use crate::cli::commands::{TauOptions, TemplateCommonOptions};
use crate::error::{CliError, DeployError};
use crate::output::print_success;
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{
    CreateDeploymentRequest, HealthCheckConfig, PersistentStorageSpec, ProbeConfig,
    ResourceRequirements, StorageBackend, StorageSpec,
};
use basilica_sdk::BasilicaClient;
use std::collections::HashMap;
use std::env as std_env;

use super::common::{create_with_retry, parse_env_vars, wait_for_ready, WaitResult};

/// Default Tau Docker image
const TAU_IMAGE: &str = "ghcr.io/one-covenant/basilica-tau:latest";

/// Default port (Tau does not expose HTTP, but a port is required)
const TAU_PORT: u32 = 8080;

/// Handle Tau deployment
pub async fn handle_tau_deploy(
    client: &BasilicaClient,
    common: TemplateCommonOptions,
    tau: TauOptions,
) -> Result<(), CliError> {
    let name = common
        .name
        .clone()
        .unwrap_or_else(|| format!("tau-{}", &uuid::Uuid::new_v4().to_string()[..8]));

    let mut env = parse_env_vars(&common.env)?;

    if let Some(token) = tau.bot_token {
        env.insert("TAU_BOT_TOKEN".to_string(), token);
    }
    if let Some(token) = tau.chutes_api_token {
        env.insert("CHUTES_API_TOKEN".to_string(), token);
    }
    if let Some(model) = tau.chat_model {
        env.insert("TAU_CHAT_MODEL".to_string(), model);
    }

    copy_env_if_missing(&mut env, "TAU_BOT_TOKEN");
    copy_env_if_missing(&mut env, "CHUTES_API_TOKEN");

    require_env(
        &env,
        "TAU_BOT_TOKEN",
        "Set it in your environment or pass --env TAU_BOT_TOKEN=...",
    )?;
    require_env(
        &env,
        "CHUTES_API_TOKEN",
        "Set it in your environment or pass --env CHUTES_API_TOKEN=...",
    )?;

    let resources = build_tau_resources(&common);

    let storage = if common.no_storage {
        None
    } else {
        Some(build_tau_storage())
    };

    let request = CreateDeploymentRequest {
        instance_name: name.clone(),
        image: TAU_IMAGE.to_string(),
        replicas: 1,
        port: TAU_PORT,
        command: Some(vec!["/usr/local/bin/basilica-entrypoint.sh".to_string()]),
        args: None,
        env: Some(env),
        resources: Some(resources),
        ttl_seconds: common.ttl,
        public: false,
        storage,
        health_check: Some(build_tau_health_check()),
        enable_billing: true,
        queue_name: None,
        suspended: false,
        priority: None,
        topology_spread: None,
        websocket: None,
        public_metadata: false,
    };

    let spinner = create_spinner(&format!("Creating Tau summons '{}'...", name));
    let response = create_with_retry(client, request).await?;
    complete_spinner_and_clear(spinner);

    let actual_name = response.instance_name.clone();

    if !common.detach {
        let result = wait_for_ready(client, &actual_name, common.timeout, "Tau").await?;

        match result {
            WaitResult::Ready(deployment) => {
                if common.json {
                    crate::output::json_output(&deployment)?;
                } else {
                    print_tau_success(&deployment, &actual_name);
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
            "Tau summons '{}' created (detached mode)",
            actual_name
        ));
        println!("  Check status: basilica summon status {}", actual_name);
    }

    Ok(())
}

fn copy_env_if_missing(env: &mut HashMap<String, String>, key: &str) {
    if env.contains_key(key) {
        return;
    }
    if let Ok(val) = std_env::var(key) {
        if !val.trim().is_empty() {
            env.insert(key.to_string(), val);
        }
    }
}

fn require_env(env: &HashMap<String, String>, key: &str, hint: &str) -> Result<(), CliError> {
    if env.contains_key(key) {
        return Ok(());
    }
    Err(CliError::Deploy(DeployError::Validation {
        message: format!("{key} is required. {hint}"),
    }))
}

fn build_tau_resources(common: &TemplateCommonOptions) -> ResourceRequirements {
    ResourceRequirements {
        cpu: "2".to_string(),
        memory: common.memory.clone(),
        cpu_request: Some("1".to_string()),
        memory_request: Some("4Gi".to_string()),
        gpus: None,
    }
}

fn build_tau_storage() -> StorageSpec {
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

fn build_tau_health_check() -> HealthCheckConfig {
    let probe = ProbeConfig {
        path: "/health".to_string(),
        port: Some(TAU_PORT as u16),
        initial_delay_seconds: 30,
        period_seconds: 10,
        timeout_seconds: 5,
        failure_threshold: 3,
    };

    HealthCheckConfig {
        liveness: Some(probe.clone()),
        readiness: Some(probe),
        startup: None,
    }
}

fn print_tau_success(deployment: &basilica_sdk::types::DeploymentResponse, name: &str) {
    print_success(&format!("Tau summons '{}' is ready!", name));
    println!("  URL: {}", deployment.url);
    println!("  Next: send a message to your Telegram bot to initialize chat_id.txt");
    println!("  Logs: basilica summon logs {}", name);
}
