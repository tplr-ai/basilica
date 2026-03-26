//! SGLang deployment template
//!
//! Provides a pre-configured deployment for SGLang inference servers.
//! Handles GPU auto-detection, storage configuration, and health checks.

use crate::cli::commands::{SglangOptions, TemplateCommonOptions};
use crate::error::{CliError, DeployError};
use crate::output::{print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{
    CreateDeploymentRequest, DeploymentResponse, GpuRequirementsSpec, HealthCheckConfig,
    PersistentStorageSpec, ProbeConfig, ResourceRequirements, StorageBackend, StorageSpec,
};
use basilica_sdk::BasilicaClient;

use super::common::{create_with_retry, parse_env_vars, wait_for_ready, WaitResult};
use super::model_size::estimate_gpu_requirements;

/// Default SGLang Docker image
const SGLANG_IMAGE: &str = "lmsysorg/sglang:latest";

/// Default model for quick testing
const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";

/// Default port for SGLang server
const SGLANG_PORT: u32 = 30000;

/// Handle SGLang deployment
pub async fn handle_sglang_deploy(
    client: &BasilicaClient,
    model: Option<String>,
    common: TemplateCommonOptions,
    sglang: SglangOptions,
) -> Result<(), CliError> {
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // Always estimate GPU requirements to get recommended GPU
    let estimated = estimate_gpu_requirements(&model);

    // Use user-specified GPU count or auto-detected
    let gpu_count = common.gpu.unwrap_or_else(|| {
        print_info(&format!(
            "Auto-detected GPU requirements: {} GPU(s), ~{}GB VRAM ({})",
            estimated.gpu_count, estimated.memory_gb, estimated.recommended_gpu
        ));
        estimated.gpu_count
    });

    // Use user-specified GPU models or recommended GPU
    let gpu_models = if common.gpu_model.is_empty() {
        vec![estimated.recommended_gpu.clone()]
    } else {
        common.gpu_model.clone()
    };

    // Generate deployment name
    let name = common
        .name
        .clone()
        .unwrap_or_else(|| generate_sglang_name(&model));

    // Build SGLang command
    let (command, args) = build_sglang_command(&model, &sglang);

    // Parse environment variables
    let env = parse_env_vars(&common.env)?;

    // Build resources
    let resources = build_sglang_resources(gpu_count, &common, &gpu_models);

    // Build storage spec (enabled by default for model caching)
    let storage = if common.no_storage {
        None
    } else {
        Some(build_sglang_storage())
    };

    // Build health check
    let health_check = Some(build_sglang_health_check());

    // Create deployment request
    let request = CreateDeploymentRequest {
        instance_name: name.clone(),
        image: SGLANG_IMAGE.to_string(),
        replicas: 1,
        port: SGLANG_PORT,
        command: Some(command),
        args: Some(args),
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
        websocket: None,
        public_metadata: false,
    };

    // Show spinner
    let spinner = create_spinner(&format!(
        "Creating SGLang summons '{}' with model '{}'...",
        name, model
    ));

    // Create deployment with retry
    let response = create_with_retry(client, request).await?;

    complete_spinner_and_clear(spinner);

    let actual_name = response.instance_name.clone();

    // Wait for ready if not detached
    if !common.detach {
        let result = wait_for_ready(client, &actual_name, common.timeout, "SGLang").await?;

        match result {
            WaitResult::Ready(deployment) => {
                if common.json {
                    crate::output::json_output(&deployment)?;
                } else {
                    print_sglang_success(&deployment, &model);
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
            "SGLang summons '{}' created (detached mode)",
            actual_name
        ));
        println!("  Check status: basilica summon status {}", actual_name);
    }

    Ok(())
}

/// Generate RFC 1123 compliant deployment name from model
fn generate_sglang_name(model: &str) -> String {
    let model_part = model
        .split('/')
        .next_back()
        .unwrap_or(model)
        .to_lowercase()
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c)
            } else if c == '-' || c == '_' || c == '.' {
                Some('-')
            } else {
                None
            }
        })
        .collect::<String>();

    let sanitized = model_part.trim_matches('-');
    let prefix = if sanitized.len() > 40 {
        &sanitized[..40]
    } else {
        sanitized
    };

    let prefix = if prefix.is_empty() { "sglang" } else { prefix };

    format!(
        "sglang-{}-{}",
        prefix,
        &uuid::Uuid::new_v4().to_string()[..8]
    )
}

/// Build SGLang launch command and arguments
fn build_sglang_command(model: &str, opts: &SglangOptions) -> (Vec<String>, Vec<String>) {
    let mut args = vec![
        "-m".to_string(),
        "sglang.launch_server".to_string(),
        "--model-path".to_string(),
        model.to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        SGLANG_PORT.to_string(),
    ];

    if let Some(tp) = opts.tensor_parallel_size {
        args.extend(["--tp".to_string(), tp.to_string()]);
    }

    if let Some(ctx_len) = opts.context_length {
        args.extend(["--context-length".to_string(), ctx_len.to_string()]);
    }

    if let Some(ref quant) = opts.quantization {
        args.extend(["--quantization".to_string(), quant.clone()]);
    }

    if let Some(mem_frac) = opts.mem_fraction_static {
        args.extend(["--mem-fraction-static".to_string(), mem_frac.to_string()]);
    }

    if opts.trust_remote_code {
        args.push("--trust-remote-code".to_string());
    }

    (vec!["python".to_string()], args)
}

/// Build resource requirements for SGLang
fn build_sglang_resources(
    gpu_count: u32,
    common: &TemplateCommonOptions,
    gpu_models: &[String],
) -> ResourceRequirements {
    let gpus = if gpu_count > 0 {
        Some(GpuRequirementsSpec {
            count: gpu_count,
            model: gpu_models.to_vec(),
            min_cuda_version: None,
            min_gpu_memory_gb: None,
            interconnect: None,
            geo: None,
            spot: None,
            infiniband: None,
        })
    } else {
        None
    };

    ResourceRequirements {
        cpu: "4".to_string(),
        memory: common.memory.clone(),
        cpu_request: Some("2".to_string()),
        memory_request: Some("8Gi".to_string()),
        gpus,
    }
}

/// Build storage spec for SGLang model caching
fn build_sglang_storage() -> StorageSpec {
    StorageSpec {
        persistent: Some(PersistentStorageSpec {
            enabled: true,
            backend: StorageBackend::R2,
            bucket: String::new(), // API assigns default
            region: Some("auto".to_string()),
            endpoint: None,
            credentials_secret: Some("basilica-r2-credentials".to_string()),
            sync_interval_ms: 1000,
            cache_size_mb: 4096, // 4GB cache for model files
            mount_path: "/root/.cache".to_string(),
        }),
    }
}

/// Build health check configuration for SGLang
fn build_sglang_health_check() -> HealthCheckConfig {
    HealthCheckConfig {
        liveness: Some(ProbeConfig {
            path: "/health".to_string(),
            port: Some(SGLANG_PORT as u16),
            initial_delay_seconds: 60,
            period_seconds: 30,
            timeout_seconds: 10,
            failure_threshold: 3,
        }),
        readiness: Some(ProbeConfig {
            path: "/health".to_string(),
            port: Some(SGLANG_PORT as u16),
            initial_delay_seconds: 30,
            period_seconds: 10,
            timeout_seconds: 5,
            failure_threshold: 3,
        }),
        startup: Some(ProbeConfig {
            path: "/health".to_string(),
            port: Some(SGLANG_PORT as u16),
            initial_delay_seconds: 0,
            period_seconds: 10,
            timeout_seconds: 5,
            failure_threshold: 60, // Allow up to 10 minutes for model loading
        }),
    }
}

/// Print SGLang summons success message
fn print_sglang_success(deployment: &DeploymentResponse, model: &str) {
    print_success(&format!(
        "SGLang summons '{}' is ready!",
        deployment.instance_name
    ));
    println!();
    println!("  Model:      {}", model);
    println!("  URL:        {}", deployment.url);
    println!("  State:      {}", deployment.state);
    println!(
        "  Replicas:   {}/{}",
        deployment.replicas.ready, deployment.replicas.desired
    );
    println!();
    println!("SGLang API endpoints:");
    println!("  Generate:   {}/generate", deployment.url);
    println!("  Chat:       {}/v1/chat/completions", deployment.url);
    println!("  Health:     {}/health", deployment.url);
    println!();
    println!("Example usage:");
    println!("  curl {}/generate \\", deployment.url);
    println!("    -H \"Content-Type: application/json\" \\");
    println!("    -d '{{\"text\": \"Hello, \", \"sampling_params\": {{\"max_new_tokens\": 64}}}}'");
    println!();
    println!("Commands:");
    println!(
        "  View status:  basilica summon status {}",
        deployment.instance_name
    );
    println!(
        "  View logs:    basilica summon logs {}",
        deployment.instance_name
    );
    println!(
        "  Delete:       basilica summon delete {}",
        deployment.instance_name
    );
}
