use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{ApiError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDeploymentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_name: Option<String>,
    pub image: String,
    pub replicas: u32,
    pub port: u32,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub resources: Option<ResourceRequirements>,
    #[serde(default)]
    pub ttl_seconds: Option<u32>,
    #[serde(default)]
    pub public: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health_check: Option<HealthCheckConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageSpec>,
    #[serde(default = "default_enable_billing")]
    pub enable_billing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_name: Option<String>,
    #[serde(default)]
    pub suspended: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
}

fn default_enable_billing() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceRequirements {
    pub cpu: String,
    pub memory: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<GpuRequirements>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuRequirements {
    pub count: u32,
    pub model: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_cuda_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_memory_gb: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistent: Option<PersistentStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistentStorageSpec {
    pub enabled: bool,
    pub backend: StorageBackend,
    pub bucket: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_secret: Option<String>,
    #[serde(default = "default_sync_interval")]
    pub sync_interval_ms: u64,
    #[serde(default = "default_cache_size")]
    pub cache_size_mb: usize,
    #[serde(default = "default_mount_path")]
    pub mount_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    R2,
    S3,
    GCS,
}

fn default_sync_interval() -> u64 {
    1000
}

fn default_cache_size() -> usize {
    1024
}

fn default_mount_path() -> String {
    "/mnt/storage".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthCheckConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liveness: Option<ProbeConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness: Option<ProbeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeConfig {
    pub path: String,
    #[serde(default = "default_initial_delay")]
    pub initial_delay_seconds: u32,
    #[serde(default = "default_period")]
    pub period_seconds: u32,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u32,
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
}

fn default_initial_delay() -> u32 {
    30
}

fn default_period() -> u32 {
    10
}

fn default_timeout() -> u32 {
    5
}

fn default_failure_threshold() -> u32 {
    3
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentResponse {
    pub instance_name: String,
    pub user_id: String,
    pub namespace: String,
    pub state: String,
    pub url: String,
    pub replicas: ReplicaStatus,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pods: Option<Vec<PodInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplicaStatus {
    pub desired: u32,
    pub ready: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PodInfo {
    pub name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteDeploymentResponse {
    pub instance_name: String,
    pub state: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentListResponse {
    pub deployments: Vec<DeploymentSummary>,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentSummary {
    pub instance_name: String,
    pub state: String,
    pub url: String,
    pub replicas: ReplicaStatus,
    pub created_at: String,
}

pub fn validate_instance_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "instance_name cannot be empty".to_string(),
        });
    }

    if name.len() > 63 {
        return Err(ApiError::InvalidRequest {
            message: format!("instance_name too long: {} characters (max 63)", name.len()),
        });
    }

    let dns_regex =
        Regex::new(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$").map_err(|e| ApiError::Internal {
            message: format!("Failed to compile regex: {}", e),
        })?;

    if !dns_regex.is_match(name) {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "instance_name '{}' must be a valid DNS label (lowercase alphanumeric and hyphens, \
                 cannot start or end with hyphen)",
                name
            ),
        });
    }

    Ok(())
}

pub fn validate_image(image: &str) -> Result<()> {
    if image.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "image cannot be empty".to_string(),
        });
    }

    if image.len() > 1024 {
        return Err(ApiError::InvalidRequest {
            message: format!("image URL too long: {} characters (max 1024)", image.len()),
        });
    }

    if image.contains(';') || image.contains('&') || image.contains('|') {
        return Err(ApiError::InvalidRequest {
            message: "image contains invalid characters (;, &, |)".to_string(),
        });
    }

    Ok(())
}

pub fn validate_port(port: u32) -> Result<()> {
    if port == 0 || port > 65535 {
        return Err(ApiError::InvalidRequest {
            message: format!("port must be in range 1-65535, got {}", port),
        });
    }

    Ok(())
}

pub fn validate_replicas(replicas: u32, max_replicas: u32) -> Result<()> {
    if replicas == 0 {
        return Err(ApiError::InvalidRequest {
            message: "replicas must be at least 1".to_string(),
        });
    }

    if replicas > max_replicas {
        return Err(ApiError::InvalidRequest {
            message: format!("replicas {} exceeds maximum of {}", replicas, max_replicas),
        });
    }

    Ok(())
}

pub fn validate_cpu_resource(cpu: &str) -> Result<()> {
    if cpu.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "cpu cannot be empty".to_string(),
        });
    }

    let cpu_regex = Regex::new(r"^[0-9]+m?$").map_err(|e| ApiError::Internal {
        message: format!("Failed to compile regex: {}", e),
    })?;

    if !cpu_regex.is_match(cpu) {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "invalid cpu format '{}' (expected format: '500m' or '2')",
                cpu
            ),
        });
    }

    Ok(())
}

pub fn validate_memory_resource(memory: &str) -> Result<()> {
    if memory.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "memory cannot be empty".to_string(),
        });
    }

    let memory_regex = Regex::new(r"^[0-9]+(Mi|Gi|M|G)?$").map_err(|e| ApiError::Internal {
        message: format!("Failed to compile regex: {}", e),
    })?;

    if !memory_regex.is_match(memory) {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "invalid memory format '{}' (expected format: '512Mi' or '2Gi')",
                memory
            ),
        });
    }

    Ok(())
}

pub fn validate_resources(resources: &ResourceRequirements) -> Result<()> {
    validate_cpu_resource(&resources.cpu)?;
    validate_memory_resource(&resources.memory)?;
    if let Some(ref gpus) = resources.gpus {
        validate_gpu_requirements(gpus)?;
    }
    Ok(())
}

pub fn validate_gpu_requirements(gpu: &GpuRequirements) -> Result<()> {
    if gpu.count == 0 {
        return Err(ApiError::InvalidRequest {
            message: "GPU count must be at least 1".to_string(),
        });
    }

    if gpu.count > 8 {
        return Err(ApiError::InvalidRequest {
            message: format!("GPU count {} exceeds maximum of 8", gpu.count),
        });
    }

    if gpu.model.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "GPU model list cannot be empty".to_string(),
        });
    }

    if gpu.model.len() > 10 {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "GPU model list too long: {} models (max 10)",
                gpu.model.len()
            ),
        });
    }

    for model in &gpu.model {
        if model.is_empty() {
            return Err(ApiError::InvalidRequest {
                message: "GPU model name cannot be empty".to_string(),
            });
        }
    }

    if let Some(ref cuda) = gpu.min_cuda_version {
        if !cuda.contains('.') {
            return Err(ApiError::InvalidRequest {
                message: format!(
                    "Invalid CUDA version format: '{}' (expected format: 'X.Y')",
                    cuda
                ),
            });
        }
    }

    if let Some(vram) = gpu.min_gpu_memory_gb {
        if vram == 0 {
            return Err(ApiError::InvalidRequest {
                message: "Minimum GPU memory must be at least 1 GB".to_string(),
            });
        }
        if vram > 256 {
            return Err(ApiError::InvalidRequest {
                message: format!("Minimum GPU memory {} GB exceeds maximum of 256 GB", vram),
            });
        }
    }

    Ok(())
}

pub fn validate_storage_spec(storage: &StorageSpec) -> Result<()> {
    if let Some(ref persistent) = storage.persistent {
        validate_persistent_storage(persistent)?;
    }
    Ok(())
}

pub fn validate_persistent_storage(storage: &PersistentStorageSpec) -> Result<()> {
    if !storage.enabled {
        return Ok(());
    }

    if storage.bucket.is_empty() {
        return Err(ApiError::InvalidRequest {
            message: "Storage bucket cannot be empty".to_string(),
        });
    }

    if storage.sync_interval_ms < 100 {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "Sync interval {} ms is too low (minimum 100 ms)",
                storage.sync_interval_ms
            ),
        });
    }

    if storage.sync_interval_ms > 60000 {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "Sync interval {} ms is too high (maximum 60000 ms)",
                storage.sync_interval_ms
            ),
        });
    }

    if storage.cache_size_mb < 512 {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "Cache size {} MB is too low (minimum 512 MB)",
                storage.cache_size_mb
            ),
        });
    }

    if storage.cache_size_mb > 16384 {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "Cache size {} MB is too high (maximum 16384 MB)",
                storage.cache_size_mb
            ),
        });
    }

    Ok(())
}

pub fn validate_create_deployment_request(
    req: &CreateDeploymentRequest,
    max_replicas: u32,
) -> Result<()> {
    if let Some(ref instance_name) = req.instance_name {
        if !instance_name.is_empty() {
            validate_instance_name(instance_name)?;
        }
    }

    validate_image(&req.image)?;
    validate_port(req.port)?;
    validate_replicas(req.replicas, max_replicas)?;

    if let Some(ref resources) = req.resources {
        validate_resources(resources)?;
    }

    if let Some(ref storage) = req.storage {
        validate_storage_spec(storage)?;
    }

    for key in req.env.keys() {
        if key.is_empty() {
            return Err(ApiError::InvalidRequest {
                message: "environment variable name cannot be empty".to_string(),
            });
        }
        if key.len() > 255 {
            return Err(ApiError::InvalidRequest {
                message: format!("environment variable name too long: {}", key),
            });
        }
    }

    Ok(())
}

pub fn sanitize_user_id(user_id: &str) -> String {
    let mut out = String::new();
    for ch in user_id.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' {
            out.push(ch);
        } else if ch.is_ascii_uppercase() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
        if out.len() >= 60 {
            break;
        }
    }
    if out.ends_with('-') {
        out.pop();
    }
    out
}

pub fn generate_instance_name() -> String {
    Uuid::new_v4().to_string()
}

pub fn generate_cr_name(instance_name: &str) -> String {
    format!("ud-{}", instance_name)
}

pub fn generate_path_prefix(instance_name: &str) -> String {
    format!("/deployments/{}", instance_name)
}

pub fn resolve_instance_name(provided: Option<String>) -> String {
    match provided {
        Some(name) if !name.is_empty() => {
            tracing::warn!(
                provided_name = %name,
                "User-provided instance_name is deprecated and will be ignored. Server will auto-generate UUID."
            );
            generate_instance_name()
        }
        _ => generate_instance_name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_instance_name_valid() {
        assert!(validate_instance_name("my-app").is_ok());
        assert!(validate_instance_name("app123").is_ok());
        assert!(validate_instance_name("a").is_ok());
        assert!(validate_instance_name("my-app-123").is_ok());
    }

    #[test]
    fn test_validate_instance_name_empty() {
        let result = validate_instance_name("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_validate_instance_name_too_long() {
        let long_name = "a".repeat(64);
        let result = validate_instance_name(&long_name);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    #[test]
    fn test_validate_instance_name_invalid_format() {
        assert!(validate_instance_name("My-App").is_err());
        assert!(validate_instance_name("my_app").is_err());
        assert!(validate_instance_name("-myapp").is_err());
        assert!(validate_instance_name("myapp-").is_err());
        assert!(validate_instance_name("my.app").is_err());
    }

    #[test]
    fn test_validate_image_valid() {
        assert!(validate_image("nginx:latest").is_ok());
        assert!(validate_image("gcr.io/project/image:tag").is_ok());
        assert!(validate_image("my-registry.com:5000/repo/image@sha256:abc").is_ok());
    }

    #[test]
    fn test_validate_image_empty() {
        assert!(validate_image("").is_err());
    }

    #[test]
    fn test_validate_image_too_long() {
        let long_image = format!("nginx:{}", "a".repeat(1020));
        assert!(validate_image(&long_image).is_err());
    }

    #[test]
    fn test_validate_image_invalid_characters() {
        assert!(validate_image("nginx;rm -rf /").is_err());
        assert!(validate_image("nginx&curl evil.com").is_err());
        assert!(validate_image("nginx|bash").is_err());
    }

    #[test]
    fn test_validate_port_valid() {
        assert!(validate_port(80).is_ok());
        assert!(validate_port(443).is_ok());
        assert!(validate_port(8080).is_ok());
        assert!(validate_port(65535).is_ok());
        assert!(validate_port(1).is_ok());
    }

    #[test]
    fn test_validate_port_invalid() {
        assert!(validate_port(0).is_err());
        assert!(validate_port(65536).is_err());
        assert!(validate_port(100000).is_err());
    }

    #[test]
    fn test_validate_replicas_valid() {
        assert!(validate_replicas(1, 10).is_ok());
        assert!(validate_replicas(5, 10).is_ok());
        assert!(validate_replicas(10, 10).is_ok());
    }

    #[test]
    fn test_validate_replicas_zero() {
        assert!(validate_replicas(0, 10).is_err());
    }

    #[test]
    fn test_validate_replicas_exceeds_max() {
        assert!(validate_replicas(11, 10).is_err());
        assert!(validate_replicas(100, 10).is_err());
    }

    #[test]
    fn test_validate_cpu_resource_valid() {
        assert!(validate_cpu_resource("500m").is_ok());
        assert!(validate_cpu_resource("1").is_ok());
        assert!(validate_cpu_resource("2").is_ok());
        assert!(validate_cpu_resource("1000m").is_ok());
    }

    #[test]
    fn test_validate_cpu_resource_invalid() {
        assert!(validate_cpu_resource("").is_err());
        assert!(validate_cpu_resource("1.5").is_err());
        assert!(validate_cpu_resource("500").is_ok());
        assert!(validate_cpu_resource("abc").is_err());
    }

    #[test]
    fn test_validate_memory_resource_valid() {
        assert!(validate_memory_resource("512Mi").is_ok());
        assert!(validate_memory_resource("2Gi").is_ok());
        assert!(validate_memory_resource("1024M").is_ok());
        assert!(validate_memory_resource("1G").is_ok());
    }

    #[test]
    fn test_validate_memory_resource_invalid() {
        assert!(validate_memory_resource("").is_err());
        assert!(validate_memory_resource("1.5Gi").is_err());
        assert!(validate_memory_resource("512").is_ok());
        assert!(validate_memory_resource("abc").is_err());
    }

    #[test]
    fn test_validate_resources() {
        let valid_resources = ResourceRequirements {
            cpu: "500m".to_string(),
            memory: "512Mi".to_string(),
            gpus: None,
        };
        assert!(validate_resources(&valid_resources).is_ok());

        let invalid_cpu = ResourceRequirements {
            cpu: "invalid".to_string(),
            memory: "512Mi".to_string(),
            gpus: None,
        };
        assert!(validate_resources(&invalid_cpu).is_err());

        let invalid_memory = ResourceRequirements {
            cpu: "500m".to_string(),
            memory: "invalid".to_string(),
            gpus: None,
        };
        assert!(validate_resources(&invalid_memory).is_err());
    }

    #[test]
    fn test_sanitize_user_id() {
        assert_eq!(sanitize_user_id("user123"), "user123");
        assert_eq!(sanitize_user_id("User123"), "user123");
        assert_eq!(sanitize_user_id("user_123"), "user-123");
        assert_eq!(sanitize_user_id("user@domain.com"), "user-domain-com");

        let long_id = "a".repeat(70);
        let sanitized = sanitize_user_id(&long_id);
        assert!(sanitized.len() <= 60);
    }

    #[test]
    fn test_sanitize_user_id_auth0_formats() {
        assert_eq!(sanitize_user_id("github|434149"), "github-434149");
        assert_eq!(
            sanitize_user_id("google-oauth2|123456789"),
            "google-oauth2-123456789"
        );
        assert_eq!(sanitize_user_id("auth0|user123"), "auth0-user123");
        assert_eq!(
            sanitize_user_id("email|user@example.com"),
            "email-user-example-com"
        );

        let result = sanitize_user_id("github|434149");
        assert!(result
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
        assert!(!result.starts_with('-'));
        assert!(!result.ends_with('-'));
    }

    #[test]
    fn test_generate_cr_name() {
        assert_eq!(generate_cr_name("my-app"), "ud-my-app");
        assert_eq!(generate_cr_name("test"), "ud-test");
    }

    #[test]
    fn test_generate_path_prefix() {
        assert_eq!(generate_path_prefix("my-app"), "/deployments/my-app");
        assert_eq!(generate_path_prefix("test"), "/deployments/test");
    }

    #[test]
    fn test_validate_create_deployment_request_valid() {
        let req = CreateDeploymentRequest {
            instance_name: Some("my-app".to_string()),
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: vec![],
            args: vec![],
            env: HashMap::new(),
            resources: Some(ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "512Mi".to_string(),
                gpus: None,
            }),
            ttl_seconds: None,
            public: false,
            health_check: None,
            storage: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
        };
        assert!(validate_create_deployment_request(&req, 10).is_ok());
    }

    #[test]
    fn test_validate_create_deployment_request_no_instance_name() {
        let req = CreateDeploymentRequest {
            instance_name: None,
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: vec![],
            args: vec![],
            env: HashMap::new(),
            resources: Some(ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "512Mi".to_string(),
                gpus: None,
            }),
            ttl_seconds: None,
            public: false,
            health_check: None,
            storage: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
        };
        assert!(validate_create_deployment_request(&req, 10).is_ok());
    }

    #[test]
    fn test_validate_create_deployment_request_invalid_instance_name() {
        let req = CreateDeploymentRequest {
            instance_name: Some("My-App".to_string()),
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: vec![],
            args: vec![],
            env: HashMap::new(),
            resources: None,
            ttl_seconds: None,
            public: false,
            health_check: None,
            storage: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
        };
        assert!(validate_create_deployment_request(&req, 10).is_err());
    }

    #[test]
    fn test_validate_create_deployment_request_env_vars() {
        let mut env = HashMap::new();
        env.insert("".to_string(), "value".to_string());

        let req = CreateDeploymentRequest {
            instance_name: Some("my-app".to_string()),
            image: "nginx:latest".to_string(),
            replicas: 1,
            port: 80,
            command: vec![],
            args: vec![],
            env,
            resources: None,
            ttl_seconds: None,
            public: false,
            health_check: None,
            storage: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
        };
        assert!(validate_create_deployment_request(&req, 10).is_err());
    }

    #[test]
    fn test_generate_instance_name() {
        let name = generate_instance_name();
        assert_eq!(name.len(), 36);
        assert!(name.contains('-'));
    }

    #[test]
    fn test_resolve_instance_name_none() {
        let name = resolve_instance_name(None);
        assert_eq!(name.len(), 36);
    }

    #[test]
    fn test_resolve_instance_name_empty() {
        let name = resolve_instance_name(Some("".to_string()));
        assert_eq!(name.len(), 36);
    }

    #[test]
    fn test_resolve_instance_name_provided() {
        let name = resolve_instance_name(Some("user-provided".to_string()));
        assert_eq!(name.len(), 36);
        assert_ne!(name, "user-provided");
    }
}
