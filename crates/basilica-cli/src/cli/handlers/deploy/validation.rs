//! Input validation for deploy command

use crate::cli::commands::{
    DeployCommand, GpuOptions, ResourceOptions, SpreadModeArg, StorageOptions,
    TopologySpreadOptions,
};
use crate::error::DeployError;

/// Minimum resource requirements for GPU workloads (default fallback)
const MIN_CPU_PER_GPU: u32 = 4;
const MIN_MEMORY_GB_PER_GPU: u32 = 16;

/// TTL limits
const MIN_TTL_SECONDS: u32 = 60;
const MAX_TTL_SECONDS: u32 = 604800; // 7 days

/// Port limits
const MIN_PORT: u16 = 1;
const MAX_PORT: u16 = 65535;

/// Allowed topology keys for security
const ALLOWED_TOPOLOGY_KEYS: &[&str] = &[
    "kubernetes.io/hostname",
    "topology.kubernetes.io/zone",
    "topology.kubernetes.io/region",
];

/// Maximum replicas for UniqueNodes mode (aligned with general replica limit)
const MAX_UNIQUE_NODES_REPLICAS: u32 = 10;

/// Validate all deployment options
pub fn validate_deployment_request(cmd: &DeployCommand) -> Result<(), DeployError> {
    validate_name(&cmd.naming.name)?;
    validate_replicas(cmd.naming.replicas)?;
    validate_ports(&cmd.networking.port)?;
    validate_resources(&cmd.resources)?;
    validate_gpu_options(&cmd.gpu)?;
    validate_gpu_resource_correlation(&cmd.gpu, &cmd.resources, cmd.lifecycle.skip_gpu_validation)?;
    validate_ttl(cmd.lifecycle.ttl)?;
    validate_env_vars(&cmd.networking.env)?;
    validate_storage(&cmd.storage)?;
    validate_topology_spread(&cmd.topology_spread, cmd.naming.replicas)?;
    Ok(())
}

/// Validate deployment name (RFC 1123 DNS label)
fn validate_name(name: &Option<String>) -> Result<(), DeployError> {
    if let Some(n) = name {
        if n.is_empty() {
            return Err(DeployError::Validation {
                message: "Deployment name cannot be empty".to_string(),
            });
        }

        if n.len() > 63 {
            return Err(DeployError::Validation {
                message: format!(
                    "Deployment name must be 63 characters or less (got {})",
                    n.len()
                ),
            });
        }

        // RFC 1123: lowercase alphanumeric + hyphens, start/end with alphanumeric
        let valid = n
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        let starts_ok = n
            .chars()
            .next()
            .map(|c| c.is_ascii_alphanumeric())
            .unwrap_or(false);
        let ends_ok = n
            .chars()
            .last()
            .map(|c| c.is_ascii_alphanumeric())
            .unwrap_or(false);

        if !valid || !starts_ok || !ends_ok {
            return Err(DeployError::Validation {
                message: "Deployment name must contain only lowercase letters, numbers, and hyphens, and must start and end with alphanumeric character".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate replica count
fn validate_replicas(replicas: u32) -> Result<(), DeployError> {
    if replicas == 0 {
        return Err(DeployError::Validation {
            message: "Replicas must be at least 1".to_string(),
        });
    }
    if replicas > 10 {
        return Err(DeployError::Validation {
            message: "Replicas cannot exceed 10".to_string(),
        });
    }
    Ok(())
}

/// Validate port specifications
fn validate_ports(ports: &[String]) -> Result<(), DeployError> {
    if ports.is_empty() {
        return Err(DeployError::Validation {
            message: "At least one port must be specified".to_string(),
        });
    }

    for port_spec in ports {
        let port_str = port_spec.split(':').next().unwrap_or(port_spec);
        let port: u16 = port_str.parse().map_err(|_| DeployError::Validation {
            message: format!("Invalid port number: {}", port_str),
        })?;

        if !(MIN_PORT..=MAX_PORT).contains(&port) {
            return Err(DeployError::Validation {
                message: format!("Port must be between {}-{}", MIN_PORT, MAX_PORT),
            });
        }
    }
    Ok(())
}

/// Validate resource specifications and ensure requests <= limits
fn validate_resources(resources: &ResourceOptions) -> Result<(), DeployError> {
    validate_cpu_format(&resources.cpu)?;
    validate_memory_format(&resources.memory)?;

    if let Some(ref cpu_req) = resources.cpu_request {
        validate_cpu_format(cpu_req)?;
        // Validate request <= limit
        let req_cores = parse_cpu_to_cores(cpu_req);
        let limit_cores = parse_cpu_to_cores(&resources.cpu);
        if req_cores > limit_cores {
            return Err(DeployError::Validation {
                message: format!(
                    "CPU request ({}) cannot exceed limit ({})",
                    cpu_req, resources.cpu
                ),
            });
        }
    }
    if let Some(ref mem_req) = resources.memory_request {
        validate_memory_format(mem_req)?;
        // Validate request <= limit
        let req_gb = parse_memory_to_gb(mem_req);
        let limit_gb = parse_memory_to_gb(&resources.memory);
        if req_gb > limit_gb {
            return Err(DeployError::Validation {
                message: format!(
                    "Memory request ({}) cannot exceed limit ({})",
                    mem_req, resources.memory
                ),
            });
        }
    }

    Ok(())
}

/// Validate CPU format (e.g., "500m", "2")
fn validate_cpu_format(cpu: &str) -> Result<(), DeployError> {
    let valid = if let Some(milli) = cpu.strip_suffix('m') {
        milli.parse::<u32>().is_ok()
    } else {
        cpu.parse::<f64>().is_ok()
    };

    if !valid {
        return Err(DeployError::Validation {
            message: format!("Invalid CPU format: {}. Use '500m' or '2'", cpu),
        });
    }
    Ok(())
}

/// Validate memory format (e.g., "512Mi", "2Gi")
fn validate_memory_format(memory: &str) -> Result<(), DeployError> {
    let suffixes = ["Mi", "Gi", "M", "G", "Ki", "K"];
    let has_suffix = suffixes.iter().any(|s| memory.ends_with(s));

    if !has_suffix {
        return Err(DeployError::Validation {
            message: format!("Invalid memory format: {}. Use '512Mi' or '2Gi'", memory),
        });
    }

    // Extract numeric part
    let num_str: String = memory.chars().take_while(|c| c.is_ascii_digit()).collect();
    if num_str.parse::<u32>().is_err() {
        return Err(DeployError::Validation {
            message: format!("Invalid memory value: {}", memory),
        });
    }

    Ok(())
}

/// Validate GPU options
fn validate_gpu_options(gpu: &GpuOptions) -> Result<(), DeployError> {
    if let Some(count) = gpu.gpu {
        if count == 0 || count > 8 {
            return Err(DeployError::Validation {
                message: "GPU count must be between 1-8".to_string(),
            });
        }
    }

    if gpu.gpu_model.len() > 10 {
        return Err(DeployError::Validation {
            message: "Cannot specify more than 10 GPU models".to_string(),
        });
    }

    for model in &gpu.gpu_model {
        if model.is_empty() {
            return Err(DeployError::Validation {
                message: "GPU model cannot be empty".to_string(),
            });
        }
    }

    if let Some(ref cuda) = gpu.cuda_version {
        if !cuda.contains('.') {
            return Err(DeployError::Validation {
                message: format!("Invalid CUDA version format: {}. Use '12.0'", cuda),
            });
        }
    }

    // Validate GPU memory range
    if let Some(mem) = gpu.gpu_memory_gb {
        if mem == 0 || mem > 256 {
            return Err(DeployError::Validation {
                message: "GPU memory must be between 1-256 GB".to_string(),
            });
        }

        // Validate against model-specific max GPU memory
        if let Some(model) = gpu.gpu_model.first() {
            let reqs = get_gpu_model_requirements(Some(model));
            if mem > reqs.max_gpu_memory_gb {
                return Err(DeployError::Validation {
                    message: format!(
                        "GPU memory {}GB exceeds {} maximum of {}GB",
                        mem,
                        model.to_uppercase(),
                        reqs.max_gpu_memory_gb
                    ),
                });
            }
        }
    }

    Ok(())
}

/// GPU model-specific resource requirements
struct GpuModelRequirements {
    min_cpu_per_gpu: u32,
    min_memory_gb_per_gpu: u32,
    max_gpu_memory_gb: u32,
}

/// Get resource requirements for specific GPU model
fn get_gpu_model_requirements(model: Option<&str>) -> GpuModelRequirements {
    match model.map(|s| s.to_uppercase()).as_deref() {
        Some("H100") | Some("H100-SXM") | Some("H100-PCIE") => {
            // H100: 80GB HBM3
            GpuModelRequirements {
                min_cpu_per_gpu: 8,
                min_memory_gb_per_gpu: 32,
                max_gpu_memory_gb: 80,
            }
        }
        Some("A100") | Some("A100-SXM") | Some("A100-PCIE") | Some("A100-80GB") => {
            // A100: 40GB or 80GB HBM2e (use 80GB as max)
            GpuModelRequirements {
                min_cpu_per_gpu: 6,
                min_memory_gb_per_gpu: 24,
                max_gpu_memory_gb: 80,
            }
        }
        Some("L40") | Some("L40S") => {
            // L40/L40S: 48GB GDDR6
            GpuModelRequirements {
                min_cpu_per_gpu: 4,
                min_memory_gb_per_gpu: 16,
                max_gpu_memory_gb: 48,
            }
        }
        Some("RTX4090") | Some("RTX-4090") | Some("4090") => {
            // RTX 4090: 24GB GDDR6X
            GpuModelRequirements {
                min_cpu_per_gpu: 4,
                min_memory_gb_per_gpu: 16,
                max_gpu_memory_gb: 24,
            }
        }
        _ => {
            // Default requirements for unspecified or unknown models
            GpuModelRequirements {
                min_cpu_per_gpu: MIN_CPU_PER_GPU,
                min_memory_gb_per_gpu: MIN_MEMORY_GB_PER_GPU,
                max_gpu_memory_gb: 256, // Allow up to 256GB for unknown models
            }
        }
    }
}

/// Validate GPU resource correlation (GPU workloads need sufficient CPU/memory)
/// Uses model-specific requirements when GPU model is specified
fn validate_gpu_resource_correlation(
    gpu: &GpuOptions,
    resources: &ResourceOptions,
    skip_validation: bool,
) -> Result<(), DeployError> {
    if skip_validation {
        return Ok(());
    }

    let gpu_count = match gpu.gpu {
        Some(count) if count > 0 => count,
        _ => return Ok(()), // No GPU, no validation needed
    };

    // Get model-specific requirements (use first model if multiple specified)
    let model = gpu.gpu_model.first().map(String::as_str);
    let requirements = get_gpu_model_requirements(model);

    let cpu_cores = parse_cpu_to_cores(&resources.cpu);
    let memory_gb = parse_memory_to_gb(&resources.memory);

    let min_cpu = (gpu_count * requirements.min_cpu_per_gpu) as f64;
    let min_memory = (gpu_count * requirements.min_memory_gb_per_gpu) as f64;

    let model_info = model.map(|m| format!(" for {} GPU", m)).unwrap_or_default();

    if cpu_cores < min_cpu {
        return Err(DeployError::GpuResourceMismatch {
            message: format!(
                "GPU deployment with {} GPU(s){} requires at least {} CPU cores (specified: {}). \
                 Use --cpu {} or higher, or --skip-gpu-validation to bypass",
                gpu_count, model_info, min_cpu as u32, resources.cpu, min_cpu as u32
            ),
        });
    }

    if memory_gb < min_memory {
        return Err(DeployError::GpuResourceMismatch {
            message: format!(
                "GPU deployment with {} GPU(s){} requires at least {}Gi memory (specified: {}). \
                 Use --memory {}Gi or higher, or --skip-gpu-validation to bypass",
                gpu_count, model_info, min_memory as u32, resources.memory, min_memory as u32
            ),
        });
    }

    Ok(())
}

/// Parse CPU string to cores (e.g., "500m" -> 0.5, "2" -> 2.0)
fn parse_cpu_to_cores(cpu: &str) -> f64 {
    if let Some(milli) = cpu.strip_suffix('m') {
        milli.parse::<f64>().map(|m| m / 1000.0).unwrap_or(0.0)
    } else {
        cpu.parse::<f64>().unwrap_or(0.0)
    }
}

/// Parse memory string to GB
/// Binary units (Ki/Mi/Gi): 1024-based (K8s standard)
/// Decimal units (K/M/G): 1000-based
fn parse_memory_to_gb(memory: &str) -> f64 {
    // Binary units (1024-based) - K8s standard
    if let Some(num_str) = memory.strip_suffix("Gi") {
        return num_str.parse::<f64>().unwrap_or(0.0);
    }
    if let Some(num_str) = memory.strip_suffix("Mi") {
        return num_str.parse::<f64>().map(|m| m / 1024.0).unwrap_or(0.0);
    }
    if let Some(num_str) = memory.strip_suffix("Ki") {
        return num_str
            .parse::<f64>()
            .map(|k| k / (1024.0 * 1024.0))
            .unwrap_or(0.0);
    }

    // Decimal units (1000-based)
    if let Some(num_str) = memory.strip_suffix('G') {
        // 1G = 1000^3 bytes = 0.931 GiB
        return num_str
            .parse::<f64>()
            .map(|g| g * 1_000_000_000.0 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);
    }
    if let Some(num_str) = memory.strip_suffix('M') {
        // 1M = 1000^2 bytes
        return num_str
            .parse::<f64>()
            .map(|m| m * 1_000_000.0 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);
    }
    if let Some(num_str) = memory.strip_suffix('K') {
        // 1K = 1000 bytes
        return num_str
            .parse::<f64>()
            .map(|k| k * 1000.0 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);
    }

    // Assume bytes if no suffix
    memory
        .parse::<f64>()
        .map(|b| b / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0)
}

/// Validate TTL
fn validate_ttl(ttl: Option<u32>) -> Result<(), DeployError> {
    if let Some(t) = ttl {
        if t < MIN_TTL_SECONDS {
            return Err(DeployError::Validation {
                message: format!("TTL must be at least {} seconds", MIN_TTL_SECONDS),
            });
        }
        if t > MAX_TTL_SECONDS {
            return Err(DeployError::Validation {
                message: format!(
                    "TTL cannot exceed {} seconds ({} days)",
                    MAX_TTL_SECONDS,
                    MAX_TTL_SECONDS / 86400
                ),
            });
        }
    }
    Ok(())
}

/// Validate environment variables
fn validate_env_vars(env: &[String]) -> Result<(), DeployError> {
    for entry in env {
        if !entry.contains('=') {
            return Err(DeployError::Validation {
                message: format!("Invalid env var format: '{}'. Use KEY=VALUE", entry),
            });
        }

        let key = entry.split('=').next().unwrap_or("");
        if key.is_empty() {
            return Err(DeployError::Validation {
                message: format!("Empty env var key in: '{}'", entry),
            });
        }
        if key.len() > 255 {
            return Err(DeployError::Validation {
                message: format!("Env var key too long (max 255): '{}'", key),
            });
        }
    }
    Ok(())
}

/// Validate storage options
fn validate_storage(storage: &StorageOptions) -> Result<(), DeployError> {
    if storage.storage {
        if storage.storage_cache_mb < 512 || storage.storage_cache_mb > 16384 {
            return Err(DeployError::Validation {
                message: "Storage cache size must be between 512-16384 MB".to_string(),
            });
        }

        if storage.storage_sync_ms < 100 || storage.storage_sync_ms > 60000 {
            return Err(DeployError::Validation {
                message: "Storage sync interval must be between 100-60000 ms".to_string(),
            });
        }

        validate_storage_path(&storage.storage_path)?;
    }
    Ok(())
}

/// Validate storage mount path format
fn validate_storage_path(path: &str) -> Result<(), DeployError> {
    if path.is_empty() {
        return Err(DeployError::Validation {
            message: "Storage mount path cannot be empty".to_string(),
        });
    }

    // Must be absolute path
    if !path.starts_with('/') {
        return Err(DeployError::Validation {
            message: format!(
                "Storage mount path must be absolute (start with '/'): {}",
                path
            ),
        });
    }

    // Check for invalid characters
    if path.contains("..") {
        return Err(DeployError::Validation {
            message: "Storage mount path cannot contain '..'".to_string(),
        });
    }

    // Prevent mounting over critical system paths
    const FORBIDDEN_PATHS: &[&str] = &[
        "/proc", "/sys", "/dev", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/usr", "/boot",
        "/root",
    ];

    for forbidden in FORBIDDEN_PATHS {
        if path == *forbidden || path.starts_with(&format!("{}/", forbidden)) {
            return Err(DeployError::Validation {
                message: format!(
                    "Cannot mount storage at system path '{}'. Use /data, /app/data, or similar",
                    path
                ),
            });
        }
    }

    // Max path length (Linux limit is 4096, but be conservative)
    if path.len() > 255 {
        return Err(DeployError::Validation {
            message: "Storage mount path too long (max 255 characters)".to_string(),
        });
    }

    Ok(())
}

/// Default topology key for hostname-based spreading
const DEFAULT_TOPOLOGY_KEY: &str = "kubernetes.io/hostname";

/// Validate topology spread configuration
pub fn validate_topology_spread(
    options: &TopologySpreadOptions,
    replicas: u32,
) -> Result<(), DeployError> {
    // Determine if unique_nodes mode is active
    let is_unique_nodes =
        options.unique_nodes || matches!(options.spread_mode, Some(SpreadModeArg::UniqueNodes));

    // Validate max_skew range (only relevant for Preferred/Required modes)
    if options.max_skew < 1 || options.max_skew > 10 {
        return Err(DeployError::Validation {
            message: format!(
                "max_skew must be between 1 and 10, got {}",
                options.max_skew
            ),
        });
    }

    // Warn if max_skew is explicitly set with UniqueNodes (it's ignored)
    if is_unique_nodes && options.max_skew != 1 {
        return Err(DeployError::Validation {
            message: format!(
                "max_skew ({}) is ignored in unique_nodes mode. \
                 UniqueNodes uses pod anti-affinity, not topology spread constraints. \
                 Remove --max-skew or use a different spread mode.",
                options.max_skew
            ),
        });
    }

    // Validate topology_key is in allowlist
    if !ALLOWED_TOPOLOGY_KEYS.contains(&options.topology_key.as_str()) {
        return Err(DeployError::Validation {
            message: format!(
                "Invalid topology_key '{}'. Allowed: {}",
                options.topology_key,
                ALLOWED_TOPOLOGY_KEYS.join(", ")
            ),
        });
    }

    // UniqueNodes mode must use kubernetes.io/hostname (enforces one-pod-per-node semantics)
    if is_unique_nodes && options.topology_key != DEFAULT_TOPOLOGY_KEY {
        return Err(DeployError::Validation {
            message: format!(
                "unique_nodes mode requires topology_key '{}' (got '{}'). \
                 UniqueNodes guarantees one pod per node; use --spread-mode required \
                 with --topology-key {} for zone/region spreading.",
                DEFAULT_TOPOLOGY_KEY, options.topology_key, options.topology_key
            ),
        });
    }

    // Validate unique_nodes mode replica limit
    if is_unique_nodes && replicas > MAX_UNIQUE_NODES_REPLICAS {
        return Err(DeployError::Validation {
            message: format!(
                "unique_nodes mode limited to {} replicas (requested {}). \
                 This mode requires one node per replica.",
                MAX_UNIQUE_NODES_REPLICAS, replicas
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_name_valid() {
        assert!(validate_name(&Some("my-app-123".to_string())).is_ok());
    }

    #[test]
    fn test_validate_name_invalid_uppercase() {
        assert!(validate_name(&Some("MyApp".to_string())).is_err());
    }

    #[test]
    fn test_validate_name_too_long() {
        let long_name = "a".repeat(64);
        assert!(validate_name(&Some(long_name)).is_err());
    }

    #[test]
    fn test_validate_gpu_resource_correlation() {
        let gpu = GpuOptions {
            gpu: Some(2),
            ..Default::default()
        };
        let resources = ResourceOptions {
            cpu: "2".to_string(),      // Only 2 cores, need 8 for 2 GPUs
            memory: "8Gi".to_string(), // Only 8Gi, need 32Gi for 2 GPUs
            ..Default::default()
        };
        assert!(validate_gpu_resource_correlation(&gpu, &resources, false).is_err());
    }

    #[test]
    fn test_validate_gpu_resource_correlation_sufficient() {
        let gpu = GpuOptions {
            gpu: Some(1),
            ..Default::default()
        };
        let resources = ResourceOptions {
            cpu: "4".to_string(),
            memory: "16Gi".to_string(),
            ..Default::default()
        };
        assert!(validate_gpu_resource_correlation(&gpu, &resources, false).is_ok());
    }

    #[test]
    fn test_validate_gpu_resource_correlation_skipped() {
        let gpu = GpuOptions {
            gpu: Some(8), // Very high GPU count
            ..Default::default()
        };
        let resources = ResourceOptions {
            cpu: "1".to_string(),      // Insufficient
            memory: "1Gi".to_string(), // Insufficient
            ..Default::default()
        };
        // Should pass when skip_validation is true
        assert!(validate_gpu_resource_correlation(&gpu, &resources, true).is_ok());
    }

    #[test]
    fn test_gpu_model_specific_requirements() {
        let gpu = GpuOptions {
            gpu: Some(1),
            gpu_model: vec!["H100".to_string()],
            ..Default::default()
        };
        let resources = ResourceOptions {
            cpu: "6".to_string(),       // H100 needs 8 cores
            memory: "24Gi".to_string(), // H100 needs 32Gi
            ..Default::default()
        };
        // Should fail due to H100-specific requirements
        assert!(validate_gpu_resource_correlation(&gpu, &resources, false).is_err());
    }

    #[test]
    fn test_gpu_memory_exceeds_model_max() {
        // RTX4090 has max 24GB - requesting 48GB should fail
        let gpu = GpuOptions {
            gpu: Some(1),
            gpu_model: vec!["RTX4090".to_string()],
            gpu_memory_gb: Some(48), // Exceeds RTX4090's 24GB
            ..Default::default()
        };
        let result = validate_gpu_options(&gpu);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds RTX4090 maximum of 24GB"));
    }

    #[test]
    fn test_gpu_memory_within_model_max() {
        // H100 has max 80GB - requesting 64GB should pass
        let gpu = GpuOptions {
            gpu: Some(1),
            gpu_model: vec!["H100".to_string()],
            gpu_memory_gb: Some(64),
            ..Default::default()
        };
        assert!(validate_gpu_options(&gpu).is_ok());
    }

    #[test]
    fn test_validate_ttl_valid() {
        assert!(validate_ttl(Some(3600)).is_ok());
    }

    #[test]
    fn test_validate_ttl_too_short() {
        assert!(validate_ttl(Some(30)).is_err());
    }

    #[test]
    fn test_validate_ttl_too_long() {
        assert!(validate_ttl(Some(1_000_000)).is_err());
    }

    #[test]
    fn test_parse_memory_to_gb() {
        // Binary units (1024-based)
        assert!((parse_memory_to_gb("1Gi") - 1.0).abs() < 0.001);
        assert!((parse_memory_to_gb("512Mi") - 0.5).abs() < 0.001);
        assert!((parse_memory_to_gb("1048576Ki") - 1.0).abs() < 0.001); // 1024*1024 Ki = 1 Gi

        // Decimal units (1000-based) - converted to GiB
        // 1G = 1,000,000,000 bytes = ~0.931 GiB
        assert!((parse_memory_to_gb("1G") - 0.931).abs() < 0.001);
        // 1000M = 1,000,000,000 bytes = ~0.931 GiB
        assert!((parse_memory_to_gb("1000M") - 0.931).abs() < 0.001);
    }

    #[test]
    fn test_parse_cpu_to_cores() {
        assert!((parse_cpu_to_cores("1000m") - 1.0).abs() < 0.001);
        assert!((parse_cpu_to_cores("500m") - 0.5).abs() < 0.001);
        assert!((parse_cpu_to_cores("2") - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_validate_topology_spread_default() {
        let options = TopologySpreadOptions::default();
        assert!(validate_topology_spread(&options, 3).is_ok());
    }

    #[test]
    fn test_validate_topology_spread_max_skew_too_low() {
        let options = TopologySpreadOptions {
            max_skew: 0,
            ..Default::default()
        };
        let result = validate_topology_spread(&options, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_skew"));
    }

    #[test]
    fn test_validate_topology_spread_max_skew_too_high() {
        let options = TopologySpreadOptions {
            max_skew: 11,
            ..Default::default()
        };
        let result = validate_topology_spread(&options, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_skew"));
    }

    #[test]
    fn test_validate_topology_spread_invalid_key() {
        let options = TopologySpreadOptions {
            topology_key: "custom.key/invalid".to_string(),
            ..Default::default()
        };
        let result = validate_topology_spread(&options, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("topology_key"));
    }

    #[test]
    fn test_validate_topology_spread_valid_zone_key() {
        let options = TopologySpreadOptions {
            topology_key: "topology.kubernetes.io/zone".to_string(),
            ..Default::default()
        };
        assert!(validate_topology_spread(&options, 3).is_ok());
    }

    #[test]
    fn test_validate_topology_spread_unique_nodes_replica_limit() {
        let options = TopologySpreadOptions {
            unique_nodes: true,
            ..Default::default()
        };
        // 10 replicas should be OK (aligned with general replica limit)
        assert!(validate_topology_spread(&options, 10).is_ok());
        // 11 replicas should fail
        let result = validate_topology_spread(&options, 11);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unique_nodes"));
    }

    #[test]
    fn test_validate_topology_spread_unique_nodes_via_mode() {
        let options = TopologySpreadOptions {
            spread_mode: Some(SpreadModeArg::UniqueNodes),
            ..Default::default()
        };
        // 11 replicas should fail for UniqueNodes mode
        let result = validate_topology_spread(&options, 11);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unique_nodes"));
    }

    #[test]
    fn test_validate_topology_spread_unique_nodes_requires_hostname_key() {
        // UniqueNodes with zone key should fail
        let options = TopologySpreadOptions {
            unique_nodes: true,
            topology_key: "topology.kubernetes.io/zone".to_string(),
            ..Default::default()
        };
        let result = validate_topology_spread(&options, 3);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("unique_nodes mode requires topology_key"));
        assert!(err_msg.contains("kubernetes.io/hostname"));
    }

    #[test]
    fn test_validate_topology_spread_unique_nodes_rejects_custom_max_skew() {
        // UniqueNodes with max_skew != 1 should fail
        let options = TopologySpreadOptions {
            unique_nodes: true,
            max_skew: 2,
            ..Default::default()
        };
        let result = validate_topology_spread(&options, 3);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_skew"));
        assert!(err_msg.contains("ignored in unique_nodes mode"));
    }

    #[test]
    fn test_validate_topology_spread_required_mode_allows_zone_key() {
        // Required mode with zone key should be OK
        let options = TopologySpreadOptions {
            spread_mode: Some(SpreadModeArg::Required),
            topology_key: "topology.kubernetes.io/zone".to_string(),
            ..Default::default()
        };
        assert!(validate_topology_spread(&options, 3).is_ok());
    }

    #[test]
    fn test_validate_topology_spread_required_mode_allows_custom_max_skew() {
        // Required mode with custom max_skew should be OK
        let options = TopologySpreadOptions {
            spread_mode: Some(SpreadModeArg::Required),
            max_skew: 3,
            ..Default::default()
        };
        assert!(validate_topology_spread(&options, 5).is_ok());
    }
}
