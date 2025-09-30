//! Container deployment management
//!
//! This module handles the orchestration of container deployments
//! including validation, resource allocation, and lifecycle management.

use anyhow::{Context, Result};
use basilica_common::utils::validate_docker_image;
use tracing::{debug, info, warn};

use super::container_client::ContainerClient;
use super::types::{ContainerInfo, ContainerSpec};

/// Container deployment manager
pub struct DeploymentManager {
    /// Deployment configuration
    config: DeploymentConfig,
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    /// Allowed container registries
    pub allowed_registries: Vec<String>,
    /// Blocked images
    pub blocked_images: Vec<String>,
    /// Default resource limits
    pub default_resource_limits: DefaultResourceLimits,
    /// Network policies
    pub network_policies: NetworkPolicies,
}

/// Default resource limits
#[derive(Debug, Clone)]
pub struct DefaultResourceLimits {
    pub max_cpu_cores: f64,
    pub max_memory_mb: i64,
    pub max_storage_mb: i64,
    pub max_gpu_count: u32,
}

/// Network policies
#[derive(Debug, Clone)]
pub struct NetworkPolicies {
    pub allowed_network_modes: Vec<String>,
    pub blocked_ports: Vec<u32>,
    pub require_network_isolation: bool,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            allowed_registries: vec![
                "docker.io".to_string(),
                "gcr.io".to_string(),
                "ghcr.io".to_string(),
            ],
            blocked_images: vec!["alpine/socat".to_string(), "nicolaka/netshoot".to_string()],
            default_resource_limits: DefaultResourceLimits {
                max_cpu_cores: 0.0,
                max_memory_mb: 0,
                max_storage_mb: 0,
                max_gpu_count: 0,
            },
            network_policies: NetworkPolicies {
                allowed_network_modes: vec!["bridge".to_string(), "none".to_string()],
                blocked_ports: vec![22, 111, 2049],
                require_network_isolation: false,
            },
        }
    }
}

impl Default for DeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self {
            config: DeploymentConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DeploymentConfig) -> Self {
        Self { config }
    }

    /// Deploy a container
    pub async fn deploy_container(
        &self,
        client: &ContainerClient,
        spec: &ContainerSpec,
        rental_id: &str,
        ssh_public_key: &str,
    ) -> Result<ContainerInfo> {
        info!("Starting container deployment for rental {}", rental_id);

        // Validate container specification
        self.validate_container_spec(spec)
            .context("Container specification validation failed")?;

        // Apply security policies
        let secured_spec = self.apply_security_policies(spec)?;

        // Deploy the container
        let container_info = client
            .deploy_container(&secured_spec, rental_id)
            .await
            .context("Failed to deploy container")?;

        // Only configure SSH if the container is expected to stay running
        let has_interactive_entrypoint = secured_spec.entrypoint.is_empty()
            || secured_spec
                .entrypoint
                .iter()
                .any(|e| e.contains("bash") || e.contains("sh"));
        let has_interactive_command = secured_spec.command.is_empty()
            || secured_spec
                .command
                .iter()
                .any(|c| c.contains("bash") || c.contains("sh"));
        let should_configure_ssh = (has_interactive_entrypoint && has_interactive_command)
            || secured_spec.ports.iter().any(|p| p.container_port == 22);

        if should_configure_ssh {
            // Give container a moment to fully start
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

            if let Err(e) = self
                .configure_container_ssh_access(
                    client,
                    &container_info.container_id,
                    ssh_public_key,
                )
                .await
            {
                warn!("Failed to configure SSH access for container: {}", e);
            }
        } else {
            debug!("Skipping SSH configuration for container with custom command");
        }

        info!(
            "Container {} deployed successfully for rental {}",
            container_info.container_id, rental_id
        );

        Ok(container_info)
    }

    /// Stop a container
    pub async fn stop_container(
        &self,
        client: &ContainerClient,
        container_id: &str,
        force: bool,
    ) -> Result<()> {
        info!("Stopping container {}", container_id);

        // First try graceful stop
        if !force {
            match client.stop_container(container_id, false).await {
                Ok(_) => {
                    info!("Container {} stopped gracefully", container_id);
                    return Ok(());
                }
                Err(e) => {
                    warn!(
                        "Graceful stop failed for container {}: {}. Trying force stop...",
                        container_id, e
                    );
                }
            }
        }

        // Force stop if needed
        client
            .stop_container(container_id, true)
            .await
            .context("Failed to force stop container")?;

        // Remove the container
        client
            .remove_container(container_id)
            .await
            .context("Failed to remove container")?;

        info!("Container {} stopped and removed", container_id);
        Ok(())
    }

    /// Validate container specification
    fn validate_container_spec(&self, spec: &ContainerSpec) -> Result<()> {
        // Validate image
        self.validate_image(&spec.image)?;

        // Validate resources
        self.validate_resources(spec)?;

        // Validate network configuration
        self.validate_network_config(spec)?;

        // Validate volumes
        self.validate_volumes(spec)?;

        // Validate ports
        self.validate_ports(spec)?;

        Ok(())
    }

    /// Validate container image
    fn validate_image(&self, image: &str) -> Result<()> {
        validate_docker_image(image)?;

        // Check if image is in blocked list
        for blocked in &self.config.blocked_images {
            if image.contains(blocked) {
                return Err(anyhow::anyhow!("Image {} is blocked", image));
            }
        }

        if !self.config.allowed_registries.is_empty() {
            let registry = if image.contains('/') {
                let first_part = image.split('/').next().unwrap_or("docker.io");
                if first_part.contains('.') || first_part.contains(':') || first_part == "localhost"
                {
                    first_part
                } else {
                    "docker.io"
                }
            } else {
                "docker.io"
            };

            if !self
                .config
                .allowed_registries
                .contains(&registry.to_string())
            {
                return Err(anyhow::anyhow!("Registry {} is not allowed", registry));
            }
        }

        Ok(())
    }

    /// Validate resource requirements
    fn validate_resources(&self, spec: &ContainerSpec) -> Result<()> {
        let limits = &self.config.default_resource_limits;

        if limits.max_cpu_cores > 0.0
            && spec.resources.cpu_cores > 0.0
            && spec.resources.cpu_cores > limits.max_cpu_cores
        {
            return Err(anyhow::anyhow!(
                "CPU cores {} exceeds limit {}",
                spec.resources.cpu_cores,
                limits.max_cpu_cores
            ));
        }

        if limits.max_memory_mb > 0
            && spec.resources.memory_mb > 0
            && spec.resources.memory_mb > limits.max_memory_mb
        {
            return Err(anyhow::anyhow!(
                "Memory {} MB exceeds limit {} MB",
                spec.resources.memory_mb,
                limits.max_memory_mb
            ));
        }

        if limits.max_storage_mb > 0
            && spec.resources.storage_mb > 0
            && spec.resources.storage_mb > limits.max_storage_mb
        {
            return Err(anyhow::anyhow!(
                "Storage {} MB exceeds limit {} MB",
                spec.resources.storage_mb,
                limits.max_storage_mb
            ));
        }

        if limits.max_gpu_count > 0
            && spec.resources.gpu_count > 0
            && spec.resources.gpu_count > limits.max_gpu_count
        {
            return Err(anyhow::anyhow!(
                "GPU count {} exceeds limit {}",
                spec.resources.gpu_count,
                limits.max_gpu_count
            ));
        }

        Ok(())
    }

    /// Validate network configuration
    fn validate_network_config(&self, spec: &ContainerSpec) -> Result<()> {
        let policies = &self.config.network_policies;

        // Check network mode
        if !policies.allowed_network_modes.contains(&spec.network.mode) {
            return Err(anyhow::anyhow!(
                "Network mode {} is not allowed",
                spec.network.mode
            ));
        }

        // Check if host network is allowed
        if spec.network.mode == "host" && policies.require_network_isolation {
            return Err(anyhow::anyhow!("Host network mode is not allowed"));
        }

        Ok(())
    }

    /// Validate volume mounts
    fn validate_volumes(&self, spec: &ContainerSpec) -> Result<()> {
        for volume in &spec.volumes {
            // Prevent mounting sensitive host paths
            let sensitive_paths = vec![
                "/etc",
                "/root",
                "/home",
                "/var/run/docker.sock",
                "/proc",
                "/sys",
                "/dev",
            ];

            let canonical_path = match std::fs::canonicalize(&volume.host_path) {
                Ok(path) => path.to_string_lossy().to_string(),
                Err(_) => volume.host_path.clone(),
            };

            for sensitive_path in sensitive_paths {
                if canonical_path.starts_with(sensitive_path) {
                    return Err(anyhow::anyhow!(
                        "Mounting {} is not allowed",
                        sensitive_path
                    ));
                }
            }

            // Ensure paths are absolute
            if !volume.host_path.starts_with('/') || !volume.container_path.starts_with('/') {
                return Err(anyhow::anyhow!("Volume paths must be absolute"));
            }
        }

        Ok(())
    }

    /// Validate port mappings
    fn validate_ports(&self, spec: &ContainerSpec) -> Result<()> {
        let blocked_ports = &self.config.network_policies.blocked_ports;

        for port in &spec.ports {
            // Check blocked ports
            if blocked_ports.contains(&port.host_port) {
                return Err(anyhow::anyhow!("Port {} is blocked", port.host_port));
            }

            // Validate port range
            if port.host_port > 65535 {
                return Err(anyhow::anyhow!("Invalid host port {}", port.host_port));
            }

            if port.container_port == 0 || port.container_port > 65535 {
                return Err(anyhow::anyhow!(
                    "Invalid container port {}",
                    port.container_port
                ));
            }

            // Validate protocol
            if port.protocol != "tcp" && port.protocol != "udp" {
                return Err(anyhow::anyhow!("Invalid protocol {}", port.protocol));
            }
        }

        Ok(())
    }

    /// Apply security policies to container specification
    fn apply_security_policies(&self, spec: &ContainerSpec) -> Result<ContainerSpec> {
        let mut secured_spec = spec.clone();

        // Add security labels
        secured_spec.labels.insert(
            "io.kubernetes.cri-o.userns-mode".to_string(),
            "auto".to_string(),
        );
        secured_spec
            .labels
            .insert("basilica.security.isolated".to_string(), "true".to_string());

        // Remove dangerous capabilities
        let dangerous_caps = [
            "CAP_SYS_ADMIN",
            "CAP_SYS_MODULE",
            "CAP_SYS_RAWIO",
            "CAP_SYS_PTRACE",
            "CAP_SYS_NICE",
            "CAP_SYS_RESOURCE",
            "CAP_NET_ADMIN",
            "CAP_NET_RAW",
        ];

        secured_spec
            .capabilities
            .retain(|cap| !dangerous_caps.contains(&cap.as_str()));

        debug!("Applied security policies to container specification");

        Ok(secured_spec)
    }

    /// Configure SSH access for the container
    async fn configure_container_ssh_access(
        &self,
        client: &ContainerClient,
        container_id: &str,
        ssh_public_key: &str,
    ) -> Result<()> {
        debug!("Configuring SSH access for container {}", container_id);

        // First, check if container is running
        let check_running = format!("docker inspect -f '{{{{.State.Running}}}}' {container_id}");
        let is_running = client
            .execute_ssh_command(&check_running)
            .await
            .unwrap_or_default()
            .trim()
            .eq("true");

        if !is_running {
            debug!(
                "Container {} is not running, skipping SSH configuration",
                container_id
            );
            return Ok(());
        }

        let check_ssh =
            format!("docker exec {container_id} which sshd 2>/dev/null || echo 'not_found'");
        let ssh_check_result = client
            .execute_ssh_command(&check_ssh)
            .await
            .unwrap_or_else(|_| "not_found".to_string());

        let needs_ssh_install =
            ssh_check_result.trim() == "not_found" || ssh_check_result.trim().is_empty();

        if needs_ssh_install {
            info!(
                "SSH not found in container {}, attempting to install...",
                container_id
            );

            // Try to detect the package manager and install SSH
            let detect_pkg_manager = format!(
                "docker exec {container_id} sh -c 'if command -v apt-get >/dev/null 2>&1; then echo apt; \
                elif command -v yum >/dev/null 2>&1; then echo yum; \
                elif command -v apk >/dev/null 2>&1; then echo apk; \
                else echo unknown; fi'"
            );

            let pkg_manager = client
                .execute_ssh_command(&detect_pkg_manager)
                .await
                .unwrap_or_else(|_| "unknown".to_string())
                .trim()
                .to_string();

            match pkg_manager.as_str() {
                "apt" => {
                    // Ubuntu/Debian based
                    info!("Installing OpenSSH on Ubuntu/Debian container");
                    let install_cmd = format!(
                        "docker exec {container_id} bash -c 'apt-get update && \
                         DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
                         mkdir -p /var/run/sshd'"
                    );
                    if let Err(e) = client.execute_ssh_command(&install_cmd).await {
                        debug!("Failed to install SSH via apt: {}", e);
                    }
                }
                "yum" => {
                    // RHEL/CentOS based
                    info!("Installing OpenSSH on RHEL/CentOS container");
                    let install_cmd = format!(
                        "docker exec {container_id} bash -c 'yum install -y openssh-server && \
                         ssh-keygen -A && mkdir -p /var/run/sshd'"
                    );
                    if let Err(e) = client.execute_ssh_command(&install_cmd).await {
                        debug!("Failed to install SSH via yum: {}", e);
                    }
                }
                "apk" => {
                    // Alpine based
                    info!("Installing OpenSSH on Alpine container");
                    let install_cmd = format!(
                        "docker exec {container_id} sh -c 'apk add --no-cache openssh-server && \
                         ssh-keygen -A && mkdir -p /var/run/sshd'"
                    );
                    if let Err(e) = client.execute_ssh_command(&install_cmd).await {
                        debug!("Failed to install SSH via apk: {}", e);
                    }
                }
                _ => {
                    debug!("Unknown package manager or unable to install SSH automatically");
                }
            }
        }

        info!("Setting up SSH key for container {}", container_id);

        let mkdir_cmd = format!(
            "docker exec {container_id} bash -c 'mkdir -p /root/.ssh && chmod 700 /root/.ssh'"
        );
        if let Err(e) = client.execute_ssh_command(&mkdir_cmd).await {
            debug!("Failed to create .ssh directory: {}", e);
            let mkdir_alt = format!(
                "docker exec {container_id} sh -c 'mkdir -p /root/.ssh && chmod 700 /root/.ssh'"
            );
            client.execute_ssh_command(&mkdir_alt).await?;
        }

        // Add the SSH public key
        let add_key_cmd = format!(
            "docker exec {container_id} bash -c 'echo \"{ssh_public_key}\" > /root/.ssh/authorized_keys && \
             chmod 600 /root/.ssh/authorized_keys'"
        );
        if let Err(e) = client.execute_ssh_command(&add_key_cmd).await {
            debug!("Failed to add SSH key with bash: {}", e);
            // Try without bash
            let add_key_alt = format!(
                "docker exec {container_id} sh -c 'echo \"{ssh_public_key}\" > /root/.ssh/authorized_keys && \
                 chmod 600 /root/.ssh/authorized_keys'"
            );
            client.execute_ssh_command(&add_key_alt).await?;
        }

        // Configure SSH to allow root login with key
        let config_ssh = format!(
            "docker exec {container_id} bash -c 'echo \"PermitRootLogin prohibit-password\" >> /etc/ssh/sshd_config && \
             echo \"PubkeyAuthentication yes\" >> /etc/ssh/sshd_config && \
             echo \"PasswordAuthentication no\" >> /etc/ssh/sshd_config'"
        );
        let _ = client.execute_ssh_command(&config_ssh).await;

        // Start SSH service (try multiple methods)
        info!("Starting SSH service in container {}", container_id);

        // Try systemctl first (for systemd-based systems)
        let start_systemctl = format!("docker exec {container_id} systemctl start ssh 2>/dev/null || systemctl start sshd 2>/dev/null");
        if client.execute_ssh_command(&start_systemctl).await.is_err() {
            // Try service command
            let start_service = format!("docker exec {container_id} service ssh start 2>/dev/null || service sshd start 2>/dev/null");
            if client.execute_ssh_command(&start_service).await.is_err() {
                // Try running sshd directly
                let start_direct = format!("docker exec -d {container_id} /usr/sbin/sshd -D");
                let _ = client.execute_ssh_command(&start_direct).await;
            }
        }

        info!("SSH access configured for container {}", container_id);
        Ok(())
    }
}
