use anyhow::Result;
use basilica_common::ssh::SshConnectionDetails;
use basilica_common::utils::validate_docker_image;
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;

const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Docker validation profile
#[derive(Debug, Clone)]
pub struct DockerProfile {
    pub service_active: bool,
    pub docker_version: Option<String>,
    pub images_pulled: Vec<String>,
    pub dind_supported: bool,
    pub gpu_runtime_supported: bool,
    pub validation_error: Option<String>,
    pub full_json: String,
}

impl DockerProfile {
    /// Create a new Docker profile from validation results
    pub fn new(
        service_active: bool,
        docker_version: Option<String>,
        images_pulled: Vec<String>,
        dind_supported: bool,
        gpu_runtime_supported: bool,
        validation_error: Option<String>,
    ) -> Self {
        let json_obj = serde_json::json!({
            "service_active": service_active,
            "docker_version": docker_version,
            "images_pulled": images_pulled,
            "dind_supported": dind_supported,
            "gpu_runtime_supported": gpu_runtime_supported,
            "validation_error": validation_error,
        });

        Self {
            service_active,
            docker_version,
            images_pulled,
            dind_supported,
            gpu_runtime_supported,
            validation_error,
            full_json: json_obj.to_string(),
        }
    }
}

/// Docker validation collector
#[derive(Clone)]
pub struct DockerCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
    docker_image: String,
    pull_timeout_secs: u64,
    service_check_timeout_secs: u64,
}

impl DockerCollector {
    /// Create a new Docker collector
    pub fn new(
        ssh_client: Arc<ValidatorSshClient>,
        persistence: Arc<SimplePersistence>,
        docker_image: String,
        pull_timeout_secs: u64,
    ) -> Self {
        Self::new_with_timeouts(
            ssh_client,
            persistence,
            docker_image,
            pull_timeout_secs,
            DEFAULT_TIMEOUT_SECS,
        )
    }

    /// Create a new Docker collector with timeouts
    pub fn new_with_timeouts(
        ssh_client: Arc<ValidatorSshClient>,
        persistence: Arc<SimplePersistence>,
        docker_image: String,
        pull_timeout_secs: u64,
        service_check_timeout_secs: u64,
    ) -> Self {
        if let Err(e) = validate_docker_image(&docker_image) {
            error!("Invalid Docker image in configuration: {}", e);
            panic!(
                "Invalid Docker image reference in configuration: {}",
                docker_image
            );
        }

        Self {
            ssh_client,
            persistence,
            docker_image,
            pull_timeout_secs,
            service_check_timeout_secs,
        }
    }

    /// Collect Docker profile from node
    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<DockerProfile> {
        info!(
            node_id = node_id,
            "[DOCKER_PROFILE] Starting Docker validation"
        );

        let mut images_pulled = Vec::new();
        let validation_error: Option<String> = None;

        // Check Docker service status
        let service_active = match self.check_docker_service(ssh_details).await {
            Ok(active) => {
                if !active {
                    let error = "Docker service is not active".to_string();
                    error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                    return Err(anyhow::anyhow!("Docker validation failed: {}", error));
                }
                active
            }
            Err(e) => {
                let error = format!("Failed to check Docker service: {}", e);
                error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                return Err(anyhow::anyhow!("Docker validation failed: {}", error));
            }
        };

        // Get Docker version
        let docker_version = match self.get_docker_version(ssh_details).await {
            Ok(version) => {
                info!(
                    node_id = node_id,
                    docker_version = version,
                    "[DOCKER_PROFILE] Docker version detected"
                );
                Some(version)
            }
            Err(e) => {
                let error = format!("Failed to get Docker version: {}", e);
                error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                return Err(anyhow::anyhow!("Docker validation failed: {}", error));
            }
        };

        // Pull Docker image
        match self
            .pull_docker_image(ssh_details, &self.docker_image)
            .await
        {
            Ok(_) => {
                images_pulled.push(self.docker_image.clone());
                info!(
                    node_id = node_id,
                    image = self.docker_image,
                    "[DOCKER_PROFILE] Successfully pulled Docker image"
                );
            }
            Err(e) => {
                let error = format!("Failed to pull Docker image {}: {}", self.docker_image, e);
                error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                return Err(anyhow::anyhow!("Docker validation failed: {}", error));
            }
        }

        // Check GPU runtime support
        let gpu_runtime_supported = match self
            .check_gpu_runtime(ssh_details, &self.docker_image)
            .await
        {
            Ok(supported) => {
                if supported {
                    info!(
                        node_id = node_id,
                        "[DOCKER_PROFILE] GPU runtime validation passed"
                    );
                    supported
                } else {
                    let error = "GPU runtime not available or not configured".to_string();
                    error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                    return Err(anyhow::anyhow!("Docker validation failed: {}", error));
                }
            }
            Err(e) => {
                let error = format!("GPU runtime check failed: {}", e);
                error!(node_id = node_id, "[DOCKER_PROFILE] {}", error);
                return Err(anyhow::anyhow!("Docker validation failed: {}", error));
            }
        };

        // Check Docker-in-Docker support (non-critical)
        let dind_supported = if service_active {
            self.check_dind_support(ssh_details).await
        } else {
            false
        };

        info!(
            node_id = node_id,
            dind_supported = dind_supported,
            gpu_runtime_supported = gpu_runtime_supported,
            "[DOCKER_PROFILE] Docker validation completed successfully"
        );

        Ok(DockerProfile::new(
            service_active,
            docker_version,
            images_pulled,
            dind_supported,
            gpu_runtime_supported,
            validation_error,
        ))
    }

    /// Store Docker profile in database
    pub async fn store(
        &self,
        miner_uid: u16,
        node_id: &str,
        docker_profile: &DockerProfile,
    ) -> Result<()> {
        self.persistence
            .store_node_docker_profile(
                miner_uid,
                node_id,
                docker_profile.service_active,
                docker_profile.docker_version.clone(),
                docker_profile.images_pulled.clone(),
                docker_profile.dind_supported,
                docker_profile.validation_error.clone(),
                &docker_profile.full_json,
            )
            .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[DOCKER_PROFILE] Stored Docker profile in database"
        );

        Ok(())
    }

    /// Collect Docker profile from node and store in database
    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<DockerProfile> {
        let docker_profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &docker_profile).await?;
        Ok(docker_profile)
    }

    /// Collect Docker profile with error handling
    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<DockerProfile> {
        info!(
            node_id = node_id,
            miner_uid = miner_uid,
            "[DOCKER_PROFILE] collect_with_fallback called - starting Docker validation"
        );

        let collect_result = match tokio::time::timeout(
            Duration::from_secs(2400), // 40 minutes timeout
            self.collect(node_id, ssh_details),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                error!(
                    node_id = node_id,
                    miner_uid = miner_uid,
                    "[DOCKER_PROFILE] Docker validation timed out after 30 minutes"
                );
                return None;
            }
        };

        match collect_result {
            Ok(profile) => {
                debug!(
                    node_id = node_id,
                    "[DOCKER_PROFILE] Docker validation succeeded, attempting to store profile"
                );
                // Try to store but don't fail if storage fails
                if let Err(e) = self.store(miner_uid, node_id, &profile).await {
                    warn!(
                        node_id = node_id,
                        error = %e,
                        "[DOCKER_PROFILE] Failed to store Docker profile: {}",
                        e
                    );
                }
                Some(profile)
            }
            Err(e) => {
                error!(
                    node_id = node_id,
                    miner_uid = miner_uid,
                    error = %e,
                    "[DOCKER_PROFILE] Docker validation failed: {}",
                    e
                );
                None
            }
        }
    }

    /// Check if Docker service is active
    async fn check_docker_service(&self, ssh_details: &SshConnectionDetails) -> Result<bool> {
        let check_timeout = Duration::from_secs(self.service_check_timeout_secs);

        let socket_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(
                    ssh_details,
                    "test -S /var/run/docker.sock && echo SOCKET_EXISTS || echo SOCKET_MISSING",
                    true,
                )
                .await
            {
                Ok(output) => {
                    if output.contains("SOCKET_EXISTS") {
                        Ok(("socket", true))
                    } else {
                        Err(())
                    }
                }
                _ => Err(()),
            }
        });

        let systemctl_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(
                    ssh_details,
                    "systemctl is-active docker 2>/dev/null; echo EXIT_CODE=$?",
                    true,
                )
                .await
            {
                Ok(output) => {
                    let output_lower = output.trim().to_lowercase();
                    if output_lower.contains("active") && output.contains("EXIT_CODE=0") {
                        Ok(("systemctl", true))
                    } else {
                        Err(())
                    }
                }
                _ => Err(()),
            }
        });

        let service_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(ssh_details, "service docker status", true)
                .await
            {
                Ok(output) => {
                    let output_lower = output.to_lowercase();
                    if output_lower.contains("running") || output_lower.contains("active") {
                        Ok(("service", true))
                    } else {
                        Err(())
                    }
                }
                _ => Err(()),
            }
        });

        let version_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(
                    ssh_details,
                    "docker -v 2>/dev/null; echo EXIT_CODE=$?",
                    true,
                )
                .await
            {
                Ok(output) => {
                    let output_lower = output.to_lowercase();
                    if (output_lower.contains("docker version")
                        || output_lower.contains("docker build"))
                        && output.contains("EXIT_CODE=0")
                    {
                        Ok(("version", true))
                    } else {
                        Err(())
                    }
                }
                _ => Err(()),
            }
        });

        let info_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(
                    ssh_details,
                    "docker info >/dev/null 2>&1; echo EXIT_CODE=$?",
                    true,
                )
                .await
            {
                Ok(output) if output.contains("EXIT_CODE=0") => Ok(("info", true)),
                _ => Err(()),
            }
        });

        let ps_future = timeout(check_timeout, async {
            match self
                .ssh_client
                .execute_command(
                    ssh_details,
                    "docker ps >/dev/null 2>&1; echo EXIT_CODE=$?",
                    true,
                )
                .await
            {
                Ok(output) if output.contains("EXIT_CODE=0") => Ok(("ps", true)),
                _ => Err(()),
            }
        });

        // Race all futures - return true as soon as any succeeds
        tokio::select! {
            Ok(Ok((method, true))) = socket_future => {
                info!("[DOCKER_PROFILE] Docker service detected via {}", method);
                Ok(true)
            }
            Ok(Ok((method, true))) = systemctl_future => {
                info!("[DOCKER_PROFILE] Docker service detected via {}", method);
                Ok(true)
            }
            Ok(Ok((method, true))) = service_future => {
                info!("[DOCKER_PROFILE] Docker service detected via {}", method);
                Ok(true)
            }
            Ok(Ok((method, true))) = version_future => {
                info!("[DOCKER_PROFILE] Docker detected via {}", method);
                Ok(true)
            }
            Ok(Ok((method, true))) = info_future => {
                info!("[DOCKER_PROFILE] Docker service detected via {}", method);
                Ok(true)
            }
            Ok(Ok((method, true))) = ps_future => {
                info!("[DOCKER_PROFILE] Docker service detected via {}", method);
                Ok(true)
            }
            else => {
                warn!("[DOCKER_PROFILE] Docker service not detected through any method");
                Ok(false)
            }
        }
    }

    /// Get Docker version
    async fn get_docker_version(&self, ssh_details: &SshConnectionDetails) -> Result<String> {
        // Try formatted version first - for later Docker versions
        match self
            .ssh_client
            .execute_command(
                ssh_details,
                "docker version --format '{{.Server.Version}}' 2>/dev/null",
                true,
            )
            .await
        {
            Ok(output) if !output.trim().is_empty() => {
                return Ok(output.trim().to_string());
            }
            _ => {}
        }

        // Fallback to parsing full version output - for older Docker versions
        let output = self
            .ssh_client
            .execute_command(
                ssh_details,
                "docker version 2>/dev/null | grep -i 'server' -A 5 | grep -i version | head -1",
                true,
            )
            .await?;

        // Parse version from output like "Version: 24.0.7" or "Server Version: 24.0.7"
        let version = output
            .lines()
            .find(|line| line.to_lowercase().contains("version"))
            .and_then(|line| line.split(':').nth(1))
            .map(|v| v.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("Could not parse Docker version from output"))?;

        if version.is_empty() {
            return Err(anyhow::anyhow!("Docker version output is empty"));
        }

        Ok(version)
    }

    /// Pull a Docker image
    async fn pull_docker_image(
        &self,
        ssh_details: &SshConnectionDetails,
        image: &str,
    ) -> Result<()> {
        validate_docker_image(image)?;

        let command = format!("docker pull {} 2>&1; echo EXIT_CODE=$?", image);
        let pull_timeout = Duration::from_secs(self.pull_timeout_secs);

        let output = timeout(pull_timeout, async {
            self.ssh_client
                .execute_command(ssh_details, &command, true)
                .await
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Docker pull timed out after {} seconds",
                self.pull_timeout_secs
            )
        })??;

        // Check exit code first
        if output.contains("EXIT_CODE=0") {
            info!("[DOCKER_PROFILE] Docker pull succeeded based on exit code");
            return Ok(());
        }

        // Check for success patterns in output
        let output_lower = output.to_lowercase();
        if output_lower.contains("status: downloaded newer image")
            || output_lower.contains("status: image is up to date")
            || output_lower.contains("pull complete")
            || output_lower.contains("already exists")
        {
            return Ok(());
        }

        self.verify_image_exists(ssh_details, image).await
    }

    /// Verify a Docker image exists
    async fn verify_image_exists(
        &self,
        ssh_details: &SshConnectionDetails,
        image: &str,
    ) -> Result<()> {
        validate_docker_image(image)?;

        let command = format!("docker images -q {}", image);
        let output = self
            .ssh_client
            .execute_command(ssh_details, &command, true)
            .await?;

        if !output.trim().is_empty() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Docker image {} not found after pull",
                image
            ))
        }
    }

    /// Check if Docker-in-Docker (DinD) is supported
    async fn check_dind_support(&self, ssh_details: &SshConnectionDetails) -> bool {
        let check_timeout = Duration::from_secs(30); // Give DinD more time to start

        // Test DinD support by running a simple docker command inside a docker:dind container
        let dind_command = r#"docker run --rm --privileged docker:dind sh -c 'dockerd-entrypoint.sh 2>/dev/null & sleep 5 && docker version >/dev/null 2>&1 && echo "supported"'"#;

        let result = timeout(check_timeout, async {
            matches!(
                self.ssh_client
                    .execute_command(ssh_details, dind_command, true)
                    .await,
                Ok(output) if output.trim().contains("supported")
            )
        })
        .await;

        match result {
            Ok(supported) => {
                if supported {
                    info!("[DOCKER_PROFILE] Docker-in-Docker (DinD) is supported");
                } else {
                    info!("[DOCKER_PROFILE] Docker-in-Docker (DinD) is not supported");
                }
                supported
            }
            Err(_) => {
                warn!("[DOCKER_PROFILE] Docker-in-Docker (DinD) check timed out");
                false
            }
        }
    }

    /// Check if GPU runtime (NVIDIA) is supported
    async fn check_gpu_runtime(
        &self,
        ssh_details: &SshConnectionDetails,
        image: &str,
    ) -> Result<bool> {
        validate_docker_image(image)?;
        let container_name = format!("basilica-eval-test-{}", Uuid::new_v4());
        let check_timeout = Duration::from_secs(60);

        let gpu_command = format!(
            "docker run --rm --name {} --label basilica.security.isolated=true --gpus all --runtime nvidia --network bridge {} nvidia-smi 2>&1; echo EXIT_CODE=$?",
            container_name, image
        );

        let result = timeout(check_timeout, async {
            self.ssh_client
                .execute_command(ssh_details, &gpu_command, true)
                .await
        })
        .await;

        let cleanup_command = format!("docker rm -f {} 2>/dev/null || true", container_name);
        let _ = self
            .ssh_client
            .execute_command(ssh_details, &cleanup_command, true)
            .await;

        match result {
            Ok(Ok(output)) => {
                if output.contains("EXIT_CODE=0") {
                    let output_lower = output.to_lowercase();
                    if output_lower.contains("nvidia-smi")
                        && (output_lower.contains("cuda")
                            || output_lower.contains("driver version")
                            || output_lower.contains("gpu")
                            || output_lower.contains("nvidia"))
                    {
                        info!("[DOCKER_PROFILE] GPU runtime (NVIDIA) is supported");
                        return Ok(true);
                    }
                }

                if output.contains("could not select device driver")
                    || output.contains("unknown runtime specified: nvidia")
                    || output.contains("could not find runtime")
                    || output.contains("docker: Error response from daemon")
                {
                    info!(
                        "[DOCKER_PROFILE] GPU runtime (NVIDIA) is not supported or not configured"
                    );
                    Ok(false)
                } else {
                    warn!(
                        "[DOCKER_PROFILE] GPU runtime check returned unexpected output: {}",
                        output.lines().take(5).collect::<Vec<_>>().join(" | ")
                    );
                    Ok(false)
                }
            }
            Ok(Err(e)) => {
                warn!("[DOCKER_PROFILE] GPU runtime check failed: {}", e);
                Ok(false)
            }
            Err(_) => {
                warn!("[DOCKER_PROFILE] GPU runtime check timed out");
                Ok(false)
            }
        }
    }

    /// Retrieve Docker profile from database
    pub async fn retrieve(&self, miner_uid: u16, node_id: &str) -> Result<Option<DockerProfile>> {
        let result = self
            .persistence
            .get_node_docker_profile(miner_uid, node_id)
            .await?;

        match result {
            Some((
                full_json,
                service_active,
                docker_version,
                images_pulled,
                dind_supported,
                validation_error,
            )) => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[DOCKER_PROFILE] Retrieved Docker profile from database"
                );

                Ok(Some(DockerProfile {
                    service_active,
                    docker_version,
                    images_pulled,
                    dind_supported,
                    gpu_runtime_supported: false, // Default for retrieved profiles without GPU info
                    validation_error,
                    full_json,
                }))
            }
            None => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[DOCKER_PROFILE] No Docker profile found in database"
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_profile_creation() {
        let profile = DockerProfile::new(
            true,
            Some("24.0.7".to_string()),
            vec!["nvidia/cuda:12.8.0-runtime-ubuntu22.04".to_string()],
            true, // dind_supported
            false,
            None,
        );

        assert!(profile.service_active);
        assert_eq!(profile.docker_version, Some("24.0.7".to_string()));
        assert_eq!(profile.images_pulled.len(), 1);
        assert_eq!(
            profile.images_pulled[0],
            "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
        );
        assert!(profile.dind_supported);
        assert!(!profile.gpu_runtime_supported); // Set to false in test
        assert!(profile.validation_error.is_none());
        assert!(profile.full_json.contains("service_active"));
        assert!(profile.full_json.contains("dind_supported"));
        assert!(profile.full_json.contains("gpu_runtime_supported"));
    }

    #[test]
    fn test_docker_profile_with_error() {
        let profile = DockerProfile::new(
            false,
            None,
            vec![],
            false, // dind_supported
            false,
            Some("Docker service not found".to_string()),
        );

        assert!(!profile.service_active);
        assert!(profile.docker_version.is_none());
        assert!(profile.images_pulled.is_empty());
        assert!(!profile.dind_supported);
        assert!(!profile.gpu_runtime_supported);
        assert_eq!(
            profile.validation_error,
            Some("Docker service not found".to_string())
        );
    }

    #[test]
    fn test_docker_profile_with_dind_not_supported() {
        let profile = DockerProfile::new(
            true,
            Some("24.0.7".to_string()),
            vec!["nginx:latest".to_string()],
            false, // DinD not supported
            false,
            None,
        );

        assert!(profile.service_active);
        assert_eq!(profile.docker_version, Some("24.0.7".to_string()));
        assert!(!profile.dind_supported);
        assert!(!profile.gpu_runtime_supported);
        assert!(profile.validation_error.is_none());
    }

    #[test]
    fn test_docker_profile_with_gpu_runtime() {
        let profile = DockerProfile::new(
            true,
            Some("24.0.7".to_string()),
            vec!["nvidia/cuda:12.8.0-runtime-ubuntu22.04".to_string()],
            false, // DinD not supported
            true,  // GPU runtime supported
            None,
        );

        assert!(profile.service_active);
        assert_eq!(profile.docker_version, Some("24.0.7".to_string()));
        assert_eq!(
            profile.images_pulled[0],
            "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
        );
        assert!(!profile.dind_supported);
        assert!(profile.gpu_runtime_supported);
        assert!(profile.validation_error.is_none());
        assert!(profile.full_json.contains("\"gpu_runtime_supported\":true"));
    }
}
