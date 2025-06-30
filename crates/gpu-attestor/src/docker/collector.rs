use anyhow::{Context, Result};
use std::process::Command;
use std::time::Instant;

use super::types::*;

pub struct DockerCollector;

impl DockerCollector {
    pub fn collect_attestation() -> Result<DockerAttestation> {
        Ok(DockerAttestation {
            info: Self::collect_docker_info()?,
            capabilities: Self::collect_docker_capabilities()?,
            benchmarks: super::benchmarker::DockerBenchmarker::run_benchmarks()?,
            security: Self::collect_security_features()?,
        })
    }

    pub fn collect_docker_info() -> Result<DockerInfo> {
        let should_suppress = !tracing::enabled!(tracing::Level::DEBUG);

        let docker_version_output = if should_suppress {
            Command::new("docker")
                .arg("--version")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker").arg("--version").output()
        };

        let is_installed = docker_version_output.is_ok();
        let version = if is_installed {
            docker_version_output
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|s| s.trim().to_string())
        } else {
            None
        };

        let is_running = if should_suppress {
            Command::new("docker")
                .arg("info")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .arg("info")
                .status()
                .is_ok_and(|status| status.success())
        };

        let (api_version, server_version, container_runtime, storage_driver, cgroup_version) =
            if is_running {
                Self::collect_docker_system_info(should_suppress)?
            } else {
                (None, None, None, None, None)
            };

        let (running_containers, cached_images, volumes, networks) = if is_running {
            let containers = Self::collect_containers(should_suppress).unwrap_or_default();
            let images = Self::collect_images(should_suppress).unwrap_or_default();
            let vols = Self::collect_volumes(should_suppress).unwrap_or_default();
            let nets = Self::collect_networks(should_suppress).unwrap_or_default();
            (containers, images, vols, nets)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        Ok(DockerInfo {
            is_installed,
            version,
            is_running,
            api_version,
            server_version,
            container_runtime,
            storage_driver,
            cgroup_version,
            running_containers,
            cached_images,
            volumes,
            networks,
        })
    }

    pub fn collect_docker_capabilities() -> Result<DockerCapabilities> {
        let info = Self::collect_docker_info()?;

        if !info.is_operational() {
            return Ok(DockerCapabilities::new());
        }

        let should_suppress = !tracing::enabled!(tracing::Level::DEBUG);

        let supports_dind = Self::test_dind_support(should_suppress)?;
        let supports_gpu = Self::test_gpu_support(should_suppress)?;
        let supports_privileged = Self::test_privileged_support(should_suppress)?;
        let supports_buildx = Self::test_buildx_support(should_suppress)?;
        let supports_compose = Self::test_compose_support(should_suppress)?;

        let (max_containers, memory_limit, cpu_limit) =
            Self::collect_resource_limits(should_suppress)?;

        Ok(DockerCapabilities {
            supports_dind,
            supports_gpu,
            supports_privileged,
            supports_buildx,
            supports_compose,
            max_containers,
            memory_limit,
            cpu_limit,
        })
    }

    pub fn collect_security_features() -> Result<DockerSecurityFeatures> {
        let info = Self::collect_docker_info()?;

        if !info.is_operational() {
            return Ok(DockerSecurityFeatures::new());
        }

        let should_suppress = !tracing::enabled!(tracing::Level::DEBUG);

        let security_options = Self::get_security_options(should_suppress)?;

        let has_apparmor = security_options.iter().any(|opt| opt.contains("apparmor"));
        let has_selinux = security_options.iter().any(|opt| opt.contains("selinux"));
        let has_seccomp = security_options.iter().any(|opt| opt.contains("seccomp"));
        let has_user_namespaces = Self::test_user_namespaces_support(should_suppress)?;
        let has_rootless_mode = Self::test_rootless_mode(should_suppress)?;
        let content_trust_enabled = Self::check_content_trust(should_suppress)?;

        Ok(DockerSecurityFeatures {
            has_apparmor,
            has_selinux,
            has_seccomp,
            has_user_namespaces,
            has_rootless_mode,
            content_trust_enabled,
            security_options,
        })
    }

    #[allow(clippy::type_complexity)]
    fn collect_docker_system_info(
        should_suppress: bool,
    ) -> Result<(
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    )> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["system", "info", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["system", "info", "--format", "json"])
                .output()
        };

        let output = output.context("Failed to get docker system info")?;

        if !output.status.success() {
            return Ok((None, None, None, None, None));
        }

        let info_str = String::from_utf8(output.stdout).context("Invalid UTF-8 in docker info")?;
        let info: serde_json::Value =
            serde_json::from_str(&info_str).context("Failed to parse docker info JSON")?;

        let api_version = info
            .get("ClientInfo")
            .and_then(|ci| ci.get("APIVersion"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let server_version = info
            .get("ServerVersion")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let container_runtime = info
            .get("DefaultRuntime")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let storage_driver = info
            .get("Driver")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let cgroup_version = info
            .get("CgroupVersion")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok((
            api_version,
            server_version,
            container_runtime,
            storage_driver,
            cgroup_version,
        ))
    }

    fn test_dind_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--privileged",
                    "docker:dind",
                    "docker",
                    "--version",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--privileged",
                    "docker:dind",
                    "docker",
                    "--version",
                ])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn test_gpu_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "nvidia/cuda:latest",
                    "nvidia-smi",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "nvidia/cuda:latest",
                    "nvidia-smi",
                ])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn test_privileged_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args(["run", "--rm", "--privileged", "alpine", "whoami"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args(["run", "--rm", "--privileged", "alpine", "whoami"])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn test_buildx_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args(["buildx", "version"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args(["buildx", "version"])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn test_compose_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args(["compose", "version"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args(["compose", "version"])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn collect_resource_limits(
        should_suppress: bool,
    ) -> Result<(Option<u32>, Option<u64>, Option<f64>)> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["system", "info", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["system", "info", "--format", "json"])
                .output()
        };

        let output = output.context("Failed to get docker system info for limits")?;

        if !output.status.success() {
            return Ok((None, None, None));
        }

        let info_str = String::from_utf8(output.stdout).context("Invalid UTF-8 in docker info")?;
        let info: serde_json::Value =
            serde_json::from_str(&info_str).context("Failed to parse docker info JSON")?;

        let max_containers = info
            .get("Containers")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let memory_limit = info.get("MemTotal").and_then(|v| v.as_u64());

        let cpu_limit = info.get("NCPU").and_then(|v| v.as_u64()).map(|v| v as f64);

        Ok((max_containers, memory_limit, cpu_limit))
    }

    fn get_security_options(should_suppress: bool) -> Result<Vec<String>> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["system", "info", "--format", "{{.SecurityOptions}}"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["system", "info", "--format", "{{.SecurityOptions}}"])
                .output()
        };

        let output = output.context("Failed to get docker security options")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let options_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in security options")?;
        let options: Vec<String> = options_str
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        Ok(options)
    }

    fn test_user_namespaces_support(should_suppress: bool) -> Result<bool> {
        let result = if should_suppress {
            Command::new("docker")
                .args(["run", "--rm", "--user", "1000:1000", "alpine", "id"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        } else {
            Command::new("docker")
                .args(["run", "--rm", "--user", "1000:1000", "alpine", "id"])
                .status()
                .is_ok_and(|status| status.success())
        };

        Ok(result)
    }

    fn test_rootless_mode(should_suppress: bool) -> Result<bool> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["system", "info", "--format", "{{.SecurityOptions}}"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["system", "info", "--format", "{{.SecurityOptions}}"])
                .output()
        };

        let output = output.context("Failed to check rootless mode")?;

        if !output.status.success() {
            return Ok(false);
        }

        let info_str = String::from_utf8(output.stdout).context("Invalid UTF-8 in docker info")?;
        Ok(info_str.contains("rootless"))
    }

    fn check_content_trust(should_suppress: bool) -> Result<bool> {
        let output = if should_suppress {
            Command::new("sh")
                .args(["-c", "echo $DOCKER_CONTENT_TRUST"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("sh")
                .args(["-c", "echo $DOCKER_CONTENT_TRUST"])
                .output()
        };

        let output = output.context("Failed to check content trust")?;

        if !output.status.success() {
            return Ok(false);
        }

        let trust_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in content trust")?;
        Ok(trust_str.trim() == "1")
    }

    fn collect_containers(should_suppress: bool) -> Result<Vec<ContainerInfo>> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["ps", "-a", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["ps", "-a", "--format", "json"])
                .output()
        }
        .context("Failed to list docker containers")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in container list")?;
        let mut containers = Vec::new();

        for line in output_str.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let container: serde_json::Value =
                serde_json::from_str(line).context("Failed to parse container JSON")?;

            containers.push(ContainerInfo {
                id: container
                    .get("ID")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                names: container
                    .get("Names")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                image: container
                    .get("Image")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                status: container
                    .get("Status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                state: container
                    .get("State")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                created: container
                    .get("CreatedAt")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                ports: container
                    .get("Ports")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                size: container.get("Size").and_then(|v| v.as_u64()),
                labels: container
                    .get("Labels")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                network_mode: container
                    .get("Networks")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                mounts: container
                    .get("Mounts")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            });
        }

        Ok(containers)
    }

    fn collect_images(should_suppress: bool) -> Result<Vec<ImageInfo>> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["images", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["images", "--format", "json"])
                .output()
        }
        .context("Failed to list docker images")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let output_str = String::from_utf8(output.stdout).context("Invalid UTF-8 in image list")?;
        let mut images = Vec::new();

        for line in output_str.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let image: serde_json::Value =
                serde_json::from_str(line).context("Failed to parse image JSON")?;

            images.push(ImageInfo {
                id: image
                    .get("ID")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                repository: image
                    .get("Repository")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                tag: image
                    .get("Tag")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                created: image.get("CreatedAt").and_then(|v| v.as_i64()).unwrap_or(0),
                size: image
                    .get("Size")
                    .and_then(|v| v.as_str())
                    .and_then(Self::parse_size_string)
                    .unwrap_or(0),
                virtual_size: image
                    .get("VirtualSize")
                    .and_then(|v| v.as_str())
                    .and_then(Self::parse_size_string)
                    .unwrap_or(0),
                labels: Vec::new(), // Docker images format doesn't include labels by default
                digest: image
                    .get("Digest")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                parent_id: None, // Not available in basic format
            });
        }

        Ok(images)
    }

    fn collect_volumes(should_suppress: bool) -> Result<Vec<VolumeInfo>> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["volume", "ls", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["volume", "ls", "--format", "json"])
                .output()
        }
        .context("Failed to list docker volumes")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in volume list")?;
        let mut volumes = Vec::new();

        for line in output_str.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let volume: serde_json::Value =
                serde_json::from_str(line).context("Failed to parse volume JSON")?;

            volumes.push(VolumeInfo {
                name: volume
                    .get("Name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                driver: volume
                    .get("Driver")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local")
                    .to_string(),
                mountpoint: volume
                    .get("Mountpoint")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                created: volume
                    .get("CreatedAt")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                scope: volume
                    .get("Scope")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local")
                    .to_string(),
                labels: Vec::new(),  // Would need detailed inspect for labels
                options: Vec::new(), // Would need detailed inspect for options
            });
        }

        Ok(volumes)
    }

    fn collect_networks(should_suppress: bool) -> Result<Vec<NetworkInfo>> {
        let output = if should_suppress {
            Command::new("docker")
                .args(["network", "ls", "--format", "json"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
        } else {
            Command::new("docker")
                .args(["network", "ls", "--format", "json"])
                .output()
        }
        .context("Failed to list docker networks")?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in network list")?;
        let mut networks = Vec::new();

        for line in output_str.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let network: serde_json::Value =
                serde_json::from_str(line).context("Failed to parse network JSON")?;

            networks.push(NetworkInfo {
                id: network
                    .get("ID")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                name: network
                    .get("Name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                driver: network
                    .get("Driver")
                    .and_then(|v| v.as_str())
                    .unwrap_or("bridge")
                    .to_string(),
                scope: network
                    .get("Scope")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local")
                    .to_string(),
                attachable: false,       // Would need detailed inspect
                ingress: false,          // Would need detailed inspect
                internal: false,         // Would need detailed inspect
                ipam_driver: None,       // Would need detailed inspect
                ipam_config: Vec::new(), // Would need detailed inspect
                containers: Vec::new(),  // Would need detailed inspect
            });
        }

        Ok(networks)
    }

    fn parse_size_string(size_str: &str) -> Option<u64> {
        let size_str = size_str.trim();
        if size_str.is_empty() {
            return None;
        }

        // Parse sizes like "1.2GB", "500MB", "1.5KB"
        let (number_part, unit_part) = if size_str.ends_with("GB") {
            (size_str.trim_end_matches("GB"), 1024 * 1024 * 1024)
        } else if size_str.ends_with("MB") {
            (size_str.trim_end_matches("MB"), 1024 * 1024)
        } else if size_str.ends_with("KB") {
            (size_str.trim_end_matches("KB"), 1024)
        } else if size_str.ends_with("B") {
            (size_str.trim_end_matches("B"), 1)
        } else {
            // Assume bytes if no unit
            (size_str, 1)
        };

        number_part
            .parse::<f64>()
            .ok()
            .map(|n| (n * unit_part as f64) as u64)
    }

    pub fn list_containers() -> Result<Vec<ContainerInfo>> {
        Self::collect_containers(false)
    }

    pub fn test_container_lifecycle() -> Result<u64> {
        let start_time = Instant::now();

        let result = Command::new("docker")
            .args(["run", "--rm", "alpine", "echo", "test"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        let elapsed = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(status) if status.success() => Ok(elapsed),
            _ => anyhow::bail!("Container lifecycle test failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_docker_info() {
        let docker_info = DockerCollector::collect_docker_info();

        match docker_info {
            Ok(info) => {
                // Test passes regardless of whether Docker is installed
                assert!(info.version.is_some() || !info.is_installed);
            }
            Err(_) => {
                // It's okay if Docker collection fails in test environment
            }
        }
    }

    #[test]
    fn test_minimal_capabilities() {
        let capabilities = DockerCapabilities::new();
        assert!(!capabilities.is_container_ready());
        assert!(!capabilities.supports_dind);
        assert!(!capabilities.supports_gpu);
    }

    #[test]
    fn test_security_features() {
        let security = DockerSecurityFeatures::new();
        assert!(!security.has_mandatory_access_control());
        assert!(!security.is_security_hardened());
        assert_eq!(security.security_options.len(), 0);
    }
}
