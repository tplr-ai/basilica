use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerAttestation {
    pub info: DockerInfo,
    pub capabilities: DockerCapabilities,
    pub benchmarks: DockerBenchmarkResults,
    pub security: DockerSecurityFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerInfo {
    pub is_installed: bool,
    pub version: Option<String>,
    pub is_running: bool,
    pub api_version: Option<String>,
    pub server_version: Option<String>,
    pub container_runtime: Option<String>,
    pub storage_driver: Option<String>,
    pub cgroup_version: Option<String>,
    pub running_containers: Vec<ContainerInfo>,
    pub cached_images: Vec<ImageInfo>,
    pub volumes: Vec<VolumeInfo>,
    pub networks: Vec<NetworkInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerCapabilities {
    pub supports_dind: bool,
    pub supports_gpu: bool,
    pub supports_privileged: bool,
    pub supports_buildx: bool,
    pub supports_compose: bool,
    pub max_containers: Option<u32>,
    pub memory_limit: Option<u64>,
    pub cpu_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerBenchmarkResults {
    pub container_start_time_ms: Option<u64>,
    pub container_stop_time_ms: Option<u64>,
    pub image_pull_time_ms: Option<u64>,
    pub volume_mount_time_ms: Option<u64>,
    pub network_performance_mbps: Option<f64>,
    pub disk_io_performance_mbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerSecurityFeatures {
    pub has_apparmor: bool,
    pub has_selinux: bool,
    pub has_seccomp: bool,
    pub has_user_namespaces: bool,
    pub has_rootless_mode: bool,
    pub content_trust_enabled: bool,
    pub security_options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub id: String,
    pub names: Vec<String>,
    pub image: String,
    pub status: String,
    pub state: String,
    pub created: i64,
    pub ports: Vec<String>,
    pub size: Option<u64>,
    pub labels: Vec<String>,
    pub network_mode: Option<String>,
    pub mounts: Vec<String>,
}

impl ContainerInfo {
    pub fn is_running(&self) -> bool {
        self.state.to_lowercase() == "running"
    }

    pub fn get_primary_name(&self) -> &str {
        self.names
            .first()
            .map(|n| n.as_str())
            .unwrap_or("<unnamed>")
    }

    pub fn has_exposed_ports(&self) -> bool {
        !self.ports.is_empty()
    }

    pub fn size_mb(&self) -> f64 {
        self.size.unwrap_or(0) as f64 / (1024.0 * 1024.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub id: String,
    pub repository: Option<String>,
    pub tag: Option<String>,
    pub created: i64,
    pub size: u64,
    pub virtual_size: u64,
    pub labels: Vec<String>,
    pub digest: Option<String>,
    pub parent_id: Option<String>,
}

impl ImageInfo {
    pub fn full_name(&self) -> String {
        match (&self.repository, &self.tag) {
            (Some(repo), Some(tag)) => format!("{repo}:{tag}"),
            (Some(repo), None) => repo.clone(),
            (None, Some(tag)) => format!("<none>:{tag}"),
            (None, None) => "<none>:<none>".to_string(),
        }
    }

    pub fn size_mb(&self) -> f64 {
        self.size as f64 / (1024.0 * 1024.0)
    }

    pub fn virtual_size_mb(&self) -> f64 {
        self.virtual_size as f64 / (1024.0 * 1024.0)
    }

    pub fn is_dangling(&self) -> bool {
        self.repository.is_none() && self.tag.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeInfo {
    pub name: String,
    pub driver: String,
    pub mountpoint: String,
    pub created: Option<String>,
    pub scope: String,
    pub labels: Vec<String>,
    pub options: Vec<String>,
}

impl VolumeInfo {
    pub fn is_local(&self) -> bool {
        self.driver == "local"
    }

    pub fn is_external(&self) -> bool {
        !self.is_local()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub id: String,
    pub name: String,
    pub driver: String,
    pub scope: String,
    pub attachable: bool,
    pub ingress: bool,
    pub internal: bool,
    pub ipam_driver: Option<String>,
    pub ipam_config: Vec<String>,
    pub containers: Vec<String>,
}

impl NetworkInfo {
    pub fn is_bridge(&self) -> bool {
        self.driver == "bridge"
    }

    pub fn is_host(&self) -> bool {
        self.driver == "host"
    }

    pub fn is_overlay(&self) -> bool {
        self.driver == "overlay"
    }

    pub fn connected_containers(&self) -> usize {
        self.containers.len()
    }
}

impl DockerAttestation {
    pub fn new() -> Self {
        Self {
            info: DockerInfo::new(),
            capabilities: DockerCapabilities::new(),
            benchmarks: DockerBenchmarkResults::new(),
            security: DockerSecurityFeatures::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.info.is_installed && self.info.is_running
    }

    pub fn has_gpu_support(&self) -> bool {
        self.capabilities.supports_gpu
    }

    pub fn can_run_privileged(&self) -> bool {
        self.capabilities.supports_privileged
    }

    pub fn security_score(&self) -> u8 {
        let mut score = 0u8;

        if self.security.has_apparmor {
            score += 15;
        }
        if self.security.has_selinux {
            score += 15;
        }
        if self.security.has_seccomp {
            score += 15;
        }
        if self.security.has_user_namespaces {
            score += 20;
        }
        if self.security.has_rootless_mode {
            score += 20;
        }
        if self.security.content_trust_enabled {
            score += 15;
        }

        score
    }

    pub fn performance_score(&self) -> u8 {
        let mut score = 0u8;

        if let Some(start_time) = self.benchmarks.container_start_time_ms {
            score += match start_time {
                0..=1000 => 25,
                1001..=3000 => 20,
                3001..=5000 => 15,
                _ => 5,
            };
        }

        if let Some(network_perf) = self.benchmarks.network_performance_mbps {
            score += match network_perf as u64 {
                1000.. => 25,
                500..=999 => 20,
                100..=499 => 15,
                _ => 5,
            };
        }

        if let Some(disk_perf) = self.benchmarks.disk_io_performance_mbps {
            score += match disk_perf as u64 {
                500.. => 25,
                200..=499 => 20,
                50..=199 => 15,
                _ => 5,
            };
        }

        if self.capabilities.supports_gpu {
            score += 25;
        }

        score
    }
}

impl DockerInfo {
    pub fn new() -> Self {
        Self {
            is_installed: false,
            version: None,
            is_running: false,
            api_version: None,
            server_version: None,
            container_runtime: None,
            storage_driver: None,
            cgroup_version: None,
            running_containers: Vec::new(),
            cached_images: Vec::new(),
            volumes: Vec::new(),
            networks: Vec::new(),
        }
    }

    pub fn total_containers(&self) -> usize {
        self.running_containers.len()
    }

    pub fn total_images(&self) -> usize {
        self.cached_images.len()
    }

    pub fn total_volumes(&self) -> usize {
        self.volumes.len()
    }

    pub fn total_networks(&self) -> usize {
        self.networks.len()
    }

    pub fn get_running_containers(&self) -> Vec<&ContainerInfo> {
        self.running_containers
            .iter()
            .filter(|c| c.is_running())
            .collect()
    }

    pub fn get_image_size_total_gb(&self) -> f64 {
        self.cached_images
            .iter()
            .map(|img| img.size as f64)
            .sum::<f64>()
            / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn is_operational(&self) -> bool {
        self.is_installed && self.is_running
    }
}

impl DockerCapabilities {
    pub fn new() -> Self {
        Self {
            supports_dind: false,
            supports_gpu: false,
            supports_privileged: false,
            supports_buildx: false,
            supports_compose: false,
            max_containers: None,
            memory_limit: None,
            cpu_limit: None,
        }
    }

    pub fn is_container_ready(&self) -> bool {
        self.supports_privileged && self.max_containers.unwrap_or(0) > 0
    }
}

impl DockerBenchmarkResults {
    pub fn new() -> Self {
        Self {
            container_start_time_ms: None,
            container_stop_time_ms: None,
            image_pull_time_ms: None,
            volume_mount_time_ms: None,
            network_performance_mbps: None,
            disk_io_performance_mbps: None,
        }
    }

    pub fn average_container_lifecycle_ms(&self) -> Option<u64> {
        match (self.container_start_time_ms, self.container_stop_time_ms) {
            (Some(start), Some(stop)) => Some((start + stop) / 2),
            _ => None,
        }
    }
}

impl DockerSecurityFeatures {
    pub fn new() -> Self {
        Self {
            has_apparmor: false,
            has_selinux: false,
            has_seccomp: false,
            has_user_namespaces: false,
            has_rootless_mode: false,
            content_trust_enabled: false,
            security_options: Vec::new(),
        }
    }

    pub fn has_mandatory_access_control(&self) -> bool {
        self.has_apparmor || self.has_selinux
    }

    pub fn is_security_hardened(&self) -> bool {
        self.has_seccomp && self.has_user_namespaces && self.has_mandatory_access_control()
    }
}

impl Default for DockerAttestation {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DockerInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DockerCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DockerBenchmarkResults {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DockerSecurityFeatures {
    fn default() -> Self {
        Self::new()
    }
}
