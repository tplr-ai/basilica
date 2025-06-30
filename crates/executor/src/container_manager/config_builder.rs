//! Container configuration builder

use crate::config::{ContainerResourceLimits, DockerConfig};
use anyhow::Result;
use bollard::{
    container::Config,
    models::{
        DeviceMapping, HostConfig, Mount, MountTypeEnum, PortBinding, RestartPolicy,
        RestartPolicyNameEnum,
    },
};

#[derive(Debug, Clone)]
pub struct ContainerConfigBuilder {
    config: DockerConfig,
}

impl ContainerConfigBuilder {
    pub fn new(config: DockerConfig) -> Self {
        Self { config }
    }

    pub fn build(
        &self,
        image: &str,
        command: &[String],
        limits: &ContainerResourceLimits,
    ) -> Result<Config<String>> {
        let host_config = self.build_host_config(limits)?;
        let config = self.build_container_config(image, command, host_config)?;
        Ok(config)
    }

    fn build_host_config(&self, limits: &ContainerResourceLimits) -> Result<HostConfig> {
        Ok(HostConfig {
            memory: Some(limits.memory_bytes as i64),
            nano_cpus: Some((limits.cpu_cores * 1_000_000_000.0) as i64),
            restart_policy: Some(RestartPolicy {
                name: Some(RestartPolicyNameEnum::NO),
                maximum_retry_count: None,
            }),
            security_opt: Some(self.build_security_options()),
            network_mode: Some(self.build_network_mode()),
            mounts: Some(self.build_mounts()),
            device_requests: self.build_device_requests(),
            devices: self.build_device_mappings(),
            port_bindings: self.build_port_bindings(),
            // ulimits: Some(self.build_ulimits()), // Commented out due to API differences
            dns: Some(self.config.network_config.dns_servers.clone()),
            cap_drop: Some(self.build_capability_drops()),
            cap_add: self.build_capability_adds(),
            // Resource constraints
            memory_swap: Some(limits.memory_bytes as i64), // No swap
            memory_swappiness: Some(0),                    // Disable swapping
            oom_kill_disable: Some(false),                 // Allow OOM killer
            pids_limit: Some(1024),                        // Limit number of processes
            shm_size: Some(64 * 1024 * 1024),              // 64MB shared memory
            ..Default::default()
        })
    }

    fn build_security_options(&self) -> Vec<String> {
        let mut security_opts = vec!["no-new-privileges:true".to_string()];

        // Only disable seccomp if GPU passthrough is enabled
        if self.config.enable_gpu_passthrough {
            security_opts.push("seccomp:unconfined".to_string());
        }

        security_opts
    }

    fn build_network_mode(&self) -> String {
        if self.config.network_config.enable_isolation {
            "none".to_string()
        } else {
            "default".to_string()
        }
    }

    fn build_mounts(&self) -> Vec<Mount> {
        let mut mounts = vec![
            // Temporary filesystem for /tmp
            Mount {
                target: Some("/tmp".to_string()),
                source: None,
                typ: Some(MountTypeEnum::TMPFS),
                read_only: Some(false),
                tmpfs_options: Some(bollard::models::MountTmpfsOptions {
                    size_bytes: Some(512 * 1024 * 1024), // 512MB limit
                    mode: Some(0o1777),
                }),
                ..Default::default()
            },
            // Workspace directory
            Mount {
                target: Some("/workspace".to_string()),
                source: None,
                typ: Some(MountTypeEnum::TMPFS),
                read_only: Some(false),
                tmpfs_options: Some(bollard::models::MountTmpfsOptions {
                    size_bytes: Some(1024 * 1024 * 1024), // 1GB limit
                    mode: Some(0o755),
                }),
                ..Default::default()
            },
        ];

        // Add read-only system mounts for security
        mounts.extend(vec![
            Mount {
                target: Some("/proc/sys".to_string()),
                source: Some("/proc/sys".to_string()),
                typ: Some(MountTypeEnum::BIND),
                read_only: Some(true),
                ..Default::default()
            },
            Mount {
                target: Some("/proc/sysrq-trigger".to_string()),
                source: Some("/proc/sysrq-trigger".to_string()),
                typ: Some(MountTypeEnum::BIND),
                read_only: Some(true),
                ..Default::default()
            },
        ]);

        mounts
    }

    fn build_device_requests(&self) -> Option<Vec<bollard::models::DeviceRequest>> {
        if self.config.enable_gpu_passthrough {
            Some(vec![bollard::models::DeviceRequest {
                driver: Some("nvidia".to_string()),
                count: Some(-1),  // All available GPUs
                device_ids: None, // Use all available
                capabilities: Some(vec![vec![
                    "gpu".to_string(),
                    "compute".to_string(),
                    "utility".to_string(),
                ]]),
                options: Some(std::collections::HashMap::from([(
                    "memory".to_string(),
                    self.format_gpu_memory_limit(),
                )])),
            }])
        } else {
            None
        }
    }

    fn build_container_config(
        &self,
        image: &str,
        command: &[String],
        host_config: HostConfig,
    ) -> Result<Config<String>> {
        Ok(Config {
            image: Some(image.to_string()),
            cmd: Some(command.to_vec()),
            working_dir: Some("/workspace".to_string()),
            env: Some(self.build_environment_variables()),
            host_config: Some(host_config),
            user: self.build_user_config(),
            ..Default::default()
        })
    }

    fn build_environment_variables(&self) -> Vec<String> {
        let mut env_vars = vec![
            "DEBIAN_FRONTEND=noninteractive".to_string(),
            "PYTHONUNBUFFERED=1".to_string(),
            "HOME=/workspace".to_string(),
            "USER=executor".to_string(),
        ];

        // Add GPU-specific environment variables if GPU is enabled
        if self.config.enable_gpu_passthrough {
            env_vars.extend(vec![
                "NVIDIA_VISIBLE_DEVICES=all".to_string(),
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility".to_string(),
                "NVIDIA_REQUIRE_CUDA=cuda>=11.0".to_string(),
            ]);
        }

        // Add network restrictions if internet is disabled
        if !self.config.network_config.allow_internet {
            env_vars.push("NO_INTERNET=1".to_string());
        }

        env_vars
    }

    fn build_device_mappings(&self) -> Option<Vec<DeviceMapping>> {
        // Only expose essential devices
        Some(vec![
            DeviceMapping {
                path_on_host: Some("/dev/null".to_string()),
                path_in_container: Some("/dev/null".to_string()),
                cgroup_permissions: Some("rwm".to_string()),
            },
            DeviceMapping {
                path_on_host: Some("/dev/zero".to_string()),
                path_in_container: Some("/dev/zero".to_string()),
                cgroup_permissions: Some("rwm".to_string()),
            },
            DeviceMapping {
                path_on_host: Some("/dev/random".to_string()),
                path_in_container: Some("/dev/random".to_string()),
                cgroup_permissions: Some("r".to_string()),
            },
            DeviceMapping {
                path_on_host: Some("/dev/urandom".to_string()),
                path_in_container: Some("/dev/urandom".to_string()),
                cgroup_permissions: Some("r".to_string()),
            },
        ])
    }

    fn build_port_bindings(
        &self,
    ) -> Option<std::collections::HashMap<String, Option<Vec<PortBinding>>>> {
        if self.config.network_config.port_mappings.is_empty() {
            return None;
        }

        let mut port_bindings = std::collections::HashMap::new();

        for mapping in &self.config.network_config.port_mappings {
            let container_port = format!("{}/{}", mapping.container_port, mapping.protocol);
            let host_binding = PortBinding {
                host_ip: Some("127.0.0.1".to_string()), // Bind only to localhost
                host_port: Some(mapping.host_port.to_string()),
            };
            port_bindings.insert(container_port, Some(vec![host_binding]));
        }

        Some(port_bindings)
    }

    // fn build_ulimits(&self) -> Vec<Ulimit> {
    //     // Commented out due to bollard API differences
    //     vec![]
    // }

    fn build_capability_drops(&self) -> Vec<String> {
        vec![
            "ALL".to_string(), // Drop all capabilities by default
        ]
    }

    fn build_capability_adds(&self) -> Option<Vec<String>> {
        // Only add essential capabilities
        Some(vec![
            "CHOWN".to_string(),
            "SETUID".to_string(),
            "SETGID".to_string(),
        ])
    }

    fn format_gpu_memory_limit(&self) -> String {
        if let Some(gpu_memory_bytes) = self.config.resource_limits.gpu_memory_bytes {
            format!("{}m", gpu_memory_bytes / (1024 * 1024)) // Convert to MB
        } else {
            "4096m".to_string() // Default 4GB
        }
    }

    fn build_user_config(&self) -> Option<String> {
        if self.config.enable_gpu_passthrough {
            // For GPU workloads, may need root access
            None
        } else {
            // Use non-root user for security
            Some("1000:1000".to_string())
        }
    }
}
