//! Unified executor manager for remote GPU machines
//!
//! This module handles:
//! - SSH-based deployment of executors to remote machines
//! - Health monitoring via gRPC
//! - Resource tracking and availability

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{interval, sleep, timeout, Instant};
use tonic::transport::{Channel, Endpoint};
use tracing::{debug, error, info, warn};

use protocol::common::ResourceUsageStats;
use protocol::executor_control::{
    executor_control_client::ExecutorControlClient, HealthCheckRequest, HealthCheckResponse,
};

use crate::config::MinerConfig;
use crate::persistence::RegistrationDb;

/// Container resource limits for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResourceLimits {
    /// Maximum memory in bytes
    pub memory_bytes: u64,
    /// Maximum CPU cores (fractional)
    pub cpu_cores: f64,
    /// Maximum GPU memory in bytes
    pub gpu_memory_bytes: Option<u64>,
    /// Maximum disk I/O bandwidth in bytes/sec
    pub disk_io_bps: Option<u64>,
    /// Maximum network bandwidth in bytes/sec
    pub network_io_bps: Option<u64>,
}

/// Configuration for a remote GPU machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMachine {
    /// Unique identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// SSH hostname or IP
    pub host: String,
    /// SSH port
    pub port: u16,
    /// SSH username
    pub username: String,
    /// Path to SSH private key
    pub private_key_path: PathBuf,
    /// Optional SSH jump host
    pub jump_host: Option<String>,
    /// SSH options
    pub ssh_options: Vec<String>,
    /// Number of GPUs
    pub gpu_count: Option<u32>,
    /// Executor gRPC port
    pub executor_port: u16,
    /// Remote data directory
    pub data_dir: Option<String>,
}

/// Executor deployment and management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorManagerConfig {
    /// Remote machines to manage
    pub machines: Vec<RemoteMachine>,
    /// Local executor binary path
    pub executor_binary: PathBuf,
    /// Executor config template
    pub config_template: String,
    /// Auto-deploy on startup
    pub auto_deploy: bool,
    /// Auto-start after deployment
    pub auto_start: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Health check timeout
    pub health_check_timeout: Duration,
    /// Max failed health checks before marking unhealthy
    pub max_retry_attempts: u32,
    /// Deployment environments
    pub environments: HashMap<String, DeploymentEnvironment>,
    /// Container orchestration configuration
    pub orchestration: Option<OrchestrationPlatform>,
    /// Binary update configuration
    pub binary_updates: Option<BinaryUpdateConfig>,
    /// Maximum parallel deployments
    pub max_parallel_deployments: usize,
    /// Rollback configuration
    pub enable_rollback: bool,
    pub snapshot_retention_days: u32,
}

/// Manages executors on remote GPU machines
#[derive(Debug)]
pub struct ExecutorManager {
    pub config: ExecutorManagerConfig,
    db: RegistrationDb,
    miner_address: String,
    miner_hotkey: String,
    state: Arc<RwLock<HashMap<String, ExecutorState>>>,
    deployment_semaphore: Arc<Semaphore>,
    snapshots: Arc<RwLock<HashMap<String, DeploymentSnapshot>>>,
}

/// State of a single executor
#[derive(Debug, Clone)]
struct ExecutorState {
    machine: RemoteMachine,
    grpc_client: Option<ExecutorControlClient<Channel>>,
    is_healthy: bool,
    last_health_check: Option<Instant>,
    failed_checks: u32,
    resources: Option<ResourceUsageStats>,
    #[allow(dead_code)]
    deployment_id: Option<String>,
    #[allow(dead_code)]
    binary_version: Option<String>,
    #[allow(dead_code)]
    last_update_check: Option<Instant>,
    #[allow(dead_code)]
    orchestration_status: Option<String>,
}

#[allow(dead_code)]
impl ExecutorManager {
    /// Create a new executor manager
    pub async fn new(config: &MinerConfig, db: RegistrationDb) -> Result<Self> {
        let machines: Vec<RemoteMachine> = if let Some(executor_config) =
            &config.remote_executor_deployment
        {
            // Use remote deployment configuration
            executor_config
                .remote_machines
                .iter()
                .map(|m| RemoteMachine {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    host: m.ssh.host.clone(),
                    port: m.ssh.port,
                    username: m.ssh.username.clone(),
                    private_key_path: m.ssh.private_key_path.clone(),
                    jump_host: m.ssh.jump_host.clone(),
                    ssh_options: m.ssh.ssh_options.clone(),
                    gpu_count: m.gpu_count,
                    executor_port: m.executor_port,
                    data_dir: m.executor_data_dir.clone(),
                })
                .collect()
        } else if !config.executor_management.executors.is_empty() {
            // Fall back to static executor configuration for backward compatibility
            config
                .executor_management
                .executors
                .iter()
                .map(|e| {
                    let (host, port) = if let Some(colon_pos) = e.grpc_address.rfind(':') {
                        let host = &e.grpc_address[..colon_pos];
                        let port = e.grpc_address[colon_pos + 1..].parse().unwrap_or(8080);
                        (host.to_string(), port)
                    } else {
                        (e.grpc_address.clone(), 8080)
                    };

                    RemoteMachine {
                        id: e.id.clone(),
                        name: e
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("Executor {}", e.id)),
                        host,
                        port: 22, // Default SSH port (not used for static executors)
                        username: "unused".to_string(), // Not used for static executors
                        private_key_path: std::path::PathBuf::from("/dev/null"), // Not used for static executors
                        jump_host: None,
                        ssh_options: vec![],
                        gpu_count: None,
                        executor_port: port,
                        data_dir: None,
                    }
                })
                .collect()
        } else {
            return Err(anyhow::anyhow!("Either remote_executor_deployment or executor_management.executors must be configured"));
        };

        let manager_config = ExecutorManagerConfig {
            machines,
            executor_binary: config
                .remote_executor_deployment
                .as_ref()
                .map(|c| c.local_executor_binary.clone())
                .unwrap_or_else(|| std::path::PathBuf::from("./target/debug/executor")),
            config_template: config
                .remote_executor_deployment
                .as_ref()
                .map(|c| c.executor_config_template.clone())
                .unwrap_or_else(|| "default_config".to_string()),
            auto_deploy: config
                .remote_executor_deployment
                .as_ref()
                .map(|c| c.auto_deploy)
                .unwrap_or(false),
            auto_start: config
                .remote_executor_deployment
                .as_ref()
                .map(|c| c.auto_start)
                .unwrap_or(false),
            health_check_interval: config
                .remote_executor_deployment
                .as_ref()
                .map(|c| c.health_check_interval)
                .unwrap_or(config.executor_management.health_check_interval),
            health_check_timeout: config.executor_management.health_check_timeout,
            max_retry_attempts: config.executor_management.max_retry_attempts,
            environments: Self::create_default_environments(),
            orchestration: None,
            binary_updates: None,
            max_parallel_deployments: 3,
            enable_rollback: true,
            snapshot_retention_days: 7,
        };

        let miner_address = format!("http://{}:{}", config.server.host, config.server.port);
        let miner_hotkey = config.bittensor.common.hotkey_name.clone();

        // Ensure at least one machine is configured
        if manager_config.machines.is_empty() {
            return Err(anyhow::anyhow!("At least one executor must be configured"));
        }

        let mut state = HashMap::new();
        for machine in &manager_config.machines {
            state.insert(
                machine.id.clone(),
                ExecutorState {
                    machine: machine.clone(),
                    grpc_client: None,
                    is_healthy: false,
                    last_health_check: None,
                    failed_checks: 0,
                    resources: None,
                    deployment_id: None,
                    binary_version: None,
                    last_update_check: None,
                    orchestration_status: None,
                },
            );
        }

        let manager = Self {
            config: manager_config.clone(),
            db,
            miner_address,
            miner_hotkey,
            state: Arc::new(RwLock::new(state)),
            deployment_semaphore: Arc::new(Semaphore::new(manager_config.max_parallel_deployments)),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
        };

        // Auto-deploy if configured
        if manager.config.auto_deploy {
            info!("Auto-deploying executors...");
            let _ = manager.deploy_all().await;
        }

        // Start update checker if configured
        if manager.config.binary_updates.is_some() {
            let update_manager = manager.clone();
            tokio::spawn(async move {
                update_manager.binary_update_loop().await;
            });
        }

        Ok(manager)
    }

    /// Deploy executors to all machines
    pub async fn deploy_all(&self) -> Result<Vec<DeploymentResult>> {
        info!("Deploying to {} machines", self.config.machines.len());

        let mut results = Vec::new();
        for machine in &self.config.machines {
            let deployment_start = Instant::now();
            let result = self.deploy_to_machine(machine).await;
            let deployment_time = deployment_start.elapsed();
            let success = result.is_ok();
            let error = result.err().map(|e| e.to_string());

            results.push(DeploymentResult {
                machine_id: machine.id.clone(),
                machine_name: machine.name.clone(),
                success,
                error,
                deployment_time,
                rollback_performed: false,
                health_check_passed: success,
                binary_version: self.get_binary_version().await.ok(),
            });
        }

        Ok(results)
    }

    /// Deploy executor to a specific machine
    async fn deploy_to_machine(&self, machine: &RemoteMachine) -> Result<()> {
        info!("Deploying to {} ({})", machine.name, machine.host);

        // 1. Test SSH connection
        self.ssh_command(machine, "echo 'SSH OK'").await?;

        // 2. Create directories
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");
        let cmds = vec![
            format!("sudo mkdir -p {}/config {}/logs", data_dir, data_dir),
            format!("sudo chown -R {} {}", machine.username, data_dir),
        ];
        for cmd in cmds {
            self.ssh_command(machine, &cmd).await?;
        }

        // 3. Copy binary
        self.scp_file(
            machine,
            self.config.executor_binary.to_str().unwrap(),
            &format!("{data_dir}/executor"),
        )
        .await?;
        self.ssh_command(machine, &format!("sudo chmod +x {data_dir}/executor"))
            .await?;

        // 4. Deploy config
        let config = self.generate_config(machine)?;
        let temp_file = format!("/tmp/executor-{}.toml", machine.id);
        tokio::fs::write(&temp_file, &config).await?;
        self.scp_file(
            machine,
            &temp_file,
            &format!("{data_dir}/config/executor.toml"),
        )
        .await?;
        tokio::fs::remove_file(&temp_file).await.ok();

        // 5. Install systemd service
        let service = self.generate_systemd_service(machine);
        let service_file = format!("/tmp/basilica-executor-{}.service", machine.id);
        tokio::fs::write(&service_file, &service).await?;
        self.scp_file(machine, &service_file, "/tmp/basilica-executor.service")
            .await?;
        self.ssh_command(
            machine,
            "sudo mv /tmp/basilica-executor.service /etc/systemd/system/",
        )
        .await?;
        self.ssh_command(machine, "sudo systemctl daemon-reload")
            .await?;
        self.ssh_command(machine, "sudo systemctl enable basilica-executor")
            .await?;
        tokio::fs::remove_file(&service_file).await.ok();

        // 6. Start if configured
        if self.config.auto_start {
            self.ssh_command(machine, "sudo systemctl start basilica-executor")
                .await?;
            tokio::time::sleep(Duration::from_secs(2)).await;

            let status = self
                .ssh_command(machine, "sudo systemctl is-active basilica-executor")
                .await?;
            if status.trim() != "active" {
                return Err(anyhow::anyhow!("Failed to start executor"));
            }
        }

        info!("Successfully deployed to {}", machine.name);
        Ok(())
    }

    /// Start health monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        let manager = self.clone();
        tokio::spawn(async move {
            info!(
                "Starting executor health monitoring for {} machines",
                manager.config.machines.len()
            );
            manager.health_check_loop().await;
        });
        Ok(())
    }

    /// Get available executors
    pub async fn list_available(&self) -> Result<Vec<AvailableExecutor>> {
        let state = self.state.read().await;
        Ok(state
            .values()
            .filter(|s| s.is_healthy)
            .map(|s| AvailableExecutor {
                id: s.machine.id.clone(),
                name: s.machine.name.clone(),
                grpc_address: format!("{}:{}", s.machine.host, s.machine.executor_port),
                resources: s.resources.clone(),
                gpu_count: s.machine.gpu_count.unwrap_or(0),
            })
            .collect())
    }

    /// Health check loop
    async fn health_check_loop(&self) {
        let mut interval = interval(self.config.health_check_interval);

        loop {
            interval.tick().await;

            let machine_ids: Vec<String> = {
                let state = self.state.read().await;
                state.keys().cloned().collect()
            };

            for id in machine_ids {
                if let Err(e) = self.check_health(&id).await {
                    warn!("Health check failed for {}: {}", id, e);
                }
            }
        }
    }

    /// Check health of a specific executor
    async fn check_health(&self, machine_id: &str) -> Result<()> {
        let client = {
            let mut state = self.state.write().await;
            let executor_state = state
                .get_mut(machine_id)
                .ok_or_else(|| anyhow::anyhow!("Machine {} not found", machine_id))?;

            // Get or create gRPC client
            if executor_state.grpc_client.is_none() {
                let endpoint = Endpoint::from_shared(format!(
                    "http://{}:{}",
                    executor_state.machine.host, executor_state.machine.executor_port
                ))?
                .timeout(self.config.health_check_timeout)
                .connect_timeout(Duration::from_secs(10));

                match endpoint.connect().await {
                    Ok(channel) => {
                        executor_state.grpc_client = Some(ExecutorControlClient::new(channel));
                    }
                    Err(e) => {
                        executor_state.failed_checks += 1;
                        executor_state.is_healthy = false;
                        return Err(e.into());
                    }
                }
            }

            executor_state.grpc_client.clone().unwrap()
        };

        // Perform health check
        let mut client = client;
        let request = HealthCheckRequest {
            requester: "miner".to_string(),
            check_type: "basic".to_string(),
        };

        match tokio::time::timeout(
            self.config.health_check_timeout,
            client.health_check(request),
        )
        .await
        {
            Ok(Ok(response)) => {
                let response = response.into_inner();
                self.handle_health_success(machine_id, response).await?;
            }
            Ok(Err(e)) => {
                self.handle_health_failure(machine_id).await?;
                return Err(e.into());
            }
            Err(_) => {
                self.handle_health_failure(machine_id).await?;
                return Err(anyhow::anyhow!("Health check timeout"));
            }
        }

        Ok(())
    }

    /// Handle successful health check
    async fn handle_health_success(
        &self,
        machine_id: &str,
        response: HealthCheckResponse,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        if let Some(executor_state) = state.get_mut(machine_id) {
            executor_state.is_healthy = response.status == "healthy";
            executor_state.last_health_check = Some(Instant::now());
            executor_state.failed_checks = 0;
            executor_state.resources = self.parse_resources(&response.resource_status);

            self.db
                .update_executor_health(machine_id, executor_state.is_healthy)
                .await?;
        }
        Ok(())
    }

    /// Handle failed health check
    async fn handle_health_failure(&self, machine_id: &str) -> Result<()> {
        let mut state = self.state.write().await;
        if let Some(executor_state) = state.get_mut(machine_id) {
            executor_state.failed_checks += 1;

            if executor_state.failed_checks >= self.config.max_retry_attempts {
                executor_state.is_healthy = false;
                executor_state.grpc_client = None; // Reset connection
                error!(
                    "Executor {} marked unhealthy after {} failures",
                    machine_id, executor_state.failed_checks
                );
            }

            self.db.update_executor_health(machine_id, false).await?;
        }
        Ok(())
    }

    /// Parse resource stats from health check response
    fn parse_resources(&self, status: &HashMap<String, String>) -> Option<ResourceUsageStats> {
        Some(ResourceUsageStats {
            cpu_percent: status.get("cpu_usage")?.parse().ok()?,
            memory_mb: status.get("memory_mb")?.parse().ok()?,
            network_rx_bytes: status.get("network_rx_bytes")?.parse().ok()?,
            network_tx_bytes: status.get("network_tx_bytes")?.parse().ok()?,
            disk_read_bytes: status.get("disk_read_bytes")?.parse().ok()?,
            disk_write_bytes: status.get("disk_write_bytes")?.parse().ok()?,
            gpu_utilization: status
                .get("gpu_utilization")?
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect::<Vec<_>>(),
            gpu_memory_mb: status
                .get("gpu_memory_mb")?
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect::<Vec<_>>(),
        })
    }

    /// Run SSH command
    async fn ssh_command(&self, machine: &RemoteMachine, cmd: &str) -> Result<String> {
        let mut ssh_cmd = "ssh".to_string();

        for opt in &machine.ssh_options {
            ssh_cmd.push_str(&format!(" -o {opt}"));
        }

        ssh_cmd.push_str(&format!(
            " -p {} -i {}",
            machine.port,
            machine.private_key_path.display()
        ));

        if let Some(jump) = &machine.jump_host {
            ssh_cmd.push_str(&format!(" -J {jump}"));
        }

        ssh_cmd.push_str(&format!(" {}@{} '{}'", machine.username, machine.host, cmd));

        let output = Command::new("sh")
            .arg("-c")
            .arg(&ssh_cmd)
            .output()
            .await
            .context("Failed to execute SSH command")?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(anyhow::anyhow!(
                "SSH command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    /// Copy file via SCP
    async fn scp_file(&self, machine: &RemoteMachine, local: &str, remote: &str) -> Result<()> {
        let mut scp_cmd = "scp".to_string();

        for opt in &machine.ssh_options {
            scp_cmd.push_str(&format!(" -o {opt}"));
        }

        scp_cmd.push_str(&format!(
            " -P {} -i {}",
            machine.port,
            machine.private_key_path.display()
        ));

        if let Some(jump) = &machine.jump_host {
            scp_cmd.push_str(&format!(" -J {jump}"));
        }

        scp_cmd.push_str(&format!(
            " {} {}@{}:{}",
            local, machine.username, machine.host, remote
        ));

        let output = Command::new("sh").arg("-c").arg(&scp_cmd).output().await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "SCP failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    /// Generate executor config from template
    fn generate_config(&self, machine: &RemoteMachine) -> Result<String> {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");

        Ok(self
            .config
            .config_template
            .replace("{EXECUTOR_ID}", &machine.id)
            .replace("{EXECUTOR_NAME}", &machine.name)
            .replace("{EXECUTOR_PORT}", &machine.executor_port.to_string())
            .replace("{MINER_HOTKEY}", &self.miner_hotkey)
            .replace("{MINER_ADDRESS}", &self.miner_address)
            .replace("{DATA_DIR}", data_dir)
            .replace("{GPU_COUNT}", &machine.gpu_count.unwrap_or(0).to_string()))
    }

    /// Generate systemd service file
    fn generate_systemd_service(&self, machine: &RemoteMachine) -> String {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");

        format!(
            r#"[Unit]
Description=Basilica Executor - {}
After=network.target

[Service]
Type=simple
User={}
WorkingDirectory={}
ExecStart={}/executor --server -c {}/config/executor.toml
Restart=always
RestartSec=10
StandardOutput=append:{}/logs/executor.log
StandardError=append:{}/logs/executor.log
Environment="NVIDIA_VISIBLE_DEVICES=all"
Environment="NVIDIA_DRIVER_CAPABILITIES=compute,utility"

[Install]
WantedBy=multi-user.target
"#,
            machine.name, machine.username, data_dir, data_dir, data_dir, data_dir, data_dir,
        )
    }

    /// Deploy SSH public key to executor machine for validator access
    pub async fn deploy_ssh_key_to_executor(
        &self,
        executor_id: &str,
        ssh_username: &str,
        public_key: &str,
    ) -> Result<()> {
        info!(
            "Deploying SSH key for user {} to executor {}",
            ssh_username, executor_id
        );

        // Find the executor's machine configuration
        let machine = self.get_executor_machine(executor_id).await?;

        // Create SSH user on the executor machine
        let create_user_cmd = format!("sudo useradd -m -s /bin/bash {ssh_username} || true");
        self.ssh_command(&machine, &create_user_cmd).await?;

        // Setup .ssh directory
        let setup_ssh_dir_cmd = format!(
            "sudo mkdir -p /home/{ssh_username}/.ssh && sudo chown {ssh_username}:{ssh_username} /home/{ssh_username}/.ssh && sudo chmod 700 /home/{ssh_username}/.ssh"
        );
        self.ssh_command(&machine, &setup_ssh_dir_cmd).await?;

        // Deploy the public key to authorized_keys
        let authorized_keys_path = format!("/home/{ssh_username}/.ssh/authorized_keys");
        let deploy_key_cmd = format!(
            "echo '{}' | sudo tee -a {} && sudo chown {} {} && sudo chmod 600 {}",
            public_key.trim(),
            authorized_keys_path,
            ssh_username,
            authorized_keys_path,
            authorized_keys_path
        );
        self.ssh_command(&machine, &deploy_key_cmd).await?;

        // Set up limited sudo access for validator operations
        let sudo_config = format!(
            "echo '{ssh_username} ALL=(ALL) NOPASSWD: /usr/local/bin/gpu-attestor, /bin/systemctl status basilica-executor' | sudo tee /etc/sudoers.d/{ssh_username}"
        );
        if let Err(e) = self.ssh_command(&machine, &sudo_config).await {
            warn!("Failed to set up sudo access for {}: {}", ssh_username, e);
            // Continue - sudo access is optional
        }

        info!(
            "SSH key successfully deployed for user {} on executor {}",
            ssh_username, executor_id
        );

        Ok(())
    }

    /// Clean up SSH configuration from executor machine
    pub async fn cleanup_ssh_key_from_executor(
        &self,
        executor_id: &str,
        ssh_username: &str,
    ) -> Result<()> {
        info!(
            "Cleaning up SSH configuration for user {} from executor {}",
            ssh_username, executor_id
        );

        // Find the executor's machine configuration
        let machine = match self.get_executor_machine(executor_id).await {
            Ok(machine) => machine,
            Err(_) => {
                warn!("Executor {} not found during cleanup", executor_id);
                return Ok(());
            }
        };

        // Remove the SSH user and their home directory
        let cleanup_user_cmd = format!("sudo userdel -r {ssh_username}");
        if let Err(e) = self.ssh_command(&machine, &cleanup_user_cmd).await {
            warn!(
                "Failed to remove user {} from executor {}: {}",
                ssh_username, executor_id, e
            );
        }

        // Remove sudo configuration
        let cleanup_sudo_cmd = format!("sudo rm -f /etc/sudoers.d/{ssh_username}");
        if let Err(e) = self.ssh_command(&machine, &cleanup_sudo_cmd).await {
            warn!(
                "Failed to remove sudo config for {} from executor {}: {}",
                ssh_username, executor_id, e
            );
        }

        info!(
            "SSH configuration cleaned up for user {} from executor {}",
            ssh_username, executor_id
        );

        Ok(())
    }

    /// Get machine configuration for a specific executor
    async fn get_executor_machine(&self, executor_id: &str) -> Result<RemoteMachine> {
        // Check configured machines first
        for machine in &self.config.machines {
            if machine.id == executor_id {
                return Ok(machine.clone());
            }
        }

        // Check static executors - create basic SSH config based on gRPC address
        let executors = self.list_available().await?;
        let executor = executors
            .iter()
            .find(|e| e.id == executor_id)
            .ok_or_else(|| anyhow::anyhow!("Executor {} not found", executor_id))?;

        // Extract host from gRPC address for SSH
        let (host, _port) = if let Some(colon_pos) = executor.grpc_address.rfind(':') {
            let host = &executor.grpc_address[..colon_pos];
            let port = executor.grpc_address[colon_pos + 1..]
                .parse()
                .unwrap_or(50051);
            (host.to_string(), port)
        } else {
            (executor.grpc_address.clone(), 50051)
        };

        // Return basic SSH configuration (assumes SSH is configured)
        Ok(RemoteMachine {
            id: executor.id.clone(),
            name: executor.name.clone(),
            host,
            port: 22,                                         // SSH port
            username: "root".to_string(),                     // Default - should be configurable
            private_key_path: PathBuf::from("~/.ssh/id_rsa"), // Default - should be configurable
            jump_host: None,
            ssh_options: vec![
                "StrictHostKeyChecking=no".to_string(),
                "UserKnownHostsFile=/dev/null".to_string(),
            ],
            gpu_count: Some(executor.gpu_count),
            executor_port: 50051,
            data_dir: Some("/opt/basilica".to_string()),
        })
    }

    /// Create default deployment environments
    fn create_default_environments() -> HashMap<String, DeploymentEnvironment> {
        let mut environments = HashMap::new();

        // Production environment
        let production = DeploymentEnvironment {
            name: "production".to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("RUST_LOG".to_string(), "info".to_string());
                vars.insert("EXECUTOR_MODE".to_string(), "production".to_string());
                vars
            },
            config_overrides: HashMap::new(),
            resource_limits: None,
            deployment_strategy: DeploymentStrategy::Sequential,
            health_check_config: HealthCheckDeploymentConfig {
                startup_timeout: Duration::from_secs(120),
                readiness_checks: 5,
                check_interval: Duration::from_secs(10),
                failure_threshold: 3,
                custom_checks: vec![CustomHealthCheck {
                    name: "grpc_health".to_string(),
                    command: "grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check"
                        .to_string(),
                    expected_output: Some("SERVING".to_string()),
                    timeout: Duration::from_secs(10),
                    retry_count: 3,
                }],
            },
        };

        // Development environment
        let development = DeploymentEnvironment {
            name: "development".to_string(),
            variables: {
                let mut vars = HashMap::new();
                vars.insert("RUST_LOG".to_string(), "debug".to_string());
                vars.insert("EXECUTOR_MODE".to_string(), "development".to_string());
                vars
            },
            config_overrides: HashMap::new(),
            resource_limits: None,
            deployment_strategy: DeploymentStrategy::Parallel { max_concurrent: 5 },
            health_check_config: HealthCheckDeploymentConfig {
                startup_timeout: Duration::from_secs(60),
                readiness_checks: 3,
                check_interval: Duration::from_secs(5),
                failure_threshold: 2,
                custom_checks: vec![],
            },
        };

        environments.insert("production".to_string(), production);
        environments.insert("development".to_string(), development);

        environments
    }

    /// Create deployment snapshot for rollback capability
    async fn create_deployment_snapshot(
        &self,
        machine: &RemoteMachine,
    ) -> Result<DeploymentSnapshot> {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");
        let timestamp = chrono::Utc::now().timestamp();

        let snapshot = DeploymentSnapshot {
            machine_id: machine.id.clone(),
            timestamp,
            binary_backup_path: format!("{data_dir}/backup/executor-snapshot-{timestamp}"),
            config_backup_path: format!("{data_dir}/backup/config-snapshot-{timestamp}.toml"),
            service_was_running: self.is_service_running(machine).await.unwrap_or(false),
            previous_version: self.get_deployed_version(machine).await.ok(),
        };

        // Create snapshot backups
        let backup_cmds = vec![
            format!("sudo mkdir -p {}/backup", data_dir),
            format!(
                "if [ -f {}/executor ]; then sudo cp {}/executor {}; fi",
                data_dir, data_dir, snapshot.binary_backup_path
            ),
            format!(
                "if [ -f {}/config/executor.toml ]; then sudo cp {}/config/executor.toml {}; fi",
                data_dir, data_dir, snapshot.config_backup_path
            ),
        ];

        for cmd in backup_cmds {
            if let Err(e) = self.ssh_command(machine, &cmd).await {
                warn!("Snapshot backup command failed: {}", e);
            }
        }

        // Store snapshot metadata
        self.snapshots
            .write()
            .await
            .insert(machine.id.clone(), snapshot.clone());

        Ok(snapshot)
    }

    /// Rollback deployment to previous state
    async fn rollback_deployment(
        &self,
        machine: &RemoteMachine,
        snapshot: &DeploymentSnapshot,
    ) -> Result<()> {
        info!("Rolling back deployment for machine {}", machine.id);

        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");

        // Stop current service
        let _ = self
            .ssh_command(machine, "sudo systemctl stop basilica-executor")
            .await;

        // Restore binary if backup exists
        let restore_binary_cmd = format!(
            "if [ -f {} ]; then sudo cp {} {}/executor && sudo chmod +x {}/executor; fi",
            snapshot.binary_backup_path, snapshot.binary_backup_path, data_dir, data_dir
        );
        if let Err(e) = self.ssh_command(machine, &restore_binary_cmd).await {
            warn!("Failed to restore binary during rollback: {}", e);
        }

        // Restore configuration if backup exists
        let restore_config_cmd = format!(
            "if [ -f {} ]; then sudo cp {} {}/config/executor.toml; fi",
            snapshot.config_backup_path, snapshot.config_backup_path, data_dir
        );
        if let Err(e) = self.ssh_command(machine, &restore_config_cmd).await {
            warn!("Failed to restore config during rollback: {}", e);
        }

        // Restart service if it was running before
        if snapshot.service_was_running {
            self.ssh_command(machine, "sudo systemctl start basilica-executor")
                .await?;

            // Wait for service to stabilize
            sleep(Duration::from_secs(10)).await;

            // Verify service is running
            let status = self
                .ssh_command(machine, "sudo systemctl is-active basilica-executor")
                .await?;
            if status.trim() != "active" {
                return Err(anyhow::anyhow!("Service failed to start after rollback"));
            }
        }

        info!("Rollback completed for machine {}", machine.id);
        Ok(())
    }

    /// Perform pre-deployment health checks
    async fn pre_deployment_health_check(&self, machine: &RemoteMachine) -> Result<()> {
        debug!(
            "Performing pre-deployment health check for {}",
            machine.name
        );

        // Check system resources
        let disk_check = self
            .ssh_command(
                machine,
                "df -h / | tail -1 | awk '{print $5}' | sed 's/%//'",
            )
            .await?;

        if let Ok(disk_usage) = disk_check.trim().parse::<u32>() {
            if disk_usage > 90 {
                return Err(anyhow::anyhow!(
                    "Insufficient disk space: {}% used",
                    disk_usage
                ));
            }
        }

        // Check memory availability
        let memory_check = self
            .ssh_command(
                machine,
                "free | grep Mem | awk '{printf \"%.0f\", ($3/$2)*100}'",
            )
            .await?;

        if let Ok(memory_usage) = memory_check.trim().parse::<u32>() {
            if memory_usage > 95 {
                return Err(anyhow::anyhow!(
                    "Insufficient memory: {}% used",
                    memory_usage
                ));
            }
        }

        Ok(())
    }

    /// Perform comprehensive post-deployment health checks
    async fn post_deployment_health_checks(
        &self,
        machine: &RemoteMachine,
        env_config: &DeploymentEnvironment,
    ) -> Result<()> {
        info!(
            "Performing post-deployment health checks for {}",
            machine.name
        );

        let health_config = &env_config.health_check_config;
        let mut check_count = 0;
        let mut consecutive_failures = 0;

        // Wait for startup
        sleep(Duration::from_secs(10)).await;

        while check_count < health_config.readiness_checks {
            let check_start = Instant::now();

            // Check service status
            let service_status = timeout(
                health_config.check_interval,
                self.ssh_command(machine, "sudo systemctl is-active basilica-executor"),
            )
            .await??;

            if service_status.trim() != "active" {
                consecutive_failures += 1;
                warn!(
                    "Service not active on {}: {}",
                    machine.name,
                    service_status.trim()
                );
            } else {
                consecutive_failures = 0;
                debug!("Service health check passed for {}", machine.name);
            }

            // Run custom health checks
            for custom_check in &health_config.custom_checks {
                if let Err(e) = self.run_custom_health_check(machine, custom_check).await {
                    warn!(
                        "Custom health check '{}' failed for {}: {}",
                        custom_check.name, machine.name, e
                    );
                    consecutive_failures += 1;
                } else {
                    debug!(
                        "Custom health check '{}' passed for {}",
                        custom_check.name, machine.name
                    );
                }
            }

            if consecutive_failures >= health_config.failure_threshold {
                return Err(anyhow::anyhow!(
                    "Health checks failed {} consecutive times",
                    consecutive_failures
                ));
            }

            check_count += 1;

            // Wait for next check if not the last one
            if check_count < health_config.readiness_checks {
                let elapsed = check_start.elapsed();
                if elapsed < health_config.check_interval {
                    sleep(health_config.check_interval - elapsed).await;
                }
            }
        }

        info!(
            "All post-deployment health checks passed for {}",
            machine.name
        );
        Ok(())
    }

    /// Run a custom health check
    async fn run_custom_health_check(
        &self,
        machine: &RemoteMachine,
        check: &CustomHealthCheck,
    ) -> Result<()> {
        let mut attempts = 0;

        while attempts < check.retry_count {
            let result = timeout(check.timeout, self.ssh_command(machine, &check.command)).await?;

            match result {
                Ok(output) => {
                    if let Some(expected) = &check.expected_output {
                        if output.contains(expected) {
                            return Ok(());
                        } else {
                            attempts += 1;
                            if attempts < check.retry_count {
                                sleep(Duration::from_secs(2)).await;
                                continue;
                            }
                            return Err(anyhow::anyhow!(
                                "Expected output '{}' not found in '{}'",
                                expected,
                                output.trim()
                            ));
                        }
                    } else {
                        return Ok(());
                    }
                }
                Err(e) => {
                    attempts += 1;
                    if attempts < check.retry_count {
                        sleep(Duration::from_secs(2)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(anyhow::anyhow!(
            "Health check failed after {} attempts",
            check.retry_count
        ))
    }

    /// Check if service is currently running
    async fn is_service_running(&self, machine: &RemoteMachine) -> Result<bool> {
        let status = self
            .ssh_command(machine, "sudo systemctl is-active basilica-executor")
            .await?;
        Ok(status.trim() == "active")
    }

    /// Get deployed binary version
    async fn get_deployed_version(&self, machine: &RemoteMachine) -> Result<String> {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");
        let version_cmd = format!("{data_dir}/executor --version 2>/dev/null || echo 'unknown'");
        let output = self.ssh_command(machine, &version_cmd).await?;
        Ok(output.trim().to_string())
    }

    /// Get current binary version
    async fn get_binary_version(&self) -> Result<String> {
        if !self.config.executor_binary.exists() {
            return Ok("unknown".to_string());
        }

        let output = Command::new(&self.config.executor_binary)
            .arg("--version")
            .output()
            .await?
            .stdout;

        Ok(String::from_utf8_lossy(&output).trim().to_string())
    }

    /// Update deployment state
    async fn update_deployment_state(
        &self,
        machine: &RemoteMachine,
        deployment_id: Option<String>,
        success: bool,
    ) {
        let mut state = self.state.write().await;
        if let Some(executor_state) = state.get_mut(&machine.id) {
            executor_state.deployment_id = deployment_id;
            if success {
                executor_state.binary_version = self.get_binary_version().await.ok();
            }
        }
    }

    /// Binary update monitoring loop
    async fn binary_update_loop(&self) {
        let Some(update_config) = &self.config.binary_updates else {
            return;
        };

        info!("Starting binary update monitoring loop");
        let mut interval = interval(update_config.version_check_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.check_for_binary_updates().await {
                error!("Binary update check failed: {}", e);
            }
        }
    }

    /// Check for available binary updates
    async fn check_for_binary_updates(&self) -> Result<()> {
        let update_config = self.config.binary_updates.as_ref().unwrap();

        let latest_version = match &update_config.update_source {
            UpdateSource::GitHub { repo, .. } => self.check_github_version(repo).await?,
            UpdateSource::Http { url_template, .. } => {
                self.check_http_version(url_template).await?
            }
            UpdateSource::Local {
                build_command,
                binary_path,
            } => self.check_local_version(build_command, binary_path).await?,
        };

        let current_version = self.get_binary_version().await?;

        if latest_version != current_version && update_config.auto_update {
            info!(
                "New version available: {} -> {}",
                current_version, latest_version
            );

            if let Err(e) = self.perform_binary_update(&latest_version).await {
                error!("Binary update failed: {}", e);
            }
        }

        Ok(())
    }

    /// Check GitHub for latest version
    async fn check_github_version(&self, _repo: &str) -> Result<String> {
        // Implementation would use GitHub API
        // For now, return placeholder
        Ok("github-latest".to_string())
    }

    /// Check HTTP endpoint for latest version
    async fn check_http_version(&self, _url_template: &str) -> Result<String> {
        // Implementation would fetch from HTTP endpoint
        // For now, return placeholder
        Ok("http-latest".to_string())
    }

    /// Check local build for latest version
    async fn check_local_version(
        &self,
        _build_command: &str,
        _binary_path: &std::path::Path,
    ) -> Result<String> {
        // Implementation would run build command and check version
        // For now, return placeholder
        Ok("local-latest".to_string())
    }

    /// Perform binary update across all machines
    async fn perform_binary_update(&self, new_version: &str) -> Result<()> {
        info!("Performing binary update to version: {}", new_version);

        let update_config = self.config.binary_updates.as_ref().unwrap();

        // Download/prepare new binary
        let updated_binary_path = self.prepare_updated_binary(new_version).await?;

        // Update all machines based on restart strategy
        match &update_config.restart_strategy {
            RestartStrategy::Immediate => {
                self.update_all_machines_immediate(&updated_binary_path)
                    .await?
            }
            RestartStrategy::Graceful { drain_timeout } => {
                self.update_all_machines_graceful(&updated_binary_path, *drain_timeout)
                    .await?
            }
            RestartStrategy::Scheduled { restart_time } => {
                self.schedule_machine_updates(&updated_binary_path, restart_time)
                    .await?
            }
        }

        info!("Binary update to version {} completed", new_version);
        Ok(())
    }

    /// Prepare updated binary
    async fn prepare_updated_binary(&self, _version: &str) -> Result<PathBuf> {
        // Implementation would download/build the new binary
        // For now, return current binary path
        Ok(self.config.executor_binary.clone())
    }

    /// Update all machines immediately
    async fn update_all_machines_immediate(&self, _binary_path: &std::path::Path) -> Result<()> {
        // Implementation would deploy new binary immediately to all machines
        info!("Immediate binary update deployment initiated");
        Ok(())
    }

    /// Update all machines gracefully
    async fn update_all_machines_graceful(
        &self,
        _binary_path: &std::path::Path,
        drain_timeout: Duration,
    ) -> Result<()> {
        // Implementation would gracefully drain and update machines
        info!(
            "Graceful binary update deployment initiated with {}s drain timeout",
            drain_timeout.as_secs()
        );
        Ok(())
    }

    /// Schedule machine updates
    async fn schedule_machine_updates(
        &self,
        _binary_path: &std::path::Path,
        restart_time: &str,
    ) -> Result<()> {
        // Implementation would schedule updates for specific time
        info!("Scheduled binary update deployment for: {}", restart_time);
        Ok(())
    }

    /// Generate configuration with environment-specific values
    fn generate_config_with_environment(
        &self,
        machine: &RemoteMachine,
        env_config: &DeploymentEnvironment,
    ) -> Result<String> {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");

        let mut config = self
            .config
            .config_template
            .replace("{EXECUTOR_ID}", &machine.id)
            .replace("{EXECUTOR_NAME}", &machine.name)
            .replace("{EXECUTOR_PORT}", &machine.executor_port.to_string())
            .replace("{MINER_HOTKEY}", &self.miner_hotkey)
            .replace("{MINER_ADDRESS}", &self.miner_address)
            .replace("{DATA_DIR}", data_dir)
            .replace("{GPU_COUNT}", &machine.gpu_count.unwrap_or(0).to_string())
            .replace("{ENVIRONMENT}", &env_config.name);

        // Apply environment variables
        for (key, value) in &env_config.variables {
            config = config.replace(&format!("{{{key}}}"), value);
        }

        // Apply configuration overrides
        for (key, value) in &env_config.config_overrides {
            config = config.replace(&format!("{{{key}}}"), value);
        }

        Ok(config)
    }

    /// Generate enhanced systemd service file with deployment metadata
    fn generate_systemd_service_enhanced(
        &self,
        machine: &RemoteMachine,
        deployment_id: &str,
        binary_version: &str,
    ) -> String {
        let data_dir = machine.data_dir.as_deref().unwrap_or("/opt/basilica");

        format!(
            r#"[Unit]
Description=Basilica Executor - {} (Deployment: {}, Version: {})
After=network.target

[Service]
Type=simple
User={}
WorkingDirectory={}
ExecStart={}/executor --server -c {}/config/executor.toml
Restart=always
RestartSec=10
StandardOutput=append:{}/logs/executor.log
StandardError=append:{}/logs/executor.log
Environment="NVIDIA_VISIBLE_DEVICES=all"
Environment="NVIDIA_DRIVER_CAPABILITIES=compute,utility"
Environment="DEPLOYMENT_ID={}"
Environment="BINARY_VERSION={}"

# Resource limits
LimitNOFILE=65536
TasksMax=4096

# Security settings
NoNewPrivileges=true
PrivateDevices=false
ProtectSystem=strict
ReadWritePaths={}
ReadWritePaths={}/logs
ReadWritePaths={}/config

[Install]
WantedBy=multi-user.target
"#,
            machine.name,
            deployment_id,
            binary_version,
            machine.username,
            data_dir,
            data_dir,
            data_dir,
            data_dir,
            data_dir,
            deployment_id,
            binary_version,
            data_dir,
            data_dir,
            data_dir,
        )
    }
}

// Clone implementation for ExecutorManager
impl Clone for ExecutorManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            db: self.db.clone(),
            miner_address: self.miner_address.clone(),
            miner_hotkey: self.miner_hotkey.clone(),
            state: self.state.clone(),
            deployment_semaphore: self.deployment_semaphore.clone(),
            snapshots: self.snapshots.clone(),
        }
    }
}

/// Available executor information
#[derive(Debug, Clone)]
pub struct AvailableExecutor {
    pub id: String,
    pub name: String,
    pub grpc_address: String,
    pub resources: Option<ResourceUsageStats>,
    pub gpu_count: u32,
}

/// Deployment result
#[derive(Debug, Clone, Serialize)]
pub struct DeploymentResult {
    pub machine_id: String,
    pub machine_name: String,
    pub success: bool,
    pub error: Option<String>,
    pub deployment_time: Duration,
    pub rollback_performed: bool,
    pub health_check_passed: bool,
    pub binary_version: Option<String>,
}

/// Deployment configuration with environment-specific values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEnvironment {
    pub name: String,
    pub variables: HashMap<String, String>,
    pub config_overrides: HashMap<String, String>,
    pub resource_limits: Option<ContainerResourceLimits>,
    pub deployment_strategy: DeploymentStrategy,
    pub health_check_config: HealthCheckDeploymentConfig,
}

/// Deployment strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Sequential deployment one machine at a time
    Sequential,
    /// Parallel deployment with specified concurrency
    Parallel { max_concurrent: usize },
    /// Rolling deployment with percentage-based rollout
    Rolling { batch_size_percent: u8 },
    /// Blue-green deployment
    BlueGreen,
}

/// Health check configuration during deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckDeploymentConfig {
    pub startup_timeout: Duration,
    pub readiness_checks: u32,
    pub check_interval: Duration,
    pub failure_threshold: u32,
    pub custom_checks: Vec<CustomHealthCheck>,
}

/// Custom health check definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHealthCheck {
    pub name: String,
    pub command: String,
    pub expected_output: Option<String>,
    pub timeout: Duration,
    pub retry_count: u32,
}

/// Rollback snapshot for failed deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeploymentSnapshot {
    pub machine_id: String,
    pub timestamp: i64,
    pub binary_backup_path: String,
    pub config_backup_path: String,
    pub service_was_running: bool,
    pub previous_version: Option<String>,
}

/// Container orchestration platform support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationPlatform {
    Kubernetes {
        kubeconfig_path: PathBuf,
        namespace: String,
        context: Option<String>,
    },
    DockerSwarm {
        manager_endpoint: String,
        stack_name: String,
    },
    Nomad {
        endpoint: String,
        datacenter: String,
        namespace: Option<String>,
    },
}

/// Binary update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryUpdateConfig {
    pub update_source: UpdateSource,
    pub version_check_interval: Duration,
    pub auto_update: bool,
    pub backup_previous_version: bool,
    pub restart_strategy: RestartStrategy,
}

/// Source for binary updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateSource {
    /// GitHub releases
    GitHub {
        repo: String,
        asset_pattern: String,
        auth_token: Option<String>,
    },
    /// Direct HTTP download
    Http {
        url_template: String,
        checksum_url: Option<String>,
    },
    /// Local build system
    Local {
        build_command: String,
        binary_path: PathBuf,
    },
}

/// Restart strategy after binary update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartStrategy {
    /// Immediate restart
    Immediate,
    /// Graceful restart with drain
    Graceful { drain_timeout: Duration },
    /// Scheduled restart
    Scheduled { restart_time: String },
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_config_generation() {
        // Add tests here
    }
}
