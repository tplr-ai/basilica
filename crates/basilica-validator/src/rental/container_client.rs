//! SSH-based Docker container client
//!
//! This module provides a client for executing Docker commands over SSH
//! to manage containers on remote node machines.

use anyhow::{Context, Result};
use basilica_common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, StandardSshClient,
};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

use super::types::{ContainerInfo, ContainerSpec, ContainerStatus, PortMapping, ResourceUsage};

/// SSH-based Docker client for container management
#[derive(Clone)]
pub struct ContainerClient {
    /// SSH client for command execution
    ssh_client: Arc<StandardSshClient>,
    /// SSH connection details for command execution
    ssh_details: SshConnectionDetails,
    /// SSH log level to control verbosity (ERROR, QUIET, FATAL, INFO, VERBOSE, DEBUG)
    ssh_log_level: Option<String>,
}

impl ContainerClient {
    /// Parse SSH connection string to extract components
    /// Handles formats like "user@host:port", "user@host", "user@[ipv6]:port", "user@[ipv6]"
    fn parse_ssh_connection(connection: &str) -> Result<(String, String, u16)> {
        let (username, host_port) = connection
            .split_once('@')
            .ok_or_else(|| anyhow::anyhow!("Invalid SSH connection format: missing '@'"))?;

        let (host, port) = if host_port.starts_with('[') {
            let closing_bracket = host_port
                .find(']')
                .ok_or_else(|| anyhow::anyhow!("Invalid IPv6 format: missing closing bracket"))?;

            let host = &host_port[1..closing_bracket];

            let port =
                match host_port.get(closing_bracket + 1..) {
                    Some(rest) if rest.starts_with(':') && rest.len() > 1 => rest[1..]
                        .parse::<u16>()
                        .map_err(|_| anyhow::anyhow!("Invalid port number: {}", &rest[1..]))?,
                    Some("") | None => 22,
                    Some(rest) => {
                        return Err(anyhow::anyhow!(
                        "Invalid IPv6 format: unexpected characters after closing bracket: '{}'",
                        rest
                    ))
                    }
                };

            (host.to_string(), port)
        } else if let Some((h, p)) = host_port.rsplit_once(':') {
            if h.contains(':') {
                (host_port.to_string(), 22)
            } else {
                let port_num = p
                    .parse::<u16>()
                    .map_err(|_| anyhow::anyhow!("Invalid port number: {}", p))?;
                (h.to_string(), port_num)
            }
        } else {
            (host_port.to_string(), 22)
        };

        Ok((username.to_string(), host, port))
    }

    /// Create a new container client with validator's private key
    pub fn new(ssh_connection: String, ssh_private_key_path: Option<PathBuf>) -> Result<Self> {
        let (username, host, port) = Self::parse_ssh_connection(&ssh_connection)?;

        let private_key_path = ssh_private_key_path
            .ok_or_else(|| anyhow::anyhow!("SSH private key path is required"))?;

        let ssh_config = SshConnectionConfig {
            strict_host_key_checking: false,
            known_hosts_file: None,
            connection_timeout: Duration::from_secs(10),
            execution_timeout: Duration::from_secs(300),
            ..Default::default()
        };

        let ssh_client = Arc::new(StandardSshClient::with_config(ssh_config));

        let ssh_details = SshConnectionDetails {
            host,
            username,
            port,
            private_key_path,
            timeout: Duration::from_secs(10),
        };

        Ok(Self {
            ssh_client,
            ssh_details,
            ssh_log_level: Some("ERROR".to_string()),
        })
    }

    /// Create a new container client with full SSH configuration
    pub fn with_ssh_config(
        ssh_connection: String,
        ssh_private_key_path: Option<PathBuf>,
        strict_host_key_checking: bool,
        known_hosts_file: Option<PathBuf>,
        ssh_log_level: Option<String>,
    ) -> Result<Self> {
        let (username, host, port) = Self::parse_ssh_connection(&ssh_connection)?;

        let private_key_path = ssh_private_key_path
            .ok_or_else(|| anyhow::anyhow!("SSH private key path is required"))?;

        let ssh_config = SshConnectionConfig {
            strict_host_key_checking,
            known_hosts_file: known_hosts_file.clone(),
            connection_timeout: Duration::from_secs(10),
            execution_timeout: Duration::from_secs(300),
            ..Default::default()
        };

        let ssh_client = Arc::new(StandardSshClient::with_config(ssh_config));

        let ssh_details = SshConnectionDetails {
            host,
            username,
            port,
            private_key_path,
            timeout: Duration::from_secs(10),
        };

        Ok(Self {
            ssh_client,
            ssh_details,
            ssh_log_level,
        })
    }

    /// Create a container client with an existing SSH client (for dependency injection)
    pub fn with_ssh_client(
        ssh_connection: String,
        ssh_client: Arc<StandardSshClient>,
        ssh_private_key_path: PathBuf,
        ssh_log_level: Option<String>,
    ) -> Result<Self> {
        let (username, host, port) = Self::parse_ssh_connection(&ssh_connection)?;

        let ssh_details = SshConnectionDetails {
            host,
            username,
            port,
            private_key_path: ssh_private_key_path,
            timeout: Duration::from_secs(10),
        };

        Ok(Self {
            ssh_client,
            ssh_details,
            ssh_log_level,
        })
    }

    /// Set SSH log level for runtime configuration
    pub fn set_ssh_log_level(&mut self, log_level: Option<String>) {
        self.ssh_log_level = log_level;
    }

    /// Execute a command over SSH using StandardSshClient
    pub async fn execute_ssh_command(&self, command: &str) -> Result<String> {
        debug!("Executing SSH command: {}", command);

        self.ssh_client
            .execute_command(&self.ssh_details, command, true)
            .await
            .context("Failed to execute SSH command")
    }

    /// Deploy a container based on the specification
    pub async fn deploy_container(
        &self,
        spec: &ContainerSpec,
        rental_id: &str,
    ) -> Result<ContainerInfo> {
        info!(
            rental_id = rental_id,
            "Deploying container for rental {rental_id}"
        );

        // Build docker run command as a string directly
        let mut docker_cmd_parts = vec!["docker", "run", "-d", "--rm"];

        // Add interactive and TTY flags if command is /bin/bash
        if spec.command.len() == 1 && spec.command[0] == "/bin/bash" {
            docker_cmd_parts.push("-it");
        }

        // Add container name with sanitized rental ID
        let sanitized_rental_id = self.sanitize_rental_id(rental_id);
        let container_name = format!("basilica-{sanitized_rental_id}");
        docker_cmd_parts.push("--name");
        docker_cmd_parts.push(&container_name);

        // Add labels
        docker_cmd_parts.push("--label");
        let rental_label = format!("basilica.rental_id={sanitized_rental_id}");
        docker_cmd_parts.push(&rental_label);

        // Collect all label strings first
        let label_strings: Vec<String> = spec
            .labels
            .iter()
            .flat_map(|(key, value)| vec!["--label".to_string(), format!("{key}={value}")])
            .collect();

        // Collect all env strings
        let env_strings: Vec<String> = spec
            .environment
            .iter()
            .flat_map(|(key, value)| vec!["-e".to_string(), format!("{key}={value}")])
            .collect();

        // Collect all port mappings
        let port_strings: Vec<String> = spec
            .ports
            .iter()
            .flat_map(|port| {
                vec![
                    "-p".to_string(),
                    if port.protocol.to_lowercase() == "udp" {
                        format!("{}:{}/udp", port.host_port, port.container_port)
                    } else {
                        format!("{}:{}", port.host_port, port.container_port)
                    },
                ]
            })
            .collect();

        // Resource limits
        let mut resource_strings = Vec::new();
        if spec.resources.cpu_cores.is_finite() && spec.resources.cpu_cores > 0.0 {
            resource_strings.push("--cpus".to_string());
            resource_strings.push(spec.resources.cpu_cores.to_string());
        }
        if spec.resources.memory_mb > 0 {
            resource_strings.push("-m".to_string());
            resource_strings.push(format!("{}m", spec.resources.memory_mb));
        }

        resource_strings.push("--gpus".to_string());
        resource_strings.push("all".to_string());
        resource_strings.push("--runtime".to_string());
        resource_strings.push("nvidia".to_string());

        // Volumes
        let volume_strings: Vec<String> = spec
            .volumes
            .iter()
            .flat_map(|volume| {
                let volume_spec = if volume.read_only {
                    format!("{}:{}:ro", volume.host_path, volume.container_path)
                } else {
                    format!("{}:{}", volume.host_path, volume.container_path)
                };
                vec!["-v".to_string(), volume_spec]
            })
            .collect();

        // Capabilities
        let cap_strings: Vec<String> = spec
            .capabilities
            .iter()
            .flat_map(|cap| vec!["--cap-add".to_string(), cap.clone()])
            .collect();

        // Network configuration
        let mut network_strings = Vec::new();
        if !spec.network.mode.is_empty() {
            network_strings.push("--network".to_string());
            network_strings.push(spec.network.mode.clone());
        }

        let dns_strings: Vec<String> = spec
            .network
            .dns
            .iter()
            .flat_map(|dns| vec!["--dns".to_string(), dns.clone()])
            .collect();

        let host_strings: Vec<String> = spec
            .network
            .extra_hosts
            .iter()
            .flat_map(|(hostname, ip)| {
                vec!["--add-host".to_string(), format!("{}:{}", hostname, ip)]
            })
            .collect();

        // Now build the final command string
        let mut final_cmd = docker_cmd_parts.join(" ");

        // Add all the collected strings
        for s in &label_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &env_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &port_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &resource_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &volume_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &cap_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &network_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &dns_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }
        for s in &host_strings {
            final_cmd.push(' ');
            final_cmd.push_str(s);
        }

        // Add entrypoint if specified (overrides image's default ENTRYPOINT)
        if !spec.entrypoint.is_empty() {
            final_cmd.push_str(" --entrypoint ");
            // If entrypoint has multiple parts, we need to quote it properly
            if spec.entrypoint.len() == 1 {
                final_cmd.push_str(&spec.entrypoint[0]);
            } else {
                // For multiple arguments, Docker expects a JSON array
                let entrypoint_json = serde_json::to_string(&spec.entrypoint)
                    .unwrap_or_else(|_| spec.entrypoint.join(" "));
                final_cmd.push_str(&entrypoint_json);
            }
        }

        // Add image
        final_cmd.push(' ');
        final_cmd.push_str(&spec.image);

        // Add command if specified
        if !spec.command.is_empty() {
            for cmd in &spec.command {
                final_cmd.push(' ');
                final_cmd.push_str(cmd);
            }
        }

        // Execute docker run
        let command = final_cmd;
        let container_id = self
            .execute_ssh_command(&command)
            .await
            .context("Failed to create container")?
            .trim()
            .to_string();

        info!(
            rental_id = rental_id,
            container_id = container_id,
            "Container {} created with ID: {}",
            container_name,
            container_id
        );

        // Get container info
        let validated_container_id = self.validate_container_id(&container_id)?;
        let inspect_cmd = format!("docker inspect {validated_container_id}");
        let inspect_output = self
            .execute_ssh_command(&inspect_cmd)
            .await
            .context("Failed to inspect container")?;

        let inspect_data: Vec<Value> = serde_json::from_str(&inspect_output)
            .context("Failed to parse container inspect data")?;

        if inspect_data.is_empty() {
            return Err(anyhow::anyhow!("Container not found after creation"));
        }

        let container_data = &inspect_data[0];

        // Extract port mappings and deduplicate
        // Docker returns multiple bindings per port (typically IPv4 and IPv6)
        let mut mapped_ports = Vec::new();
        let mut seen_ports = std::collections::HashSet::new();

        if let Some(ports) = container_data["NetworkSettings"]["Ports"].as_object() {
            for (container_port_proto, bindings) in ports {
                if let Some(bindings_arr) = bindings.as_array() {
                    for binding in bindings_arr {
                        if let (Some(host_port), Some(container_port)) = (
                            binding["HostPort"].as_str(),
                            container_port_proto.split('/').next(),
                        ) {
                            let protocol = container_port_proto
                                .split('/')
                                .nth(1)
                                .unwrap_or("tcp")
                                .to_string();

                            let container_port_num: u32 = container_port.parse().unwrap_or(0);
                            let host_port_num: u32 = host_port.parse().unwrap_or(0);

                            // Create a unique key for this port mapping
                            let key = (container_port_num, host_port_num, protocol.clone());

                            // Only add if we haven't seen this exact mapping before
                            if seen_ports.insert(key) {
                                mapped_ports.push(PortMapping {
                                    container_port: container_port_num,
                                    host_port: host_port_num,
                                    protocol,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ContainerInfo {
            container_id: container_id.clone(),
            container_name,
            mapped_ports,
            status: "running".to_string(),
            labels: spec.labels.clone(),
        })
    }

    /// Get container status
    pub async fn get_container_status(&self, container_id: &str) -> Result<ContainerStatus> {
        let validated_container_id = self.validate_container_id(container_id)?;
        let inspect_cmd = format!("docker inspect {validated_container_id}");
        let output = self
            .execute_ssh_command(&inspect_cmd)
            .await
            .context("Failed to inspect container")?;

        let data: Vec<Value> = serde_json::from_str(&output)?;
        if data.is_empty() {
            return Err(anyhow::anyhow!("Container not found"));
        }

        let container = &data[0];
        let state = &container["State"];

        Ok(ContainerStatus {
            container_id: container_id.to_string(),
            state: state["Status"].as_str().unwrap_or("unknown").to_string(),
            exit_code: state["ExitCode"].as_i64().map(|c| c as i32),
            health: container["State"]["Health"]["Status"]
                .as_str()
                .unwrap_or("none")
                .to_string(),
            started_at: state["StartedAt"]
                .as_str()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc)),
            finished_at: state["FinishedAt"]
                .as_str()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc)),
        })
    }

    /// Get container resource usage
    pub async fn get_resource_usage(&self, container_id: &str) -> Result<ResourceUsage> {
        let validated_container_id = self.validate_container_id(container_id)?;
        let stats_cmd = format!("docker stats {validated_container_id} --no-stream --format json");
        let output = self
            .execute_ssh_command(&stats_cmd)
            .await
            .context("Failed to get container stats")?;

        let stats: Value = serde_json::from_str(&output)?;

        // Parse CPU percentage
        let cpu_percent = stats["CPUPerc"]
            .as_str()
            .and_then(|s| s.trim_end_matches('%').parse::<f64>().ok())
            .unwrap_or(0.0);

        // Parse memory usage
        let mem_usage = stats["MemUsage"].as_str().unwrap_or("0MiB / 0MiB");
        let memory_mb = self.parse_memory_usage(mem_usage);

        // Get network and disk I/O
        let net_io = stats["NetIO"].as_str().unwrap_or("0B / 0B");
        let (network_rx_bytes, network_tx_bytes) = self.parse_network_io(net_io);

        let block_io = stats["BlockIO"].as_str().unwrap_or("0B / 0B");
        let (disk_read_bytes, disk_write_bytes) = self.parse_block_io(block_io);

        // TODO: Get GPU usage if available
        let gpu_usage = Vec::new();

        Ok(ResourceUsage {
            cpu_percent,
            memory_mb,
            disk_read_bytes,
            disk_write_bytes,
            network_rx_bytes,
            network_tx_bytes,
            gpu_usage,
        })
    }

    /// Stop a container
    pub async fn stop_container(&self, container_id: &str, force: bool) -> Result<()> {
        let validated_container_id = self.validate_container_id(container_id)?;
        let stop_cmd = if force {
            format!("docker kill {validated_container_id}")
        } else {
            format!("docker stop {validated_container_id}")
        };

        self.execute_ssh_command(&stop_cmd)
            .await
            .context("Failed to stop container")?;

        info!("Container {} stopped", container_id);
        Ok(())
    }

    /// Remove a container
    pub async fn remove_container(&self, container_id: &str) -> Result<()> {
        let validated_container_id = self.validate_container_id(container_id)?;
        // There are some cases where just having `--rm` flag during container creation is
        // not enough, so we also specify the `-v` flag to remove the volume associated with the container.
        let rm_cmd = format!("docker rm -v -f {validated_container_id}");

        self.execute_ssh_command(&rm_cmd)
            .await
            .context("Failed to remove container")?;

        info!("Container {} removed", container_id);
        Ok(())
    }

    /// Restart a container (docker restart)
    pub async fn restart_container(&self, container_id: &str, timeout_secs: u64) -> Result<()> {
        let validated_container_id = self.validate_container_id(container_id)?;
        let restart_cmd = format!(
            "docker restart -t {} {}",
            timeout_secs, validated_container_id
        );

        self.execute_ssh_command(&restart_cmd)
            .await
            .context("Failed to restart container")?;

        info!("Container {} restart initiated", container_id);

        // Wait for container to be running (with timeout)
        self.wait_for_container_running(container_id, Duration::from_secs(30))
            .await?;

        Ok(())
    }

    /// Wait for container to reach running state
    async fn wait_for_container_running(
        &self,
        container_id: &str,
        timeout: Duration,
    ) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!(
                    "Timeout waiting for container {} to start after restart",
                    container_id
                ));
            }

            let status = self.get_container_status(container_id).await?;
            if status.state == "running" {
                info!("Container {} is running", container_id);
                return Ok(());
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// Stream container logs using StandardSshClient streaming
    pub async fn stream_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> Result<tokio::process::Child> {
        let mut docker_cmd_parts = vec!["docker".to_string(), "logs".to_string()];

        if follow {
            docker_cmd_parts.push("-f".to_string());
        }

        if let Some(lines) = tail_lines {
            docker_cmd_parts.push("--tail".to_string());
            docker_cmd_parts.push(lines.to_string());
        }

        docker_cmd_parts.push("--timestamps".to_string());

        let validated_container_id = self.validate_container_id(container_id)?;
        docker_cmd_parts.push(validated_container_id.to_string());

        let docker_cmd = docker_cmd_parts.join(" ");

        debug!("Streaming container logs: {}", docker_cmd);

        self.ssh_client
            .execute_command_streaming(&self.ssh_details, &docker_cmd)
            .await
            .context("Failed to start log streaming")
    }

    /// Parse memory usage string (e.g., "100MiB / 1GiB")
    fn parse_memory_usage(&self, mem_usage: &str) -> i64 {
        let parts: Vec<&str> = mem_usage.split(" / ").collect();
        if parts.is_empty() {
            return 0;
        }

        let used = parts[0].trim();
        self.parse_size_string(used)
    }

    /// Parse network I/O string (e.g., "1.5MB / 2.3MB")
    fn parse_network_io(&self, net_io: &str) -> (i64, i64) {
        let parts: Vec<&str> = net_io.split(" / ").collect();
        if parts.len() != 2 {
            return (0, 0);
        }

        let rx = self.parse_size_string(parts[0].trim());
        let tx = self.parse_size_string(parts[1].trim());
        (rx, tx)
    }

    /// Parse block I/O string
    fn parse_block_io(&self, block_io: &str) -> (i64, i64) {
        self.parse_network_io(block_io)
    }

    /// Validate and sanitize container ID to prevent command injection
    fn validate_container_id<'a>(&self, container_id: &'a str) -> Result<&'a str> {
        if container_id.is_empty() {
            return Err(anyhow::anyhow!("Container ID cannot be empty"));
        }

        if !container_id.chars().all(|c| c.is_alphanumeric()) {
            return Err(anyhow::anyhow!("Invalid container ID format"));
        }

        if container_id.len() > 64 {
            return Err(anyhow::anyhow!("Container ID too long"));
        }

        Ok(container_id)
    }

    /// Sanitize rental ID for use in container names
    fn sanitize_rental_id(&self, rental_id: &str) -> String {
        rental_id
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .take(32) // Limit length
            .collect()
    }

    /// Parse size string with units (e.g., "100MB", "1.5GiB")
    fn parse_size_string(&self, size_str: &str) -> i64 {
        let size_str = size_str.trim();

        // Extract number and unit
        let (num_str, unit) = if let Some(idx) = size_str.find(|c: char| c.is_alphabetic()) {
            (&size_str[..idx], &size_str[idx..])
        } else {
            (size_str, "")
        };

        let num: f64 = num_str.parse().unwrap_or(0.0);

        let multiplier = match unit.to_uppercase().as_str() {
            "B" => 1,
            "KB" | "KIB" => 1024,
            "MB" | "MIB" => 1024 * 1024,
            "GB" | "GIB" => 1024 * 1024 * 1024,
            _ => 1,
        };

        (num * multiplier as f64) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ssh_connection_ipv4_with_port() {
        let result = ContainerClient::parse_ssh_connection("user@192.168.1.100:2222");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "192.168.1.100");
        assert_eq!(port, 2222);
    }

    #[test]
    fn test_parse_ssh_connection_ipv4_default_port() {
        let result = ContainerClient::parse_ssh_connection("user@192.168.1.100");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "192.168.1.100");
        assert_eq!(port, 22);
    }

    #[test]
    fn test_parse_ssh_connection_hostname_with_port() {
        let result = ContainerClient::parse_ssh_connection("admin@example.com:3000");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "admin");
        assert_eq!(host, "example.com");
        assert_eq!(port, 3000);
    }

    #[test]
    fn test_parse_ssh_connection_hostname_default_port() {
        let result = ContainerClient::parse_ssh_connection("admin@example.com");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "admin");
        assert_eq!(host, "example.com");
        assert_eq!(port, 22);
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_bracketed_with_port() {
        let result = ContainerClient::parse_ssh_connection("user@[2001:db8::1]:2222");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "2001:db8::1");
        assert_eq!(port, 2222);
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_bracketed_default_port() {
        let result = ContainerClient::parse_ssh_connection("user@[2001:db8::1]");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "2001:db8::1");
        assert_eq!(port, 22);
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_full_address_bracketed() {
        let result = ContainerClient::parse_ssh_connection(
            "root@[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:8080",
        );
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "root");
        assert_eq!(host, "2001:0db8:85a3:0000:0000:8a2e:0370:7334");
        assert_eq!(port, 8080);
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_bare_default_port() {
        let result = ContainerClient::parse_ssh_connection("user@2001:db8::1");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "2001:db8::1");
        assert_eq!(port, 22);
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_localhost_bracketed() {
        let result = ContainerClient::parse_ssh_connection("user@[::1]:3000");
        assert!(result.is_ok());
        let (username, host, port) = result.unwrap();
        assert_eq!(username, "user");
        assert_eq!(host, "::1");
        assert_eq!(port, 3000);
    }

    #[test]
    fn test_parse_ssh_connection_missing_at_sign() {
        let result = ContainerClient::parse_ssh_connection("user-example.com:2222");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing '@'"));
    }

    #[test]
    fn test_parse_ssh_connection_invalid_port() {
        let result = ContainerClient::parse_ssh_connection("user@example.com:abc");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid port number"));
    }

    #[test]
    fn test_parse_ssh_connection_port_out_of_range() {
        let result = ContainerClient::parse_ssh_connection("user@example.com:99999");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid port number"));
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_missing_closing_bracket() {
        let result = ContainerClient::parse_ssh_connection("user@[2001:db8::1:2222");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing closing bracket"));
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_invalid_characters_after_bracket() {
        let result = ContainerClient::parse_ssh_connection("user@[2001:db8::1]abc");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unexpected characters after closing bracket"));
    }

    #[test]
    fn test_parse_ssh_connection_ipv6_bracketed_invalid_port() {
        let result = ContainerClient::parse_ssh_connection("user@[2001:db8::1]:invalid");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid port number"));
    }

    #[test]
    fn test_parse_ssh_connection_hostname_with_invalid_port() {
        let result = ContainerClient::parse_ssh_connection("user@host:name");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid port number"));
    }

    #[test]
    fn test_validate_container_id_valid() {
        let temp_dir = tempfile::tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key").unwrap();

        let client = ContainerClient::new("user@example.com".to_string(), Some(key_path)).unwrap();

        let result = client.validate_container_id("abc123def456");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "abc123def456");
    }

    #[test]
    fn test_validate_container_id_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key").unwrap();

        let client = ContainerClient::new("user@example.com".to_string(), Some(key_path)).unwrap();

        let result = client.validate_container_id("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_validate_container_id_invalid_characters() {
        let temp_dir = tempfile::tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key").unwrap();

        let client = ContainerClient::new("user@example.com".to_string(), Some(key_path)).unwrap();

        let result = client.validate_container_id("abc-123");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid container ID format"));
    }

    #[test]
    fn test_sanitize_rental_id() {
        let temp_dir = tempfile::tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key").unwrap();

        let client = ContainerClient::new("user@example.com".to_string(), Some(key_path)).unwrap();

        assert_eq!(client.sanitize_rental_id("rental-123"), "rental-123");
        assert_eq!(client.sanitize_rental_id("rental@#$123"), "rental123");
        assert_eq!(
            client.sanitize_rental_id("a".repeat(40).as_str()),
            "a".repeat(32)
        );
    }
}
