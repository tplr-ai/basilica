use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use basilica_common::ssh::SshConnectionDetails;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatProfile {
    pub is_accessible: bool,
    pub test_port: u16,
    pub test_path: String,
    pub container_id: Option<String>,
    pub response_content: Option<String>,
    pub test_timestamp: DateTime<Utc>,
    pub full_json: String,
    pub error_message: Option<String>,
}

#[derive(Clone)]
pub struct NatCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
    http_timeout_secs: u64,
    max_retries: u32,
}

impl NatCollector {
    pub fn new(ssh_client: Arc<ValidatorSshClient>, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            ssh_client,
            persistence,
            http_timeout_secs: 10,
            max_retries: 3,
        }
    }

    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NatProfile> {
        info!(node_id = node_id, "[NAT] Starting NAT validation");

        let test_id = Uuid::new_v4().to_string().replace("-", "");
        let test_content = format!("NAT_TEST_{}", test_id);

        debug!(
            node_id = node_id,
            test_id = test_id,
            "[NAT] Generated test parameters"
        );

        let container_name = format!("nat_test_{}", test_id);
        let start_container_cmd = format!(
            r#"docker run -d --name {container_name} -p 0:8000 --rm python:3-alpine sh -c 'echo "{test_content}" > /tmp/test.txt && cd /tmp && python -m http.server 8000 --bind 0.0.0.0'"#,
        );

        let container_result = self
            .ssh_client
            .execute_command(ssh_details, &start_container_cmd, true)
            .await;

        let container_id = match container_result {
            Ok(output) => {
                let id = output.trim().to_string();
                if id.is_empty() || id.contains("Error") {
                    error!(
                        node_id = node_id,
                        output = output,
                        "[NAT] Failed to start container"
                    );
                    return Err(anyhow::anyhow!(
                        "Failed to start Docker container: {}",
                        output
                    ));
                }
                info!(
                    node_id = node_id,
                    container_id = id,
                    "[NAT] Container started successfully"
                );
                id
            }
            Err(e) => {
                error!(
                    node_id = node_id,
                    error = %e,
                    "[NAT] Failed to start container"
                );
                return Err(anyhow::anyhow!("Failed to start Docker container: {}", e));
            }
        };

        let ready = self
            .wait_for_container_ready(ssh_details, &container_id, node_id)
            .await;

        if !ready {
            self.cleanup_container(ssh_details, &container_id, node_id)
                .await;
            return Err(anyhow::anyhow!(
                "Container failed to become ready within timeout"
            ));
        }

        let get_port_cmd = format!("docker port {container_name} 8000 | cut -d':' -f2");
        let test_port = match self
            .ssh_client
            .execute_command(ssh_details, &get_port_cmd, true)
            .await
        {
            Ok(output) => {
                let port_str = output.trim();
                match port_str.parse::<u16>() {
                    Ok(port) => {
                        info!(
                            node_id = node_id,
                            allocated_port = port,
                            "[NAT] OS allocated port for container"
                        );
                        port
                    }
                    Err(e) => {
                        error!(
                            node_id = node_id,
                            output = port_str,
                            error = %e,
                            "[NAT] Failed to parse allocated port"
                        );
                        self.cleanup_container(ssh_details, &container_id, node_id)
                            .await;
                        return Err(anyhow::anyhow!("Failed to parse allocated port: {}", e));
                    }
                }
            }
            Err(e) => {
                error!(
                    node_id = node_id,
                    error = %e,
                    "[NAT] Failed to get allocated port from Docker"
                );
                self.cleanup_container(ssh_details, &container_id, node_id)
                    .await;
                return Err(anyhow::anyhow!("Failed to get allocated port: {}", e));
            }
        };

        let test_url = format!("http://{}:{}/test.txt", ssh_details.host, test_port);
        let mut is_accessible = false;
        let mut last_error = None;
        let mut response_content = None;

        for attempt in 0..self.max_retries {
            if attempt > 0 {
                let delay = Duration::from_secs(1 << attempt);
                debug!(
                    node_id = node_id,
                    attempt = attempt + 1,
                    delay_secs = delay.as_secs(),
                    "[NAT] Retrying after delay"
                );
                tokio::time::sleep(delay).await;
            }

            debug!(
                node_id = node_id,
                url = test_url,
                attempt = attempt + 1,
                "[NAT] Testing HTTP connectivity"
            );

            let http_client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(self.http_timeout_secs))
                .danger_accept_invalid_certs(true)
                .build()
            {
                Ok(client) => client,
                Err(e) => {
                    last_error = Some(format!("Failed to create HTTP client: {}", e));
                    continue;
                }
            };

            match http_client.get(&test_url).send().await {
                Ok(response) => match response.text().await {
                    Ok(text) => {
                        let text = text.trim();
                        if text == test_content {
                            is_accessible = true;
                            response_content = Some(text.to_string());
                            info!(
                                node_id = node_id,
                                attempt = attempt + 1,
                                "[NAT] Connectivity test successful"
                            );
                            break;
                        } else {
                            last_error = Some(format!(
                                "Response mismatch: expected '{}', got '{}'",
                                test_content, text
                            ));
                            debug!(
                                node_id = node_id,
                                expected = test_content,
                                received = text,
                                "[NAT] Response content mismatch"
                            );
                        }
                    }
                    Err(e) => {
                        last_error = Some(format!("Failed to read response: {}", e));
                    }
                },
                Err(e) => {
                    last_error = Some(format!("HTTP request failed: {}", e));
                    debug!(
                        node_id = node_id,
                        error = %e,
                        "[NAT] HTTP request failed"
                    );
                }
            }
        }

        self.cleanup_container(ssh_details, &container_id, node_id)
            .await;

        if is_accessible {
            info!(
                node_id = node_id,
                port = test_port,
                "[NAT] NAT validation successful - node is accessible"
            );

            Ok(NatProfile {
                is_accessible: true,
                test_port,
                test_path: "/test.txt".to_string(),
                container_id: Some(container_id.clone()),
                response_content,
                test_timestamp: Utc::now(),
                full_json: serde_json::json!({
                    "test_id": test_id,
                    "test_port": test_port,
                    "test_content": test_content,
                    "is_accessible": true,
                    "container_id": container_id,
                })
                .to_string(),
                error_message: None,
            })
        } else {
            let error_msg = last_error.unwrap_or_else(|| "Unknown error".to_string());
            error!(
                node_id = node_id,
                error = error_msg,
                "[NAT] NAT validation failed"
            );

            Err(anyhow::anyhow!(
                "NAT validation failed: node not accessible - {}",
                error_msg
            ))
        }
    }

    async fn wait_for_container_ready(
        &self,
        ssh_details: &SshConnectionDetails,
        container_id: &str,
        node_id: &str,
    ) -> bool {
        for attempt in 0..20 {
            tokio::time::sleep(Duration::from_millis(500)).await;

            let check_cmd = format!(
                "docker inspect --format='{{{{.State.Running}}}}' {} 2>/dev/null",
                container_id
            );

            match self
                .ssh_client
                .execute_command(ssh_details, &check_cmd, true)
                .await
            {
                Ok(output) => {
                    if output.trim() == "true" {
                        debug!(
                            node_id = node_id,
                            attempt = attempt + 1,
                            "[NAT] Container is running"
                        );

                        tokio::time::sleep(Duration::from_millis(500)).await;
                        return true;
                    }
                }
                Err(e) => {
                    debug!(
                        node_id = node_id,
                        error = %e,
                        "[NAT] Failed to check container status"
                    );
                }
            }
        }

        warn!(
            node_id = node_id,
            "[NAT] Container failed to become ready within timeout"
        );
        false
    }

    async fn cleanup_container(
        &self,
        ssh_details: &SshConnectionDetails,
        container_id: &str,
        node_id: &str,
    ) {
        let container_name = if container_id.starts_with("nat_test_") {
            container_id.to_string()
        } else {
            "nat_test_*".to_string()
        };

        let cleanup_cmd = format!(
            "docker rm -f {} 2>/dev/null || docker rm -f {} 2>/dev/null || true",
            container_id, container_name
        );

        if let Err(e) = self
            .ssh_client
            .execute_command(ssh_details, &cleanup_cmd, false)
            .await
        {
            debug!(
                node_id = node_id,
                error = %e,
                "[NAT] Failed to cleanup container"
            );
        }
    }

    pub async fn store(&self, miner_uid: u16, node_id: &str, profile: &NatProfile) -> Result<()> {
        debug!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[NAT] Storing NAT profile"
        );

        self.persistence
            .store_node_nat_profile(miner_uid, node_id, profile)
            .await
            .context("Failed to store NAT profile")?;

        Ok(())
    }

    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NatProfile> {
        let profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &profile).await?;
        Ok(profile)
    }

    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<NatProfile> {
        match self
            .collect_and_store(node_id, miner_uid, ssh_details)
            .await
        {
            Ok(profile) => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    is_accessible = profile.is_accessible,
                    "[NAT] NAT validation completed successfully"
                );
                Some(profile)
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %e,
                    "[NAT] NAT validation failed: {}",
                    e
                );
                None
            }
        }
    }

    pub async fn retrieve(&self, miner_uid: u16, node_id: &str) -> Result<Option<NatProfile>> {
        self.persistence
            .get_node_nat_profile(miner_uid, node_id)
            .await
            .context("Failed to retrieve NAT profile")
    }
}
