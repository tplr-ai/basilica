//! Executor gRPC Client for Miner
//!
//! Provides gRPC communication between miner and executors for SSH key management
//! and other control operations.

use anyhow::{Context, Result};
use protocol::executor_control::{
    executor_control_client::ExecutorControlClient, HealthCheckRequest, HealthCheckResponse,
    ProvisionAccessRequest, ProvisionAccessResponse,
};
use std::time::Duration;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

/// Configuration for executor gRPC client
#[derive(Debug, Clone)]
pub struct ExecutorGrpcConfig {
    /// Timeout for gRPC calls
    pub timeout: Duration,
    /// Number of retry attempts
    pub max_retries: u32,
    /// Whether to use TLS for gRPC connections
    pub use_tls: bool,
}

impl Default for ExecutorGrpcConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            use_tls: false,
        }
    }
}

/// gRPC client for communicating with executors
pub struct ExecutorGrpcClient {
    config: ExecutorGrpcConfig,
}

impl ExecutorGrpcClient {
    /// Create a new executor gRPC client
    pub fn new(config: ExecutorGrpcConfig) -> Self {
        Self { config }
    }

    /// Connect to executor and provision validator SSH access
    pub async fn provision_validator_access(
        &self,
        executor_endpoint: &str,
        validator_hotkey: &str,
        ssh_public_key: &str,
        duration_seconds: u64,
    ) -> Result<ProvisionAccessResponse> {
        let grpc_endpoint = self.build_grpc_endpoint(executor_endpoint)?;
        info!(
            "Provisioning validator access via gRPC to executor at {}",
            grpc_endpoint
        );

        // Create channel with timeout
        let channel = Channel::from_shared(grpc_endpoint.clone())
            .with_context(|| format!("Invalid gRPC endpoint: {grpc_endpoint}"))?
            .connect_timeout(self.config.timeout)
            .timeout(self.config.timeout)
            .connect()
            .await
            .with_context(|| format!("Failed to connect to executor at {grpc_endpoint}"))?;

        let request = ProvisionAccessRequest {
            validator_hotkey: validator_hotkey.to_string(),
            ssh_public_key: ssh_public_key.to_string(),
            access_token: String::new(), // Not needed for SSH access
            duration_seconds,
            access_type: "ssh".to_string(),
            config: std::collections::HashMap::new(),
        };

        debug!(
            "Sending ProvisionAccessRequest for validator {} to executor",
            validator_hotkey
        );

        // DEBUG: Log the SSH public key being sent to executor
        debug!(
            "SSH public key being sent to executor: '{}' (length: {} chars)",
            ssh_public_key,
            ssh_public_key.len()
        );

        // Make gRPC call with retry logic
        let response = self
            .retry_grpc_call(|| {
                let channel = channel.clone();
                let request = request.clone();
                async move {
                    let mut client = ExecutorControlClient::new(channel);
                    client
                        .provision_validator_access(request)
                        .await
                        .map_err(|e| anyhow::anyhow!("Provision access failed: {}", e))
                }
            })
            .await?;

        let response = response.into_inner();

        if !response.success {
            let error_msg = response
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Access provisioning failed: {}", error_msg));
        }

        info!(
            "Successfully provisioned SSH access for validator {} on executor",
            validator_hotkey
        );

        Ok(response)
    }

    /// Health check executor
    pub async fn health_check(&self, executor_endpoint: &str) -> Result<HealthCheckResponse> {
        let grpc_endpoint = self.build_grpc_endpoint(executor_endpoint)?;

        let channel = Channel::from_shared(grpc_endpoint.clone())
            .with_context(|| format!("Invalid gRPC endpoint: {grpc_endpoint}"))?
            .connect_timeout(self.config.timeout)
            .timeout(self.config.timeout)
            .connect()
            .await
            .with_context(|| format!("Failed to connect to executor at {grpc_endpoint}"))?;

        let request = HealthCheckRequest {
            requester: "miner".to_string(),
            check_type: "basic".to_string(),
        };

        let response = self
            .retry_grpc_call(|| {
                let channel = channel.clone();
                let request = request.clone();
                async move {
                    let mut client = ExecutorControlClient::new(channel);
                    client
                        .health_check(request)
                        .await
                        .map_err(|e| anyhow::anyhow!("Health check failed: {}", e))
                }
            })
            .await?;

        Ok(response.into_inner())
    }

    /// Build gRPC endpoint from executor address
    fn build_grpc_endpoint(&self, executor_endpoint: &str) -> Result<String> {
        // executor_endpoint is like "185.26.8.109:50051"
        let scheme = if self.config.use_tls { "https" } else { "http" };

        // If it already has a scheme, use as-is, otherwise add scheme
        if executor_endpoint.starts_with("http://") || executor_endpoint.starts_with("https://") {
            Ok(executor_endpoint.to_string())
        } else {
            Ok(format!("{}://{}", scheme, executor_endpoint))
        }
    }

    /// Retry a gRPC call with exponential backoff
    async fn retry_grpc_call<F, Fut, T>(&self, mut call: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut backoff = Duration::from_millis(500);

        loop {
            match call().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.config.max_retries {
                        return Err(e.context(format!(
                            "Failed after {} attempts with exponential backoff",
                            self.config.max_retries
                        )));
                    }

                    warn!(
                        "gRPC call failed (attempt {}/{}): {}. Retrying in {:?}",
                        attempt, self.config.max_retries, e, backoff
                    );

                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(10));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_grpc_endpoint() {
        let client = ExecutorGrpcClient::new(ExecutorGrpcConfig::default());

        assert_eq!(
            client.build_grpc_endpoint("185.26.8.109:50051").unwrap(),
            "http://185.26.8.109:50051"
        );

        assert_eq!(
            client
                .build_grpc_endpoint("https://executor.example.com:50051")
                .unwrap(),
            "https://executor.example.com:50051"
        );
    }

    #[test]
    fn test_build_grpc_endpoint_with_tls() {
        let config = ExecutorGrpcConfig {
            use_tls: true,
            ..Default::default()
        };
        let client = ExecutorGrpcClient::new(config);

        assert_eq!(
            client.build_grpc_endpoint("185.26.8.109:50051").unwrap(),
            "https://185.26.8.109:50051"
        );
    }
}
