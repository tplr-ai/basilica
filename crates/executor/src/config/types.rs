//! Core configuration types and main executor configuration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use common::config::{loader, LoggingConfig, MetricsConfig, ServerConfig};
use common::identity::Hotkey;
use std::str::FromStr;

use super::{DockerConfig, SystemConfig};
use crate::validation_session::ValidatorConfig;

/// Advertised endpoint configuration for executor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutorAdvertisedEndpoint {
    /// Advertised gRPC endpoint for validator access
    pub grpc_endpoint: Option<String>,
    /// Advertised SSH endpoint for validator SSH access
    pub ssh_endpoint: Option<String>,
    /// Advertised health check endpoint
    pub health_endpoint: Option<String>,
    /// Force TLS for advertised endpoints
    #[serde(default)]
    pub force_tls: bool,
    /// Custom port mappings for different services
    #[serde(default)]
    pub port_mappings: HashMap<String, u16>,
}

/// Main executor configuration
///
/// Aggregates all configuration sections following the Single Responsibility Principle.
/// Each section is handled by its own module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Server configuration
    pub server: ServerConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// System monitoring configuration
    pub system: SystemConfig,

    /// Docker configuration
    pub docker: DockerConfig,

    /// Validator access configuration
    pub validator: ValidatorConfig,

    /// Managing miner hotkey (for authentication)
    pub managing_miner_hotkey: Hotkey,

    /// Advertised endpoint configuration
    #[serde(default)]
    pub advertised_endpoint: ExecutorAdvertisedEndpoint,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 50051,
                ..Default::default()
            },
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            system: SystemConfig::default(),
            docker: DockerConfig::default(),
            validator: ValidatorConfig::default(),
            // Use a simple constructor for default - in production this would come from config
            managing_miner_hotkey: Hotkey::from_str(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            )
            .unwrap(), // Default Alice hotkey
            advertised_endpoint: ExecutorAdvertisedEndpoint::default(),
        }
    }
}

impl ExecutorConfig {
    /// Load configuration using common loader
    pub fn load() -> Result<Self> {
        Ok(loader::load_config::<Self>()?)
    }

    /// Load configuration from specific file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        Ok(loader::load_from_file::<Self>(path)?)
    }

    /// Get the advertised gRPC endpoint for miner registration
    pub fn get_advertised_grpc_endpoint(&self) -> String {
        self.advertised_endpoint
            .grpc_endpoint
            .as_ref()
            .unwrap_or(&self.server.advertised_url("http"))
            .clone()
    }

    /// Get the advertised SSH endpoint for validator access
    pub fn get_advertised_ssh_endpoint(&self) -> String {
        if let Some(ref ssh_endpoint) = self.advertised_endpoint.ssh_endpoint {
            ssh_endpoint.clone()
        } else {
            let advertised_host = self
                .server
                .advertised_host
                .as_ref()
                .unwrap_or(&self.server.host);
            let ssh_port = self
                .advertised_endpoint
                .port_mappings
                .get("ssh")
                .copied()
                .unwrap_or(22);
            format!("ssh://{advertised_host}:{ssh_port}")
        }
    }

    /// Get the advertised health check endpoint
    pub fn get_advertised_health_endpoint(&self) -> String {
        if let Some(ref health_endpoint) = self.advertised_endpoint.health_endpoint {
            health_endpoint.clone()
        } else {
            let advertised_host = self
                .server
                .advertised_host
                .as_ref()
                .unwrap_or(&self.server.host);
            let health_port = self
                .advertised_endpoint
                .port_mappings
                .get("health")
                .copied()
                .unwrap_or(self.server.advertised_port.unwrap_or(self.server.port) + 1);
            format!("http://{advertised_host}:{health_port}/health")
        }
    }

    /// Validate advertised endpoint configuration
    pub fn validate_advertised_endpoints(&self) -> Result<(), String> {
        self.server.validate_advertised_config()?;

        if let Some(ref grpc_endpoint) = self.advertised_endpoint.grpc_endpoint {
            if !grpc_endpoint.starts_with("http://") && !grpc_endpoint.starts_with("https://") {
                return Err("gRPC endpoint must start with http:// or https://".to_string());
            }
        }

        if let Some(ref ssh_endpoint) = self.advertised_endpoint.ssh_endpoint {
            if !ssh_endpoint.starts_with("ssh://") {
                return Err("SSH endpoint must start with ssh://".to_string());
            }
        }

        for (service, port) in &self.advertised_endpoint.port_mappings {
            if *port == 0 {
                return Err(format!(
                    "Port mapping for service '{service}' cannot be zero"
                ));
            }
        }

        Ok(())
    }
}
