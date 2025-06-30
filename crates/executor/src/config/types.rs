//! Core configuration types and main executor configuration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use common::config::{loader, LoggingConfig, MetricsConfig, ServerConfig};
use common::identity::Hotkey;
use std::str::FromStr;

use super::{DockerConfig, SystemConfig};
use crate::validation_session::ValidatorConfig;

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
}
