//! TUI configuration management

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// TUI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiConfig {
    /// API base URL
    #[serde(default = "default_api_url")]
    pub api_url: String,

    /// Theme preference
    #[serde(default)]
    pub theme: ThemePreference,

    /// Refresh intervals in seconds
    #[serde(default)]
    pub refresh: RefreshConfig,

    /// Miner-specific configuration
    #[serde(default)]
    pub miner: MinerTuiConfig,
}

fn default_api_url() -> String {
    "https://api.basilica.ai".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemePreference {
    #[default]
    Dark,
    Light,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshConfig {
    /// Balance refresh interval in seconds
    #[serde(default = "default_balance_interval")]
    pub balance: u64,

    /// Rentals list refresh interval in seconds
    #[serde(default = "default_rentals_interval")]
    pub rentals: u64,

    /// Metrics refresh interval in seconds
    #[serde(default = "default_metrics_interval")]
    pub metrics: u64,
}

fn default_balance_interval() -> u64 {
    30
}

fn default_rentals_interval() -> u64 {
    10
}

fn default_metrics_interval() -> u64 {
    5
}

impl Default for RefreshConfig {
    fn default() -> Self {
        Self {
            balance: default_balance_interval(),
            rentals: default_rentals_interval(),
            metrics: default_metrics_interval(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinerTuiConfig {
    /// Path to miner configuration file
    pub config_path: Option<String>,

    /// Miner metrics endpoint
    pub metrics_url: Option<String>,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            api_url: default_api_url(),
            theme: ThemePreference::default(),
            refresh: RefreshConfig::default(),
            miner: MinerTuiConfig::default(),
        }
    }
}

impl TuiConfig {
    /// Load configuration from file or defaults
    pub fn load(path: Option<&Path>) -> Result<Self> {
        if let Some(path) = path {
            if path.exists() {
                let content = std::fs::read_to_string(path)?;
                let config: TuiConfig = toml::from_str(&content)?;
                return Ok(config);
            }
        }

        // Try default config location
        if let Some(config_dir) = directories::ProjectDirs::from("ai", "basilica", "basilica-tui") {
            let config_path = config_dir.config_dir().join("config.toml");
            if config_path.exists() {
                let content = std::fs::read_to_string(&config_path)?;
                let config: TuiConfig = toml::from_str(&content)?;
                return Ok(config);
            }
        }

        Ok(Self::default())
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Save to default config location
    pub fn save_default(&self) -> Result<()> {
        if let Some(config_dir) = directories::ProjectDirs::from("ai", "basilica", "basilica-tui") {
            let config_path = config_dir.config_dir().join("config.toml");
            self.save(&config_path)?;
        }
        Ok(())
    }

    /// Get default config path
    pub fn default_path() -> Option<std::path::PathBuf> {
        directories::ProjectDirs::from("ai", "basilica", "basilica-tui")
            .map(|dirs| dirs.config_dir().join("config.toml"))
    }
}

