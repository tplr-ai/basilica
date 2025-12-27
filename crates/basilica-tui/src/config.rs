//! TUI configuration management
#![allow(dead_code)]

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
    // Check environment variable first
    std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "https://api.basilica.ai".to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        // Clear env var for this test to ensure default is used
        std::env::remove_var("BASILICA_API_URL");
        let config = TuiConfig::default();
        assert_eq!(config.api_url, "https://api.basilica.ai");
        assert!(matches!(config.theme, ThemePreference::Dark));
        assert_eq!(config.refresh.balance, 30);
        assert_eq!(config.refresh.rentals, 10);
        assert_eq!(config.refresh.metrics, 5);
    }

    #[test]
    fn test_config_serialization() {
        let config = TuiConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();

        assert!(toml_str.contains("api_url"));
        assert!(toml_str.contains("basilica.ai"));

        // Deserialize back
        let parsed: TuiConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.api_url, config.api_url);
    }

    #[test]
    fn test_config_load_from_file() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(
            temp,
            r#"
api_url = "http://localhost:8080"
theme = "light"

[refresh]
balance = 60
rentals = 30
metrics = 10
"#
        )
        .unwrap();

        let config = TuiConfig::load(Some(temp.path())).unwrap();
        assert_eq!(config.api_url, "http://localhost:8080");
        assert!(matches!(config.theme, ThemePreference::Light));
        assert_eq!(config.refresh.balance, 60);
        assert_eq!(config.refresh.rentals, 30);
    }

    #[test]
    fn test_config_load_nonexistent_returns_default() {
        std::env::remove_var("BASILICA_API_URL");
        let config = TuiConfig::load(Some(Path::new("/nonexistent/path/config.toml"))).unwrap();
        assert_eq!(config.api_url, "https://api.basilica.ai");
    }

    #[test]
    fn test_config_save_and_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let original = TuiConfig {
            api_url: "http://test.local".to_string(),
            theme: ThemePreference::System,
            refresh: RefreshConfig {
                balance: 120,
                rentals: 60,
                metrics: 15,
            },
            miner: MinerTuiConfig {
                config_path: Some("/etc/miner.toml".to_string()),
                metrics_url: Some("http://localhost:9090".to_string()),
            },
        };

        original.save(&config_path).unwrap();
        let loaded = TuiConfig::load(Some(&config_path)).unwrap();

        assert_eq!(loaded.api_url, original.api_url);
        assert_eq!(loaded.refresh.balance, 120);
        assert_eq!(
            loaded.miner.config_path,
            Some("/etc/miner.toml".to_string())
        );
    }

    #[test]
    fn test_theme_preference_variants() {
        // Test via full config deserialization
        let dark_config: TuiConfig = toml::from_str("theme = \"dark\"").unwrap();
        let light_config: TuiConfig = toml::from_str("theme = \"light\"").unwrap();
        let system_config: TuiConfig = toml::from_str("theme = \"system\"").unwrap();

        assert!(matches!(dark_config.theme, ThemePreference::Dark));
        assert!(matches!(light_config.theme, ThemePreference::Light));
        assert!(matches!(system_config.theme, ThemePreference::System));
    }

    #[test]
    fn test_refresh_config_defaults() {
        let refresh = RefreshConfig::default();
        assert_eq!(refresh.balance, 30);
        assert_eq!(refresh.rentals, 10);
        assert_eq!(refresh.metrics, 5);
    }
}
