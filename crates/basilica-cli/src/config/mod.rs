//! Configuration management for the Basilica CLI

use basilica_common::config::loader;
use color_eyre::eyre::{eyre, WrapErr};
use etcetera::{choose_base_strategy, BaseStrategy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::CliError;

/// CLI configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CliConfig {
    /// API configuration
    pub api: ApiConfig,

    /// SSH configuration
    pub ssh: SshConfig,

    /// Default image configuration
    pub image: ImageConfig,

    /// Wallet configuration
    pub wallet: WalletConfig,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Base URL for the Basilica API
    pub base_url: String,

    /// Request timeout in seconds
    #[serde(default = "default_api_request_timeout")]
    pub request_timeout: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.basilica.ai".to_string(),
            request_timeout: 900,
        }
    }
}

/// SSH configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshConfig {
    /// Default SSH public key path
    pub key_path: PathBuf,
    /// SSH private key path
    pub private_key_path: PathBuf,
    /// SSH connection timeout in seconds (default: 30)
    #[serde(default = "default_ssh_timeout")]
    pub connection_timeout: u64,
}

fn default_ssh_timeout() -> u64 {
    30
}

fn default_api_request_timeout() -> u64 {
    120
}

impl Default for SshConfig {
    fn default() -> Self {
        Self {
            key_path: PathBuf::from("~/.ssh/basilica_ed25519.pub"),
            private_key_path: PathBuf::from("~/.ssh/basilica_ed25519"),
            connection_timeout: 30,
        }
    }
}

impl SshConfig {
    /// Check if SSH keys exist at the configured paths
    pub fn ssh_keys_exist(&self) -> bool {
        self.key_path.exists() && self.private_key_path.exists()
    }

    /// Check if SSH keys are missing (neither exists)
    pub fn ssh_keys_missing(&self) -> bool {
        !self.key_path.exists() && !self.private_key_path.exists()
    }

    /// Check if SSH key pair is incomplete (only one exists)
    pub fn ssh_keys_incomplete(&self) -> bool {
        self.key_path.exists() != self.private_key_path.exists()
    }
}

/// Docker image configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Default Docker image name
    pub name: String,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            name: "nvidia/cuda:12.8.0-runtime-ubuntu22.04".to_string(),
        }
    }
}

/// Wallet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletConfig {
    /// Default wallet name
    pub default_wallet: String,

    /// Base wallet directory path (wallets are located at base_wallet_path/{wallet_name})
    pub base_wallet_path: PathBuf,
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            default_wallet: "default".to_string(),
            base_wallet_path: PathBuf::from("~/.bittensor/wallets"),
        }
    }
}

/// Create auth configuration for OAuth flows with specific port
/// This bridges the gap between constants and the auth module's requirements
/// The port must be one of the registered callback ports
pub fn create_auth_config_with_port(port: u16) -> crate::auth::types::AuthConfig {
    // Validate that the port is in the allowed list (for OAuth flow)
    // Device flow uses port 0 and doesn't need validation
    if port != 0 {
        let allowed_ports = basilica_common::auth0_callback_ports();
        if !allowed_ports.contains(&port) {
            tracing::warn!(
                "Port {} is not in the list of registered callback ports: {:?}",
                port,
                allowed_ports
            );
        }
    }

    // Use constants from basilica-common
    let domain = basilica_common::auth0_domain();

    crate::auth::types::AuthConfig {
        client_id: basilica_common::auth0_client_id().to_string(),
        auth_endpoint: format!("https://{}/authorize", domain),
        token_endpoint: format!("https://{}/oauth/token", domain),
        device_auth_endpoint: Some(format!("https://{}/oauth/device/code", domain)),
        revoke_endpoint: Some(format!("https://{}/oauth/revoke", domain)),
        redirect_uri: format!("http://localhost:{}/auth/callback", port),
        scopes: vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "offline_access".to_string(), // Required for refresh tokens
            "rentals:*".to_string(),      // All rental operations
            "executors:list".to_string(), // List available executors
            "keys:create".to_string(),    // Create API keys
            "keys:list".to_string(),      // List API keys
            "keys:revoke".to_string(),    // Revoke API keys
        ],
        additional_params: std::collections::HashMap::new(),
    }
}

/// Cache data structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CliCache {
    /// Registration information
    pub registration: Option<RegistrationCache>,
}

/// Registration cache data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationCache {
    /// Hotwallet address for credits
    pub hotwallet: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl CliConfig {
    /// Load configuration using the common loader pattern
    pub fn load() -> Result<Self, CliError> {
        let mut config = loader::load_config::<Self>().wrap_err("Failed to load config")?;
        config.expand_paths();
        Ok(config)
    }

    /// Load configuration from specific file using the common loader pattern
    pub fn load_from_file(path: &Path) -> Result<Self, CliError> {
        let mut config =
            loader::load_from_file::<Self>(path).wrap_err("Failed to load config from file")?;
        config.expand_paths();
        Ok(config)
    }

    /// Expand tilde (~) in path fields
    fn expand_paths(&mut self) {
        if let Some(path_str) = self.ssh.key_path.to_str() {
            let expanded = shellexpand::tilde(path_str);
            self.ssh.key_path = PathBuf::from(expanded.as_ref());
        }
        if let Some(path_str) = self.ssh.private_key_path.to_str() {
            let expanded = shellexpand::tilde(path_str);
            self.ssh.private_key_path = PathBuf::from(expanded.as_ref());
        }
        if let Some(path_str) = self.wallet.base_wallet_path.to_str() {
            let expanded = shellexpand::tilde(path_str);
            self.wallet.base_wallet_path = PathBuf::from(expanded.as_ref());
        }
    }

    /// Compress paths by replacing home directory with tilde for serialization
    fn compress_paths(&self) -> Self {
        let home_dir = if let Ok(strategy) = choose_base_strategy() {
            strategy.home_dir().to_path_buf()
        } else {
            return self.clone(); // If we can't get home dir, return as-is
        };

        let mut config = self.clone();

        // Compress SSH paths
        config.ssh.key_path = Self::compress_path(&config.ssh.key_path, &home_dir);
        config.ssh.private_key_path = Self::compress_path(&config.ssh.private_key_path, &home_dir);

        // Compress wallet path
        config.wallet.base_wallet_path =
            Self::compress_path(&config.wallet.base_wallet_path, &home_dir);

        config
    }

    /// Helper function to compress a single path
    fn compress_path(path: &Path, home_dir: &std::path::PathBuf) -> PathBuf {
        if let Ok(relative_path) = path.strip_prefix(home_dir) {
            // Path is under home directory, replace with tilde
            let mut tilde_path = std::path::PathBuf::from("~");
            tilde_path.push(relative_path);
            tilde_path
        } else {
            // Path is not under home directory, keep as-is
            path.to_path_buf()
        }
    }

    /// Save configuration to specific path
    pub async fn save_to_path(&self, path: &Path) -> Result<(), CliError> {
        debug!("Saving configuration to: {}", path.display());

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| eyre!("Failed to create directory: {}", e))?;
        }

        // Compress paths to use tilde notation before serialization
        let compressed_config = self.compress_paths();

        let content =
            toml::to_string_pretty(&compressed_config).wrap_err("Failed to serialize config")?;

        tokio::fs::write(path, content)
            .await
            .map_err(|e| eyre!("Failed to write config file: {}", e))?;

        info!("Configuration saved successfully");
        Ok(())
    }

    /// Get all configuration as key-value pairs
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Get home directory for path compression
        let home_dir = if let Ok(strategy) = choose_base_strategy() {
            Some(strategy.home_dir().to_path_buf())
        } else {
            None
        };

        map.insert("api.base_url".to_string(), self.api.base_url.clone());

        // Compress SSH key paths
        let ssh_key_path = if let Some(ref home) = home_dir {
            Self::compress_path(&self.ssh.key_path, home)
        } else {
            self.ssh.key_path.clone()
        };
        map.insert(
            "ssh.key_path".to_string(),
            ssh_key_path.to_string_lossy().to_string(),
        );

        let ssh_private_key_path = if let Some(ref home) = home_dir {
            Self::compress_path(&self.ssh.private_key_path, home)
        } else {
            self.ssh.private_key_path.clone()
        };
        map.insert(
            "ssh.private_key_path".to_string(),
            ssh_private_key_path.to_string_lossy().to_string(),
        );

        map.insert(
            "ssh.connection_timeout".to_string(),
            self.ssh.connection_timeout.to_string(),
        );
        map.insert("image.name".to_string(), self.image.name.clone());
        map.insert(
            "wallet.default_wallet".to_string(),
            self.wallet.default_wallet.clone(),
        );

        // Compress wallet base path
        let wallet_base_path = if let Some(ref home) = home_dir {
            Self::compress_path(&self.wallet.base_wallet_path, home)
        } else {
            self.wallet.base_wallet_path.clone()
        };
        map.insert(
            "wallet.base_wallet_path".to_string(),
            wallet_base_path.to_string_lossy().to_string(),
        );

        map
    }

    /// Get configuration directory
    pub fn config_dir() -> Result<PathBuf, CliError> {
        let strategy = choose_base_strategy().map_err(|e| -> crate::error::CliError {
            eyre!("Failed to determine base directories: {}", e).into()
        })?;
        Ok(strategy.config_dir().join("basilica"))
    }

    /// Get data directory
    pub fn data_dir() -> Result<PathBuf, CliError> {
        let strategy = choose_base_strategy().map_err(|e| -> crate::error::CliError {
            eyre!("Failed to determine base directories: {}", e).into()
        })?;
        Ok(strategy.data_dir().join("basilica"))
    }

    /// Get cache file path
    pub fn cache_path() -> Result<PathBuf, CliError> {
        let config_dir = Self::config_dir()?;
        Ok(config_dir.join("cache.json"))
    }

    /// Get default config file path
    pub fn default_config_path() -> Result<PathBuf, CliError> {
        let config_dir = Self::config_dir()?;
        Ok(config_dir.join("config.toml"))
    }

    /// Check if config file exists at default location
    pub fn config_exists() -> Result<bool, CliError> {
        let path = Self::default_config_path()?;
        Ok(path.exists())
    }
}

impl CliCache {
    /// Load cache from default location
    pub async fn load() -> Result<Self, CliError> {
        let cache_path = CliConfig::cache_path()?;
        Self::load_from_file(&cache_path).await
    }

    /// Load cache from specific path
    pub async fn load_from_file(path: &Path) -> Result<Self, CliError> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| eyre!("Failed to read cache file: {}", e))?;

        let cache: Self =
            serde_json::from_str(&content).map_err(|e| eyre!("Failed to parse cache: {}", e))?;

        Ok(cache)
    }

    /// Save cache to default location
    pub async fn save(&self) -> Result<(), CliError> {
        let cache_path = CliConfig::cache_path()?;
        self.save_to_path(&cache_path).await
    }

    /// Save cache to specific path
    pub async fn save_to_path(&self, path: &Path) -> Result<(), CliError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| eyre!("Failed to create directory: {}", e))?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| eyre!("Failed to serialize cache: {}", e))?;

        tokio::fs::write(path, content)
            .await
            .map_err(|e| eyre!("Failed to write cache file: {}", e))?;

        Ok(())
    }
}
