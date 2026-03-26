//! # Configuration Types
//!
//! Common configuration structures and implementations for all Basilca components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::error::{BasilicaError, ConfigurationError};

/// Default port for the validator's bidding gRPC service.
pub const DEFAULT_BID_GRPC_PORT: u16 = 50052;

// Re-export BittensorConfig from the bittensor crate as the canonical source
pub use bittensor::BittensorConfig;

/// gRPC server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcServerConfig {
    /// Listen address
    pub listen_address: String,

    /// Optional TLS certificate path
    pub tls_cert_path: Option<PathBuf>,

    /// Optional TLS key path
    pub tls_key_path: Option<PathBuf>,

    /// Maximum concurrent connections
    pub max_connections: u32,

    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            listen_address: "127.0.0.1:50051".to_string(),
            tls_cert_path: None,
            tls_key_path: None,
            max_connections: 1000,
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Database configuration shared across all crates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database connection URL
    pub url: String,

    /// Maximum number of connections in the pool
    pub max_connections: u32,

    /// Minimum number of connections in the pool
    pub min_connections: u32,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Idle timeout for connections
    pub idle_timeout: Option<Duration>,

    /// Maximum lifetime for connections
    pub max_lifetime: Option<Duration>,

    /// Whether to run migrations on startup
    pub run_migrations: bool,

    /// SSL/TLS configuration
    pub ssl_config: Option<SslConfig>,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite::memory:".to_string(),
            max_connections: 10,
            min_connections: 1,
            connect_timeout: Duration::from_secs(30),
            idle_timeout: Some(Duration::from_secs(600)),
            max_lifetime: Some(Duration::from_secs(3600)),
            run_migrations: true,
            ssl_config: None,
        }
    }
}

/// SSL/TLS configuration for database connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfig {
    pub ca_cert_path: Option<PathBuf>,
    pub client_cert_path: Option<PathBuf>,
    pub client_key_path: Option<PathBuf>,
    pub verify_hostname: bool,
}

/// Enhanced server configuration with advertised address support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Internal binding host (default: "0.0.0.0")
    pub host: String,

    /// Internal binding port
    pub port: u16,

    /// Optional advertised host (external address for client connections)
    #[serde(default)]
    pub advertised_host: Option<String>,

    /// Optional advertised port (external port for client connections)
    #[serde(default)]
    pub advertised_port: Option<u16>,

    /// Maximum number of concurrent connections
    pub max_connections: u32,

    /// Request timeout
    pub request_timeout: Duration,

    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,

    /// Enable TLS for advertised endpoint
    #[serde(default)]
    pub advertised_tls: bool,

    /// Enable TLS
    pub tls_enabled: bool,

    /// TLS configuration
    pub tls_config: Option<TlsConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            advertised_host: None,
            advertised_port: None,
            max_connections: 1000,
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(60),
            advertised_tls: false,
            tls_enabled: false,
            tls_config: None,
        }
    }
}

impl ServerConfig {
    /// Get the listening address (internal binding)
    pub fn listen_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Get the advertised address (external endpoint for clients)
    pub fn advertised_address(&self) -> String {
        let advertised_host = self.advertised_host.as_ref().unwrap_or(&self.host);
        let advertised_port = self.advertised_port.unwrap_or(self.port);
        format!("{advertised_host}:{advertised_port}")
    }

    /// Get the full advertised URL with protocol
    pub fn advertised_url(&self, default_protocol: &str) -> String {
        let protocol = if self.advertised_tls {
            "https"
        } else {
            default_protocol
        };
        format!("{}://{}", protocol, self.advertised_address())
    }

    /// Check if advertised address differs from listening address
    pub fn has_address_separation(&self) -> bool {
        self.advertised_host.is_some() || self.advertised_port.is_some()
    }

    /// Validate configuration consistency
    pub fn validate_advertised_config(&self) -> Result<(), String> {
        if self.port == 0 {
            return Err("Port cannot be zero".to_string());
        }

        if let Some(advertised_port) = self.advertised_port {
            if advertised_port == 0 {
                return Err("Advertised port cannot be zero".to_string());
            }
        }

        if let Some(ref advertised_host) = self.advertised_host {
            if advertised_host.is_empty() {
                return Err("Advertised host cannot be empty".to_string());
            }
        }

        Ok(())
    }
}

/// TLS configuration for servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
    pub ca_cert_path: Option<PathBuf>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Log format (json, pretty, compact)
    pub format: String,

    /// Whether to log to stdout
    pub stdout: bool,

    /// Optional file to log to
    pub file: Option<PathBuf>,

    /// Maximum log file size before rotation
    pub max_file_size: Option<u64>,

    /// Number of log files to keep
    pub max_files: Option<u32>,

    /// Additional log targets and their levels
    pub targets: HashMap<String, String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            stdout: true,
            file: None,
            max_file_size: Some(100 * 1024 * 1024), // 100MB
            max_files: Some(5),
            targets: HashMap::new(),
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,

    /// Metrics collection interval
    pub collection_interval: Duration,

    /// Prometheus exporter configuration
    pub prometheus: Option<PrometheusConfig>,

    /// Additional metric labels to add to all metrics
    pub default_labels: HashMap<String, String>,

    /// Metric retention period
    pub retention_period: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(30),
            prometheus: Some(PrometheusConfig::default()),
            default_labels: HashMap::new(),
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

/// Prometheus exporter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Host to bind Prometheus exporter to
    pub host: String,

    /// Port for Prometheus exporter
    pub port: u16,

    /// Path for metrics endpoint
    pub path: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9090,
            path: "/metrics".to_string(),
        }
    }
}

/// Common configuration validation trait
pub trait ConfigValidation {
    type Error: BasilicaError;

    /// Validate the configuration
    fn validate(&self) -> Result<(), Self::Error>;

    /// Get configuration warnings (non-fatal issues)
    fn warnings(&self) -> Vec<String> {
        Vec::new()
    }
}

// Configuration validation implementations for common configs
impl ConfigValidation for DatabaseConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        if self.url.is_empty() {
            return Err(ConfigurationError::InvalidValue {
                key: "url".to_string(),
                value: self.url.clone(),
                reason: "Database URL cannot be empty".to_string(),
            });
        }

        if self.max_connections == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "max_connections".to_string(),
                value: self.max_connections.to_string(),
                reason: "Max connections must be greater than 0".to_string(),
            });
        }

        if self.min_connections > self.max_connections {
            return Err(ConfigurationError::InvalidValue {
                key: "min_connections".to_string(),
                value: self.min_connections.to_string(),
                reason: "Min connections cannot be greater than max connections".to_string(),
            });
        }

        Ok(())
    }
}

impl ConfigValidation for ServerConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        // Use the new advertised config validation
        if let Err(msg) = self.validate_advertised_config() {
            return Err(ConfigurationError::InvalidValue {
                key: "server_config".to_string(),
                value: "advertised_address".to_string(),
                reason: msg,
            });
        }

        if self.tls_enabled && self.tls_config.is_none() {
            return Err(ConfigurationError::MissingRequired {
                key: "tls_config".to_string(),
            });
        }

        Ok(())
    }
}

impl ConfigValidation for GrpcServerConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        if self.listen_address.is_empty() {
            return Err(ConfigurationError::InvalidValue {
                key: "listen_address".to_string(),
                value: self.listen_address.clone(),
                reason: "Listen address cannot be empty".to_string(),
            });
        }

        if self.max_connections == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "max_connections".to_string(),
                value: self.max_connections.to_string(),
                reason: "Max connections must be greater than 0".to_string(),
            });
        }

        // Validate TLS config if cert path is provided
        if let Some(_cert_path) = &self.tls_cert_path {
            if self.tls_key_path.is_none() {
                return Err(ConfigurationError::MissingRequired {
                    key: "tls_key_path".to_string(),
                });
            }
        }

        if let Some(_key_path) = &self.tls_key_path {
            if self.tls_cert_path.is_none() {
                return Err(ConfigurationError::MissingRequired {
                    key: "tls_cert_path".to_string(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bittensor_config_defaults() {
        // Verify BittensorConfig is properly re-exported
        let config = BittensorConfig::default();
        assert_eq!(config.wallet_name, "default");
        assert_eq!(config.network, "finney");
    }

    #[test]
    fn test_bittensor_config_endpoint_resolution() {
        // Test finney network
        let finney_config = BittensorConfig {
            network: "finney".to_string(),
            chain_endpoint: None,
            ..Default::default()
        };
        assert_eq!(
            finney_config.get_chain_endpoint(),
            "wss://entrypoint-finney.opentensor.ai:443"
        );

        // Test custom endpoint override
        let custom_config = BittensorConfig {
            network: "finney".to_string(),
            chain_endpoint: Some("wss://custom.endpoint.com:443".to_string()),
            ..Default::default()
        };
        assert_eq!(
            custom_config.get_chain_endpoint(),
            "wss://custom.endpoint.com:443"
        );
    }
}
