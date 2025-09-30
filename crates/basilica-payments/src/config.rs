use anyhow::Result;
use basilica_common::error::ConfigurationError;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentsConfig {
    pub service: ServiceConfig,
    pub database: DatabaseConfig,
    pub grpc: GrpcConfig,
    pub http: HttpConfig,
    pub blockchain: BlockchainConfig,
    pub treasury: TreasuryConfig,
    pub price_oracle: PriceOracleConfig,
    pub billing: BillingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub name: String,
    pub environment: String,
    pub log_level: String,
    pub metrics_enabled: bool,
    pub service_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout_seconds: u64,
    pub acquire_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub max_lifetime_seconds: u64,
    pub enable_ssl: bool,
    pub ssl_ca_cert_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    pub listen_address: String,
    pub port: u16,
    pub max_message_size: usize,
    pub keepalive_interval_seconds: Option<u64>,
    pub keepalive_timeout_seconds: Option<u64>,
    pub tls_enabled: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
    pub max_concurrent_requests: Option<usize>,
    pub max_concurrent_streams: Option<u32>,
    pub request_timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    pub listen_address: String,
    pub port: u16,
    pub cors_enabled: bool,
    pub cors_allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainConfig {
    pub websocket_url: String,
    /// Optional fallback websocket endpoints for failover
    #[serde(default)]
    pub fallback_websocket_urls: Vec<String>,
    pub ss58_prefix: u16,
    pub connection_timeout_seconds: u64,
    pub retry_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreasuryConfig {
    pub aead_key_hex: String,
    pub tao_decimals: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceOracleConfig {
    pub update_interval_seconds: u64,
    pub max_price_age_seconds: u64,
    pub request_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    pub grpc_endpoint: String,
    pub connection_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
}

impl Default for PaymentsConfig {
    fn default() -> Self {
        Self {
            service: ServiceConfig {
                name: "basilica-payments".to_string(),
                environment: "development".to_string(),
                log_level: "info".to_string(),
                metrics_enabled: true,
                service_id: Uuid::new_v4().to_string(),
            },
            database: DatabaseConfig {
                url: "postgres://payments@localhost:5432/basilica_payments".to_string(),
                max_connections: 32,
                min_connections: 5,
                connect_timeout_seconds: 30,
                acquire_timeout_seconds: 30,
                idle_timeout_seconds: 600,
                max_lifetime_seconds: 1800,
                enable_ssl: false,
                ssl_ca_cert_path: None,
            },
            grpc: GrpcConfig {
                listen_address: "0.0.0.0".to_string(),
                port: 50061,
                max_message_size: 4 * 1024 * 1024, // 4MB
                keepalive_interval_seconds: Some(300),
                keepalive_timeout_seconds: Some(20),
                tls_enabled: false,
                tls_cert_path: None,
                tls_key_path: None,
                max_concurrent_requests: Some(1000),
                max_concurrent_streams: Some(100),
                request_timeout_seconds: Some(60),
            },
            http: HttpConfig {
                listen_address: "0.0.0.0".to_string(),
                port: 8082,
                cors_enabled: true,
                cors_allowed_origins: vec!["*".to_string()],
            },
            blockchain: BlockchainConfig {
                websocket_url: "ws://localhost:9944".to_string(),
                fallback_websocket_urls: Vec::new(),
                ss58_prefix: 42,
                connection_timeout_seconds: 30,
                retry_interval_seconds: 5,
            },
            treasury: TreasuryConfig {
                aead_key_hex: "0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                tao_decimals: 9,
            },
            price_oracle: PriceOracleConfig {
                update_interval_seconds: 60,
                max_price_age_seconds: 300,
                request_timeout_seconds: 10,
            },
            billing: BillingConfig {
                grpc_endpoint: "http://localhost:50051".to_string(),
                connection_timeout_seconds: 30,
                request_timeout_seconds: 60,
            },
        }
    }
}

impl PaymentsConfig {
    pub fn load(path_override: Option<PathBuf>) -> Result<PaymentsConfig, ConfigurationError> {
        let default_config = PaymentsConfig::default();

        let mut figment = Figment::from(Serialized::defaults(default_config));

        if let Some(path) = path_override {
            if path.exists() {
                figment = figment.merge(Toml::file(&path));
            }
        } else {
            let default_path = PathBuf::from("payments.toml");
            if default_path.exists() {
                figment = figment.merge(Toml::file(default_path));
            }
        }

        figment = figment.merge(Env::prefixed("PAYMENTS_").split("__"));

        figment
            .extract()
            .map_err(|e| ConfigurationError::ParseError {
                details: e.to_string(),
            })
    }

    pub fn load_from_file(path: &Path) -> Result<PaymentsConfig, ConfigurationError> {
        Self::load(Some(path.to_path_buf()))
    }

    pub fn apply_env_overrides(
        config: &mut PaymentsConfig,
        prefix: &str,
    ) -> Result<(), ConfigurationError> {
        let figment = Figment::from(Serialized::defaults(config.clone()))
            .merge(Env::prefixed(prefix).split("__"));

        *config = figment
            .extract()
            .map_err(|e| ConfigurationError::ParseError {
                details: e.to_string(),
            })?;

        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if self.database.url.is_empty() {
            return Err(ConfigurationError::InvalidValue {
                key: "database.url".to_string(),
                value: String::new(),
                reason: "Database URL cannot be empty".to_string(),
            });
        }

        if self.database.max_connections < self.database.min_connections {
            return Err(ConfigurationError::ValidationFailed {
                details: format!(
                    "database.max_connections ({}) must be >= min_connections ({})",
                    self.database.max_connections, self.database.min_connections
                ),
            });
        }

        if self.grpc.port == 0 {
            return Err(ConfigurationError::ValidationFailed {
                details: "grpc.port must be non-zero".to_string(),
            });
        }

        if self.grpc.tls_enabled
            && (self.grpc.tls_cert_path.is_none() || self.grpc.tls_key_path.is_none())
        {
            return Err(ConfigurationError::ValidationFailed {
                details: "TLS cert and key paths required when TLS is enabled".to_string(),
            });
        }

        if self.treasury.aead_key_hex.is_empty() {
            return Err(ConfigurationError::ValidationFailed {
                details: "treasury.aead_key_hex must not be empty".to_string(),
            });
        }

        if self.blockchain.websocket_url.is_empty() {
            return Err(ConfigurationError::ValidationFailed {
                details: "blockchain.websocket_url must not be empty".to_string(),
            });
        }

        Ok(())
    }

    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.database.enable_ssl && self.service.environment == "production" {
            warnings.push("Database SSL is disabled in production environment".to_string());
        }

        if !self.grpc.tls_enabled && self.service.environment == "production" {
            warnings.push("gRPC TLS is disabled in production environment".to_string());
        }

        if self.treasury.aead_key_hex
            == "0000000000000000000000000000000000000000000000000000000000000000"
        {
            warnings
                .push("Using default AEAD key - generate a secure key for production".to_string());
        }

        warnings
    }

    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.database.connect_timeout_seconds)
    }

    pub fn acquire_timeout(&self) -> Duration {
        Duration::from_secs(self.database.acquire_timeout_seconds)
    }

    pub fn idle_timeout(&self) -> Duration {
        Duration::from_secs(self.database.idle_timeout_seconds)
    }

    pub fn max_lifetime(&self) -> Duration {
        Duration::from_secs(self.database.max_lifetime_seconds)
    }

    pub fn blockchain_connection_timeout(&self) -> Duration {
        Duration::from_secs(self.blockchain.connection_timeout_seconds)
    }

    pub fn blockchain_retry_interval(&self) -> Duration {
        Duration::from_secs(self.blockchain.retry_interval_seconds)
    }

    pub fn price_oracle_update_interval(&self) -> Duration {
        Duration::from_secs(self.price_oracle.update_interval_seconds)
    }

    pub fn price_oracle_max_age(&self) -> Duration {
        Duration::from_secs(self.price_oracle.max_price_age_seconds)
    }

    pub fn price_oracle_request_timeout(&self) -> Duration {
        Duration::from_secs(self.price_oracle.request_timeout_seconds)
    }

    pub fn billing_connection_timeout(&self) -> Duration {
        Duration::from_secs(self.billing.connection_timeout_seconds)
    }

    pub fn billing_request_timeout(&self) -> Duration {
        Duration::from_secs(self.billing.request_timeout_seconds)
    }
}
