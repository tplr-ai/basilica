//! Configuration utilities for integration tests
//!
//! This module provides centralized configuration management for test scenarios,
//! allowing tests to be run against different environments (local, staging, production).

use std::env;

/// Configuration for service endpoints
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub payments_endpoint: String,
    pub billing_endpoint: String,
    pub validator_endpoint: String,
    pub miner_endpoint: String,
    pub node_endpoint: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            payments_endpoint: "http://localhost:50061".to_string(),
            billing_endpoint: "http://localhost:50051".to_string(),
            validator_endpoint: "http://localhost:50052".to_string(),
            miner_endpoint: "http://localhost:50053".to_string(),
            node_endpoint: "http://localhost:50054".to_string(),
        }
    }
}

impl TestConfig {
    /// Create configuration from environment variables with fallback to defaults
    pub fn from_env() -> Self {
        Self {
            payments_endpoint: env::var("PAYMENTS_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:50061".to_string()),
            billing_endpoint: env::var("BILLING_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:50051".to_string()),
            validator_endpoint: env::var("VALIDATOR_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:50052".to_string()),
            miner_endpoint: env::var("MINER_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:50053".to_string()),
            node_endpoint: env::var("NODE_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:50054".to_string()),
        }
    }

    /// Check if services are available by attempting to connect
    pub async fn check_service_availability(&self) -> ServiceAvailability {
        use tokio::net::TcpStream;
        use tokio::time::{timeout, Duration};

        let check_endpoint = |endpoint: String| async move {
            if let Ok(url) = endpoint.parse::<url::Url>() {
                if let Some(host) = url.host_str() {
                    let port = url.port().unwrap_or(match url.scheme() {
                        "https" => 443,
                        "http" => 80,
                        _ => return false,
                    });

                    timeout(Duration::from_secs(2), TcpStream::connect((host, port)))
                        .await
                        .is_ok()
                } else {
                    false
                }
            } else {
                false
            }
        };

        ServiceAvailability {
            payments: check_endpoint(self.payments_endpoint.clone()).await,
            billing: check_endpoint(self.billing_endpoint.clone()).await,
            validator: check_endpoint(self.validator_endpoint.clone()).await,
            miner: check_endpoint(self.miner_endpoint.clone()).await,
            node: check_endpoint(self.node_endpoint.clone()).await,
        }
    }
}

/// Service availability status
#[derive(Debug)]
pub struct ServiceAvailability {
    pub payments: bool,
    pub billing: bool,
    pub validator: bool,
    pub miner: bool,
    pub node: bool,
}

impl ServiceAvailability {
    pub fn payments_and_billing_available(&self) -> bool {
        self.payments && self.billing
    }

    pub fn all_available(&self) -> bool {
        self.payments && self.billing && self.validator && self.miner && self.node
    }

    pub fn report(&self) {
        println!("Service Availability Report:");
        println!("  Payments: {}", if self.payments { "✓" } else { "✗" });
        println!("  Billing: {}", if self.billing { "✓" } else { "✗" });
        println!("  Validator: {}", if self.validator { "✓" } else { "✗" });
        println!("  Miner: {}", if self.miner { "✓" } else { "✗" });
        println!("  Node: {}", if self.node { "✓" } else { "✗" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = TestConfig::default();
        assert_eq!(config.payments_endpoint, "http://localhost:50061");
        assert_eq!(config.billing_endpoint, "http://localhost:50051");
    }

    #[test]
    fn test_config_from_env() {
        env::set_var("PAYMENTS_ENDPOINT", "http://test:8080");
        env::set_var("BILLING_ENDPOINT", "http://test:8081");

        let config = TestConfig::from_env();
        assert_eq!(config.payments_endpoint, "http://test:8080");
        assert_eq!(config.billing_endpoint, "http://test:8081");

        env::remove_var("PAYMENTS_ENDPOINT");
        env::remove_var("BILLING_ENDPOINT");
    }
}
