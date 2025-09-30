//! Connection pool for managing multiple blockchain connections with automatic failover

use crate::connect::health::{ConnectionPoolTrait, HealthChecker};
use crate::error::{BittensorError, RetryConfig};
use crate::retry::ExponentialBackoff;
use futures::future::join_all;
use std::sync::Arc;
use std::time::Duration;
use subxt::{OnlineClient, PolkadotConfig};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Type alias for chain client
type ChainClient = OnlineClient<PolkadotConfig>;

/// Connection pool managing multiple blockchain connections with automatic failover
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    endpoints: Arc<Vec<String>>,
    connections: Arc<RwLock<Vec<Arc<ChainClient>>>>,
    health_checker: Arc<HealthChecker>,
    #[doc(hidden)]
    pub max_connections: usize,
    #[doc(hidden)]
    pub retry_config: RetryConfig,
}

impl ConnectionPool {
    /// Creates a new connection pool
    ///
    /// # Arguments
    /// * `endpoints` - List of WebSocket endpoints to connect to
    /// * `max_connections` - Maximum number of concurrent connections to maintain
    pub fn new(endpoints: Vec<String>, max_connections: usize) -> Self {
        Self {
            endpoints: Arc::new(endpoints),
            connections: Arc::new(RwLock::new(Vec::new())),
            health_checker: Arc::new(HealthChecker::default()),
            max_connections,
            retry_config: RetryConfig::network(),
        }
    }

    /// Initialize the connection pool with at least one working connection
    pub async fn initialize(&self) -> Result<(), BittensorError> {
        let mut connections = Vec::with_capacity(self.max_connections);
        let endpoints_to_try = self
            .endpoints
            .iter()
            .take(self.max_connections)
            .collect::<Vec<_>>();

        if endpoints_to_try.is_empty() {
            return Err(BittensorError::ConfigError {
                field: "endpoints".to_string(),
                message: "No endpoints configured".to_string(),
            });
        }

        // Try to establish connections in parallel
        let connection_futures = endpoints_to_try
            .iter()
            .map(|endpoint| self.create_connection(endpoint));

        let results = join_all(connection_futures).await;

        for (endpoint, result) in endpoints_to_try.into_iter().zip(results) {
            match result {
                Ok(client) => {
                    info!("Successfully connected to {}", endpoint);
                    connections.push(Arc::new(client));
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", endpoint, e);
                }
            }
        }

        if connections.is_empty() {
            error!("Failed to establish any connections to chain endpoints");
            return Err(BittensorError::NetworkError {
                message: "Failed to establish any connections".to_string(),
            });
        }

        info!(
            "Initialized connection pool with {} connections",
            connections.len()
        );
        *self.connections.write().await = connections;
        Ok(())
    }

    /// Get a healthy client from the pool, reconnecting if necessary
    pub async fn get_healthy_client(&self) -> Result<Arc<ChainClient>, BittensorError> {
        // Fast path: check existing connections
        {
            let connections = self.connections.read().await;
            for conn in connections.iter() {
                if self.health_checker.is_healthy(conn).await {
                    return Ok(Arc::clone(conn));
                }
            }
        }

        // Slow path: all connections unhealthy, trigger reconnection
        warn!("All connections unhealthy, attempting reconnection");
        self.reconnect_with_backoff().await
    }

    /// Reconnect to endpoints with exponential backoff
    pub async fn reconnect_with_backoff(&self) -> Result<Arc<ChainClient>, BittensorError> {
        let mut backoff = ExponentialBackoff::new(self.retry_config.clone());
        let mut last_error = None;

        while let Some(delay) = backoff.next_delay() {
            debug!("Waiting {:?} before reconnection attempt", delay);
            tokio::time::sleep(delay).await;

            match self.try_reconnect().await {
                Ok(client) => {
                    info!("Successfully reconnected to chain");
                    return Ok(client);
                }
                Err(e) => {
                    warn!("Reconnection attempt {} failed: {}", backoff.attempts(), e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| BittensorError::NetworkError {
            message: "Failed to reconnect after maximum attempts".to_string(),
        }))
    }

    /// Attempt to reconnect to any available endpoint
    async fn try_reconnect(&self) -> Result<Arc<ChainClient>, BittensorError> {
        // Try endpoints in order of priority
        for endpoint in self.endpoints.iter() {
            match self.create_connection(endpoint).await {
                Ok(client) => {
                    let client_arc = Arc::new(client);

                    // Update connection pool atomically
                    let mut connections = self.connections.write().await;
                    connections.clear();
                    connections.push(Arc::clone(&client_arc));

                    return Ok(client_arc);
                }
                Err(e) => {
                    debug!("Failed to connect to {}: {}", endpoint, e);
                }
            }
        }

        Err(BittensorError::NetworkError {
            message: "Failed to connect to any endpoint".to_string(),
        })
    }

    /// Create a new connection to the specified endpoint
    async fn create_connection(&self, endpoint: &str) -> Result<ChainClient, BittensorError> {
        let timeout_duration = Duration::from_secs(30);

        tokio::time::timeout(
            timeout_duration,
            OnlineClient::<PolkadotConfig>::from_url(endpoint),
        )
        .await
        .map_err(|_| BittensorError::RpcTimeoutError {
            message: format!("Connection to {} timed out", endpoint),
            timeout: timeout_duration,
        })?
        .map_err(|e| BittensorError::RpcConnectionError {
            message: format!("Failed to connect to {}: {}", endpoint, e),
        })
    }

    /// Get the current number of healthy connections
    pub async fn healthy_connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        let mut count = 0;

        for conn in connections.iter() {
            if self.health_checker.is_healthy(conn).await {
                count += 1;
            }
        }

        count
    }

    /// Force refresh all connections
    pub async fn refresh_connections(&self) -> Result<(), BittensorError> {
        info!("Refreshing all connections");
        self.initialize().await
    }

    /// Get total number of connections (healthy and unhealthy)
    pub async fn total_connections(&self) -> usize {
        self.connections.read().await.len()
    }
}

/// Builder pattern for better ergonomics
pub struct ConnectionPoolBuilder {
    endpoints: Vec<String>,
    max_connections: usize,
    retry_config: Option<RetryConfig>,
    health_checker: Option<HealthChecker>,
}

impl ConnectionPoolBuilder {
    pub fn new(endpoints: Vec<String>) -> Self {
        Self {
            endpoints,
            max_connections: 3,
            retry_config: None,
            health_checker: None,
        }
    }

    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    pub fn health_checker(mut self, checker: HealthChecker) -> Self {
        self.health_checker = Some(checker);
        self
    }

    pub fn build(self) -> ConnectionPool {
        let mut pool = ConnectionPool::new(self.endpoints, self.max_connections);

        if let Some(config) = self.retry_config {
            pool.retry_config = config;
        }

        if let Some(checker) = self.health_checker {
            pool.health_checker = Arc::new(checker);
        }

        pool
    }
}

// Implement the trait for health checking
#[async_trait::async_trait]
impl ConnectionPoolTrait for ConnectionPool {
    async fn connections(&self) -> Arc<RwLock<Vec<Arc<ChainClient>>>> {
        Arc::clone(&self.connections)
    }

    async fn reconnect_with_backoff(&self) -> Result<Arc<ChainClient>, BittensorError> {
        ConnectionPool::reconnect_with_backoff(self).await
    }

    async fn get_healthy_client(&self) -> Result<Arc<ChainClient>, BittensorError> {
        ConnectionPool::get_healthy_client(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn setup_mock_server() -> MockServer {
        MockServer::start().await
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let endpoints = vec!["wss://test.endpoint:443".to_string()];
        let pool = ConnectionPool::new(endpoints.clone(), 3);

        assert_eq!(pool.endpoints.len(), 1);
        assert_eq!(pool.max_connections, 3);
    }

    #[tokio::test]
    async fn test_connection_pool_builder() {
        let endpoints = vec!["wss://test.endpoint:443".to_string()];
        let pool = ConnectionPoolBuilder::new(endpoints.clone())
            .max_connections(5)
            .retry_config(RetryConfig::transient())
            .build();

        assert_eq!(pool.endpoints.len(), 1);
        assert_eq!(pool.max_connections, 5);
    }

    #[tokio::test]
    async fn test_empty_endpoints_initialization() {
        let pool = ConnectionPool::new(vec![], 3);
        let result = pool.initialize().await;

        assert!(result.is_err());
        if let Err(BittensorError::ConfigError { field, .. }) = result {
            assert_eq!(field, "endpoints");
        } else {
            panic!("Expected ConfigError");
        }
    }

    #[tokio::test]
    async fn test_connection_pool_initialization_with_mock() {
        let mock_server = setup_mock_server().await;

        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        // Note: This test would need actual WebSocket mocking which is complex
        // For real testing, we'd need to mock the subxt client properly
        let endpoints = vec![format!("ws://{}", mock_server.address())];
        let pool = ConnectionPool::new(endpoints, 1);

        // This will fail as we can't easily mock WebSocket connections
        // In production, you'd use integration tests or more sophisticated mocking
        let result = pool.initialize().await;
        assert!(result.is_err()); // Expected as we can't mock WS properly
    }

    #[tokio::test]
    async fn test_healthy_connection_count() {
        let pool = ConnectionPool::new(vec!["wss://test.endpoint:443".to_string()], 3);
        let count = pool.healthy_connection_count().await;
        assert_eq!(count, 0); // No connections established yet
    }

    #[tokio::test]
    async fn test_total_connections() {
        let pool = ConnectionPool::new(vec!["wss://test.endpoint:443".to_string()], 3);
        let count = pool.total_connections().await;
        assert_eq!(count, 0); // No connections established yet
    }

    #[tokio::test]
    async fn test_get_healthy_client_no_connections() {
        let pool = ConnectionPool::new(vec!["wss://invalid.endpoint:443".to_string()], 1);
        let result = pool.get_healthy_client().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reconnect_with_backoff() {
        let pool = ConnectionPool::new(vec!["wss://invalid.endpoint:443".to_string()], 1);

        // Override retry config to make test faster
        let mut pool = pool;
        pool.retry_config = RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(20),
            backoff_multiplier: 1.5,
            jitter: false,
        };

        let result = pool.reconnect_with_backoff().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_multiple_endpoints_fallback() {
        let endpoints = vec![
            "wss://invalid1.endpoint:443".to_string(),
            "wss://invalid2.endpoint:443".to_string(),
            "wss://invalid3.endpoint:443".to_string(),
        ];

        let pool = ConnectionPool::new(endpoints, 3);
        let result = pool.try_reconnect().await;
        assert!(result.is_err()); // All endpoints are invalid
    }

    #[tokio::test]
    async fn test_create_connection_timeout() {
        let pool = ConnectionPool::new(vec!["wss://10.255.255.1:443".to_string()], 1);

        // This IP should not be routable, causing a timeout
        let result = pool.create_connection("wss://10.255.255.1:443").await;
        assert!(result.is_err());

        if let Err(BittensorError::RpcTimeoutError { .. }) = result {
            // Expected
        } else {
            panic!("Expected RpcTimeoutError");
        }
    }
}
