//! Connection subsystem: pooling, health checks, connection state, retries, and monitors.
//!
//! This module groups all connection-related primitives behind a cohesive API while
//! re-exporting items to keep the public surface stable.

pub mod health;
pub mod monitor;
pub mod pool;
pub mod state;

// Re-export core types from submodules
pub use crate::error::RetryConfig;
pub use crate::retry::{CircuitBreaker, ExponentialBackoff, RetryExecutor};
pub use health::{ConnectionPoolTrait, HealthCheckMetrics, HealthChecker};
pub use monitor::{BlockchainEventHandler, BlockchainMonitor};
pub use pool::{ConnectionPool, ConnectionPoolBuilder};
pub use state::{ConnectionManager, ConnectionMetricsSnapshot, ConnectionState};

/// Common imports for connection-related code
pub mod prelude {
    pub use super::{
        BlockchainMonitor, CircuitBreaker, ConnectionManager, ConnectionMetricsSnapshot,
        ConnectionPool, ConnectionPoolBuilder, ConnectionPoolTrait, ConnectionState,
        ExponentialBackoff, HealthCheckMetrics, HealthChecker, RetryConfig, RetryExecutor,
    };
}
