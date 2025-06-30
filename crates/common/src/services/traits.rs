//! # Service Traits
//!
//! Core traits for service lifecycle management and dependency injection.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use crate::error::BasilcaError;

/// Core service lifecycle management
#[async_trait]
pub trait Service: Send + Sync {
    type Config: Send + Sync;
    type Error: BasilcaError;

    /// Start the service with given configuration
    async fn start(&mut self, config: Self::Config) -> Result<(), Self::Error>;

    /// Stop the service gracefully
    async fn stop(&mut self) -> Result<(), Self::Error>;

    /// Check if service is currently running
    fn is_running(&self) -> bool;

    /// Get service status information
    async fn status(&self) -> ServiceStatus;

    /// Perform health check
    async fn health_check(&self) -> Result<HealthStatus, Self::Error>;

    /// Get service name for identification
    fn name(&self) -> &str;
}

/// Service dependency injection container
pub trait ServiceContainer {
    type Error: BasilcaError;

    /// Register a service in the container
    fn register<T: Service + 'static>(&mut self, name: &str, service: T)
        -> Result<(), Self::Error>;

    /// Get a service by name
    fn get<T: Service + 'static>(&self, name: &str) -> Option<&T>;

    /// Get a mutable service by name
    fn get_mut<T: Service + 'static>(&mut self, name: &str) -> Option<&mut T>;

    /// Start all registered services
    #[allow(async_fn_in_trait)]
    async fn start_all(&mut self) -> Result<(), Self::Error>;

    /// Stop all registered services
    #[allow(async_fn_in_trait)]
    async fn stop_all(&mut self) -> Result<(), Self::Error>;

    /// Get status of all services
    #[allow(async_fn_in_trait)]
    async fn status_all(&self) -> HashMap<String, ServiceStatus>;
}

/// Service health monitoring
#[async_trait]
pub trait HealthMonitor: Send + Sync {
    type Error: BasilcaError;

    /// Start monitoring a service
    async fn monitor_service<T: Service>(&mut self, service: &T) -> Result<(), Self::Error>;

    /// Stop monitoring a service
    async fn stop_monitoring(&mut self, service_name: &str) -> Result<(), Self::Error>;

    /// Get health status for all monitored services
    async fn get_health_status(&self) -> Result<HashMap<String, HealthStatus>, Self::Error>;

    /// Set health check interval
    fn set_check_interval(&mut self, interval: Duration);
}

/// Graceful shutdown coordination
#[async_trait]
pub trait GracefulShutdown: Send + Sync {
    /// Register a shutdown hook
    fn register_shutdown_hook(&mut self, name: String, hook: Box<dyn ShutdownHook>);

    /// Initiate graceful shutdown
    async fn shutdown(&mut self, timeout: Duration) -> Result<(), anyhow::Error>;

    /// Check if shutdown is in progress
    fn is_shutting_down(&self) -> bool;
}

/// Shutdown hook for cleanup operations
#[async_trait]
pub trait ShutdownHook: Send + Sync {
    /// Execute cleanup operations
    async fn execute(&self) -> Result<(), anyhow::Error>;

    /// Get hook priority (lower numbers execute first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Service status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub state: ServiceState,
    pub started_at: Option<SystemTime>,
    pub uptime: Option<Duration>,
    pub restart_count: u32,
    pub last_error: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Service state enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ServiceState {
    /// Service is not started
    Stopped,
    /// Service is starting up
    Starting,
    /// Service is running normally
    Running,
    /// Service is stopping
    Stopping,
    /// Service encountered an error
    Error,
    /// Service is in maintenance mode
    Maintenance,
}

/// Health status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub service_name: String,
    pub status: HealthState,
    pub checked_at: SystemTime,
    pub response_time: Duration,
    pub message: Option<String>,
    pub details: HashMap<String, String>,
}

/// Health state enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthState {
    /// Service is healthy
    Healthy,
    /// Service is degraded but functional
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Health check failed
    Unknown,
}

/// Configuration provider for services
pub trait ConfigProvider<T> {
    type Error: BasilcaError;

    /// Load configuration
    fn load_config(&self) -> Result<T, Self::Error>;

    /// Watch for configuration changes
    #[allow(async_fn_in_trait)]
    async fn watch_config(&self) -> Result<tokio::sync::watch::Receiver<T>, Self::Error>;

    /// Validate configuration
    fn validate_config(&self, config: &T) -> Result<(), Self::Error>;
}

/// Service builder pattern for dependency injection
pub struct ServiceBuilder<T: Service> {
    service: T,
    dependencies: Vec<String>,
}

impl<T: Service> ServiceBuilder<T> {
    pub fn new(service: T) -> Self {
        Self {
            service,
            dependencies: Vec::new(),
        }
    }

    pub fn depends_on(mut self, service_name: &str) -> Self {
        self.dependencies.push(service_name.to_string());
        self
    }

    pub fn build(self) -> (T, ServiceDescriptor) {
        let descriptor = ServiceDescriptor {
            name: self.service.name().to_string(),
            dependencies: self.dependencies,
        };
        (self.service, descriptor)
    }
}

/// Service dependency descriptor
#[derive(Debug, Clone)]
pub struct ServiceDescriptor {
    pub name: String,
    pub dependencies: Vec<String>,
}
