//! Basilica Executor Library
//!
//! Core executor functionality and CLI interface.

pub mod cli;
pub mod config;
pub mod container_manager;
pub mod grpc_server;
pub mod journal;
pub mod system_monitor;
pub mod validation_session;

pub use config::ExecutorConfig;

use anyhow::Result;
use common::identity::ExecutorId;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use tracing::info;
use validation_session::ValidationSessionService;

pub struct ExecutorState {
    pub id: ExecutorId,
    pub config: ExecutorConfig,
    pub system_monitor: Arc<system_monitor::SystemMonitor>,
    pub container_manager: container_manager::ContainerManager,
    pub validation_service: Option<Arc<ValidationSessionService>>,
    pub validation_session: Arc<validation_session::ValidationSessionService>,
    pub active_challenges: Arc<AtomicU32>,
}

impl ExecutorState {
    pub async fn new(config: ExecutorConfig) -> Result<Self> {
        let id = ExecutorId::new();
        info!("Initializing executor with ID: {}", id);

        let system_monitor = Arc::new(system_monitor::SystemMonitor::new(config.system.clone())?);

        let container_manager =
            container_manager::ContainerManager::new(config.docker.clone()).await?;

        let validation_session = Arc::new(ValidationSessionService::new(config.validator.clone())?);

        let validation_service = if config.validator.enabled {
            Some(validation_session.clone())
        } else {
            None
        };

        Ok(Self {
            id,
            config,
            system_monitor,
            container_manager,
            validation_service,
            validation_session,
            active_challenges: Arc::new(AtomicU32::new(0)),
        })
    }

    pub async fn health_check(&self) -> Result<()> {
        info!("Running executor health check...");

        self.system_monitor.health_check().await?;

        self.container_manager.health_check().await?;

        if let Some(validation_service) = &self.validation_service {
            let cleaned = validation_service.cleanup_access().await?;
            if cleaned > 0 {
                info!(
                    "Cleaned up {} orphaned validation access records during health check",
                    cleaned
                );
            }
        }

        info!("All executor components healthy");
        Ok(())
    }
}
