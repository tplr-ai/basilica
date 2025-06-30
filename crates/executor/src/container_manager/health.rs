//! Container manager health checking functionality

use anyhow::Result;
use bollard::Docker;
use tracing::info;

#[derive(Debug, Clone)]
pub struct HealthChecker {
    docker: Docker,
}

impl HealthChecker {
    pub fn new(docker: Docker) -> Self {
        Self { docker }
    }

    pub async fn health_check(&self) -> Result<()> {
        info!("Running container manager health check");

        let _version = self.docker.version().await?;

        let containers = self.docker.list_containers::<String>(None).await?;
        info!("Health check: {} containers found", containers.len());

        info!("Container manager health check passed");
        Ok(())
    }
}
