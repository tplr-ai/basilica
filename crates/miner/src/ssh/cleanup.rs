//! Background cleanup service for SSH resources

use anyhow::Result;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, info};

use super::{MinerSshConfig, ValidatorAccessService};

/// Background service for cleaning up expired SSH sessions and resources
#[derive(Debug)]
pub struct SshCleanupService {
    validator_access: ValidatorAccessService,
    cleanup_interval: Duration,
}

impl SshCleanupService {
    /// Create a new cleanup service
    pub fn new(validator_access: ValidatorAccessService, config: &MinerSshConfig) -> Self {
        Self {
            validator_access,
            cleanup_interval: Duration::from_secs(config.cleanup_interval),
        }
    }

    /// Start the background cleanup task
    pub async fn start_cleanup_task(self) -> Result<()> {
        info!(
            "Starting SSH cleanup service with interval: {:?}",
            self.cleanup_interval
        );

        tokio::spawn(async move {
            self.cleanup_loop().await;
        });

        Ok(())
    }

    /// Main cleanup loop
    async fn cleanup_loop(self) {
        let mut interval = interval(self.cleanup_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.perform_cleanup().await {
                error!("Error during SSH cleanup: {}", e);
            }
        }
    }

    /// Perform one cleanup cycle
    async fn perform_cleanup(&self) -> Result<()> {
        debug!("Starting SSH cleanup cycle");

        let stats = self.validator_access.cleanup_expired().await?;

        if stats.cleaned_keys > 0 || stats.cleaned_users > 0 {
            info!(
                "SSH cleanup completed: {} keys, {} users cleaned, {} errors",
                stats.cleaned_keys, stats.cleaned_users, stats.errors_encountered
            );
        } else {
            debug!("SSH cleanup completed: no expired resources found");
        }

        Ok(())
    }
}
