//! # Session Cleanup Service
//!
//! Provides periodic cleanup of expired JWT tokens and validator sessions.

use anyhow::Result;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info};

use crate::auth::JwtAuthService;

/// Session cleanup service
pub struct SessionCleanupService {
    jwt_service: Arc<JwtAuthService>,
    cleanup_interval: Duration,
}

impl SessionCleanupService {
    /// Create a new session cleanup service
    pub fn new(jwt_service: Arc<JwtAuthService>, cleanup_interval: Duration) -> Self {
        Self {
            jwt_service,
            cleanup_interval,
        }
    }

    /// Start the cleanup service
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting session cleanup service with interval: {:?}",
            self.cleanup_interval
        );

        let mut interval = interval(self.cleanup_interval);

        loop {
            interval.tick().await;

            debug!("Running session cleanup");

            match self.jwt_service.cleanup_expired().await {
                Ok(()) => {
                    debug!("Session cleanup completed successfully");
                }
                Err(e) => {
                    error!("Failed to cleanup expired sessions: {}", e);
                }
            }
        }
    }
}

/// Background task to run the cleanup service
pub async fn run_cleanup_service(
    jwt_service: Arc<JwtAuthService>,
    cleanup_interval: Duration,
) -> Result<()> {
    let service = SessionCleanupService::new(jwt_service, cleanup_interval);
    service.start().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    use common::identity::Hotkey;

    #[tokio::test]
    async fn test_session_cleanup_service() {
        let jwt_service = Arc::new(
            JwtAuthService::new(
                "test_secret_key_that_is_long_enough_for_security",
                "test-miner".to_string(),
                "test-miner".to_string(),
                ChronoDuration::milliseconds(100), // Very short expiration for testing
            )
            .unwrap(),
        );

        // Create some sessions
        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();

        // Generate tokens
        for i in 0..3 {
            let session_id = format!("test-session-{i}");
            let _token = jwt_service
                .generate_token(&hotkey, &session_id, vec![], None)
                .await
                .unwrap();
        }

        // Verify sessions exist
        let sessions = jwt_service.list_active_sessions().await;
        assert_eq!(sessions.len(), 3);

        // Wait for sessions to expire
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Run cleanup
        jwt_service.cleanup_expired().await.unwrap();

        // Verify sessions are cleaned up
        let sessions = jwt_service.list_active_sessions().await;
        assert_eq!(sessions.len(), 0);
    }
}
