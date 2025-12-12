//! Federation integration for basilica-api
//!
//! Provides integration with the multi-cluster federation system

use basilica_federation::{FederationApi, FederationConfig};
use std::sync::Arc;

/// Federation client for basilica-api
pub struct FederationClient {
    api: Arc<FederationApi>,
}

impl FederationClient {
    /// Create a new federation client
    pub async fn new(config: FederationConfig) -> crate::Result<Self> {
        let api = FederationApi::new(config).await
            .map_err(|e| crate::error::ApiError::Internal(format!("Federation error: {}", e)))?;
        
        Ok(Self {
            api: Arc::new(api),
        })
    }
    
    /// Get federation API reference
    pub fn api(&self) -> &Arc<FederationApi> {
        &self.api
    }
    
    /// Check if federation is enabled
    pub fn is_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_federation_client_creation() {
        let config = FederationConfig::default();
        let client = FederationClient::new(config).await;
        assert!(client.is_ok());
    }
}

