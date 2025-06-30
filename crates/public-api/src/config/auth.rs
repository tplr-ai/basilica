//! Authentication configuration

use serde::{Deserialize, Serialize};

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// API key header name
    pub api_key_header: String,

    /// JWT secret key
    pub jwt_secret: String,

    /// JWT expiration in hours
    pub jwt_expiration_hours: u64,

    /// Enable anonymous access (with lower rate limits)
    pub allow_anonymous: bool,

    /// Master API keys for admin access
    pub master_api_keys: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_key_header: "X-API-Key".to_string(),
            jwt_secret: "change-me-in-production".to_string(),
            jwt_expiration_hours: 24,
            allow_anonymous: true,
            master_api_keys: vec![],
        }
    }
}
