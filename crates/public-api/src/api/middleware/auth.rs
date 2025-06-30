//! Authentication middleware

use crate::{
    api::types::{ApiKeyInfo, ApiKeyTier},
    error::Error,
    server::AppState,
};
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

/// Authentication middleware
#[derive(Clone)]
pub struct AuthMiddleware {
    state: AppState,
}

impl AuthMiddleware {
    /// Create new authentication middleware
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Middleware handler
    pub async fn handle(
        State(state): State<AppState>,
        mut req: Request,
        next: Next,
    ) -> Result<Response, Error> {
        // Extract API key from header
        let api_key = req
            .headers()
            .get(&state.config.auth.api_key_header)
            .and_then(|v| v.to_str().ok());

        // Validate API key
        match api_key {
            Some(key) => {
                // Check if it's a master key
                if state.config.auth.master_api_keys.contains(&key.to_string()) {
                    let key_info = ApiKeyInfo {
                        key_id: "master".to_string(),
                        tier: ApiKeyTier::Enterprise,
                        rate_limit_override: None,
                    };
                    req.extensions_mut().insert(Arc::new(key_info));
                } else {
                    // Validate API key format and determine tier
                    let (tier, rate_limit_override) =
                        if key.starts_with("sk_enterprise_") && key.len() == 32 {
                            (ApiKeyTier::Enterprise, Some(6000))
                        } else if (key.starts_with("sk_premium_") || key.starts_with("sk_live_"))
                            && key.len() >= 24
                        {
                            (ApiKeyTier::Premium, None)
                        } else if key.starts_with("sk_test_") && key.len() >= 20 {
                            (ApiKeyTier::Free, Some(300))
                        } else {
                            // Invalid key format
                            return Err(Error::Authentication {
                                message: "Invalid API key format".to_string(),
                            });
                        };

                    let key_info = ApiKeyInfo {
                        key_id: key.to_string(),
                        tier,
                        rate_limit_override,
                    };
                    req.extensions_mut().insert(Arc::new(key_info));
                }
            }
            None => {
                // Check if anonymous access is allowed
                if state.config.auth.allow_anonymous {
                    let key_info = ApiKeyInfo {
                        key_id: "anonymous".to_string(),
                        tier: ApiKeyTier::Free,
                        rate_limit_override: None,
                    };
                    req.extensions_mut().insert(Arc::new(key_info));
                } else {
                    return Err(Error::Authentication {
                        message: "API key required".to_string(),
                    });
                }
            }
        }

        Ok(next.run(req).await)
    }
}
