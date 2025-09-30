//! Unified authentication middleware for JWT and API Key authentication
//!
//! This module provides a unified authentication middleware that supports both
//! Auth0 JWT tokens and API keys (Personal Access Tokens).

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use basilica_common::{auth0_audience, auth0_domain, auth0_issuer};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::{
    api::auth::{
        api_keys,
        jwt_validator::{fetch_jwks, validate_jwt_with_options, verify_audience, verify_issuer},
    },
    error::ApiError,
    server::AppState,
};

/// Authentication-specific details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthDetails {
    /// JWT authentication via Auth0 with full user info
    Jwt {
        email: String,
        email_verified: bool,
        name: String,
    },
    /// API Key authentication
    ApiKey,
}

/// Unified authentication context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    /// User ID (subject from JWT or user_id from API key)
    pub user_id: String,

    /// Scopes/permissions
    pub scopes: Vec<String>,

    /// Authentication-specific details
    pub details: AuthDetails,
}

impl AuthContext {
    /// Check if the user has a specific scope (supports wildcards)
    pub fn has_scope(&self, required_scope: &str) -> bool {
        self.scopes.iter().any(|scope| {
            // Exact match
            if scope == required_scope {
                return true;
            }

            // Wildcard match (e.g., "rentals:*" matches "rentals:view", "rentals:create", etc.)
            if scope.ends_with(":*") {
                let prefix = &scope[..scope.len() - 1]; // Remove the '*'
                required_scope.starts_with(prefix)
            } else {
                false
            }
        })
    }

    /// Check if authenticated via JWT
    pub fn is_jwt(&self) -> bool {
        matches!(self.details, AuthDetails::Jwt { .. })
    }

    /// Check if authenticated via API key
    pub fn is_api_key(&self) -> bool {
        matches!(self.details, AuthDetails::ApiKey)
    }
}

/// Unified authentication middleware that handles both JWT and API key authentication
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, Response> {
    // Extract the Authorization header
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "Missing Authorization header".to_string(),
                },
            )
                .into_response()
        })?;

    // Remove "Bearer " prefix if present
    let token = auth_header.strip_prefix("Bearer ").unwrap_or(auth_header);

    // Check if it's an API key or JWT
    let auth_context = if token.starts_with("basilica_") {
        // API Key authentication
        match api_keys::verify_api_key(&state.db, token).await {
            Ok(verified) => {
                debug!(
                    "API key authentication successful for user: {}",
                    verified.user_id
                );
                AuthContext {
                    user_id: verified.user_id,
                    scopes: verified.scopes,
                    details: AuthDetails::ApiKey,
                }
            }
            Err(e) => {
                warn!("API key authentication failed: {}", e);
                return Err((
                    StatusCode::UNAUTHORIZED,
                    ApiError::Authentication {
                        message: "Invalid API key".to_string(),
                    },
                )
                    .into_response());
            }
        }
    } else {
        // JWT authentication (existing Auth0 logic)
        debug!("Attempting JWT authentication");

        // Fetch the JWKS from Auth0
        let jwks = fetch_jwks(auth0_domain()).await.map_err(|e| {
            warn!("Failed to fetch JWKS from Auth0: {}", e);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                ApiError::Authentication {
                    message: "Unable to verify token - authentication service unavailable"
                        .to_string(),
                },
            )
                .into_response()
        })?;

        // Validate the JWT token
        let claims = validate_jwt_with_options(token, &jwks, None).map_err(|e| {
            warn!("JWT validation failed: {}", e);
            (
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "Invalid token".to_string(),
                },
            )
                .into_response()
        })?;

        // Verify audience matches our API identifier
        if let Err(e) = verify_audience(&claims, auth0_audience()) {
            warn!("Audience verification failed: {}", e);
            return Err((
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "Token not authorized for this API".to_string(),
                },
            )
                .into_response());
        }

        // Verify issuer matches Auth0 domain
        if let Err(e) = verify_issuer(&claims, auth0_issuer()) {
            warn!("Issuer verification failed: {}", e);
            return Err((
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "Token issued by unauthorized provider".to_string(),
                },
            )
                .into_response());
        }

        debug!(
            "JWT authentication successful for user: {}. Scopes: {:?}",
            claims.sub, claims.scope
        );

        // Parse scopes from space-separated string
        let scopes = claims
            .scope
            .as_ref()
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default();

        // Extract user info from JWT claims
        let email = claims
            .custom
            .get("email")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("{}@auth0.user", claims.sub));

        let email_verified = claims
            .custom
            .get("email_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let name = claims
            .custom
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| claims.sub.clone());

        AuthContext {
            user_id: claims.sub.clone(),
            scopes,
            details: AuthDetails::Jwt {
                email,
                email_verified,
                name,
            },
        }
    };

    // Store auth context in request extensions for use by handlers
    req.extensions_mut().insert(auth_context);

    // Continue to the next middleware/handler
    Ok(next.run(req).await)
}

/// Extract authentication context from request extensions
///
/// This should be called from handlers that are protected by auth_middleware
pub fn get_auth_context(req: &Request) -> Option<&AuthContext> {
    req.extensions().get::<AuthContext>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_context_has_scope() {
        let context = AuthContext {
            user_id: "user123".to_string(),
            scopes: vec![
                "rentals:view".to_string(),
                "rentals:create".to_string(),
                "admin:*".to_string(),
            ],
            details: AuthDetails::Jwt {
                email: "user@example.com".to_string(),
                email_verified: true,
                name: "Test User".to_string(),
            },
        };

        // Test exact matches
        assert!(context.has_scope("rentals:view"));
        assert!(context.has_scope("rentals:create"));
        assert!(!context.has_scope("rentals:delete"));

        // Test wildcard matches
        assert!(context.has_scope("admin:read"));
        assert!(context.has_scope("admin:write"));
        assert!(context.has_scope("admin:delete"));
        assert!(!context.has_scope("user:read"));
    }

    #[test]
    fn test_auth_details_helpers() {
        let jwt_context = AuthContext {
            user_id: "user123".to_string(),
            scopes: vec![],
            details: AuthDetails::Jwt {
                email: "user@example.com".to_string(),
                email_verified: true,
                name: "Test User".to_string(),
            },
        };

        assert!(jwt_context.is_jwt());
        assert!(!jwt_context.is_api_key());

        let api_key_context = AuthContext {
            user_id: "user456".to_string(),
            scopes: vec![],
            details: AuthDetails::ApiKey,
        };

        assert!(!api_key_context.is_jwt());
        assert!(api_key_context.is_api_key());
    }

    #[test]
    fn test_auth_details_equality() {
        assert_eq!(AuthDetails::ApiKey, AuthDetails::ApiKey);
        assert_ne!(
            AuthDetails::ApiKey,
            AuthDetails::Jwt {
                email: "test@example.com".to_string(),
                email_verified: true,
                name: "Test".to_string(),
            }
        );
    }
}
