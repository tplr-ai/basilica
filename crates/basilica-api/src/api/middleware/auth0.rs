//! Auth0 JWT authentication middleware for the Basilica API
//!
//! This module provides middleware for validating Auth0 JWT tokens
//! using remote JWKS validation.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use basilica_common::{auth0_audience, auth0_domain, auth0_issuer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

use crate::{
    api::auth::jwt_validator::{
        fetch_jwks, validate_jwt_with_options, verify_audience, verify_issuer,
    },
    error::ApiError,
    server::AppState,
};

/// Auth0 user claims extracted from JWT token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Auth0Claims {
    /// Subject (user ID from Auth0)
    pub sub: String,

    /// Token audience
    pub aud: serde_json::Value,

    /// Token issuer
    pub iss: String,

    /// Expiration timestamp
    pub exp: u64,

    /// Issued at timestamp
    pub iat: u64,

    /// Token scope/permissions
    #[serde(default)]
    pub scope: Option<String>,

    /// User email (if available)
    #[serde(default)]
    pub email: Option<String>,

    /// Email verification status
    #[serde(default)]
    pub email_verified: Option<bool>,

    /// User name (if available)
    #[serde(default)]
    pub name: Option<String>,

    /// Custom claims
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Extract bearer token from Authorization header
fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .and_then(|auth_header| {
            let mut parts = auth_header.splitn(2, char::is_whitespace);
            let scheme = parts.next()?.trim();
            let token = parts.next()?.trim();

            if scheme.eq_ignore_ascii_case("bearer") && !token.is_empty() {
                Some(token.to_string())
            } else {
                None
            }
        })
}

/// Auth0 authentication middleware
///
/// This middleware validates JWT tokens issued by Auth0 using JWKS validation.
/// It verifies:
/// - Token signature using Auth0's public keys
/// - Token expiration
/// - Audience matches our API identifier
/// - Issuer matches Auth0 domain
pub async fn auth0_middleware(
    State(_state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, Response> {
    debug!("Auth0 middleware: processing request");

    // Extract bearer token from Authorization header
    let token = match extract_bearer_token(req.headers()) {
        Some(token) => token,
        None => {
            warn!("Auth0 middleware: No bearer token found in Authorization header");
            return Err((
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "No authentication token provided. Please include a valid JWT token in the Authorization header".to_string(),
                },
            ).into_response());
        }
    };

    // Fetch JWKS from Auth0 (with caching)
    let jwks = match fetch_jwks(auth0_domain()).await {
        Ok(jwks) => jwks,
        Err(e) => {
            warn!("Auth0 middleware: Failed to fetch JWKS: {}", e);
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                ApiError::Internal {
                    message: "Authentication service temporarily unavailable".to_string(),
                },
            )
                .into_response());
        }
    };

    // Validate JWT token
    let claims = match validate_jwt_with_options(&token, &jwks, None) {
        Ok(claims) => claims,
        Err(e) => {
            warn!("Auth0 middleware: JWT validation failed: {}", e);
            return Err((
                StatusCode::UNAUTHORIZED,
                ApiError::Authentication {
                    message: "Invalid authentication token".to_string(),
                },
            )
                .into_response());
        }
    };

    // Verify audience matches our API identifier
    if let Err(e) = verify_audience(&claims, auth0_audience()) {
        warn!("Auth0 middleware: Audience verification failed: {}", e);
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
        warn!("Auth0 middleware: Issuer verification failed: {}", e);
        return Err((
            StatusCode::UNAUTHORIZED,
            ApiError::Authentication {
                message: "Token issued by unauthorized provider".to_string(),
            },
        )
            .into_response());
    }

    debug!(
        "Auth0 middleware: Successfully validated token for user: {}. Scopes: {:?}",
        claims.sub, claims.scope
    );

    // Convert to Auth0Claims for easier access in handlers
    let auth0_claims = Auth0Claims {
        sub: claims.sub.clone(),
        aud: claims.aud.clone(),
        iss: claims.iss.clone(),
        exp: claims.exp,
        iat: claims.iat,
        scope: claims.scope.clone(),
        email: claims
            .custom
            .get("email")
            .and_then(|v| v.as_str())
            .map(String::from),
        email_verified: claims
            .custom
            .get("email_verified")
            .and_then(|v| v.as_bool()),
        name: claims
            .custom
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from),
        custom: claims.custom.clone(),
    };

    // Store claims in request extensions for use by handlers
    req.extensions_mut().insert(auth0_claims);

    // Continue to the next middleware/handler
    Ok(next.run(req).await)
}

/// Extract Auth0 user claims from request extensions
///
/// This should be called from handlers that are protected by auth0_middleware
pub fn get_auth0_claims(req: &Request) -> Option<&Auth0Claims> {
    req.extensions().get::<Auth0Claims>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bearer_token() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderValue::from_static("Bearer test_token_123"),
        );

        assert_eq!(
            extract_bearer_token(&headers),
            Some("test_token_123".to_string())
        );

        // Test without Bearer prefix
        headers.insert(
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderValue::from_static("test_token_123"),
        );
        assert_eq!(extract_bearer_token(&headers), None);

        // Test empty headers
        let empty_headers = axum::http::HeaderMap::new();
        assert_eq!(extract_bearer_token(&empty_headers), None);
    }

    // Helper function for testing scope matching
    fn has_scope(claims: &Auth0Claims, required_scope: &str) -> bool {
        if let Some(scope) = &claims.scope {
            scope.split_whitespace().any(|s| {
                // Exact match
                if s == required_scope {
                    return true;
                }

                // Wildcard match (e.g., "rentals:*" matches "rentals:view", "rentals:create", etc.)
                if s.ends_with(":*") {
                    let prefix = &s[..s.len() - 1]; // Remove the "*"
                    return required_scope.starts_with(prefix);
                }

                false
            })
        } else {
            false
        }
    }

    #[test]
    fn test_has_scope() {
        let claims = Auth0Claims {
            sub: "test_user".to_string(),
            aud: serde_json::Value::String("basilica-api".to_string()),
            iss: auth0_issuer().to_string(),
            exp: 9999999999,
            iat: 1234567890,
            scope: Some("read:profile write:profile admin".to_string()),
            email: None,
            email_verified: None,
            name: None,
            custom: HashMap::new(),
        };

        assert!(has_scope(&claims, "read:profile"));
        assert!(has_scope(&claims, "write:profile"));
        assert!(has_scope(&claims, "admin"));
        assert!(!has_scope(&claims, "delete:profile"));

        // Test with no scope
        let claims_no_scope = Auth0Claims {
            scope: None,
            ..claims.clone()
        };
        assert!(!has_scope(&claims_no_scope, "read:profile"));
    }

    #[test]
    fn test_has_scope_wildcard() {
        // Test wildcard scope matching
        let claims = Auth0Claims {
            sub: "test_user".to_string(),
            aud: serde_json::Value::String("basilica-api".to_string()),
            iss: auth0_issuer().to_string(),
            exp: 9999999999,
            iat: 1234567890,
            scope: Some("rentals:* executors:list".to_string()),
            email: None,
            email_verified: None,
            name: None,
            custom: HashMap::new(),
        };

        // Wildcard should match all rentals: scopes
        assert!(has_scope(&claims, "rentals:list"));
        assert!(has_scope(&claims, "rentals:create"));
        assert!(has_scope(&claims, "rentals:view"));
        assert!(has_scope(&claims, "rentals:stop"));
        assert!(has_scope(&claims, "rentals:logs"));

        // Should also match exact wildcard
        assert!(has_scope(&claims, "rentals:*"));

        // Should match other exact scopes
        assert!(has_scope(&claims, "executors:list"));

        // Should not match unrelated scopes
        assert!(!has_scope(&claims, "users:list"));
        assert!(!has_scope(&claims, "admin"));
    }
}
