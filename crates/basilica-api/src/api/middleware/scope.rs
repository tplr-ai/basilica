//! Scope validation middleware for OAuth-based authorization
//!
//! This middleware validates that authenticated users have the required scopes
//! to access specific API endpoints.

use axum::{
    extract::Request,
    http::{Method, StatusCode},
    middleware::Next,
    response::Response,
};
use tracing::{debug, warn};

use super::auth::get_auth_context;

/// Scope validation middleware
///
/// Checks if the authenticated user has the required scope for the requested endpoint
pub async fn scope_validation_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    // Get the required scope for this route
    let required_scope = match get_required_scope(&req) {
        Some(scope) => scope,
        None => {
            // Route not explicitly configured - deny access for security
            warn!(
                "Access denied for unconfigured route: {} {}",
                req.method(),
                req.uri().path()
            );
            return Err(StatusCode::FORBIDDEN);
        }
    };

    // If empty scope, just require authentication (already validated by auth0 middleware)
    if required_scope.is_empty() {
        debug!(
            "Route {} {} requires authentication only (no specific scope)",
            req.method(),
            req.uri().path()
        );
        return Ok(next.run(req).await);
    }

    // Get the user's auth context from the request extensions
    let auth_context = match get_auth_context(&req) {
        Some(context) => context,
        None => {
            warn!("No authentication context found in request for scope validation");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Check if the user has the required scope
    if !auth_context.has_scope(&required_scope) {
        warn!(
            "User {} lacks required scope '{}' for {} {}. User's scopes: {:?}",
            auth_context.user_id,
            required_scope,
            req.method(),
            req.uri().path(),
            auth_context.scopes
        );
        return Err(StatusCode::FORBIDDEN);
    }

    debug!(
        "User {} authorized with scope '{}' for {} {}",
        auth_context.user_id,
        required_scope,
        req.method(),
        req.uri().path()
    );

    Ok(next.run(req).await)
}

/// Get the required scope for a given route
///
/// Maps HTTP method and path combinations to their required OAuth scopes.
/// Returns Some(scope) for configured routes, or None for unconfigured routes
/// which will be rejected by the middleware.
fn get_required_scope(req: &Request) -> Option<String> {
    let path = req.uri().path();
    let method = req.method();

    match (method, path) {
        // Rental endpoints
        (&Method::GET, "/rentals") => Some("rentals:list".to_string()),
        (&Method::POST, "/rentals") => Some("rentals:create".to_string()),
        (&Method::DELETE, p) if p.starts_with("/rentals/") && !p.contains("/logs") => {
            Some("rentals:stop".to_string())
        }
        (&Method::GET, p) if p.starts_with("/rentals/") && p.ends_with("/logs") => {
            Some("rentals:logs".to_string())
        }
        (&Method::GET, p) if p.starts_with("/rentals/") => Some("rentals:view".to_string()),

        // Executor endpoints
        (&Method::GET, "/executors") => Some("executors:list".to_string()),

        // API Key management endpoints
        (&Method::POST, "/api-keys") => Some("keys:create".to_string()),
        (&Method::GET, "/api-keys") => Some("keys:list".to_string()),
        (&Method::DELETE, p) if p.starts_with("/api-keys/") => Some("keys:revoke".to_string()),

        // Health check requires authentication but no specific scope
        // We use an empty string to indicate "authenticated but no specific scope required"
        (&Method::GET, "/health") => Some(String::new()),

        // Disable access to routes that are not explicitly configured to avoid unintentional access
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    #[test]
    fn test_required_scope_mapping() {
        // Test rental endpoints
        let req = Request::builder()
            .method(Method::GET)
            .uri("/rentals")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("rentals:list".to_string()));

        let req = Request::builder()
            .method(Method::POST)
            .uri("/rentals")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("rentals:create".to_string()));

        let req = Request::builder()
            .method(Method::GET)
            .uri("/rentals/123")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("rentals:view".to_string()));

        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/rentals/123")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("rentals:stop".to_string()));

        let req = Request::builder()
            .method(Method::GET)
            .uri("/rentals/123/logs")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("rentals:logs".to_string()));

        // Test executor endpoint
        let req = Request::builder()
            .method(Method::GET)
            .uri("/executors")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some("executors:list".to_string()));

        // Test health endpoint (requires authentication but no specific scope)
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), Some(String::new()));
    }

    #[test]
    fn test_unknown_routes_rejected() {
        // Test unknown path
        let req = Request::builder()
            .method(Method::GET)
            .uri("/unknown")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), None);

        // Test unknown method on known path
        let req = Request::builder()
            .method(Method::PATCH)
            .uri("/rentals")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), None);

        // Test completely random path
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/nonexistent")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), None);

        // Test PUT on rental (not configured)
        let req = Request::builder()
            .method(Method::PUT)
            .uri("/rentals/123")
            .body(Body::empty())
            .unwrap();
        assert_eq!(get_required_scope(&req), None);
    }
}
