use crate::api::types::ApiError;
use axum::{
    body::Body,
    extract::State,
    http::{header::AUTHORIZATION, HeaderMap, HeaderValue, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

pub async fn require_api_key(
    State(expected_api_key): State<Arc<str>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match extract_api_key(request.headers()) {
        Some(provided_api_key) if provided_api_key == expected_api_key.as_ref() => {
            next.run(request).await
        }
        _ => ApiError::Unauthorized.into_response(),
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-api-key")
        .and_then(header_value_to_str)
        .or_else(|| {
            headers
                .get(AUTHORIZATION)
                .and_then(header_value_to_str)
                .and_then(|value| {
                    value
                        .strip_prefix("Bearer ")
                        .or_else(|| value.strip_prefix("bearer "))
                })
        })
}

fn header_value_to_str(value: &HeaderValue) -> Option<&str> {
    value
        .to_str()
        .ok()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

pub async fn reject_unconfigured_protected_routes(
    _request: Request<Body>,
    _next: Next,
) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(serde_json::json!({
            "error": "Protected validator API routes are disabled until api.api_key is configured or api.allow_unauthenticated_routes=true is set explicitly.",
            "timestamp": chrono::Utc::now()
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use tower::util::ServiceExt;

    fn protected_router() -> Router {
        Router::new()
            .route("/protected", get(|| async { StatusCode::OK }))
            .route_layer(middleware::from_fn_with_state(
                Arc::<str>::from("secret-key"),
                require_api_key,
            ))
    }

    #[tokio::test]
    async fn protected_routes_require_api_key() {
        let response = protected_router()
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn x_api_key_header_is_accepted() {
        let response = protected_router()
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header("x-api-key", "secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bearer_header_is_accepted() {
        let response = protected_router()
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .header(AUTHORIZATION, "Bearer secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn public_routes_stay_accessible_without_api_key() {
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .merge(protected_router());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn disabled_protected_routes_return_service_unavailable() {
        let app = Router::new()
            .route("/protected", get(|| async { StatusCode::OK }))
            .route_layer(middleware::from_fn(reject_unconfigured_protected_routes));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/protected")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
