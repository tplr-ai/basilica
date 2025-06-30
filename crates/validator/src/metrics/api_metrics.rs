//! API-specific metrics for Validator HTTP endpoints

use anyhow::Result;
use axum::{extract::Request, middleware::Next, response::Response};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;

use crate::metrics::ValidatorPrometheusMetrics;

/// API metrics collector for Validator HTTP endpoints
pub struct ValidatorApiMetrics {
    prometheus: Arc<ValidatorPrometheusMetrics>,
}

impl ValidatorApiMetrics {
    /// Create new API metrics collector
    pub fn new(prometheus: Arc<ValidatorPrometheusMetrics>) -> Result<Self> {
        Ok(Self { prometheus })
    }

    /// Record API request metrics
    pub fn record_request(
        &self,
        method: &str,
        path: &str,
        status_code: u16,
        duration: Duration,
        response_size: usize,
    ) {
        self.prometheus
            .record_http_request(method, path, status_code, duration, response_size);

        debug!(
            "API request: {} {} -> {} in {:?} ({} bytes)",
            method, path, status_code, duration, response_size
        );
    }

    /// Record specific endpoint metrics
    pub fn record_endpoint_metrics(
        &self,
        endpoint: &str,
        operation: &str,
        success: bool,
        duration: Duration,
    ) {
        // Record general API metrics
        let status_code = if success { 200 } else { 500 };
        self.record_request("POST", endpoint, status_code, duration, 0);

        debug!(
            "Endpoint operation: {} {} success={} duration={:?}",
            endpoint, operation, success, duration
        );
    }

    /// Create timing middleware for automatic metrics collection
    pub fn create_timing_middleware() -> impl Fn(
        Request,
        Next,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response, axum::http::StatusCode>> + Send>,
    > + Clone {
        move |request: Request, next: Next| {
            Box::pin(async move {
                let start = Instant::now();
                let _method = request.method().to_string();
                let _uri = request.uri().path().to_string();

                let response = next.run(request).await;
                let duration = start.elapsed();
                let _status_code = response.status().as_u16();

                // Get response size from headers if available
                let response_size = response
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);

                // Note: In a real implementation, you'd inject the metrics instance here
                // For now, we'll use the global metrics registry
                use metrics::{counter, histogram};

                counter!("basilica_validator_http_requests_total").increment(1);
                histogram!("basilica_validator_http_request_duration_seconds")
                    .record(duration.as_secs_f64());

                if response_size > 0 {
                    histogram!("basilica_validator_http_response_size_bytes")
                        .record(response_size as f64);
                }

                Ok(response)
            })
        }
    }
}

/// Request timer for manual timing
pub struct ApiRequestTimer {
    start: Instant,
    method: String,
    path: String,
    metrics: Arc<ValidatorApiMetrics>,
}

impl ApiRequestTimer {
    /// Create new request timer
    pub fn new(method: String, path: String, metrics: Arc<ValidatorApiMetrics>) -> Self {
        Self {
            start: Instant::now(),
            method,
            path,
            metrics,
        }
    }

    /// Finish timing and record metrics
    pub fn finish(self, status_code: u16, response_size: usize) {
        let duration = self.start.elapsed();
        self.metrics.record_request(
            &self.method,
            &self.path,
            status_code,
            duration,
            response_size,
        );
    }
}

/// API endpoint categories for metrics labeling
pub mod endpoints {
    pub const HEALTH: &str = "/health";
    pub const METRICS: &str = "/metrics";
    pub const CAPACITY: &str = "/api/v1/capacity";
    pub const LOGS: &str = "/api/v1/logs";
    pub const RENTALS: &str = "/api/v1/rentals";
    pub const VALIDATION: &str = "/api/v1/validation";
    pub const STATS: &str = "/api/v1/stats";
}

/// HTTP method constants
pub mod methods {
    pub const GET: &str = "GET";
    pub const POST: &str = "POST";
    pub const PUT: &str = "PUT";
    pub const DELETE: &str = "DELETE";
    pub const PATCH: &str = "PATCH";
}

/// Status code ranges for metrics
pub mod status_ranges {
    pub fn is_success(code: u16) -> bool {
        (200..300).contains(&code)
    }

    pub fn is_client_error(code: u16) -> bool {
        (400..500).contains(&code)
    }

    pub fn is_server_error(code: u16) -> bool {
        (500..600).contains(&code)
    }

    pub fn get_range_label(code: u16) -> &'static str {
        match code {
            200..=299 => "2xx",
            300..=399 => "3xx",
            400..=499 => "4xx",
            500..=599 => "5xx",
            _ => "other",
        }
    }
}
