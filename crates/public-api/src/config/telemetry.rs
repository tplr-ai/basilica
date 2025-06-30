//! Telemetry configuration

use serde::{Deserialize, Serialize};

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable Prometheus metrics
    pub enable_metrics: bool,

    /// Metrics endpoint path
    pub metrics_path: String,

    /// Enable distributed tracing
    pub enable_tracing: bool,

    /// Tracing sample rate (0.0 to 1.0)
    pub trace_sample_rate: f64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_path: "/metrics".to_string(),
            enable_tracing: true,
            trace_sample_rate: 0.1,
        }
    }
}
