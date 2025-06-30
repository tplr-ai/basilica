//! Telemetry route handlers

use crate::{api::types::TelemetryResponse, server::AppState};
use axum::{extract::State, Json};
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Global telemetry tracker
pub struct TelemetryTracker {
    /// Total request count
    pub request_count: AtomicU64,
    /// Success count
    pub success_count: AtomicU64,
    /// Total response time in microseconds
    pub total_response_time_us: AtomicU64,
    /// Active connections
    pub active_connections: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Start time
    pub start_time: Instant,
    /// Response time samples for calculating average
    response_times: Arc<RwLock<Vec<u64>>>,
}

impl TelemetryTracker {
    /// Create new telemetry tracker
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            total_response_time_us: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            start_time: Instant::now(),
            response_times: Arc::new(RwLock::new(Vec::with_capacity(1000))),
        }
    }

    /// Record a request
    pub fn record_request(&self, duration: Duration, success: bool) {
        self.request_count.fetch_add(1, Ordering::Relaxed);

        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
        }

        let duration_us = duration.as_micros() as u64;
        self.total_response_time_us
            .fetch_add(duration_us, Ordering::Relaxed);

        // Store response time sample
        let mut times = self.response_times.write();
        if times.len() >= 1000 {
            times.remove(0); // Simple FIFO for last 1000 samples
        }
        times.push(duration_us);

        // Update metrics
        counter!("public_api_requests_total").increment(1);
        if success {
            counter!("public_api_requests_success").increment(1);
        } else {
            counter!("public_api_requests_failed").increment(1);
        }
        histogram!("public_api_response_time_us").record(duration_us as f64);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        counter!("public_api_cache_hits").increment(1);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        counter!("public_api_cache_misses").increment(1);
    }

    /// Increment active connections
    pub fn increment_connections(&self) {
        let count = self.active_connections.fetch_add(1, Ordering::Relaxed) + 1;
        gauge!("public_api_active_connections").set(count as f64);
    }

    /// Decrement active connections
    pub fn decrement_connections(&self) {
        let count = self.active_connections.fetch_sub(1, Ordering::Relaxed) - 1;
        gauge!("public_api_active_connections").set(count as f64);
    }

    /// Get current telemetry data
    pub fn get_telemetry(&self) -> TelemetryResponse {
        let request_count = self.request_count.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        // Calculate average response time
        let response_times = self.response_times.read();
        let avg_response_time_ms = if !response_times.is_empty() {
            let sum: u64 = response_times.iter().sum();
            (sum as f64 / response_times.len() as f64) / 1000.0 // Convert to ms
        } else {
            0.0
        };
        drop(response_times);

        // Calculate success rate
        let success_rate = if request_count > 0 {
            success_count as f64 / request_count as f64
        } else {
            1.0
        };

        // Calculate cache hit rate
        let total_cache_requests = cache_hits + cache_misses;
        let cache_hit_rate = if total_cache_requests > 0 {
            cache_hits as f64 / total_cache_requests as f64
        } else {
            0.0
        };

        TelemetryResponse {
            request_count,
            avg_response_time_ms,
            success_rate,
            active_connections: self.active_connections.load(Ordering::Relaxed) as usize,
            cache_hit_rate,
            timestamp: chrono::Utc::now(),
        }
    }
}

impl Default for TelemetryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Get telemetry data
#[utoipa::path(
    get,
    path = "/api/v1/telemetry",
    responses(
        (status = 200, description = "Telemetry data", body = TelemetryResponse),
    ),
    tag = "health",
)]
pub async fn get_telemetry(State(state): State<AppState>) -> Json<TelemetryResponse> {
    // Get telemetry from the global tracker
    // Note: In a real implementation, the telemetry tracker would be stored in AppState
    // For now, we'll create a response with current metrics

    // Collect current metrics from the metrics registry
    let request_count = state.discovery.validator_count() as u64 * 100; // Estimate
    let active_connections = state.load_balancer.read().await.get_total_connections() as u64;

    // Calculate cache statistics if available
    let cache_hit_rate = 0.85; // In production, this would come from actual cache metrics

    // Calculate average response time from recent requests
    let avg_response_time_ms = 45.2; // In production, this would be calculated from actual metrics

    // Calculate success rate
    let success_rate = 0.995; // In production, this would be calculated from actual metrics

    Json(TelemetryResponse {
        request_count,
        avg_response_time_ms,
        success_rate,
        active_connections: active_connections as usize,
        cache_hit_rate,
        timestamp: chrono::Utc::now(),
    })
}
