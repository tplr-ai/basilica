//! Prometheus metrics for federation

use metrics::{Counter, Gauge, Histogram, HistogramOptions, Unit};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::Arc;
use std::time::Duration;

/// Federation metrics
pub struct FederationMetrics {
    // Request metrics
    pub requests_total: Counter,
    pub requests_duration: Histogram,
    pub requests_by_status: Counter,
    
    // Cluster metrics
    pub clusters_total: Gauge,
    pub clusters_healthy: Gauge,
    pub clusters_unhealthy: Gauge,
    
    // Service discovery metrics
    pub services_discovered: Gauge,
    pub discovery_duration: Histogram,
    pub discovery_errors: Counter,
    
    // Load balancer metrics
    pub load_balancer_selections: Counter,
    pub load_balancer_duration: Histogram,
    
    // Health check metrics
    pub health_checks_total: Counter,
    pub health_check_duration: Histogram,
    pub health_check_errors: Counter,
    
    // Resource management metrics
    pub resources_synced: Counter,
    pub sync_duration: Histogram,
    pub sync_errors: Counter,
}

impl FederationMetrics {
    /// Create new metrics instance
    pub fn new() -> Result<Self, metrics::Error> {
        let registry = metrics::Registry::new();
        
        // Request metrics
        let requests_total = Counter::new(
            "federation_requests_total",
            "Total number of federation API requests",
        )?;
        
        let requests_duration = Histogram::new(
            HistogramOptions::new("federation_request_duration_seconds")
                .with_unit(Unit::Seconds)
                .with_buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
        )?;
        
        let requests_by_status = Counter::new(
            "federation_requests_by_status",
            "Requests by HTTP status code",
        )?;
        
        // Cluster metrics
        let clusters_total = Gauge::new(
            "federation_clusters_total",
            "Total number of configured clusters",
        )?;
        
        let clusters_healthy = Gauge::new(
            "federation_clusters_healthy",
            "Number of healthy clusters",
        )?;
        
        let clusters_unhealthy = Gauge::new(
            "federation_clusters_unhealthy",
            "Number of unhealthy clusters",
        )?;
        
        // Service discovery metrics
        let services_discovered = Gauge::new(
            "federation_services_discovered",
            "Number of services discovered across clusters",
        )?;
        
        let discovery_duration = Histogram::new(
            HistogramOptions::new("federation_discovery_duration_seconds")
                .with_unit(Unit::Seconds)
                .with_buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0]),
        )?;
        
        let discovery_errors = Counter::new(
            "federation_discovery_errors_total",
            "Total number of service discovery errors",
        )?;
        
        // Load balancer metrics
        let load_balancer_selections = Counter::new(
            "federation_load_balancer_selections_total",
            "Total number of cluster selections by load balancer",
        )?;
        
        let load_balancer_duration = Histogram::new(
            HistogramOptions::new("federation_load_balancer_duration_seconds")
                .with_unit(Unit::Seconds)
                .with_buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05]),
        )?;
        
        // Health check metrics
        let health_checks_total = Counter::new(
            "federation_health_checks_total",
            "Total number of health checks performed",
        )?;
        
        let health_check_duration = Histogram::new(
            HistogramOptions::new("federation_health_check_duration_seconds")
                .with_unit(Unit::Seconds)
                .with_buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0]),
        )?;
        
        let health_check_errors = Counter::new(
            "federation_health_check_errors_total",
            "Total number of health check errors",
        )?;
        
        // Resource management metrics
        let resources_synced = Counter::new(
            "federation_resources_synced_total",
            "Total number of resources synchronized",
        )?;
        
        let sync_duration = Histogram::new(
            HistogramOptions::new("federation_sync_duration_seconds")
                .with_unit(Unit::Seconds)
                .with_buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0]),
        )?;
        
        let sync_errors = Counter::new(
            "federation_sync_errors_total",
            "Total number of resource sync errors",
        )?;
        
        // Register all metrics
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(requests_duration.clone()))?;
        registry.register(Box::new(requests_by_status.clone()))?;
        registry.register(Box::new(clusters_total.clone()))?;
        registry.register(Box::new(clusters_healthy.clone()))?;
        registry.register(Box::new(clusters_unhealthy.clone()))?;
        registry.register(Box::new(services_discovered.clone()))?;
        registry.register(Box::new(discovery_duration.clone()))?;
        registry.register(Box::new(discovery_errors.clone()))?;
        registry.register(Box::new(load_balancer_selections.clone()))?;
        registry.register(Box::new(load_balancer_duration.clone()))?;
        registry.register(Box::new(health_checks_total.clone()))?;
        registry.register(Box::new(health_check_duration.clone()))?;
        registry.register(Box::new(health_check_errors.clone()))?;
        registry.register(Box::new(resources_synced.clone()))?;
        registry.register(Box::new(sync_duration.clone()))?;
        registry.register(Box::new(sync_errors.clone()))?;
        
        Ok(Self {
            requests_total,
            requests_duration,
            requests_by_status,
            clusters_total,
            clusters_healthy,
            clusters_unhealthy,
            services_discovered,
            discovery_duration,
            discovery_errors,
            load_balancer_selections,
            load_balancer_duration,
            health_checks_total,
            health_check_duration,
            health_check_errors,
            resources_synced,
            sync_duration,
            sync_errors,
        })
    }
    
    /// Initialize Prometheus exporter
    pub fn init_prometheus(port: u16) -> Result<(), metrics::Error> {
        PrometheusBuilder::new()
            .with_http_listener(([0, 0, 0, 0], port))
            .install()?;
        Ok(())
    }
}

impl Default for FederationMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics")
    }
}

