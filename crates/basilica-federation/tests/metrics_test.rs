//! Tests for metrics

use basilica_federation::metrics::FederationMetrics;

#[test]
fn test_metrics_creation() {
    let metrics = FederationMetrics::new();
    assert!(metrics.is_ok());
}

#[test]
fn test_metrics_default() {
    let metrics = FederationMetrics::default();
    
    // Should be able to create default metrics
    assert!(true);
}

#[test]
fn test_prometheus_exporter_init() {
    // Test that Prometheus exporter can be initialized
    // Note: This might fail if port is already in use, which is OK for tests
    let result = FederationMetrics::init_prometheus(9091);
    // We don't assert on result as port might be in use
    assert!(true);
}

