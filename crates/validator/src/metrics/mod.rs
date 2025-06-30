//! Validator Prometheus Metrics Implementation
//!
//! Provides comprehensive business and operational metrics for the Basilica Validator component.
//! Implements Prometheus metrics according to the Grafana dashboard compatibility requirements.

pub mod api_metrics;
pub mod business_metrics;
pub mod prometheus_metrics;

pub use api_metrics::*;
pub use business_metrics::*;
pub use prometheus_metrics::*;

use anyhow::Result;
use common::config::MetricsConfig;
use std::sync::Arc;

/// Complete validator metrics collection system
#[derive(Clone)]
pub struct ValidatorMetrics {
    prometheus: Arc<ValidatorPrometheusMetrics>,
    business: Arc<ValidatorBusinessMetrics>,
    api: Arc<ValidatorApiMetrics>,
    config: MetricsConfig,
}

impl ValidatorMetrics {
    /// Initialize validator metrics system
    pub fn new(config: MetricsConfig) -> Result<Self> {
        let prometheus = Arc::new(ValidatorPrometheusMetrics::new()?);
        let business = Arc::new(ValidatorBusinessMetrics::new(prometheus.clone())?);
        let api = Arc::new(ValidatorApiMetrics::new(prometheus.clone())?);

        Ok(Self {
            prometheus,
            business,
            api,
            config,
        })
    }

    /// Get Prometheus metrics instance
    pub fn prometheus(&self) -> Arc<ValidatorPrometheusMetrics> {
        self.prometheus.clone()
    }

    /// Get business metrics instance
    pub fn business(&self) -> Arc<ValidatorBusinessMetrics> {
        self.business.clone()
    }

    /// Get API metrics instance
    pub fn api(&self) -> Arc<ValidatorApiMetrics> {
        self.api.clone()
    }

    /// Start metrics server
    pub async fn start_server(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Metrics collection disabled");
            return Ok(());
        }

        let prometheus_config = self
            .config
            .prometheus
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Prometheus configuration is required"))?;
        let address = format!("{}:{}", prometheus_config.host, prometheus_config.port);

        // Install Prometheus exporter
        let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let socket_addr: std::net::SocketAddr = address.parse()?;
        builder.with_http_listener(socket_addr).install()?;

        tracing::info!("Metrics server started on http://{}/metrics", address);

        // Start metrics collection task
        let prometheus = self.prometheus.clone();
        let interval = self.config.collection_interval.as_secs();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));

            loop {
                ticker.tick().await;
                prometheus.collect_system_metrics().await;
            }
        });

        Ok(())
    }
}
