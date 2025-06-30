//! Miner Prometheus Metrics Implementation
//!
//! Provides comprehensive business and operational metrics for the Basilica Miner component.
//! Tracks executor management, validator interactions, SSH sessions, and deployment operations.

pub mod business_metrics;
pub mod executor_metrics;
pub mod prometheus_metrics;

pub use business_metrics::*;
pub use executor_metrics::*;
pub use prometheus_metrics::*;

use anyhow::Result;
use common::config::MetricsConfig;
use std::sync::Arc;

/// Complete miner metrics collection system
#[derive(Clone)]
pub struct MinerMetrics {
    prometheus: Arc<MinerPrometheusMetrics>,
    business: Arc<MinerBusinessMetrics>,
    executor: Arc<MinerExecutorMetrics>,
    config: MetricsConfig,
}

impl MinerMetrics {
    /// Initialize miner metrics system
    pub fn new(config: MetricsConfig) -> Result<Self> {
        let prometheus = Arc::new(MinerPrometheusMetrics::new()?);
        let business = Arc::new(MinerBusinessMetrics::new(prometheus.clone())?);
        let executor = Arc::new(MinerExecutorMetrics::new(prometheus.clone())?);

        Ok(Self {
            prometheus,
            business,
            executor,
            config,
        })
    }

    /// Get Prometheus metrics instance
    pub fn prometheus(&self) -> Arc<MinerPrometheusMetrics> {
        self.prometheus.clone()
    }

    /// Get business metrics instance
    pub fn business(&self) -> Arc<MinerBusinessMetrics> {
        self.business.clone()
    }

    /// Get executor metrics instance
    pub fn executor(&self) -> Arc<MinerExecutorMetrics> {
        self.executor.clone()
    }

    /// Start metrics server
    pub async fn start_server(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Miner metrics collection disabled");
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

        tracing::info!("Miner metrics server started on http://{}/metrics", address);

        // Start metrics collection task
        let prometheus = self.prometheus.clone();
        let interval = self.config.collection_interval.as_secs();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));

            loop {
                ticker.tick().await;
                prometheus.collect_periodic_metrics().await;
            }
        });

        Ok(())
    }
}
