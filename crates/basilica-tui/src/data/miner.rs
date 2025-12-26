//! Miner mode data fetching
//!
//! Data structures for miner mode. Will be connected to miner metrics in future.
#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Miner mode data state
#[derive(Debug, Default, Clone)]
pub struct MinerData {
    /// Registered nodes
    pub nodes: Vec<NodeInfo>,
    /// Validator assignments
    pub validators: Vec<ValidatorInfo>,
    /// Miner identity
    pub miner_info: Option<MinerInfo>,
    /// Earnings data
    pub earnings: EarningsData,
    /// Recent log entries
    pub logs: Vec<LogEntry>,
    /// Loading states
    pub loading: LoadingState,
    /// Historical metrics for sparklines
    pub metrics_history: MetricsHistory,
}

/// Historical metrics data for visualization
#[derive(Debug, Default, Clone)]
pub struct MetricsHistory {
    /// GPU utilization history (last N samples)
    pub gpu_utilization: Vec<f64>,
    /// Memory utilization history
    pub memory_utilization: Vec<f64>,
    /// Revenue history (daily)
    pub daily_revenue: Vec<f64>,
    /// Request rate history
    pub request_rate: Vec<f64>,
    /// Max samples to keep
    pub max_samples: usize,
}

impl MetricsHistory {
    pub fn new(max_samples: usize) -> Self {
        Self {
            gpu_utilization: Vec::with_capacity(max_samples),
            memory_utilization: Vec::with_capacity(max_samples),
            daily_revenue: Vec::with_capacity(max_samples),
            request_rate: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn push_gpu_util(&mut self, value: f64) {
        if self.gpu_utilization.len() >= self.max_samples {
            self.gpu_utilization.remove(0);
        }
        self.gpu_utilization.push(value);
    }

    pub fn push_memory_util(&mut self, value: f64) {
        if self.memory_utilization.len() >= self.max_samples {
            self.memory_utilization.remove(0);
        }
        self.memory_utilization.push(value);
    }

    pub fn push_revenue(&mut self, value: f64) {
        if self.daily_revenue.len() >= self.max_samples {
            self.daily_revenue.remove(0);
        }
        self.daily_revenue.push(value);
    }
}

#[derive(Debug, Default, Clone)]
pub struct LoadingState {
    pub nodes: bool,
    pub validators: bool,
    pub earnings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub gpu_type: String,
    pub gpu_count: u32,
    pub status: NodeStatus,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
    pub assigned_gpus: u32,
    pub uptime_hours: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum NodeStatus {
    Healthy,
    Warning,
    Offline,
    #[default]
    Unknown,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub name: String,
    pub hotkey: String,
    pub stake: f64,
    pub status: ValidatorStatus,
    pub assigned_gpus: u32,
    pub assigned_nodes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum ValidatorStatus {
    Active,
    Pending,
    #[default]
    Inactive,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerInfo {
    pub uid: u16,
    pub hotkey: String,
    pub stake: f64,
    pub netuid: u16,
    pub network: String,
    pub axon_port: u16,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EarningsData {
    pub current_rate_per_hour: f64,
    pub today: f64,
    pub this_week: f64,
    pub this_month: f64,
    pub revenue_history: Vec<f64>,
    pub payments: Vec<PaymentInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentInfo {
    pub date: String,
    pub validator: String,
    pub description: String,
    pub amount: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub source: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl MinerData {
    /// Create new miner data instance
    pub fn new() -> Self {
        Self {
            metrics_history: MetricsHistory::new(30), // Keep 30 samples
            ..Default::default()
        }
    }

    /// Refresh all miner data
    pub async fn refresh_all(&mut self, _config_path: Option<&str>) -> Result<()> {
        self.loading = LoadingState {
            nodes: true,
            validators: true,
            earnings: true,
        };

        // TODO: Implement actual data fetching from miner metrics/gRPC
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        self.loading = LoadingState::default();
        Ok(())
    }

    /// Refresh node status only
    pub async fn refresh_nodes(&mut self) -> Result<()> {
        self.loading.nodes = true;
        // TODO: Implement actual node status fetching
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        self.loading.nodes = false;
        Ok(())
    }

    /// Poll and update metrics
    pub async fn poll_metrics(&mut self, _metrics_url: Option<&str>) -> Result<()> {
        // TODO: Fetch from Prometheus endpoint
        // For now, simulate metrics with some variance
        let gpu_util = 70.0 + (fastrand::f64() * 20.0);
        let mem_util = 60.0 + (fastrand::f64() * 25.0);

        self.metrics_history.push_gpu_util(gpu_util);
        self.metrics_history.push_memory_util(mem_util);

        Ok(())
    }

    /// Get total GPU count across all nodes
    pub fn total_gpus(&self) -> u32 {
        self.nodes.iter().map(|n| n.gpu_count).sum()
    }

    /// Get count of healthy nodes
    pub fn healthy_nodes(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count()
    }

    /// Get count of assigned GPUs
    pub fn assigned_gpus(&self) -> u32 {
        self.nodes.iter().map(|n| n.assigned_gpus).sum()
    }

    /// Get average GPU utilization
    pub fn avg_gpu_utilization(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes.iter().map(|n| n.gpu_utilization).sum::<f64>() / self.nodes.len() as f64
    }

    /// Get average memory utilization
    pub fn avg_memory_utilization(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes.iter().map(|n| n.memory_utilization).sum::<f64>() / self.nodes.len() as f64
    }
}
