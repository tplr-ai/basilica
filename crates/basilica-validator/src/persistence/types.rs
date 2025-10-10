use chrono::{DateTime, Utc};
use serde_json::Value;

/// Node statistics derived from verification logs
#[derive(Debug, Clone)]
pub struct NodeStats {
    pub node_id: String,
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub average_score: Option<f64>,
    pub average_duration_ms: Option<f64>,
    pub first_verification: Option<DateTime<Utc>>,
    pub last_verification: Option<DateTime<Utc>>,
}

impl NodeStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }
}

/// Available capacity entry
#[derive(Debug, Clone)]
pub struct CapacityEntry {
    pub node_id: String,
    pub verification_score: f64,
    pub success_rate: f64,
    pub last_verification: DateTime<Utc>,
    pub hardware_info: Value,
    pub total_verifications: u64,
}

/// Miner data for listings
#[derive(Debug, Clone)]
pub struct MinerData {
    pub miner_id: String,
    pub hotkey: String,
    pub endpoint: String,
    pub node_count: u32,
    pub verification_score: f64,
    pub uptime_percentage: f64,
    pub last_seen: DateTime<Utc>,
    pub registered_at: DateTime<Utc>,
    pub node_info: Value,
}

/// Miner health data
#[derive(Debug, Clone)]
pub struct MinerHealthData {
    pub last_health_check: DateTime<Utc>,
    pub node_health: Vec<NodeHealthData>,
}

/// Node health data
#[derive(Debug, Clone)]
pub struct NodeHealthData {
    pub node_id: String,
    pub status: String,
    pub last_seen: DateTime<Utc>,
}

/// Node details for miner listings
#[derive(Debug, Clone)]
pub struct NodeData {
    pub node_id: String,
    pub gpu_specs: Vec<crate::api::types::GpuSpec>,
    pub cpu_specs: crate::api::types::CpuSpec,
    pub location: Option<String>,
}

/// Available node data for rental listings
#[derive(Debug, Clone)]
pub struct AvailableNodeData {
    pub node_id: String,
    pub miner_id: String,
    pub gpu_specs: Vec<crate::api::types::GpuSpec>,
    pub cpu_specs: crate::api::types::CpuSpec,
    pub location: Option<String>,
    pub verification_score: f64,
    pub uptime_percentage: f64,
    pub status: Option<String>,
    pub download_mbps: Option<f64>,
    pub upload_mbps: Option<f64>,
    pub speed_test_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Node metric data for initializing metrics
#[derive(Debug, Clone)]
pub struct NodeMetricData {
    pub node_id: String,
    pub miner_id: String,
    pub miner_uid: u16,
    pub gpu_name: Option<String>,
    pub has_active_rental: bool,
}

/// Filter criteria for querying rentals
#[derive(Default)]
pub struct RentalFilter {
    pub rental_id: Option<String>,
    pub validator_hotkey: Option<String>,
    pub exclude_states: Option<Vec<crate::rental::RentalState>>,
    pub order_by_created_desc: bool,
}
