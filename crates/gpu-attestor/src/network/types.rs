//! Network benchmark data types and structures

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpInfo {
    pub ip: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBenchmarkResults {
    pub latency_tests: Vec<LatencyTest>,
    pub throughput_tests: Vec<ThroughputTest>,
    pub packet_loss_test: PacketLossTest,
    pub dns_resolution_test: DnsResolutionTest,
    pub port_connectivity_tests: Vec<PortConnectivityTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyTest {
    pub target_host: String,
    pub target_ip: String,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub packet_loss_percent: f64,
    pub samples: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputTest {
    pub test_type: String,
    pub target_host: String,
    pub duration_seconds: u64,
    pub throughput_mbps: f64,
    pub bytes_transferred: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketLossTest {
    pub target_host: String,
    pub packets_sent: u32,
    pub packets_received: u32,
    pub packet_loss_percent: f64,
    pub test_duration_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsResolutionTest {
    pub hostname: String,
    pub resolution_time_ms: f64,
    pub resolved_ips: Vec<String>,
    pub dns_server: String,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConnectivityTest {
    pub host: String,
    pub port: u16,
    pub protocol: String,
    pub is_reachable: bool,
    pub connection_time_ms: Option<f64>,
    pub error_message: Option<String>,
}

impl NetworkBenchmarkResults {
    pub fn new() -> Self {
        Self {
            latency_tests: Vec::new(),
            throughput_tests: Vec::new(),
            packet_loss_test: PacketLossTest::default(),
            dns_resolution_test: DnsResolutionTest::default(),
            port_connectivity_tests: Vec::new(),
        }
    }

    pub fn avg_latency(&self) -> Option<f64> {
        if self.latency_tests.is_empty() {
            return None;
        }
        let sum: f64 = self.latency_tests.iter().map(|t| t.avg_latency_ms).sum();
        Some(sum / self.latency_tests.len() as f64)
    }

    pub fn avg_throughput(&self) -> Option<f64> {
        if self.throughput_tests.is_empty() {
            return None;
        }
        let sum: f64 = self
            .throughput_tests
            .iter()
            .map(|t| t.throughput_mbps)
            .sum();
        Some(sum / self.throughput_tests.len() as f64)
    }

    pub fn overall_packet_loss(&self) -> f64 {
        self.packet_loss_test.packet_loss_percent
    }

    pub fn connectivity_score(&self) -> f64 {
        let mut score = 0.0;
        let mut total_tests = 0;

        // Latency tests contribution (40%)
        if !self.latency_tests.is_empty() {
            let successful_latency_tests = self
                .latency_tests
                .iter()
                .filter(|t| t.packet_loss_percent < 50.0)
                .count();
            score += (successful_latency_tests as f64 / self.latency_tests.len() as f64) * 40.0;
            total_tests += 1;
        }

        // Throughput tests contribution (30%)
        if !self.throughput_tests.is_empty() {
            let successful_throughput_tests = self
                .throughput_tests
                .iter()
                .filter(|t| t.throughput_mbps > 0.1) // At least 0.1 Mbps
                .count();
            score +=
                (successful_throughput_tests as f64 / self.throughput_tests.len() as f64) * 30.0;
            total_tests += 1;
        }

        // DNS resolution contribution (15%)
        if self.dns_resolution_test.success {
            score += 15.0;
        }
        total_tests += 1;

        // Port connectivity contribution (15%)
        if !self.port_connectivity_tests.is_empty() {
            let successful_port_tests = self
                .port_connectivity_tests
                .iter()
                .filter(|t| t.is_reachable)
                .count();
            score +=
                (successful_port_tests as f64 / self.port_connectivity_tests.len() as f64) * 15.0;
            total_tests += 1;
        }

        if total_tests == 0 {
            0.0
        } else {
            score
        }
    }
}

impl Default for NetworkBenchmarkResults {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PacketLossTest {
    fn default() -> Self {
        Self {
            target_host: "8.8.8.8".to_string(),
            packets_sent: 0,
            packets_received: 0,
            packet_loss_percent: 0.0,
            test_duration_seconds: 0,
        }
    }
}

impl Default for DnsResolutionTest {
    fn default() -> Self {
        Self {
            hostname: "google.com".to_string(),
            resolution_time_ms: 0.0,
            resolved_ips: Vec::new(),
            dns_server: "8.8.8.8".to_string(),
            success: false,
        }
    }
}

impl LatencyTest {
    pub fn new(target_host: String, target_ip: String) -> Self {
        Self {
            target_host,
            target_ip,
            avg_latency_ms: 0.0,
            min_latency_ms: 0.0,
            max_latency_ms: 0.0,
            packet_loss_percent: 0.0,
            samples: 0,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.packet_loss_percent < 5.0 && self.avg_latency_ms < 100.0
    }
}

impl ThroughputTest {
    pub fn new(test_type: String, target_host: String) -> Self {
        Self {
            test_type,
            target_host,
            duration_seconds: 0,
            throughput_mbps: 0.0,
            bytes_transferred: 0,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.throughput_mbps > 1.0 // At least 1 Mbps
    }
}

impl PortConnectivityTest {
    pub fn new(host: String, port: u16, protocol: String) -> Self {
        Self {
            host,
            port,
            protocol,
            is_reachable: false,
            connection_time_ms: None,
            error_message: None,
        }
    }

    pub fn with_success(mut self, connection_time_ms: f64) -> Self {
        self.is_reachable = true;
        self.connection_time_ms = Some(connection_time_ms);
        self.error_message = None;
        self
    }

    pub fn with_error(mut self, error: String) -> Self {
        self.is_reachable = false;
        self.connection_time_ms = None;
        self.error_message = Some(error);
        self
    }
}
