//! Network benchmarking and testing module
//!
//! This module provides comprehensive network performance testing capabilities
//! including latency, throughput, packet loss, DNS resolution, and port connectivity tests.

pub mod benchmarker;
pub mod ipinfo;
pub mod types;

pub use benchmarker::NetworkBenchmarker;
pub use ipinfo::IpInfoCollector;
pub use types::{
    DnsResolutionTest, IpInfo, LatencyTest, NetworkBenchmarkResults, PacketLossTest,
    PortConnectivityTest, ThroughputTest,
};

/// Get network performance summary
///
/// Analyzes benchmark results and provides a simple performance overview
pub fn get_network_summary(results: &NetworkBenchmarkResults) -> NetworkSummary {
    NetworkSummary {
        avg_latency_ms: results.avg_latency(),
        avg_throughput_mbps: results.avg_throughput(),
        packet_loss_percent: results.overall_packet_loss(),
        dns_working: results.dns_resolution_test.success,
        connectivity_score: results.connectivity_score(),
        total_tests_run: results.latency_tests.len()
            + results.throughput_tests.len()
            + results.port_connectivity_tests.len()
            + 2, // DNS + packet loss tests
        successful_tests: count_successful_tests(results),
    }
}

/// Network performance summary
#[derive(Debug, Clone)]
pub struct NetworkSummary {
    pub avg_latency_ms: Option<f64>,
    pub avg_throughput_mbps: Option<f64>,
    pub packet_loss_percent: f64,
    pub dns_working: bool,
    pub connectivity_score: f64,
    pub total_tests_run: usize,
    pub successful_tests: usize,
}

impl NetworkSummary {
    /// Check if network performance is considered healthy
    pub fn is_healthy(&self) -> bool {
        self.connectivity_score >= 75.0
            && self.packet_loss_percent < 5.0
            && self.dns_working
            && self.avg_latency_ms.is_some_and(|l| l < 100.0)
    }

    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_tests_run == 0 {
            0.0
        } else {
            (self.successful_tests as f64 / self.total_tests_run as f64) * 100.0
        }
    }

    /// Get a simple performance grade (A-F)
    pub fn performance_grade(&self) -> char {
        match self.connectivity_score {
            score if score >= 90.0 => 'A',
            score if score >= 80.0 => 'B',
            score if score >= 70.0 => 'C',
            score if score >= 60.0 => 'D',
            _ => 'F',
        }
    }
}

/// Count successful tests in benchmark results
fn count_successful_tests(results: &NetworkBenchmarkResults) -> usize {
    let mut count = 0;

    // Count successful latency tests (< 50% packet loss)
    count += results
        .latency_tests
        .iter()
        .filter(|t| t.packet_loss_percent < 50.0)
        .count();

    // Count successful throughput tests (> 0.1 Mbps)
    count += results
        .throughput_tests
        .iter()
        .filter(|t| t.throughput_mbps > 0.1)
        .count();

    // Count successful port connectivity tests
    count += results
        .port_connectivity_tests
        .iter()
        .filter(|t| t.is_reachable)
        .count();

    // Add DNS test success
    if results.dns_resolution_test.success {
        count += 1;
    }

    // Add packet loss test success (< 10% loss)
    if results.packet_loss_test.packet_loss_percent < 10.0 {
        count += 1;
    }

    count
}

/// Create a minimal network benchmark result for testing
pub fn create_minimal_benchmark_result() -> NetworkBenchmarkResults {
    NetworkBenchmarkResults {
        latency_tests: Vec::new(),
        throughput_tests: Vec::new(),
        packet_loss_test: PacketLossTest {
            target_host: "8.8.8.8".to_string(),
            packets_sent: 0,
            packets_received: 0,
            packet_loss_percent: 0.0,
            test_duration_seconds: 0,
        },
        dns_resolution_test: DnsResolutionTest {
            hostname: "google.com".to_string(),
            resolution_time_ms: 0.0,
            resolved_ips: Vec::new(),
            dns_server: "8.8.8.8".to_string(),
            success: false,
        },
        port_connectivity_tests: Vec::new(),
    }
}
