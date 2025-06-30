//! Network benchmarking implementation

use anyhow::{Context, Result};
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};
use tokio::time::timeout;

use super::types::*;

pub struct NetworkBenchmarker;

impl NetworkBenchmarker {
    pub async fn run_comprehensive_benchmark() -> Result<NetworkBenchmarkResults> {
        let mut results = NetworkBenchmarkResults::new();

        let latency_targets = [
            ("Google DNS", "8.8.8.8"),
            ("Cloudflare DNS", "1.1.1.1"),
            ("Google", "google.com"),
        ];

        for (name, target) in latency_targets {
            if let Ok(latency_test) = Self::test_latency(name, target).await {
                results.latency_tests.push(latency_test);
            }
        }

        // Run throughput tests
        let throughput_targets = [
            ("huggingface-mlfoundations-dclm-baseline-1.0-parquet-global-shard_01_of_10-local-shard_0_of_10-shard_00000000_processed.parquet-144MB", "https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/resolve/main/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.parquet"),
            ("dagshub-s3-fast-ai-imageclas-cifar10-129MB", "https://dagshub.com/DagsHub-Datasets/fast-ai-imageclas-dataset/raw/79dce8650aff92c30bbe1305f760d740fc2c7f76/s3:/fast-ai-imageclas/cifar10.tgz"),
            // ("dagshub-gcp-OperationSavta-SavtaDepth-cifar10-2.8GB", "https://dagshub.com/OperationSavta/SavtaDepth/raw/685eb747641f1f3c42711320441cfaa32fa5c1e9/src/data/raw/nyu_depth_v2_labeled.mat"),
        ];

        for (test_type, url) in throughput_targets {
            if let Ok(throughput_test) = Self::test_throughput(test_type, url).await {
                results.throughput_tests.push(throughput_test);
            }
        }

        // Run packet loss test
        if let Ok(packet_loss_test) = Self::test_packet_loss("8.8.8.8", 10).await {
            results.packet_loss_test = packet_loss_test;
        }

        // Run DNS resolution test
        if let Ok(dns_test) = Self::test_dns_resolution("google.com", "8.8.8.8").await {
            results.dns_resolution_test = dns_test;
        }

        Ok(results)
    }

    async fn test_latency(name: &str, target: &str) -> Result<LatencyTest> {
        let mut latencies = Vec::new();
        let mut successful_pings = 0;
        let total_pings = 5; // Reduced for better performance

        for _ in 0..total_pings {
            let ping_start = Instant::now();

            match Self::ping_host(target).await {
                Ok(_) => {
                    let latency = ping_start.elapsed().as_millis() as f64;
                    latencies.push(latency);
                    successful_pings += 1;
                }
                Err(_) => {
                    // Ping failed, don't add to latencies
                }
            }

            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let packet_loss_percent =
            ((total_pings - successful_pings) as f64 / total_pings as f64) * 100.0;

        let (avg_latency, min_latency, max_latency) = if !latencies.is_empty() {
            let sum: f64 = latencies.iter().sum();
            let avg = sum / latencies.len() as f64;
            let min = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = latencies.iter().fold(0.0_f64, |a, &b| a.max(b));
            (avg, min, max)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Resolve IP address
        let target_ip = if target.parse::<IpAddr>().is_ok() {
            target.to_string()
        } else {
            Self::resolve_hostname(target)
                .await
                .unwrap_or_else(|_| target.to_string())
        };

        Ok(LatencyTest {
            target_host: name.to_string(),
            target_ip,
            avg_latency_ms: avg_latency,
            min_latency_ms: min_latency,
            max_latency_ms: max_latency,
            packet_loss_percent,
            samples: successful_pings,
        })
    }

    async fn test_throughput(test_type: &str, url: &str) -> Result<ThroughputTest> {
        let start_time = Instant::now();

        // Use reqwest for simple and reliable HTTP/HTTPS downloads
        let response = timeout(Duration::from_secs(60), reqwest::get(url))
            .await
            .context("Throughput test timed out")?
            .context("Failed to make HTTP request")?;

        let bytes = timeout(Duration::from_secs(60), response.bytes())
            .await
            .context("Download timed out")?
            .context("Failed to download response body")?;

        let total_time = start_time.elapsed();
        let bytes_transferred = bytes.len() as u64;
        let throughput_bps = bytes_transferred as f64 / total_time.as_secs_f64();
        let throughput_mbps = throughput_bps * 8.0 / (1024.0 * 1024.0); // Convert to Mbps

        Ok(ThroughputTest {
            test_type: test_type.to_string(),
            target_host: Self::extract_host_from_url(url).unwrap_or("unknown".to_string()),
            duration_seconds: total_time.as_secs(),
            throughput_mbps,
            bytes_transferred,
        })
    }

    async fn test_packet_loss(target: &str, count: u32) -> Result<PacketLossTest> {
        let start_time = Instant::now();
        let mut packets_received = 0;

        for _ in 0..count {
            if Self::ping_host(target).await.is_ok() {
                packets_received += 1;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let test_duration = start_time.elapsed();
        let packet_loss_percent = ((count - packets_received) as f64 / count as f64) * 100.0;

        Ok(PacketLossTest {
            target_host: target.to_string(),
            packets_sent: count,
            packets_received,
            packet_loss_percent,
            test_duration_seconds: test_duration.as_secs(),
        })
    }

    async fn test_dns_resolution(hostname: &str, dns_server: &str) -> Result<DnsResolutionTest> {
        let start_time = Instant::now();

        match Self::resolve_hostname(hostname).await {
            Ok(ip) => {
                let resolution_time = start_time.elapsed();
                Ok(DnsResolutionTest {
                    hostname: hostname.to_string(),
                    resolution_time_ms: resolution_time.as_millis() as f64,
                    resolved_ips: vec![ip],
                    dns_server: dns_server.to_string(),
                    success: true,
                })
            }
            Err(_) => {
                let resolution_time = start_time.elapsed();
                Ok(DnsResolutionTest {
                    hostname: hostname.to_string(),
                    resolution_time_ms: resolution_time.as_millis() as f64,
                    resolved_ips: Vec::new(),
                    dns_server: dns_server.to_string(),
                    success: false,
                })
            }
        }
    }
}

// Helper methods
impl NetworkBenchmarker {
    pub async fn ping_host(target: &str) -> Result<()> {
        let output = timeout(
            Duration::from_secs(5),
            tokio::process::Command::new("ping")
                .args(["-c", "1", "-W", "2", target])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output(),
        )
        .await
        .context("Ping command timed out")?
        .context("Failed to execute ping command")?;

        if output.status.success() {
            Ok(())
        } else {
            anyhow::bail!("Ping failed");
        }
    }

    async fn resolve_hostname(hostname: &str) -> Result<String> {
        use tokio::net::lookup_host;

        let addresses: Vec<SocketAddr> = lookup_host((hostname, 80)).await?.collect();

        addresses
            .first()
            .map(|addr| addr.ip().to_string())
            .context("No IP addresses found for hostname")
    }

    fn extract_host_from_url(url: &str) -> Option<String> {
        if let Some(protocol_end) = url.find("://") {
            let after_protocol = &url[protocol_end + 3..];
            if let Some(slash_pos) = after_protocol.find('/') {
                Some(after_protocol[..slash_pos].to_string())
            } else {
                Some(after_protocol.to_string())
            }
        } else {
            None
        }
    }
}
