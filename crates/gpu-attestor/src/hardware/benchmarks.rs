use anyhow::{Context, Result};
use std::fs;

use super::types::BenchmarkResults;

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    pub fn run_all() -> Result<BenchmarkResults> {
        Ok(BenchmarkResults {
            cpu_benchmark_score: Self::benchmark_cpu()?,
            memory_bandwidth_mbps: Self::benchmark_memory()?,
            disk_sequential_read_mbps: Self::benchmark_disk_read()?,
            disk_sequential_write_mbps: Self::benchmark_disk_write()?,
            network_throughput_mbps: Self::benchmark_network().ok(),
        })
    }

    pub fn benchmark_cpu() -> Result<f64> {
        let start = std::time::Instant::now();
        let mut result = 0u64;

        // Simple CPU-intensive calculation
        for i in 0..1_000_000 {
            result = result.wrapping_add(i * i);
        }

        let duration = start.elapsed();
        let score = 1_000_000.0 / duration.as_secs_f64();

        Ok(score)
    }

    pub fn benchmark_memory() -> Result<f64> {
        let size = 1024 * 1024; // 1MB
        let data = vec![0u8; size];
        let mut copy = vec![0u8; size];

        let start = std::time::Instant::now();
        for _ in 0..100 {
            copy.copy_from_slice(&data);
        }
        let duration = start.elapsed();

        let bytes_copied = size * 100;
        let bandwidth_bps = bytes_copied as f64 / duration.as_secs_f64();
        let bandwidth_mbps = bandwidth_bps / (1024.0 * 1024.0);

        Ok(bandwidth_mbps)
    }

    pub fn benchmark_disk_read() -> Result<f64> {
        let temp_file = "/tmp/gpu_attestor_read_benchmark";
        let data = vec![0u8; 1024 * 1024]; // 1MB
        fs::write(temp_file, &data).context("Failed to create benchmark file")?;

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _content = fs::read(temp_file).context("Failed to read benchmark file")?;
        }
        let duration = start.elapsed();

        fs::remove_file(temp_file).ok();

        let bytes_read = data.len() * 10;
        let bandwidth_bps = bytes_read as f64 / duration.as_secs_f64();
        let bandwidth_mbps = bandwidth_bps / (1024.0 * 1024.0);

        Ok(bandwidth_mbps)
    }

    pub fn benchmark_disk_write() -> Result<f64> {
        let temp_file = "/tmp/gpu_attestor_write_benchmark";
        let data = vec![42u8; 1024 * 1024]; // 1MB

        let start = std::time::Instant::now();
        for i in 0..10 {
            let temp_path = format!("{temp_file}.{i}");
            fs::write(&temp_path, &data).context("Failed to write benchmark file")?;
        }
        let duration = start.elapsed();

        // Cleanup
        for i in 0..10 {
            fs::remove_file(format!("{temp_file}.{i}")).ok();
        }

        let bytes_written = data.len() * 10;
        let bandwidth_bps = bytes_written as f64 / duration.as_secs_f64();
        let bandwidth_mbps = bandwidth_bps / (1024.0 * 1024.0);

        Ok(bandwidth_mbps)
    }

    pub fn benchmark_network() -> Result<f64> {
        // This would require a remote endpoint to test against
        // For now, return a placeholder value
        Ok(100.0) // Placeholder 100 Mbps
    }

    pub fn benchmark_cpu_with_iterations(iterations: u64) -> Result<f64> {
        let start = std::time::Instant::now();
        let mut result = 0u64;

        for i in 0..iterations {
            result = result.wrapping_add(i * i);
        }

        let duration = start.elapsed();
        let score = iterations as f64 / duration.as_secs_f64();

        Ok(score)
    }

    pub fn benchmark_memory_with_size(size_mb: usize) -> Result<f64> {
        let size = size_mb * 1024 * 1024;
        let data = vec![0u8; size];
        let mut copy = vec![0u8; size];

        let start = std::time::Instant::now();
        copy.copy_from_slice(&data);
        let duration = start.elapsed();

        let bandwidth_bps = size as f64 / duration.as_secs_f64();
        let bandwidth_mbps = bandwidth_bps / (1024.0 * 1024.0);

        Ok(bandwidth_mbps)
    }

    pub fn benchmark_disk_with_size(size_mb: usize) -> Result<(f64, f64)> {
        let temp_file = "/tmp/gpu_attestor_size_benchmark";
        let data = vec![42u8; size_mb * 1024 * 1024];

        // Write benchmark
        let start = std::time::Instant::now();
        fs::write(temp_file, &data).context("Failed to write benchmark file")?;
        let write_duration = start.elapsed();

        // Read benchmark
        let start = std::time::Instant::now();
        let _content = fs::read(temp_file).context("Failed to read benchmark file")?;
        let read_duration = start.elapsed();

        fs::remove_file(temp_file).ok();

        let bytes = data.len() as f64;
        let write_bandwidth_bps = bytes / write_duration.as_secs_f64();
        let read_bandwidth_bps = bytes / read_duration.as_secs_f64();

        let write_bandwidth_mbps = write_bandwidth_bps / (1024.0 * 1024.0);
        let read_bandwidth_mbps = read_bandwidth_bps / (1024.0 * 1024.0);

        Ok((read_bandwidth_mbps, write_bandwidth_mbps))
    }

    pub fn quick_benchmark() -> Result<BenchmarkResults> {
        Ok(BenchmarkResults {
            cpu_benchmark_score: Self::benchmark_cpu_with_iterations(100_000)?,
            memory_bandwidth_mbps: Self::benchmark_memory_with_size(1)?, // 1MB
            disk_sequential_read_mbps: Self::benchmark_disk_with_size(1)?.0, // 1MB read
            disk_sequential_write_mbps: Self::benchmark_disk_with_size(1)?.1, // 1MB write
            network_throughput_mbps: None, // Skip network for quick benchmark
        })
    }

    pub fn comprehensive_benchmark() -> Result<BenchmarkResults> {
        Ok(BenchmarkResults {
            cpu_benchmark_score: Self::benchmark_cpu_with_iterations(10_000_000)?,
            memory_bandwidth_mbps: Self::benchmark_memory_with_size(100)?, // 100MB
            disk_sequential_read_mbps: Self::benchmark_disk_with_size(100)?.0, // 100MB read
            disk_sequential_write_mbps: Self::benchmark_disk_with_size(100)?.1, // 100MB write
            network_throughput_mbps: Self::benchmark_network().ok(),
        })
    }
}
