//! OS performance benchmarking implementation

use anyhow::{Context, Result};
use std::fs;
use std::process;
use std::time::Instant;

use super::types::OsPerformanceMetrics;

pub struct OsBenchmarker;

impl OsBenchmarker {
    pub fn run_benchmarks() -> Result<OsPerformanceMetrics> {
        let start = Instant::now();
        let initial_stats = Self::read_proc_stat()?;

        std::thread::sleep(std::time::Duration::from_millis(100));

        let final_stats = Self::read_proc_stat()?;
        let duration = start.elapsed().as_secs_f64();

        Ok(OsPerformanceMetrics {
            context_switches_per_second: Self::calculate_context_switches(
                &initial_stats,
                &final_stats,
                duration,
            )?,
            interrupts_per_second: Self::calculate_interrupts(
                &initial_stats,
                &final_stats,
                duration,
            )?,
            cpu_idle_percentage: Self::calculate_cpu_idle(&initial_stats, &final_stats)?,
            memory_fragmentation: Self::calculate_memory_fragmentation()?,
            io_wait_percentage: Self::calculate_io_wait(&initial_stats, &final_stats)?,
            system_call_latency_us: Self::measure_syscall_latency()?,
            scheduler_latency_us: Self::measure_scheduler_latency()?,
            filesystem_latency_us: Self::measure_filesystem_latency()?,
        })
    }

    pub fn quick_benchmark() -> Result<OsPerformanceMetrics> {
        Ok(OsPerformanceMetrics {
            context_switches_per_second: Self::estimate_context_switches()?,
            interrupts_per_second: Self::estimate_interrupts()?,
            cpu_idle_percentage: Self::get_current_cpu_idle()?,
            memory_fragmentation: Self::calculate_memory_fragmentation()?,
            io_wait_percentage: Self::get_current_io_wait()?,
            system_call_latency_us: Self::measure_syscall_latency()?,
            scheduler_latency_us: 0.0,
            filesystem_latency_us: 0.0,
        })
    }

    fn read_proc_stat() -> Result<ProcStat> {
        let content = fs::read_to_string("/proc/stat").context("Failed to read /proc/stat")?;

        let mut stat = ProcStat::default();

        for line in content.lines() {
            if line.starts_with("ctxt ") {
                stat.context_switches = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("intr ") {
                stat.interrupts = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("cpu ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 8 {
                    stat.cpu_user = parts[1].parse().unwrap_or(0);
                    stat.cpu_nice = parts[2].parse().unwrap_or(0);
                    stat.cpu_system = parts[3].parse().unwrap_or(0);
                    stat.cpu_idle = parts[4].parse().unwrap_or(0);
                    stat.cpu_iowait = parts[5].parse().unwrap_or(0);
                    stat.cpu_irq = parts[6].parse().unwrap_or(0);
                    stat.cpu_softirq = parts[7].parse().unwrap_or(0);
                }
            }
        }

        Ok(stat)
    }

    fn calculate_context_switches(
        initial: &ProcStat,
        final_stat: &ProcStat,
        duration: f64,
    ) -> Result<f64> {
        let diff = final_stat
            .context_switches
            .saturating_sub(initial.context_switches);
        Ok(diff as f64 / duration)
    }

    fn calculate_interrupts(
        initial: &ProcStat,
        final_stat: &ProcStat,
        duration: f64,
    ) -> Result<f64> {
        let diff = final_stat.interrupts.saturating_sub(initial.interrupts);
        Ok(diff as f64 / duration)
    }

    fn calculate_cpu_idle(initial: &ProcStat, final_stat: &ProcStat) -> Result<f64> {
        let initial_total = initial.cpu_total();
        let final_total = final_stat.cpu_total();

        let total_diff = final_total.saturating_sub(initial_total);
        let idle_diff = final_stat.cpu_idle.saturating_sub(initial.cpu_idle);

        if total_diff == 0 {
            return Ok(0.0);
        }

        Ok((idle_diff as f64 / total_diff as f64) * 100.0)
    }

    fn calculate_io_wait(initial: &ProcStat, final_stat: &ProcStat) -> Result<f64> {
        let initial_total = initial.cpu_total();
        let final_total = final_stat.cpu_total();

        let total_diff = final_total.saturating_sub(initial_total);
        let iowait_diff = final_stat.cpu_iowait.saturating_sub(initial.cpu_iowait);

        if total_diff == 0 {
            return Ok(0.0);
        }

        Ok((iowait_diff as f64 / total_diff as f64) * 100.0)
    }

    fn calculate_memory_fragmentation() -> Result<f64> {
        let content =
            fs::read_to_string("/proc/buddyinfo").context("Failed to read /proc/buddyinfo")?;

        let mut total_free_pages = 0u64;
        let mut large_free_pages = 0u64;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 11 {
                for (i, &part) in parts[4..].iter().enumerate() {
                    if let Ok(count) = part.parse::<u64>() {
                        let pages = count * (1u64 << i);
                        total_free_pages += pages;
                        if i >= 8 {
                            large_free_pages += pages;
                        }
                    }
                }
            }
        }

        if total_free_pages == 0 {
            return Ok(0.0);
        }

        let fragmentation = 100.0 - ((large_free_pages as f64 / total_free_pages as f64) * 100.0);
        Ok(fragmentation.clamp(0.0, 100.0))
    }

    fn measure_syscall_latency() -> Result<f64> {
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = process::id();
        }

        let duration = start.elapsed();
        Ok(duration.as_micros() as f64 / iterations as f64)
    }

    fn measure_scheduler_latency() -> Result<f64> {
        let iterations = 100;
        let mut total_latency = 0.0;

        for _ in 0..iterations {
            let start = Instant::now();
            std::thread::yield_now();
            let duration = start.elapsed();
            total_latency += duration.as_micros() as f64;
        }

        Ok(total_latency / iterations as f64)
    }

    fn measure_filesystem_latency() -> Result<f64> {
        let test_file = "/tmp/gpu_attestor_fs_latency_test";
        let test_data = b"test_data_for_filesystem_latency_measurement";
        let iterations = 100;
        let mut total_latency = 0.0;

        for _ in 0..iterations {
            let start = Instant::now();
            fs::write(test_file, test_data).context("Failed to write test file")?;
            let _ = fs::read(test_file);
            fs::remove_file(test_file).ok();
            let duration = start.elapsed();
            total_latency += duration.as_micros() as f64;
        }

        Ok(total_latency / iterations as f64)
    }

    fn estimate_context_switches() -> Result<f64> {
        let content = fs::read_to_string("/proc/stat").context("Failed to read /proc/stat")?;

        for line in content.lines() {
            if line.starts_with("ctxt ") {
                let count: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return Ok(count as f64 / 100.0);
            }
        }
        Ok(0.0)
    }

    fn estimate_interrupts() -> Result<f64> {
        let content = fs::read_to_string("/proc/stat").context("Failed to read /proc/stat")?;

        for line in content.lines() {
            if line.starts_with("intr ") {
                let count: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return Ok(count as f64 / 100.0);
            }
        }
        Ok(0.0)
    }

    fn get_current_cpu_idle() -> Result<f64> {
        let stat = Self::read_proc_stat()?;
        let total = stat.cpu_total();

        if total == 0 {
            return Ok(0.0);
        }

        Ok((stat.cpu_idle as f64 / total as f64) * 100.0)
    }

    fn get_current_io_wait() -> Result<f64> {
        let stat = Self::read_proc_stat()?;
        let total = stat.cpu_total();

        if total == 0 {
            return Ok(0.0);
        }

        Ok((stat.cpu_iowait as f64 / total as f64) * 100.0)
    }
}

#[derive(Debug, Default)]
struct ProcStat {
    context_switches: u64,
    interrupts: u64,
    cpu_user: u64,
    cpu_nice: u64,
    cpu_system: u64,
    cpu_idle: u64,
    cpu_iowait: u64,
    cpu_irq: u64,
    cpu_softirq: u64,
}

impl ProcStat {
    fn cpu_total(&self) -> u64 {
        self.cpu_user
            + self.cpu_nice
            + self.cpu_system
            + self.cpu_idle
            + self.cpu_iowait
            + self.cpu_irq
            + self.cpu_softirq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_benchmarks() {
        let result = OsBenchmarker::run_benchmarks();
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.context_switches_per_second >= 0.0);
        assert!(metrics.interrupts_per_second >= 0.0);
        assert!(metrics.cpu_idle_percentage >= 0.0);
        assert!(metrics.cpu_idle_percentage <= 100.0);
    }

    #[test]
    fn test_quick_benchmark() {
        let result = OsBenchmarker::quick_benchmark();
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.context_switches_per_second >= 0.0);
        assert!(metrics.cpu_idle_percentage >= 0.0);
    }

    #[test]
    fn test_syscall_latency() {
        let result = OsBenchmarker::measure_syscall_latency();
        assert!(result.is_ok());

        let latency = result.unwrap();
        assert!(latency > 0.0);
        assert!(latency < 1000.0);
    }

    #[test]
    fn test_memory_fragmentation() {
        let result = OsBenchmarker::calculate_memory_fragmentation();
        if let Ok(fragmentation) = result {
            assert!(fragmentation >= 0.0);
            assert!(fragmentation <= 100.0);
        }
    }
}
