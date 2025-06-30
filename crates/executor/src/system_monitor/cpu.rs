//! CPU monitoring functionality

use super::types::CpuInfo;
use anyhow::Result;
use sysinfo::System;

/// CPU monitoring handler
#[derive(Debug)]
pub struct CpuMonitor;

impl CpuMonitor {
    /// Create new CPU monitor
    pub fn new() -> Self {
        Self
    }

    /// Get CPU information
    pub fn get_cpu_info(&self, system: &System) -> Result<CpuInfo> {
        let cpu = system.global_cpu_info();

        Ok(CpuInfo {
            usage_percent: cpu.cpu_usage(),
            cores: system.cpus().len(),
            frequency_mhz: cpu.frequency(),
            model: cpu.brand().to_string(),
            vendor: cpu.vendor_id().to_string(),
            temperature_celsius: self.get_cpu_temperature(system),
        })
    }

    /// Get CPU temperature from thermal sensors
    fn get_cpu_temperature(&self, _system: &System) -> Option<f32> {
        // Try to read CPU temperature from common Linux thermal sensors
        // This is a simple implementation that reads from /sys/class/thermal
        use std::fs;

        // Common thermal zone paths for CPU temperature
        let thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp", // Usually CPU on most systems
            "/sys/class/thermal/thermal_zone1/temp", // Alternative CPU zone
            "/sys/devices/platform/coretemp.0/hwmon/hwmon1/temp1_input", // Intel Core
            "/sys/devices/platform/k10temp.0/hwmon/hwmon0/temp1_input", // AMD
        ];

        for path in &thermal_paths {
            if let Ok(temp_str) = fs::read_to_string(path) {
                if let Ok(temp_millidegrees) = temp_str.trim().parse::<f32>() {
                    // Convert from millidegrees to degrees Celsius
                    let temp_celsius = temp_millidegrees / 1000.0;

                    // Validate temperature is reasonable (between 0°C and 150°C)
                    if temp_celsius > 0.0 && temp_celsius < 150.0 {
                        return Some(temp_celsius);
                    }
                }
            }
        }

        // Temperature reading not available or not supported
        None
    }
}

impl Default for CpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}
