//! Memory monitoring functionality

use super::types::MemoryInfo;
use anyhow::Result;
use sysinfo::System;

/// Memory monitoring handler
#[derive(Debug)]
pub struct MemoryMonitor;

impl MemoryMonitor {
    /// Create new memory monitor
    pub fn new() -> Self {
        Self
    }

    /// Get memory information
    pub fn get_memory_info(&self, system: &System) -> Result<MemoryInfo> {
        let total = system.total_memory();
        let used = system.used_memory();
        let available = system.available_memory();
        let usage_percent = (used as f32 / total as f32) * 100.0;

        Ok(MemoryInfo {
            total_bytes: total,
            used_bytes: used,
            available_bytes: available,
            usage_percent,
            swap_total_bytes: system.total_swap(),
            swap_used_bytes: system.used_swap(),
        })
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}
