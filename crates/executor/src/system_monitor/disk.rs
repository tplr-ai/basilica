//! Disk monitoring functionality

use super::types::DiskInfo;
use anyhow::Result;
use sysinfo::Disks;

/// Disk monitoring handler
#[derive(Debug)]
pub struct DiskMonitor;

impl DiskMonitor {
    /// Create new disk monitor
    pub fn new() -> Self {
        Self
    }

    /// Get disk information
    pub fn get_disk_info(&self) -> Result<Vec<DiskInfo>> {
        let mut disks = Vec::new();

        // For sysinfo 0.30+, disks are accessed via Disks struct
        let disk_manager = Disks::new_with_refreshed_list();
        for disk in &disk_manager {
            let total = disk.total_space();
            let available = disk.available_space();
            let used = total - available;
            let usage_percent = if total > 0 {
                (used as f32 / total as f32) * 100.0
            } else {
                0.0
            };

            disks.push(DiskInfo {
                name: disk.name().to_string_lossy().to_string(),
                mount_point: disk.mount_point().to_string_lossy().to_string(),
                total_bytes: total,
                used_bytes: used,
                available_bytes: available,
                usage_percent,
                filesystem: format!("{:?}", disk.file_system()),
            });
        }

        Ok(disks)
    }
}

impl Default for DiskMonitor {
    fn default() -> Self {
        Self::new()
    }
}
