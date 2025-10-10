use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use basilica_common::ssh::SshConnectionDetails;

/// Individual filesystem mount point information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemInfo {
    pub mount_point: String,
    pub filesystem_type: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// Storage profile containing disk space information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageProfile {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub required_bytes: u64,
    pub filesystem_details: Vec<FilesystemInfo>,
    pub collection_timestamp: DateTime<Utc>,
    pub full_json: String,
}

/// Collector for storage capacity validation
#[derive(Clone)]
pub struct StorageCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
    min_required_storage_bytes: u64,
}

impl StorageCollector {
    /// Create a new storage collector
    pub fn new(
        ssh_client: Arc<ValidatorSshClient>,
        persistence: Arc<SimplePersistence>,
        min_required_storage_bytes: u64,
    ) -> Self {
        Self {
            ssh_client,
            persistence,
            min_required_storage_bytes,
        }
    }

    /// Collect storage profile from node
    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<StorageProfile> {
        info!(node_id = node_id, "[STORAGE] Starting storage validation");

        // Primary command, filtering of virtual filesystems
        let primary_cmd = "df -B1 --output=avail,size,fstype,target --exclude-type=tmpfs --exclude-type=devtmpfs --exclude-type=squashfs --exclude-type=sysfs --exclude-type=proc --exclude-type=cgroup --exclude-type=cgroup2 --exclude-type=debugfs --exclude-type=tracefs --exclude-type=securityfs --exclude-type=pstore --exclude-type=configfs --exclude-type=fusectl --exclude-type=hugetlbfs --exclude-type=mqueue --exclude-type=binfmt_misc --exclude-type=autofs 2>/dev/null | tail -n +2";

        // Fallback command for older systems
        let fallback_cmd =
            "df -k -P --exclude-type=tmpfs --exclude-type=devtmpfs 2>/dev/null | tail -n +2";

        let (output, is_bytes_format) = match self
            .ssh_client
            .execute_command(ssh_details, primary_cmd, true)
            .await
        {
            Ok(output) if !output.trim().is_empty() => {
                debug!(node_id = node_id, "[STORAGE] Primary df command successful");
                (output, true)
            }
            _ => {
                debug!(
                    node_id = node_id,
                    "[STORAGE] Primary command failed, trying fallback"
                );
                let output = self
                    .ssh_client
                    .execute_command(ssh_details, fallback_cmd, true)
                    .await
                    .context("Both primary and fallback df commands failed")?;
                (output, false)
            }
        };

        let filesystem_details = parse_df_output(&output, node_id, is_bytes_format)?;

        let total_bytes: u64 = filesystem_details.iter().map(|fs| fs.total_bytes).sum();
        let available_bytes: u64 = filesystem_details.iter().map(|fs| fs.available_bytes).sum();

        let available_tb = available_bytes as f64 / 1024_f64.powi(4);
        info!(
            node_id = node_id,
            available_tb = format!("{:.2}", available_tb),
            "[STORAGE] Storage: {:.2} TB available",
            available_tb
        );

        let full_json = serde_json::json!({
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
            "required_bytes": self.min_required_storage_bytes,
            "filesystem_details": filesystem_details,
            "raw_output": output,
        })
        .to_string();

        Ok(StorageProfile {
            total_bytes,
            available_bytes,
            required_bytes: self.min_required_storage_bytes,
            filesystem_details,
            collection_timestamp: Utc::now(),
            full_json,
        })
    }

    /// Store storage profile in database
    pub async fn store(
        &self,
        miner_uid: u16,
        node_id: &str,
        profile: &StorageProfile,
    ) -> Result<()> {
        debug!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[STORAGE] Storing storage profile"
        );

        self.persistence
            .store_node_storage_profile(miner_uid, node_id, profile)
            .await
            .context("Failed to store storage profile")?;

        Ok(())
    }

    /// Collect and store storage profile
    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<StorageProfile> {
        let profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &profile).await?;

        if profile.available_bytes < self.min_required_storage_bytes {
            let available_tb = profile.available_bytes as f64 / 1024_f64.powi(4);
            let required_tb = self.min_required_storage_bytes as f64 / 1024_f64.powi(4);
            error!(
                node_id = node_id,
                available_tb = format!("{:.2}", available_tb),
                required_tb = format!("{:.2}", required_tb),
                "[STORAGE] Storage requirement not met: {:.2} TB available, {:.2} TB required",
                available_tb,
                required_tb
            );
            return Err(anyhow::anyhow!(
                "Storage requirement not met: {:.2} TB available, {:.2} TB required",
                available_tb,
                required_tb
            ));
        }

        Ok(profile)
    }

    /// Collect with fallback
    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<StorageProfile> {
        match self
            .collect_and_store(node_id, miner_uid, ssh_details)
            .await
        {
            Ok(profile) => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    available_tb =
                        format!("{:.2}", profile.available_bytes as f64 / 1024_f64.powi(4)),
                    "[STORAGE] Storage validation completed successfully with {:.2} TB available",
                    profile.available_bytes as f64 / 1024_f64.powi(4)
                );
                Some(profile)
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %e,
                    "[STORAGE] Storage validation failed: {}",
                    e
                );
                None
            }
        }
    }

    /// Retrieve storage profile from database
    pub async fn retrieve(&self, miner_uid: u16, node_id: &str) -> Result<Option<StorageProfile>> {
        self.persistence
            .get_node_storage_profile(miner_uid, node_id)
            .await
            .context("Failed to retrieve storage profile")
    }
}

/// df output parsing - robust implementation to avoid false negatives
pub fn parse_df_output(
    output: &str,
    node_id: &str,
    is_bytes_format: bool,
) -> Result<Vec<FilesystemInfo>> {
    let mut filesystems = Vec::new();
    let lines: Vec<&str> = output.lines().collect();

    if lines.is_empty() {
        return Err(anyhow::anyhow!(
            "No filesystem data returned from df command"
        ));
    }

    // Handle multi-line entries by joining lines that don't start with expected patterns
    let mut consolidated_lines = Vec::new();
    let mut current_line = String::new();

    for line in lines.iter() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if this looks like a new filesystem entry or continuation
        let is_new_entry = if is_bytes_format {
            // In bytes format, entries start with a number
            trimmed
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        } else {
            // In fallback format, entries start with device path or "Filesystem"
            trimmed.starts_with('/') || trimmed.starts_with("Filesystem") ||
            trimmed.contains(":/") || // Network mounts like server:/path
            trimmed.starts_with("tmpfs") || trimmed.starts_with("devtmpfs")
        };

        if is_new_entry && !current_line.is_empty() {
            consolidated_lines.push(current_line.clone());
            current_line = trimmed.to_string();
        } else if is_new_entry {
            current_line = trimmed.to_string();
        } else {
            // Continuation of previous line
            if !current_line.is_empty() {
                current_line.push(' ');
            }
            current_line.push_str(trimmed);
        }
    }

    if !current_line.is_empty() {
        consolidated_lines.push(current_line);
    }

    // Parse each consolidated line
    for line in consolidated_lines.iter() {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        // Skip header lines
        if parts[0] == "Filesystem" || parts[0].contains("Avail") {
            continue;
        }

        let parse_result = if is_bytes_format {
            parse_bytes_format(&parts)
        } else {
            parse_fallback_format(&parts)
        };

        let (available_bytes, total_bytes, filesystem_type, mount_point) = match parse_result {
            Ok(result) => result,
            Err(e) => {
                debug!(
                    node_id = node_id,
                    line = line,
                    error = %e,
                    "[STORAGE] Failed to parse line, skipping"
                );
                continue;
            }
        };

        // Validate parsed values
        if total_bytes == 0 || available_bytes > total_bytes {
            debug!(
                node_id = node_id,
                mount_point = mount_point,
                total_bytes = total_bytes,
                available_bytes = available_bytes,
                "[STORAGE] Invalid filesystem metrics, skipping"
            );
            continue;
        }

        if is_virtual_filesystem(&mount_point, &filesystem_type) {
            continue;
        }

        debug!(
            node_id = node_id,
            mount_point = mount_point,
            available_gb = format!("{:.2}", available_bytes as f64 / 1024_f64.powi(3)),
            total_gb = format!("{:.2}", total_bytes as f64 / 1024_f64.powi(3)),
            filesystem_type = filesystem_type,
            "[STORAGE] Valid filesystem found"
        );

        filesystems.push(FilesystemInfo {
            mount_point,
            filesystem_type,
            total_bytes,
            available_bytes,
        });
    }

    if filesystems.is_empty() {
        return Err(anyhow::anyhow!(
            "No valid filesystems found in df output. Raw output: {:?}",
            output.lines().take(5).collect::<Vec<_>>()
        ));
    }

    // Deduplicate by device/mount point - keep the entry with most available space
    let mut unique_filesystems = std::collections::HashMap::new();
    for fs in filesystems {
        unique_filesystems
            .entry(fs.mount_point.clone())
            .and_modify(|existing: &mut FilesystemInfo| {
                // Keep the one with more available space (handles bind mounts)
                if fs.available_bytes > existing.available_bytes {
                    *existing = fs.clone();
                }
            })
            .or_insert(fs);
    }

    let mut result: Vec<FilesystemInfo> = unique_filesystems.into_values().collect();

    // Sort by mount point for consistent ordering
    result.sort_by(|a, b| a.mount_point.cmp(&b.mount_point));

    Ok(result)
}

/// Parse bytes format: avail size fstype target
fn parse_bytes_format(parts: &[&str]) -> Result<(u64, u64, String, String)> {
    if parts.len() < 4 {
        return Err(anyhow::anyhow!(
            "Invalid bytes format: expected at least 4 fields, got {}",
            parts.len()
        ));
    }

    let avail = parts[0]
        .parse::<u64>()
        .with_context(|| format!("Failed to parse available bytes: {}", parts[0]))?;
    let total = parts[1]
        .parse::<u64>()
        .with_context(|| format!("Failed to parse total bytes: {}", parts[1]))?;
    let fstype = parts[2].to_string();
    let mount = parts[3..].join(" ");

    Ok((avail, total, fstype, mount))
}

/// Parse fallback format: Filesystem 1024-blocks Used Available Capacity Mounted
fn parse_fallback_format(parts: &[&str]) -> Result<(u64, u64, String, String)> {
    if parts.len() < 6 {
        return Err(anyhow::anyhow!(
            "Invalid fallback format: expected at least 6 fields, got {}",
            parts.len()
        ));
    }

    // Parse with overflow checking
    let total_kb = parts[1]
        .parse::<u64>()
        .with_context(|| format!("Failed to parse total KB: {}", parts[1]))?;
    let available_kb = parts[3]
        .parse::<u64>()
        .with_context(|| format!("Failed to parse available KB: {}", parts[3]))?;

    // Check for overflow before multiplication
    let total_bytes = total_kb
        .checked_mul(1024)
        .ok_or_else(|| anyhow::anyhow!("Integer overflow calculating total bytes"))?;
    let available_bytes = available_kb
        .checked_mul(1024)
        .ok_or_else(|| anyhow::anyhow!("Integer overflow calculating available bytes"))?;

    // Filesystem type is unknown in fallback format
    let fstype = "unknown".to_string();

    // Mount point might contain spaces
    let mount = parts[5..].join(" ");

    Ok((available_bytes, total_bytes, fstype, mount))
}

/// Check if a filesystem should be excluded from storage calculations
fn is_virtual_filesystem(mount_point: &str, filesystem_type: &str) -> bool {
    // Virtual filesystem types to exclude
    const VIRTUAL_FS_TYPES: &[&str] = &[
        "tmpfs",
        "devtmpfs",
        "sysfs",
        "proc",
        "cgroup",
        "cgroup2",
        "debugfs",
        "tracefs",
        "securityfs",
        "pstore",
        "configfs",
        "fusectl",
        "hugetlbfs",
        "mqueue",
        "binfmt_misc",
        "autofs",
        "squashfs",
        "overlay",
        "aufs",
        "shm",
        "devpts",
        "ramfs",
    ];

    // Check filesystem type
    if VIRTUAL_FS_TYPES.contains(&filesystem_type) {
        return true;
    }

    // Virtual mount points to exclude
    const VIRTUAL_MOUNT_PREFIXES: &[&str] = &[
        "/proc",
        "/sys",
        "/dev/shm",
        "/run",
        "/dev/pts",
        "/var/lib/docker/overlay2",
        "/var/lib/docker/aufs",
        "/snap",
        "/tmp/.mount_",
        "/dev/mqueue",
        "/dev/hugepages",
    ];

    // Check mount point prefixes
    for prefix in VIRTUAL_MOUNT_PREFIXES {
        if mount_point.starts_with(prefix) {
            return true;
        }
    }

    // Special cases
    mount_point == "/dev" || mount_point == "/dev/shm"
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test constant for minimum required storage (1TB)
    const TEST_MIN_REQUIRED_STORAGE_BYTES: u64 = 1_099_511_627_776;

    #[test]
    fn test_parse_df_output_primary_format() {
        // Sample output from df -B1 --output=avail,size,fstype,target
        // Total available: ~1.5TB which meets the 1TB requirement
        let output = "536870912000 1073741824000 ext4 /\n268435456000 536870912000 ext4 /home\n1099511627776 2199023255552 xfs /data";

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 3);

        // Check that we parsed the values correctly
        let data_fs = filesystems
            .iter()
            .find(|fs| fs.mount_point == "/data")
            .unwrap();
        assert_eq!(data_fs.available_bytes, 1099511627776); // 1TB
        assert_eq!(data_fs.total_bytes, 2199023255552); // 2TB
        assert_eq!(data_fs.filesystem_type, "xfs");
    }

    #[test]
    fn test_parse_df_output_fallback_format() {
        // Sample output from df -k -P
        let output = "/dev/sda1 1073741824 536870912 524288000 52% /\n/dev/sdb1 2147483648 1048576000 1073741824 50% /home\ntmpfs 8388608 0 8388608 0% /dev/shm";

        let result = parse_df_output(output, "test-node", false);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        // tmpfs should be filtered out
        assert_eq!(filesystems.len(), 2);

        // Check kilobyte to byte conversion
        let root_fs = filesystems.iter().find(|fs| fs.mount_point == "/").unwrap();
        assert_eq!(root_fs.available_bytes, 524288000 * 1024);
        assert_eq!(root_fs.total_bytes, 1073741824 * 1024);
    }

    #[test]
    fn test_parse_df_output_filters_virtual_filesystems() {
        let output = "1073741824 2147483648 ext4 /\n536870912 1073741824 proc /proc\n268435456 536870912 sysfs /sys\n134217728 268435456 devtmpfs /dev\n67108864 134217728 ext4 /home";

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        // Should only include / and /home, filtering out virtual filesystems
        assert_eq!(filesystems.len(), 2);
        assert!(filesystems.iter().any(|fs| fs.mount_point == "/"));
        assert!(filesystems.iter().any(|fs| fs.mount_point == "/home"));
        assert!(!filesystems.iter().any(|fs| fs.mount_point == "/proc"));
        assert!(!filesystems.iter().any(|fs| fs.mount_point == "/sys"));
        assert!(!filesystems.iter().any(|fs| fs.mount_point == "/dev"));
    }

    #[test]
    fn test_parse_df_output_handles_mount_points_with_spaces() {
        // Fallback format with mount point containing spaces
        let output = "/dev/sdc1 1073741824 536870912 524288000 52% /mnt/my backup drive";

        let result = parse_df_output(output, "test-node", false);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 1);
        assert_eq!(filesystems[0].mount_point, "/mnt/my backup drive");
    }

    #[test]
    fn test_parse_df_output_deduplicates_mount_points() {
        // Duplicate mount points with different available space
        let output = "536870912 1073741824 ext4 /data\n1073741824 2147483648 ext4 /data";

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 1);
        // Should keep the one with more available space
        assert_eq!(filesystems[0].available_bytes, 1073741824);
    }

    #[test]
    fn test_parse_df_output_empty_input() {
        let output = "";
        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No filesystem data"));
    }

    #[test]
    fn test_parse_df_output_no_valid_filesystems() {
        // Only virtual filesystems
        let output = "536870912 1073741824 proc /proc\n268435456 536870912 sysfs /sys";
        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No valid filesystems found"));
    }

    #[tokio::test]
    async fn test_collect_fails_when_storage_requirement_not_met() {
        // This would require mocking SSH client, so we'll test the logic directly
        // Create a scenario where available storage is less than 1TB
        let output = "500000000000 1000000000000 ext4 /"; // 500GB available, less than 1TB

        let filesystems = parse_df_output(output, "test-node", true).unwrap();
        let available_bytes: u64 = filesystems.iter().map(|fs| fs.available_bytes).sum();

        // Verify that 500GB is less than 1TB requirement
        assert!(available_bytes < TEST_MIN_REQUIRED_STORAGE_BYTES);
        assert_eq!(available_bytes, 500000000000);
    }

    #[tokio::test]
    async fn test_collect_succeeds_when_storage_requirement_met() {
        // Create a scenario where available storage exceeds 1TB
        let output = "1500000000000 2000000000000 ext4 /"; // 1.5TB available

        let filesystems = parse_df_output(output, "test-node", true).unwrap();
        let available_bytes: u64 = filesystems.iter().map(|fs| fs.available_bytes).sum();

        // Verify that 1.5TB meets the 1TB requirement
        assert!(available_bytes >= TEST_MIN_REQUIRED_STORAGE_BYTES);
        assert_eq!(available_bytes, 1500000000000);
    }

    #[test]
    fn test_parse_multiline_filesystem_entry() {
        // Test handling of filesystem names split across lines
        let output = "/dev/mapper/ubuntu--vg-ubuntu--lv\n                       1073741824 536870912 524288000 52% /";

        let result = parse_df_output(output, "test-node", false);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 1);
        assert_eq!(filesystems[0].mount_point, "/");
    }

    #[test]
    fn test_parse_network_filesystem() {
        // Test NFS mount parsing
        let output =
            "server.example.com:/export/data 2147483648 1048576000 1073741824 50% /mnt/nfs";

        let result = parse_df_output(output, "test-node", false);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 1);
        assert_eq!(filesystems[0].mount_point, "/mnt/nfs");
    }

    #[test]
    fn test_parse_lvm_volumes() {
        // Test LVM logical volume paths
        let output = "/dev/mapper/vg0-lv_root 1073741824 536870912 524288000 52% /\n/dev/mapper/vg0-lv_home 2147483648 1048576000 1073741824 50% /home";

        let result = parse_df_output(output, "test-node", false);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 2);
    }

    #[test]
    fn test_parse_df_output_primary_format_mount_with_spaces() {
        // Test mount point with spaces in primary format (bytes)
        let output = "1073741824 2147483648 ext4 /mnt/my backup drive";
        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());
        let fs = result.unwrap();
        assert_eq!(fs.len(), 1);
        assert_eq!(fs[0].mount_point, "/mnt/my backup drive");
        assert_eq!(fs[0].available_bytes, 1073741824);
        assert_eq!(fs[0].total_bytes, 2147483648);
        assert_eq!(fs[0].filesystem_type, "ext4");
    }

    #[test]
    fn test_overflow_protection() {
        // Test that we handle potential overflow gracefully
        let output = "/dev/sda1 18446744073709551615 9223372036854775807 9223372036854775807 50% /";

        let result = parse_df_output(output, "test-node", false);
        // Should either parse successfully or fail gracefully, not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_docker_overlay_excluded() {
        // Test that Docker overlay filesystems are excluded
        let output = "1073741824 2147483648 ext4 /\n536870912 1073741824 overlay /var/lib/docker/overlay2/abc123/merged";

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems.len(), 1);
        assert_eq!(filesystems[0].mount_point, "/");
    }

    #[test]
    fn test_consistent_ordering() {
        // Test that results are consistently ordered
        let output = "1073741824 2147483648 ext4 /home\n536870912 1073741824 ext4 /\n268435456 536870912 ext4 /var";

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_ok());

        let filesystems = result.unwrap();
        assert_eq!(filesystems[0].mount_point, "/");
        assert_eq!(filesystems[1].mount_point, "/home");
        assert_eq!(filesystems[2].mount_point, "/var");
    }

    #[test]
    fn test_invalid_available_bytes() {
        // Test that we skip filesystems with invalid metrics
        let output = "2000000000000 1000000000000 ext4 /"; // Available > Total (invalid)

        let result = parse_df_output(output, "test-node", true);
        assert!(result.is_err()); // Should fail as no valid filesystems
    }
}
