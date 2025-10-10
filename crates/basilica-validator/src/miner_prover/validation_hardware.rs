//! Hardware Profile Collection and Parsing Module
//!
//! This module handles the collection and parsing of hardware information
//! from nodes using the `lshw` command. It extracts key hardware metrics
//! including CPU model, core count, RAM, and disk capacity.

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use anyhow::Result;
use basilica_common::ssh::SshConnectionDetails;
use std::sync::Arc;
use tracing::{info, warn};

/// Hardware profile information extracted from lshw
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_model: Option<String>,
    pub cpu_cores: Option<i32>,
    pub ram_gb: Option<i32>,
    pub disk_gb: Option<i32>,
    pub full_json: String,
}

impl HardwareProfile {
    /// Parse lshw JSON output to extract hardware information
    pub fn from_lshw_json(json_str: &str) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse lshw JSON: {}", e))?;

        let mut cpu_model: Option<String> = None;
        let mut cpu_cores: Option<i32> = None;
        let mut ram_gb: Option<i32> = None;
        let mut disk_gb: Option<i32> = None;

        // Find processor nodes and sum all cores
        let mut processor_nodes = Vec::new();
        find_nodes_by_class(&json, "processor", &mut processor_nodes);

        // Get CPU model from first processor
        if let Some(first_processor) = processor_nodes.first() {
            cpu_model = first_processor
                .get("product")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }

        // Sum cores from all processors
        let mut total_cores = 0;
        let mut has_cores_field = false;

        for processor in &processor_nodes {
            if let Some(cores) = processor
                .get("configuration")
                .and_then(|c| c.get("cores"))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<i32>().ok())
            {
                total_cores += cores;
                has_cores_field = true;
            }
        }

        // If no cores field found in configuration, fall back to counting processor nodes
        // This handles virtualized environments where each vCPU appears as a separate processor
        if !has_cores_field && !processor_nodes.is_empty() {
            total_cores = processor_nodes.len() as i32;
        }

        if total_cores > 0 {
            cpu_cores = Some(total_cores);
        }

        // Find memory nodes - only use "System Memory" to avoid double counting
        let mut memory_nodes = Vec::new();
        find_nodes_by_class(&json, "memory", &mut memory_nodes);
        let mut total_memory_bytes: i64 = 0;

        // Look for the "System Memory" node specifically to avoid counting individual banks
        // Note: Some systems use "System memory" (lowercase 'm') so we do case-insensitive comparison
        for memory_node in &memory_nodes {
            if let Some(description) = memory_node.get("description").and_then(|v| v.as_str()) {
                if description.eq_ignore_ascii_case("System Memory") {
                    if let Some(size) = memory_node.get("size").and_then(|v| v.as_i64()) {
                        total_memory_bytes = size;
                        break; // Use only the System Memory total
                    }
                }
            }
        }

        if total_memory_bytes > 0 {
            ram_gb = Some((total_memory_bytes / (1024 * 1024 * 1024)) as i32);
        }

        // Find disk nodes and sum sizes
        let mut disk_nodes = Vec::new();
        find_nodes_by_class(&json, "disk", &mut disk_nodes);
        let mut total_disk_bytes: i64 = 0;

        for disk_node in &disk_nodes {
            if let Some(size) = disk_node.get("size").and_then(|v| v.as_i64()) {
                total_disk_bytes += size;
            } else {
                // For virtual disks without size, try to find volume/partition information
                // Look for children that might be volumes or partitions with size
                if let Some(children) = disk_node.get("children").and_then(|v| v.as_array()) {
                    for child in children {
                        if let Some(size) = child.get("size").and_then(|v| v.as_i64()) {
                            // Check if it's a volume or partition (not another disk)
                            if let Some(class) = child.get("class").and_then(|v| v.as_str()) {
                                if class == "volume" {
                                    total_disk_bytes += size;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also look for volume nodes directly as some systems report them separately
        let mut volume_nodes = Vec::new();
        find_nodes_by_class(&json, "volume", &mut volume_nodes);

        // Only add volumes if we haven't found disk sizes (to avoid double counting)
        if total_disk_bytes == 0 {
            for volume_node in &volume_nodes {
                if let Some(size) = volume_node.get("size").and_then(|v| v.as_i64()) {
                    total_disk_bytes += size;
                }
            }
        }

        if total_disk_bytes > 0 {
            disk_gb = Some((total_disk_bytes / (1024 * 1024 * 1024)) as i32);
        }

        Ok(HardwareProfile {
            cpu_model,
            cpu_cores,
            ram_gb,
            disk_gb,
            full_json: json_str.to_string(),
        })
    }
}

/// Helper function to recursively search for nodes by class in lshw JSON
fn find_nodes_by_class<'a>(
    node: &'a serde_json::Value,
    class_name: &str,
    results: &mut Vec<&'a serde_json::Value>,
) {
    if let Some(class) = node.get("class") {
        if class.as_str() == Some(class_name) {
            results.push(node);
        }
    }
    if let Some(children) = node.get("children") {
        if let Some(arr) = children.as_array() {
            for child in arr {
                find_nodes_by_class(child, class_name, results);
            }
        }
    }
}

/// Hardware collector for gathering hardware profiles from nodes
#[derive(Clone)]
pub struct HardwareCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
}

impl HardwareCollector {
    /// Create a new hardware collector
    pub fn new(ssh_client: Arc<ValidatorSshClient>, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            ssh_client,
            persistence,
        }
    }

    /// Collect hardware profile from node
    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<HardwareProfile> {
        info!(
            node_id = node_id,
            "[HARDWARE_PROFILE] Starting hardware profile collection"
        );

        // Ensure lshw is installed
        self.ssh_client
            .ensure_installed(ssh_details, "lshw", "lshw")
            .await?;

        let lshw_output = self
            .ssh_client
            .execute_command(ssh_details, "lshw -json -quiet -sanitize", true)
            .await?;

        // Parse the output
        let hardware_profile = HardwareProfile::from_lshw_json(&lshw_output)?;

        // Log the collected information
        let cpu_info = format!(
            "{} ({} cores)",
            hardware_profile
                .cpu_model
                .as_deref()
                .unwrap_or("Unknown CPU"),
            hardware_profile.cpu_cores.unwrap_or(0)
        );

        let mem_info = format!(
            "{}GB RAM, {}GB Disk",
            hardware_profile.ram_gb.unwrap_or(0),
            hardware_profile.disk_gb.unwrap_or(0)
        );

        info!(
            node_id = node_id,
            cpu_info = cpu_info,
            mem_info = mem_info,
            "[HARDWARE_PROFILE] Successfully collected hardware profile"
        );

        Ok(hardware_profile)
    }

    /// Store hardware profile in database
    pub async fn store(
        &self,
        miner_uid: u16,
        node_id: &str,
        hardware_profile: &HardwareProfile,
    ) -> Result<()> {
        self.persistence
            .store_node_hardware_profile(
                miner_uid,
                node_id,
                hardware_profile.cpu_model.clone(),
                hardware_profile.cpu_cores,
                hardware_profile.ram_gb,
                hardware_profile.disk_gb,
                &hardware_profile.full_json,
            )
            .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[HARDWARE_PROFILE] Stored hardware profile in database"
        );

        Ok(())
    }

    /// Collect hardware profile from node and store in database
    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<HardwareProfile> {
        let hardware_profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &hardware_profile).await?;
        Ok(hardware_profile)
    }

    /// Collect hardware profile with error handling (non-critical operation)
    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<HardwareProfile> {
        match self.collect(node_id, ssh_details).await {
            Ok(profile) => {
                // Try to store but don't fail if storage fails
                if let Err(e) = self.store(miner_uid, node_id, &profile).await {
                    warn!(
                        miner_uid = miner_uid,
                        node_id = node_id,
                        error = %e,
                        "[HARDWARE_PROFILE] Failed to store hardware profile (non-critical)"
                    );
                }
                Some(profile)
            }
            Err(e) => {
                warn!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %e,
                    "[HARDWARE_PROFILE] Failed to collect hardware profile (non-critical)"
                );
                None
            }
        }
    }

    /// Retrieve hardware profile from database
    pub async fn retrieve(&self, miner_uid: u16, node_id: &str) -> Result<Option<HardwareProfile>> {
        let result = self
            .persistence
            .get_node_hardware_profile(miner_uid, node_id)
            .await?;

        match result {
            Some((full_json, _cpu_model, _cpu_cores, _ram_gb, _disk_gb)) => {
                let profile = HardwareProfile::from_lshw_json(&full_json)?;

                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[HARDWARE_PROFILE] Retrieved hardware profile from database"
                );

                Ok(Some(profile))
            }
            None => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[HARDWARE_PROFILE] No hardware profile found in database"
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    struct TestCase {
        name: &'static str,
        file_name: &'static str,
        expected_cpu_model: Option<&'static str>,
        expected_cpu_cores: Option<i32>,
        expected_ram_gb: Option<i32>,
        expected_disk_gb: Option<i32>,
    }

    #[test]
    fn test_parse_production_lshw() {
        // Build path to test fixture
        let mut fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture_path.push("tests");
        fixture_path.push("fixtures");
        fixture_path.push("lshw");
        fixture_path.push("production_no_cores.json");

        // Read the test file
        let production_json = fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
            panic!(
                "Failed to read test fixture '{}': {}",
                fixture_path.display(),
                e
            )
        });

        println!("\n=== Testing: Production JSON (no cores in config) ===");

        let result = HardwareProfile::from_lshw_json(&production_json);
        assert!(result.is_ok(), "Should parse production JSON successfully");

        let profile = result.unwrap();
        println!("Parsed Hardware Profile for production JSON:");
        println!("  CPU Model: {:?}", profile.cpu_model);
        println!("  CPU Cores: {:?}", profile.cpu_cores);
        println!("  RAM GB: {:?}", profile.ram_gb);
        println!("  Disk GB: {:?}", profile.disk_gb);

        // Verify the fixes work
        assert_eq!(
            profile.cpu_model,
            Some("AMD EPYC 9554 64-Core Processor".to_string()),
            "Should extract CPU model"
        );
        assert_eq!(
            profile.cpu_cores,
            Some(2),
            "Should count 2 processor nodes as fallback when no cores field"
        );
        assert_eq!(
            profile.ram_gb,
            Some(180),
            "Should extract RAM with case-insensitive 'System memory'"
        );
    }

    #[test]
    fn test_parse_lshw_outputs() {
        // Define test cases for different hardware configurations
        let test_cases = vec![
            TestCase {
                name: "A100 SXM5 with 30 vCPUs",
                file_name: "a100_sxm5_30vcpu.json",
                expected_cpu_model: Some("Intel(R) Xeon(R) Platinum 8462Y+"),
                expected_cpu_cores: Some(30),
                expected_ram_gb: Some(120),
                expected_disk_gb: Some(250),
            },
            TestCase {
                name: "AMD EPYC 9554 with B200 GPU and 31 vCPUs",
                file_name: "amd_epyc_b200_31vcpu.json",
                expected_cpu_model: Some("AMD EPYC 9554 64-Core Processor"),
                expected_cpu_cores: Some(14), // 2 processors × 7 cores each
                expected_ram_gb: Some(180),   // 193273528320 bytes = ~180 GB
                expected_disk_gb: Some(100),  // Root volume is 107GB, rounded to 100
            },
        ];

        for test_case in test_cases {
            println!("\n=== Testing: {} ===", test_case.name);

            // Build path to test fixture
            let mut fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            fixture_path.push("tests");
            fixture_path.push("fixtures");
            fixture_path.push("lshw");
            fixture_path.push(test_case.file_name);

            // Read the test file
            let json_content = fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
                panic!(
                    "Failed to read test fixture '{}': {}",
                    fixture_path.display(),
                    e
                )
            });

            // Parse the output
            let result = HardwareProfile::from_lshw_json(&json_content);
            assert!(
                result.is_ok(),
                "[{}] Failed to parse lshw output: {:?}",
                test_case.name,
                result.err()
            );

            let hardware_profile = result.unwrap();

            // Print the parsed values for debugging
            println!("Parsed Hardware Profile for '{}':", test_case.name);
            println!("  CPU Model: {:?}", hardware_profile.cpu_model);
            println!("  CPU Cores: {:?}", hardware_profile.cpu_cores);
            println!("  RAM GB: {:?}", hardware_profile.ram_gb);
            println!("  Disk GB: {:?}", hardware_profile.disk_gb);

            // Verify CPU model
            if let Some(expected_model) = test_case.expected_cpu_model {
                assert_eq!(
                    hardware_profile.cpu_model,
                    Some(expected_model.to_string()),
                    "[{}] CPU model mismatch",
                    test_case.name
                );
            }

            // Verify CPU cores
            if let Some(expected_cores) = test_case.expected_cpu_cores {
                assert_eq!(
                    hardware_profile.cpu_cores,
                    Some(expected_cores),
                    "[{}] CPU cores mismatch",
                    test_case.name
                );
            }

            // Verify RAM
            if let Some(expected_ram) = test_case.expected_ram_gb {
                assert_eq!(
                    hardware_profile.ram_gb,
                    Some(expected_ram),
                    "[{}] RAM mismatch",
                    test_case.name
                );
            }

            // Verify Disk (if expected)
            match (test_case.expected_disk_gb, hardware_profile.disk_gb) {
                (Some(expected), Some(actual)) => {
                    assert_eq!(actual, expected, "[{}] Disk size mismatch", test_case.name);
                }
                (Some(expected), None) => {
                    panic!(
                        "[{}] Expected {} GB disk but found none",
                        test_case.name, expected
                    );
                }
                (None, Some(actual)) => {
                    println!(
                        "[{}] Found disk with {} GB (not expected in test case)",
                        test_case.name, actual
                    );
                }
                (None, None) => {
                    println!("[{}] No disk found (as expected)", test_case.name);
                }
            }
        }
    }
}
