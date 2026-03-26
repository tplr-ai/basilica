use std::collections::BTreeMap;

use crate::miner_prover::types::NodeResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeProfileSpec {
    pub provider: String,
    pub region: String,
    pub gpu: NodeGpu,
    pub cpu: NodeCpu,
    pub memory_gb: u32,
    pub storage_gb: u32,
    pub network_gbps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeGpu {
    pub model: String,
    pub count: u32,
    pub memory_gb: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCpu {
    pub model: String,
    pub cores: u32,
}

#[derive(Debug, Clone)]
pub struct NodeProfileInput<'a> {
    pub provider: &'a str,
    pub region: &'a str,
    pub node_result: &'a NodeResult,
}

pub fn to_node_profile_spec(input: &NodeProfileInput<'_>) -> NodeProfileSpec {
    let nr = input.node_result;
    let gpu_count = nr.gpu_infos.len() as u32;
    let gpu_model = nr
        .gpu_infos
        .first()
        .map(|g| g.gpu_name.clone())
        .unwrap_or_else(|| nr.gpu_name.clone());
    let gpu_mem = nr
        .gpu_infos
        .first()
        .map(|g| g.gpu_memory_gb as u32)
        .unwrap_or(0);
    let memory_gb = nr.memory_info.total_gb as u32;
    let cpu = NodeCpu {
        model: nr.cpu_info.model.clone(),
        cores: nr.cpu_info.cores,
    };
    let gpu = NodeGpu {
        model: gpu_model,
        count: gpu_count,
        memory_gb: gpu_mem,
    };
    NodeProfileSpec {
        provider: input.provider.to_string(),
        region: input.region.to_string(),
        gpu,
        cpu,
        memory_gb,
        storage_gb: 0,
        network_gbps: 1,
    }
}

/// Determine which node group a node should be assigned to based on strategy
pub fn assign_node_group(node_id: &str, config: &crate::config::NodeGroupConfig) -> &'static str {
    // Strategy 1: Force override if specified in config
    if let Some(ref force_group) = config.force_group {
        return match force_group.as_str() {
            "jobs" => "jobs",
            "rentals" => "rentals",
            _ => "rentals", // default
        };
    }

    // Strategy 2: Use configured strategy
    match config.strategy.as_str() {
        "round-robin" => {
            // Use hash of node_id to deterministically assign groups
            // This ensures consistent assignment across restarts
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            node_id.hash(&mut hasher);
            let hash = hasher.finish();

            // Use configured percentage split
            let jobs_percentage = config.jobs_percentage.min(100);

            if (hash % 100) < jobs_percentage {
                "jobs"
            } else {
                "rentals"
            }
        }
        "all-jobs" => "jobs",
        "all-rentals" => "rentals",
        _ => "rentals", // default to rentals
    }
}

/// Produce Kubernetes node labels from a validation result and context.
pub fn labels_from_validation(
    nr: &NodeResult,
    provider: &str,
    region: &str,
    node_group: Option<&str>,
) -> BTreeMap<String, String> {
    let mut labels = BTreeMap::new();

    // CRITICAL: Mark as miner node (distinguishes from control plane)
    labels.insert("basilica.ai/node-role".into(), "miner".into());
    labels.insert("basilica.ai/validated".into(), "true".into());
    labels.insert("basilica.ai/provider".into(), provider.to_string());
    labels.insert("basilica.ai/region".into(), region.to_string());

    // Node group for workload isolation (jobs vs rentals)
    if let Some(group) = node_group {
        labels.insert("basilica.ai/node-group".into(), group.to_string());
    }

    let model = nr
        .gpu_infos
        .first()
        .map(|g| g.gpu_name.clone())
        .unwrap_or_else(|| nr.gpu_name.clone());
    labels.insert("basilica.ai/gpu-model".into(), model.clone());
    labels.insert(
        "basilica.ai/gpu-count".into(),
        nr.gpu_infos.len().to_string(),
    );
    labels.insert(
        "basilica.ai/gpu-mem".into(),
        nr.gpu_infos
            .first()
            .map(|g| g.gpu_memory_gb as u32)
            .unwrap_or(0)
            .to_string(),
    );

    // Add compute tier based on GPU model (handle vendor prefixes)
    let tier = if model.contains("H100") || model.contains("H200") || model.contains("B200") {
        "premium"
    } else if model.contains("A100") {
        "high"
    } else if model.contains("A10G") || model.contains("T4") || model.contains("L4") {
        "standard"
    } else {
        "basic"
    };
    labels.insert("basilica.ai/compute-tier".into(), tier.into());

    labels
}

/// Suggested taint when node is not validated.
pub fn taint_for_non_validated() -> (&'static str, &'static str) {
    ("basilica.ai/validated", "NoSchedule")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miner_prover::types::{
        BinaryCpuInfo, BinaryMemoryInfo, BinaryNetworkInfo, CompressedMatrix, GpuInfo,
        NetworkInterface, NodeResult, SmUtilizationStats,
    };

    fn sample_node_result() -> NodeResult {
        NodeResult {
            gpu_name: "NVIDIA A100".into(),
            gpu_uuid: "GPU-XYZ".into(),
            gpu_infos: vec![GpuInfo {
                index: 0,
                gpu_name: "NVIDIA A100".into(),
                gpu_uuid: "GPU-XYZ".into(),
                gpu_memory_gb: 80.0,
                computation_time_ns: 0,
                memory_bandwidth_gbps: 0.0,
                sm_utilization: SmUtilizationStats {
                    min_utilization: 0.0,
                    max_utilization: 0.0,
                    avg_utilization: 0.0,
                    per_sm_stats: vec![],
                },
                active_sms: 0,
                total_sms: 0,
                anti_debug_passed: true,
            }],
            cpu_info: BinaryCpuInfo {
                model: "AMD EPYC".into(),
                cores: 64,
                threads: 128,
                frequency_mhz: 0,
            },
            memory_info: BinaryMemoryInfo {
                total_gb: 256.0,
                available_gb: 0.0,
            },
            network_info: BinaryNetworkInfo {
                interfaces: vec![NetworkInterface {
                    name: "eth0".into(),
                    mac_address: "aa:bb".into(),
                    ip_addresses: vec!["10.0.0.2".into()],
                }],
            },
            cpu_pow: None,
            storage_pow: None,
            matrix_c: CompressedMatrix {
                rows: 0,
                cols: 0,
                data: vec![],
            },
            computation_time_ns: 0,
            checksum: [0u8; 32],
            sm_utilization: SmUtilizationStats {
                min_utilization: 0.0,
                max_utilization: 0.0,
                avg_utilization: 0.0,
                per_sm_stats: vec![],
            },
            active_sms: 0,
            total_sms: 0,
            memory_bandwidth_gbps: 0.0,
            anti_debug_passed: true,
            timing_fingerprint: 0,
        }
    }

    #[test]
    fn maps_to_node_profile_spec() {
        let nr = sample_node_result();
        let input = NodeProfileInput {
            provider: "onprem",
            region: "us-east-1",
            node_result: &nr,
        };
        let spec = to_node_profile_spec(&input);
        assert_eq!(spec.provider, "onprem");
        assert_eq!(spec.region, "us-east-1");
        assert_eq!(spec.gpu.model, "NVIDIA A100");
        assert_eq!(spec.gpu.count, 1);
        assert_eq!(spec.gpu.memory_gb, 80);
        assert_eq!(spec.cpu.model, "AMD EPYC");
        assert_eq!(spec.cpu.cores, 64);
        assert_eq!(spec.memory_gb, 256);
    }

    #[test]
    fn produces_k8s_labels() {
        let nr = sample_node_result();
        let labels = labels_from_validation(&nr, "onprem", "us-east-1", Some("rentals"));
        assert_eq!(labels.get("basilica.ai/node-role").unwrap(), "miner");
        assert_eq!(labels.get("basilica.ai/validated").unwrap(), "true");
        assert_eq!(labels.get("basilica.ai/provider").unwrap(), "onprem");
        assert_eq!(labels.get("basilica.ai/region").unwrap(), "us-east-1");
        assert_eq!(labels.get("basilica.ai/gpu-model").unwrap(), "NVIDIA A100");
        assert_eq!(labels.get("basilica.ai/gpu-count").unwrap(), "1");
        assert_eq!(labels.get("basilica.ai/gpu-mem").unwrap(), "80");
        assert_eq!(labels.get("basilica.ai/compute-tier").unwrap(), "high");
        assert_eq!(labels.get("basilica.ai/node-group").unwrap(), "rentals");
    }

    #[test]
    fn test_node_group_force_override() {
        let config = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 50,
            force_group: Some("jobs".to_string()),
        };

        // Should always return jobs when forced
        assert_eq!(assign_node_group("node-1", &config), "jobs");
        assert_eq!(assign_node_group("node-2", &config), "jobs");
        assert_eq!(assign_node_group("node-3", &config), "jobs");

        // Test force rentals
        let config_rentals = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 50,
            force_group: Some("rentals".to_string()),
        };
        assert_eq!(assign_node_group("node-1", &config_rentals), "rentals");
        assert_eq!(assign_node_group("node-2", &config_rentals), "rentals");
    }

    #[test]
    fn test_node_group_all_jobs_strategy() {
        let config = crate::config::NodeGroupConfig {
            strategy: "all-jobs".to_string(),
            jobs_percentage: 50,
            force_group: None,
        };

        // Should always return jobs
        assert_eq!(assign_node_group("node-1", &config), "jobs");
        assert_eq!(assign_node_group("node-2", &config), "jobs");
        assert_eq!(assign_node_group("node-3", &config), "jobs");
    }

    #[test]
    fn test_node_group_all_rentals_strategy() {
        let config = crate::config::NodeGroupConfig {
            strategy: "all-rentals".to_string(),
            jobs_percentage: 50,
            force_group: None,
        };

        // Should always return rentals
        assert_eq!(assign_node_group("node-1", &config), "rentals");
        assert_eq!(assign_node_group("node-2", &config), "rentals");
        assert_eq!(assign_node_group("node-3", &config), "rentals");
    }

    #[test]
    fn test_node_group_round_robin_deterministic() {
        let config = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 50,
            force_group: None,
        };

        // Same node ID should always return same group
        let node_id = "test-node-123";
        let first_assignment = assign_node_group(node_id, &config);
        let second_assignment = assign_node_group(node_id, &config);
        assert_eq!(first_assignment, second_assignment);
    }

    #[test]
    fn test_node_group_round_robin_distribution() {
        let config = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 30, // 30% jobs, 70% rentals
            force_group: None,
        };

        // Test multiple nodes to verify distribution
        let mut jobs_count = 0;
        let mut rentals_count = 0;

        for i in 0..100 {
            let node_id = format!("node-{}", i);
            match assign_node_group(&node_id, &config) {
                "jobs" => jobs_count += 1,
                "rentals" => rentals_count += 1,
                _ => panic!("unexpected group"),
            }
        }

        // Should be roughly 30/70 split (allow some variance due to hashing)
        assert!(
            jobs_count > 20 && jobs_count < 40,
            "jobs_count: {}",
            jobs_count
        );
        assert!(
            rentals_count > 60 && rentals_count < 80,
            "rentals_count: {}",
            rentals_count
        );
    }

    #[test]
    fn test_node_group_round_robin_extreme_percentages() {
        // Test 0% jobs (all rentals)
        let config_0 = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 0,
            force_group: None,
        };

        for i in 0..10 {
            let node_id = format!("node-{}", i);
            assert_eq!(assign_node_group(&node_id, &config_0), "rentals");
        }

        // Test 100% jobs
        let config_100 = crate::config::NodeGroupConfig {
            strategy: "round-robin".to_string(),
            jobs_percentage: 100,
            force_group: None,
        };

        for i in 0..10 {
            let node_id = format!("node-{}", i);
            assert_eq!(assign_node_group(&node_id, &config_100), "jobs");
        }
    }

    #[test]
    fn test_node_group_default_strategy() {
        let config = crate::config::NodeGroupConfig {
            strategy: "unknown-strategy".to_string(),
            jobs_percentage: 50,
            force_group: None,
        };

        // Unknown strategy should default to rentals
        assert_eq!(assign_node_group("node-1", &config), "rentals");
    }
}
