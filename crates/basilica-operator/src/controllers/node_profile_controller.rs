//! Node Profile Controller for GPU node validation and registration.
//!
//! This controller watches Kubernetes nodes and:
//! 1. Detects nodes with Basilica datacenter labels
//! 2. Normalizes NFD (Node Feature Discovery) labels to Basilica format
//! 3. Validates required GPU labels are present
//! 4. Creates a BasilicaNodeProfile CRD for the node
//! 5. Applies validation labels and removes the unvalidated taint

use crate::crd::basilica_node_profile::{
    BasilicaNodeProfile, BasilicaNodeProfileSpec, BasilicaNodeProfileStatus, NodeCpu, NodeGpu,
};
use crate::k8s_client::K8sClient;
use crate::labels::{basilica, extract_nfd_labels, has_nfd_gpu_labels};
use crate::metrics;
use anyhow::Result;
use k8s_openapi::api::core::v1::Node;
use k8s_openapi::chrono::Utc;
use std::collections::BTreeMap;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct NodeProfileController<C: K8sClient> {
    pub client: C,
}

impl<C: K8sClient> NodeProfileController<C> {
    pub fn new(client: C) -> Self {
        Self { client }
    }

    /// Build the set of labels applied to validated nodes.
    fn build_validation_labels() -> BTreeMap<String, String> {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::VALIDATED.to_string(), "true".to_string());
        labels.insert(basilica::NODE_ROLE.to_string(), "miner".to_string());
        labels.insert(
            basilica::NODE_GROUP.to_string(),
            "user-deployments".to_string(),
        );
        labels
    }

    /// Validate that all required Basilica GPU labels are present and valid.
    fn validate_node_labels(labels: &BTreeMap<String, String>) -> Result<()> {
        // Node type
        let node_type = labels
            .get(basilica::NODE_TYPE)
            .ok_or_else(|| anyhow::anyhow!("Missing required label: {}", basilica::NODE_TYPE))?;

        if node_type != "gpu" {
            return Err(anyhow::anyhow!(
                "Invalid node-type: expected 'gpu', got '{}'",
                node_type
            ));
        }

        // Datacenter
        let datacenter = labels
            .get(basilica::DATACENTER)
            .ok_or_else(|| anyhow::anyhow!("Missing required label: {}", basilica::DATACENTER))?;

        if datacenter.is_empty() {
            return Err(anyhow::anyhow!("Invalid datacenter label: cannot be empty"));
        }

        // GPU model
        let gpu_model = labels
            .get(basilica::GPU_MODEL)
            .ok_or_else(|| anyhow::anyhow!("Missing required label: {}", basilica::GPU_MODEL))?;

        if gpu_model.is_empty() {
            return Err(anyhow::anyhow!("Invalid gpu-model label: cannot be empty"));
        }

        // GPU count
        let gpu_count_str = labels
            .get(basilica::GPU_COUNT)
            .ok_or_else(|| anyhow::anyhow!("Missing required label: {}", basilica::GPU_COUNT))?;

        let gpu_count = gpu_count_str.parse::<u32>().map_err(|_| {
            anyhow::anyhow!(
                "Invalid gpu-count label: must be a positive integer, got '{}'",
                gpu_count_str
            )
        })?;

        if gpu_count == 0 {
            return Err(anyhow::anyhow!("Invalid gpu-count: must be greater than 0"));
        }

        // GPU memory
        let gpu_memory_str = labels.get(basilica::GPU_MEMORY_GB).ok_or_else(|| {
            anyhow::anyhow!("Missing required label: {}", basilica::GPU_MEMORY_GB)
        })?;

        let gpu_memory = gpu_memory_str.parse::<u32>().map_err(|_| {
            anyhow::anyhow!(
                "Invalid gpu-memory-gb label: must be a positive integer, got '{}'",
                gpu_memory_str
            )
        })?;

        if gpu_memory == 0 {
            return Err(anyhow::anyhow!(
                "Invalid gpu-memory-gb: must be greater than 0"
            ));
        }

        // Driver version
        let driver_version = labels.get(basilica::DRIVER_VERSION).ok_or_else(|| {
            anyhow::anyhow!("Missing required label: {}", basilica::DRIVER_VERSION)
        })?;

        if driver_version.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid driver-version label: cannot be empty"
            ));
        }

        // CUDA version
        let cuda_version = labels
            .get(basilica::CUDA_VERSION)
            .ok_or_else(|| anyhow::anyhow!("Missing required label: {}", basilica::CUDA_VERSION))?;

        if cuda_version.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid cuda-version label: cannot be empty"
            ));
        }

        Ok(())
    }

    /// Apply labels to a node atomically. Combines NFD-derived and validation labels
    /// into a single patch operation to reduce race conditions.
    async fn apply_labels_atomic(
        &self,
        node_name: &str,
        nfd_labels: &BTreeMap<String, String>,
        validation_labels: &BTreeMap<String, String>,
    ) -> Result<()> {
        let mut all_labels = nfd_labels.clone();
        all_labels.extend(validation_labels.clone());

        if all_labels.is_empty() {
            return Ok(());
        }

        self.client
            .add_node_labels(node_name, &all_labels)
            .await
            .map_err(|e| {
                // Check if node was deleted during reconciliation
                if e.to_string().contains("NotFound") {
                    info!(node = %node_name, "Node deleted during reconciliation, ignoring");
                    return anyhow::anyhow!("Node {} deleted during reconciliation", node_name);
                }
                error!(node = %node_name, error = %e, "Failed to apply labels");
                e
            })
    }

    /// Remove taint from node with graceful handling of deleted nodes.
    async fn remove_taint_graceful(&self, node_name: &str, taint_key: &str) -> Result<()> {
        match self.client.remove_node_taint(node_name, taint_key).await {
            Ok(()) => Ok(()),
            Err(e) => {
                if e.to_string().contains("NotFound") {
                    info!(node = %node_name, "Node deleted during reconciliation, ignoring taint removal");
                    Ok(())
                } else {
                    error!(node = %node_name, error = %e, "Failed to remove taint");
                    Err(e)
                }
            }
        }
    }

    /// Reconcile a node: validate labels, create profile, apply labels, remove taint.
    pub async fn reconcile(&self, node: &Node) -> Result<()> {
        let node_name = node
            .metadata
            .name
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Node has no name"))?;

        debug!(node = %node_name, "NodeProfileController: reconciling node");

        let labels = node
            .metadata
            .labels
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Node has no labels"))?;

        // Check for datacenter label - required for Basilica-managed nodes
        let datacenter_id = match labels.get(basilica::DATACENTER) {
            Some(id) => id,
            None => {
                debug!(node = %node_name, "Skipping node without datacenter label");
                return Ok(());
            }
        };

        debug!(node = %node_name, datacenter = %datacenter_id, "Node has datacenter label");

        // Extract NFD labels once - avoid double extraction
        let has_nfd = has_nfd_gpu_labels(labels);
        let nfd_derived_labels = if has_nfd {
            let derived = extract_nfd_labels(labels);
            if !derived.is_empty() {
                info!(
                    node = %node_name,
                    nfd_labels = ?derived.keys().collect::<Vec<_>>(),
                    "Detected NFD labels, will apply normalized Basilica labels"
                );
                // Record metric for NFD label conversion
                metrics::record_node_nfd_conversion(node_name);
            }
            derived
        } else {
            BTreeMap::new()
        };

        // Create effective labels by merging original with NFD-derived (without re-extracting)
        let mut effective_labels = labels.clone();
        for (key, value) in &nfd_derived_labels {
            effective_labels.entry(key.clone()).or_insert(value.clone());
        }

        debug!(node = %node_name, "Validating node labels (including NFD-derived)");

        if let Err(e) = Self::validate_node_labels(&effective_labels) {
            warn!(node = %node_name, error = %e, "Node label validation failed");
            return Err(e);
        }

        debug!(node = %node_name, "Node labels validated successfully");

        if !is_node_ready(node) {
            debug!(node = %node_name, "Skipping node that is not Ready");
            return Ok(());
        }

        info!(node = %node_name, datacenter = %datacenter_id, "Processing GPU node for validation");

        // Extract values from effective_labels
        let node_id = effective_labels
            .get(basilica::NODE_ID)
            .map(|s| s.as_str())
            .unwrap_or(node_name);

        let gpu_model = effective_labels
            .get(basilica::GPU_MODEL)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let gpu_count = effective_labels
            .get(basilica::GPU_COUNT)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let gpu_memory_gb = effective_labels
            .get(basilica::GPU_MEMORY_GB)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let (cpu_cores, memory_gb, storage_gb) = node
            .status
            .as_ref()
            .and_then(|s| s.capacity.as_ref())
            .map(|capacity| {
                let cores = capacity
                    .get("cpu")
                    .and_then(|q| q.0.parse::<i64>().ok())
                    .unwrap_or(0);
                let memory_bytes = capacity
                    .get("memory")
                    .and_then(|q| parse_memory_quantity(&q.0))
                    .unwrap_or(0);
                let storage_bytes = capacity
                    .get("ephemeral-storage")
                    .and_then(|q| parse_storage_quantity(&q.0))
                    .unwrap_or(0);
                (
                    cores as u32,
                    (memory_bytes / (1024 * 1024 * 1024)) as u32,
                    (storage_bytes / (1024 * 1024 * 1024)) as u32,
                )
            })
            .unwrap_or((0, 0, 0));

        let cpu_model = labels
            .get("node.kubernetes.io/instance-type")
            .map(|s| s.as_str())
            .unwrap_or("datacenter-gpu-node");

        let node_uid = node
            .metadata
            .uid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Node has no UID"))?;

        let owner_reference = k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference {
            api_version: "v1".to_string(),
            kind: "Node".to_string(),
            name: node_name.clone(),
            uid: node_uid.clone(),
            controller: Some(true),
            block_owner_deletion: Some(true),
        };

        let profile = BasilicaNodeProfile {
            metadata: k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta {
                name: Some(node_id.to_string()),
                namespace: Some("basilica-system".to_string()),
                owner_references: Some(vec![owner_reference]),
                ..Default::default()
            },
            spec: BasilicaNodeProfileSpec {
                provider: "datacenter".to_string(),
                region: datacenter_id.to_string(),
                gpu: NodeGpu {
                    model: gpu_model.to_string(),
                    count: gpu_count,
                    memory_gb: gpu_memory_gb,
                    cc_capable: false, // TODO: Detect GPU CC capability from node labels
                },
                cpu: NodeCpu {
                    model: cpu_model.to_string(),
                    cores: cpu_cores,
                },
                memory_gb,
                storage_gb,
                network_gbps: 10,
                tee: None, // TEE status populated by validator after attestation
            },
            status: Some(BasilicaNodeProfileStatus {
                kube_node_name: Some(node_name.clone()),
                last_validated: Some(Utc::now().to_rfc3339()),
                health: Some("Active".to_string()),
                tee_verified: false,
                tee_error: None,
            }),
        };

        debug!(node = %node_name, node_id = %node_id, "Creating BasilicaNodeProfile");

        self.client
            .create_node_profile("basilica-system", &profile)
            .await
            .or_else(|e| {
                if e.to_string().contains("AlreadyExists") {
                    debug!(node = %node_name, "NodeProfile already exists, continuing");
                    Ok(profile)
                } else {
                    error!(node = %node_name, error = %e, "Failed to create NodeProfile");
                    Err(e)
                }
            })?;

        info!(node = %node_name, "NodeProfile created successfully");

        // Apply all labels atomically (NFD-derived + validation)
        let validation_labels = Self::build_validation_labels();
        info!(
            node = %node_name,
            nfd_count = nfd_derived_labels.len(),
            validation_count = validation_labels.len(),
            "Applying labels atomically"
        );

        self.apply_labels_atomic(node_name, &nfd_derived_labels, &validation_labels)
            .await?;

        info!(node = %node_name, "Labels applied, removing unvalidated taint");

        self.remove_taint_graceful(node_name, basilica::UNVALIDATED_TAINT)
            .await?;

        // Record successful validation metric
        metrics::record_node_validation(node_name, datacenter_id);

        info!(node = %node_name, datacenter = %datacenter_id, "Successfully validated GPU node and removed taint");

        Ok(())
    }
}

fn parse_memory_quantity(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(num_str) = s.strip_suffix("Ki") {
        num_str.parse::<i64>().ok().map(|n| n * 1024)
    } else if let Some(num_str) = s.strip_suffix("Mi") {
        num_str.parse::<i64>().ok().map(|n| n * 1024 * 1024)
    } else if let Some(num_str) = s.strip_suffix("Gi") {
        num_str.parse::<i64>().ok().map(|n| n * 1024 * 1024 * 1024)
    } else if let Some(num_str) = s.strip_suffix("Ti") {
        num_str
            .parse::<i64>()
            .ok()
            .map(|n| n * 1024 * 1024 * 1024 * 1024)
    } else {
        s.parse::<i64>().ok()
    }
}

fn parse_storage_quantity(s: &str) -> Option<i64> {
    parse_memory_quantity(s)
}

fn is_node_ready(node: &Node) -> bool {
    node.status
        .as_ref()
        .and_then(|s| s.conditions.as_ref())
        .map(|conditions| {
            conditions
                .iter()
                .any(|c| c.type_ == "Ready" && c.status == "True")
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k8s_client::MockK8sClient;
    use crate::labels::{
        basilica, extract_nfd_labels as labels_extract_nfd, has_nfd_gpu_labels as labels_has_nfd,
        nfd,
    };
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn node_with_datacenter_label_creates_profile_and_applies_labels() {
        let client = MockK8sClient::default();
        let mut labels = BTreeMap::new();
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::NODE_ID.to_string(), "node-1".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "A100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "4".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "535.104.05".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.2".to_string());

        let node = Node {
            metadata: ObjectMeta {
                name: Some("test-node".to_string()),
                uid: Some("test-node-uid".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            status: Some(k8s_openapi::api::core::v1::NodeStatus {
                conditions: Some(vec![k8s_openapi::api::core::v1::NodeCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                capacity: Some({
                    let mut cap = BTreeMap::new();
                    cap.insert(
                        "cpu".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity("32".to_string()),
                    );
                    cap.insert(
                        "memory".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "256Gi".to_string(),
                        ),
                    );
                    cap.insert(
                        "ephemeral-storage".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "2000Gi".to_string(),
                        ),
                    );
                    cap
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        client.add_node("test-node").await;

        let ctrl = NodeProfileController::new(client.clone());
        ctrl.reconcile(&node).await.unwrap();

        // Verify NodeProfile was created
        let profile = client
            .get_node_profile("basilica-system", "node-1")
            .await
            .unwrap();
        assert_eq!(profile.spec.provider, "datacenter");
        assert_eq!(profile.spec.region, "dc-1");
        assert_eq!(profile.spec.gpu.model, "A100");
        assert_eq!(profile.spec.gpu.count, 4);
        assert_eq!(profile.spec.gpu.memory_gb, 80);
        assert_eq!(profile.spec.cpu.cores, 32);
        assert_eq!(profile.spec.memory_gb, 256);
        assert_eq!(profile.spec.storage_gb, 2000);

        // Verify validation labels were applied to node
        let updated_node = client.get_node("test-node").await.unwrap();
        let node_labels = updated_node.metadata.labels.unwrap();
        assert_eq!(
            node_labels.get(basilica::VALIDATED),
            Some(&"true".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::NODE_ROLE),
            Some(&"miner".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::NODE_GROUP),
            Some(&"user-deployments".to_string())
        );
    }

    #[tokio::test]
    async fn node_without_datacenter_label_skipped() {
        let client = MockK8sClient::default();
        let node = Node {
            metadata: ObjectMeta {
                name: Some("test-node".to_string()),
                labels: Some(BTreeMap::new()),
                ..Default::default()
            },
            status: Some(k8s_openapi::api::core::v1::NodeStatus {
                conditions: Some(vec![k8s_openapi::api::core::v1::NodeCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let ctrl = NodeProfileController::new(client.clone());
        ctrl.reconcile(&node).await.unwrap();

        assert!(client
            .get_node_profile("basilica-system", "test-node")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn node_not_ready_skipped() {
        let client = MockK8sClient::default();
        let mut labels = BTreeMap::new();
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "A100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "4".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "535.104.05".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.2".to_string());

        let node = Node {
            metadata: ObjectMeta {
                name: Some("test-node".to_string()),
                uid: Some("test-node-uid-2".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            status: Some(k8s_openapi::api::core::v1::NodeStatus {
                conditions: Some(vec![k8s_openapi::api::core::v1::NodeCondition {
                    type_: "Ready".to_string(),
                    status: "False".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let ctrl = NodeProfileController::new(client.clone());
        ctrl.reconcile(&node).await.unwrap();

        assert!(client
            .get_node_profile("basilica-system", "test-node")
            .await
            .is_err());
    }

    #[test]
    fn test_extract_nfd_labels_full_conversion() {
        let mut labels = BTreeMap::new();
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "8".to_string());
        labels.insert(nfd::GPU_MEMORY.to_string(), "81920".to_string());
        labels.insert(nfd::CUDA_DRIVER_FULL.to_string(), "535.104.05".to_string());
        labels.insert(nfd::CUDA_RUNTIME_FULL.to_string(), "12.2".to_string());
        labels.insert(nfd::CUDA_RUNTIME_MAJOR.to_string(), "12".to_string());

        let result = labels_extract_nfd(&labels);

        assert_eq!(
            result.get(basilica::GPU_MODEL),
            Some(&"TESLAA100SXM480GB".to_string())
        );
        assert_eq!(result.get(basilica::GPU_COUNT), Some(&"8".to_string()));
        assert_eq!(result.get(basilica::GPU_MEMORY_GB), Some(&"80".to_string()));
        assert_eq!(
            result.get(basilica::DRIVER_VERSION),
            Some(&"535.104.05".to_string())
        );
        assert_eq!(
            result.get(basilica::CUDA_VERSION),
            Some(&"12.2".to_string())
        );
        assert_eq!(result.get(basilica::CUDA_MAJOR), Some(&"12".to_string()));
        assert_eq!(result.get(basilica::NODE_TYPE), Some(&"gpu".to_string()));
    }

    #[test]
    fn test_extract_nfd_labels_skips_existing_basilica_labels() {
        let mut labels = BTreeMap::new();
        // NFD labels
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "8".to_string());
        // Existing Basilica labels should NOT be overwritten
        labels.insert(basilica::GPU_MODEL.to_string(), "H100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "4".to_string());

        let result = labels_extract_nfd(&labels);

        // NFD extraction should skip fields that already have Basilica labels
        assert!(!result.contains_key(basilica::GPU_MODEL));
        assert!(!result.contains_key(basilica::GPU_COUNT));
    }

    #[test]
    fn test_has_nfd_gpu_labels() {
        let mut labels = BTreeMap::new();
        assert!(!labels_has_nfd(&labels));

        labels.insert(nfd::GPU_PRODUCT.to_string(), "Tesla-A100".to_string());
        assert!(labels_has_nfd(&labels));

        labels.clear();
        labels.insert(nfd::GPU_COUNT.to_string(), "4".to_string());
        assert!(labels_has_nfd(&labels));

        labels.clear();
        labels.insert(nfd::PCI_NVIDIA_PRESENT.to_string(), "true".to_string());
        assert!(labels_has_nfd(&labels));
    }

    #[test]
    fn test_validate_node_labels_uses_constants() {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "A100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "4".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "535.104.05".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.2".to_string());

        let result = NodeProfileController::<MockK8sClient>::validate_node_labels(&labels);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_node_labels_invalid_gpu_count() {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "A100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "invalid".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "535.104.05".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.2".to_string());

        let result = NodeProfileController::<MockK8sClient>::validate_node_labels(&labels);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("gpu-count"));
    }

    #[test]
    fn test_validate_node_labels_zero_gpu_count() {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "A100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "0".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "535.104.05".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.2".to_string());

        let result = NodeProfileController::<MockK8sClient>::validate_node_labels(&labels);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("greater than 0"));
    }

    #[tokio::test]
    async fn node_with_nfd_labels_creates_profile_and_applies_normalized_labels() {
        let client = MockK8sClient::default();
        let mut labels = BTreeMap::new();
        // Minimal Basilica labels (datacenter only - required)
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        // NFD labels (to be normalized)
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "4".to_string());
        labels.insert(nfd::GPU_MEMORY.to_string(), "81920".to_string());
        labels.insert(nfd::CUDA_DRIVER_FULL.to_string(), "535.104.05".to_string());
        labels.insert(nfd::CUDA_RUNTIME_FULL.to_string(), "12.2".to_string());
        labels.insert(nfd::CUDA_RUNTIME_MAJOR.to_string(), "12".to_string());

        let node = Node {
            metadata: ObjectMeta {
                name: Some("nfd-node".to_string()),
                uid: Some("nfd-node-uid".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            status: Some(k8s_openapi::api::core::v1::NodeStatus {
                conditions: Some(vec![k8s_openapi::api::core::v1::NodeCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                capacity: Some({
                    let mut cap = BTreeMap::new();
                    cap.insert(
                        "cpu".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity("64".to_string()),
                    );
                    cap.insert(
                        "memory".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "512Gi".to_string(),
                        ),
                    );
                    cap.insert(
                        "ephemeral-storage".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "4000Gi".to_string(),
                        ),
                    );
                    cap
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        client.add_node("nfd-node").await;

        let ctrl = NodeProfileController::new(client.clone());
        ctrl.reconcile(&node).await.unwrap();

        // Verify NodeProfile was created with NFD-derived values
        let profile = client
            .get_node_profile("basilica-system", "nfd-node")
            .await
            .unwrap();
        assert_eq!(profile.spec.provider, "datacenter");
        assert_eq!(profile.spec.region, "dc-1");
        assert_eq!(profile.spec.gpu.model, "TESLAA100SXM480GB");
        assert_eq!(profile.spec.gpu.count, 4);
        assert_eq!(profile.spec.gpu.memory_gb, 80);
        assert_eq!(profile.spec.cpu.cores, 64);
        assert_eq!(profile.spec.memory_gb, 512);
        assert_eq!(profile.spec.storage_gb, 4000);

        // Verify all labels were applied atomically to node
        let updated_node = client.get_node("nfd-node").await.unwrap();
        let node_labels = updated_node.metadata.labels.unwrap();

        // NFD-derived Basilica labels
        assert_eq!(
            node_labels.get(basilica::GPU_MODEL),
            Some(&"TESLAA100SXM480GB".to_string())
        );
        assert_eq!(node_labels.get(basilica::GPU_COUNT), Some(&"4".to_string()));
        assert_eq!(
            node_labels.get(basilica::GPU_MEMORY_GB),
            Some(&"80".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::DRIVER_VERSION),
            Some(&"535.104.05".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::CUDA_VERSION),
            Some(&"12.2".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::NODE_TYPE),
            Some(&"gpu".to_string())
        );

        // Validation labels
        assert_eq!(
            node_labels.get(basilica::VALIDATED),
            Some(&"true".to_string())
        );
        assert_eq!(
            node_labels.get(basilica::NODE_ROLE),
            Some(&"miner".to_string())
        );
    }

    #[tokio::test]
    async fn node_with_mixed_labels_preserves_existing_basilica_labels() {
        let client = MockK8sClient::default();
        let mut labels = BTreeMap::new();
        // Existing Basilica labels
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
        labels.insert(basilica::GPU_MODEL.to_string(), "H100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "8".to_string());
        labels.insert(basilica::GPU_MEMORY_GB.to_string(), "80".to_string());
        labels.insert(
            basilica::DRIVER_VERSION.to_string(),
            "545.29.06".to_string(),
        );
        labels.insert(basilica::CUDA_VERSION.to_string(), "12.3".to_string());
        // NFD labels (should not override existing)
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "4".to_string());

        let node = Node {
            metadata: ObjectMeta {
                name: Some("mixed-node".to_string()),
                uid: Some("mixed-node-uid".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            status: Some(k8s_openapi::api::core::v1::NodeStatus {
                conditions: Some(vec![k8s_openapi::api::core::v1::NodeCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                capacity: Some({
                    let mut cap = BTreeMap::new();
                    cap.insert(
                        "cpu".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity("64".to_string()),
                    );
                    cap.insert(
                        "memory".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "512Gi".to_string(),
                        ),
                    );
                    cap.insert(
                        "ephemeral-storage".to_string(),
                        k8s_openapi::apimachinery::pkg::api::resource::Quantity(
                            "4000Gi".to_string(),
                        ),
                    );
                    cap
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        client.add_node("mixed-node").await;

        let ctrl = NodeProfileController::new(client.clone());
        ctrl.reconcile(&node).await.unwrap();

        // Verify NodeProfile used existing Basilica labels (NOT NFD)
        let profile = client
            .get_node_profile("basilica-system", "mixed-node")
            .await
            .unwrap();
        assert_eq!(profile.spec.gpu.model, "H100");
        assert_eq!(profile.spec.gpu.count, 8);
    }

    #[test]
    fn test_build_validation_labels_uses_constants() {
        let labels = NodeProfileController::<MockK8sClient>::build_validation_labels();

        assert!(labels.contains_key(basilica::VALIDATED));
        assert!(labels.contains_key(basilica::NODE_ROLE));
        assert!(labels.contains_key(basilica::NODE_GROUP));
        assert_eq!(labels.get(basilica::VALIDATED), Some(&"true".to_string()));
    }
}
