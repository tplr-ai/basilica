//! Label constants and utilities for Kubernetes node and pod labels.
//!
//! This module provides centralized label key constants for both:
//! - NVIDIA GPU Feature Discovery (NFD/GFD) standard labels
//! - Basilica platform-specific labels
//!
//! The operator normalizes NFD labels to Basilica labels during node validation,
//! ensuring consistent label-based scheduling across heterogeneous node sources.

use std::borrow::Cow;
use std::collections::BTreeMap;

/// NFD/GFD standard labels (nvidia.com/*).
///
/// These labels are automatically applied by NVIDIA GPU Feature Discovery
/// when deployed in a Kubernetes cluster.
pub mod nfd {
    /// GPU product name (e.g., "Tesla-A100-SXM4-80GB")
    pub const GPU_PRODUCT: &str = "nvidia.com/gpu.product";

    /// GPU count as string
    pub const GPU_COUNT: &str = "nvidia.com/gpu.count";

    /// GPU memory in MiB (e.g., "81920")
    pub const GPU_MEMORY: &str = "nvidia.com/gpu.memory";

    /// GPU architecture family (e.g., "ampere", "hopper")
    pub const GPU_FAMILY: &str = "nvidia.com/gpu.family";

    /// GPU compute capability major version
    pub const GPU_COMPUTE_MAJOR: &str = "nvidia.com/gpu.compute.major";

    /// GPU compute capability minor version
    pub const GPU_COMPUTE_MINOR: &str = "nvidia.com/gpu.compute.minor";

    /// CUDA driver version major
    pub const CUDA_DRIVER_MAJOR: &str = "nvidia.com/cuda.driver-version.major";

    /// CUDA driver version full (e.g., "535.104.05")
    pub const CUDA_DRIVER_FULL: &str = "nvidia.com/cuda.driver-version.full";

    /// CUDA runtime version major
    pub const CUDA_RUNTIME_MAJOR: &str = "nvidia.com/cuda.runtime-version.major";

    /// CUDA runtime version full (e.g., "12.2")
    pub const CUDA_RUNTIME_FULL: &str = "nvidia.com/cuda.runtime-version.full";

    /// PCI device presence indicator from base NFD (NVIDIA vendor ID 10de)
    pub const PCI_NVIDIA_PRESENT: &str = "feature.node.kubernetes.io/pci-10de.present";

    /// NFD version label (if present)
    pub const NFD_VERSION: &str = "feature.node.kubernetes.io/nfd.version";
}

/// Basilica platform labels (basilica.ai/*).
///
/// These labels are used for node selection, affinity rules, and
/// platform-specific metadata across all Basilica controllers.
pub mod basilica {
    /// Node type (e.g., "gpu")
    pub const NODE_TYPE: &str = "basilica.ai/node-type";

    /// Datacenter/region identifier
    pub const DATACENTER: &str = "basilica.ai/datacenter";

    /// Node identifier
    pub const NODE_ID: &str = "basilica.ai/node-id";

    /// GPU model short name (e.g., "A100", "H100")
    pub const GPU_MODEL: &str = "basilica.ai/gpu-model";

    /// GPU count
    pub const GPU_COUNT: &str = "basilica.ai/gpu-count";

    /// GPU memory in GB
    pub const GPU_MEMORY_GB: &str = "basilica.ai/gpu-memory-gb";

    /// GPU architecture family (e.g., "ampere", "hopper")
    pub const GPU_FAMILY: &str = "basilica.ai/gpu-family";

    /// GPU compute capability (e.g., "8.0", "9.0")
    pub const GPU_COMPUTE_CAPABILITY: &str = "basilica.ai/gpu-compute-capability";

    /// NVIDIA driver version
    pub const DRIVER_VERSION: &str = "basilica.ai/driver-version";

    /// CUDA version (e.g., "12.2")
    pub const CUDA_VERSION: &str = "basilica.ai/cuda-version";

    /// CUDA major version (e.g., "12")
    pub const CUDA_MAJOR: &str = "basilica.ai/cuda-major";

    /// Node role (e.g., "miner")
    pub const NODE_ROLE: &str = "basilica.ai/node-role";

    /// Node validated status
    pub const VALIDATED: &str = "basilica.ai/validated";

    /// Node group for workload isolation
    pub const NODE_GROUP: &str = "basilica.ai/node-group";

    /// Workloads-only marker
    pub const WORKLOADS_ONLY: &str = "basilica.ai/workloads-only";

    /// Unvalidated taint key
    pub const UNVALIDATED_TAINT: &str = "basilica.ai/unvalidated";

    /// NFD version detected on node
    pub const NFD_VERSION: &str = "basilica.ai/nfd-version";

    // TEE (Trusted Execution Environment) labels
    /// Whether node is TEE verified
    pub const TEE_VERIFIED: &str = "basilica.ai/tee-verified";

    /// Whether Intel TDX is available on the node
    pub const TDX_AVAILABLE: &str = "basilica.ai/tdx-available";

    /// Whether GPU Confidential Compute mode is enabled
    pub const GPU_CC_ENABLED: &str = "basilica.ai/gpu-cc-enabled";

    /// MRTD measurement (build-time TD measurement)
    pub const TDX_MRTD: &str = "basilica.ai/tdx-mrtd";

    /// Last TEE verification timestamp
    pub const TEE_VERIFIED_AT: &str = "basilica.ai/tee-verified-at";
}

/// Known GPU model patterns for extraction.
/// Each entry is (pattern_to_match, short_name).
const GPU_MODEL_PATTERNS: &[(&str, &str)] = &[
    // Data center GPUs (ordered by specificity)
    ("H200", "H200"),
    ("H100", "H100"),
    ("A100", "A100"),
    ("A10G", "A10G"),
    ("A30", "A30"),
    ("A40", "A40"),
    ("A10", "A10"),
    ("L40S", "L40S"),
    ("L40", "L40"),
    ("L4", "L4"),
    ("V100", "V100"),
    ("T4", "T4"),
    ("P100", "P100"),
    ("P40", "P40"),
    // Consumer GPUs
    ("RTX-4090", "RTX-4090"),
    ("RTX-4080", "RTX-4080"),
    ("RTX-4070", "RTX-4070"),
    ("RTX-3090", "RTX-3090"),
    ("RTX-3080", "RTX-3080"),
    ("RTX 4090", "RTX-4090"),
    ("RTX 4080", "RTX-4080"),
    ("RTX 3090", "RTX-3090"),
    ("RTX 3080", "RTX-3080"),
];

/// Normalize GPU model name for consistent matching.
/// Removes non-alphanumeric characters and converts to uppercase.
/// This matches the autoscaler's normalization for consistency.
///
/// # Examples
///
/// ```
/// use basilica_operator::labels::normalize_gpu_model;
///
/// assert_eq!(normalize_gpu_model("A100-40GB"), "A10040GB");
/// assert_eq!(normalize_gpu_model("H100"), "H100");
/// assert_eq!(normalize_gpu_model("rtx-4090"), "RTX4090");
/// ```
pub fn normalize_gpu_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return "UNKNOWN".to_string();
    }
    trimmed
        .to_uppercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Extract short GPU model name from NFD full product name.
///
/// NFD provides full product names like "Tesla-A100-SXM4-80GB" or "NVIDIA-H100-80GB-HBM3".
/// This function extracts the commonly-used short model identifier.
///
/// Returns a `Cow<'static, str>` to avoid allocations when a known pattern matches.
///
/// # Examples
///
/// ```
/// use basilica_operator::labels::extract_short_gpu_model;
///
/// assert_eq!(extract_short_gpu_model("Tesla-A100-SXM4-80GB").as_ref(), "A100");
/// assert_eq!(extract_short_gpu_model("NVIDIA-H100-80GB-HBM3").as_ref(), "H100");
/// assert_eq!(extract_short_gpu_model("Tesla-V100-PCIE-32GB").as_ref(), "V100");
/// ```
pub fn extract_short_gpu_model(nfd_product: &str) -> Cow<'static, str> {
    let product = nfd_product.trim();

    if product.is_empty() {
        return Cow::Borrowed("unknown");
    }

    // Case-insensitive matching by converting to uppercase
    let product_upper = product.to_uppercase();
    for (pattern, short_name) in GPU_MODEL_PATTERNS {
        if product_upper.contains(pattern) {
            return Cow::Borrowed(short_name);
        }
    }

    // Fallback: return sanitized original (replace spaces with hyphens)
    Cow::Owned(product.replace(' ', "-"))
}

/// Normalize a list of GPU model names for nodeAffinity matching.
///
/// Applies `normalize_gpu_model` to each model in the list, converting
/// user-specified formats to normalized forms (uppercase, alphanumeric only).
/// Duplicates are removed while preserving order.
///
/// # Examples
///
/// ```
/// use basilica_operator::labels::normalize_gpu_models;
///
/// let models = vec!["A100-40GB".to_string(), "H100-80GB".to_string()];
/// let normalized = normalize_gpu_models(&models);
/// assert_eq!(normalized, vec!["A10040GB", "H10080GB"]);
///
/// // Duplicates are removed
/// let dupes = vec!["A100-40GB".to_string(), "a100-40gb".to_string()];
/// let deduped = normalize_gpu_models(&dupes);
/// assert_eq!(deduped, vec!["A10040GB"]);
/// ```
pub fn normalize_gpu_models(models: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    models
        .iter()
        .map(|m| normalize_gpu_model(m))
        .filter(|m| seen.insert(m.clone()))
        .collect()
}

/// Convert GPU memory from MiB (NFD format) to GB (Basilica format).
///
/// NFD reports memory in MiB (e.g., "81920" for 80GB).
/// Basilica uses GB for simplicity. Uses proper rounding.
///
/// Returns `None` for invalid input or values that would overflow.
///
/// # Examples
///
/// ```
/// use basilica_operator::labels::mib_to_gb;
///
/// assert_eq!(mib_to_gb("81920"), Some(80));  // 80 GB
/// assert_eq!(mib_to_gb("40960"), Some(40));  // 40 GB
/// assert_eq!(mib_to_gb("invalid"), None);
/// ```
pub fn mib_to_gb(mib_str: &str) -> Option<u32> {
    let trimmed = mib_str.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mib: u64 = trimmed.parse().ok()?;
    // 1 GB = 1024 MiB, round to nearest GB
    let gb = (mib + 512) / 1024;

    // Safe conversion to u32 (handles overflow)
    u32::try_from(gb).ok()
}

/// Check if node has NFD GPU labels that can be normalized.
///
/// Returns `true` if the node has any of the key NFD GPU labels:
/// - `nvidia.com/gpu.product`
/// - `nvidia.com/gpu.count`
/// - `feature.node.kubernetes.io/pci-10de.present`
pub fn has_nfd_gpu_labels(labels: &BTreeMap<String, String>) -> bool {
    labels.contains_key(nfd::GPU_PRODUCT)
        || labels.contains_key(nfd::GPU_COUNT)
        || labels.contains_key(nfd::PCI_NVIDIA_PRESENT)
}

/// Extract NFD version if present.
///
/// Returns the NFD version string if the version label exists.
pub fn get_nfd_version(labels: &BTreeMap<String, String>) -> Option<&str> {
    labels.get(nfd::NFD_VERSION).map(|s| s.as_str())
}

/// Extract NFD labels and convert to Basilica label format.
///
/// This function reads NFD labels from the node and generates corresponding
/// Basilica labels. It respects existing Basilica labels and will NOT overwrite them.
///
/// # Label Mappings
///
/// | NFD Label | Basilica Label |
/// |-----------|----------------|
/// | `nvidia.com/gpu.product` | `basilica.ai/gpu-model` (normalized) |
/// | `nvidia.com/gpu.count` | `basilica.ai/gpu-count` |
/// | `nvidia.com/gpu.memory` | `basilica.ai/gpu-memory-gb` (MiB -> GB) |
/// | `nvidia.com/gpu.family` | `basilica.ai/gpu-family` |
/// | `nvidia.com/gpu.compute.major/minor` | `basilica.ai/gpu-compute-capability` |
/// | `nvidia.com/cuda.driver-version.full` | `basilica.ai/driver-version` |
/// | `nvidia.com/cuda.runtime-version.full` | `basilica.ai/cuda-version` |
/// | `nvidia.com/cuda.runtime-version.major` | `basilica.ai/cuda-major` |
///
/// # Arguments
///
/// * `labels` - The node's current labels
///
/// # Returns
///
/// A `BTreeMap` containing only the NFD-derived Basilica labels that should be applied.
/// Labels that already exist on the node are NOT included.
pub fn extract_nfd_labels(labels: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut normalized = BTreeMap::new();

    // GPU model: nvidia.com/gpu.product -> basilica.ai/gpu-model
    if let Some(nfd_product) = labels.get(nfd::GPU_PRODUCT) {
        if labels.get(basilica::GPU_MODEL).is_none() {
            let model = normalize_gpu_model(nfd_product);
            normalized.insert(basilica::GPU_MODEL.to_string(), model);
        }
    }

    // GPU count: nvidia.com/gpu.count -> basilica.ai/gpu-count
    if let Some(nfd_count) = labels.get(nfd::GPU_COUNT) {
        if labels.get(basilica::GPU_COUNT).is_none() {
            // Validate it's a valid number before copying
            if nfd_count.parse::<u32>().is_ok() {
                normalized.insert(basilica::GPU_COUNT.to_string(), nfd_count.clone());
            }
        }
    }

    // GPU memory: nvidia.com/gpu.memory (MiB) -> basilica.ai/gpu-memory-gb (GB)
    if let Some(nfd_memory) = labels.get(nfd::GPU_MEMORY) {
        if labels.get(basilica::GPU_MEMORY_GB).is_none() {
            if let Some(gb) = mib_to_gb(nfd_memory) {
                normalized.insert(basilica::GPU_MEMORY_GB.to_string(), gb.to_string());
            }
        }
    }

    // GPU family: nvidia.com/gpu.family -> basilica.ai/gpu-family
    if let Some(nfd_family) = labels.get(nfd::GPU_FAMILY) {
        if labels.get(basilica::GPU_FAMILY).is_none() {
            normalized.insert(basilica::GPU_FAMILY.to_string(), nfd_family.clone());
        }
    }

    // Compute capability: nvidia.com/gpu.compute.major + minor -> basilica.ai/gpu-compute-capability
    if let (Some(major), Some(minor)) = (
        labels.get(nfd::GPU_COMPUTE_MAJOR),
        labels.get(nfd::GPU_COMPUTE_MINOR),
    ) {
        if labels.get(basilica::GPU_COMPUTE_CAPABILITY).is_none() {
            normalized.insert(
                basilica::GPU_COMPUTE_CAPABILITY.to_string(),
                format!("{}.{}", major, minor),
            );
        }
    }

    // Driver version: nvidia.com/cuda.driver-version.full -> basilica.ai/driver-version
    if let Some(nfd_driver) = labels.get(nfd::CUDA_DRIVER_FULL) {
        if labels.get(basilica::DRIVER_VERSION).is_none() {
            normalized.insert(basilica::DRIVER_VERSION.to_string(), nfd_driver.clone());
        }
    }

    // CUDA version: nvidia.com/cuda.runtime-version.full -> basilica.ai/cuda-version
    if let Some(nfd_cuda) = labels.get(nfd::CUDA_RUNTIME_FULL) {
        if labels.get(basilica::CUDA_VERSION).is_none() {
            normalized.insert(basilica::CUDA_VERSION.to_string(), nfd_cuda.clone());
        }
    }

    // CUDA major: nvidia.com/cuda.runtime-version.major -> basilica.ai/cuda-major
    if let Some(nfd_cuda_major) = labels.get(nfd::CUDA_RUNTIME_MAJOR) {
        if labels.get(basilica::CUDA_MAJOR).is_none() {
            normalized.insert(basilica::CUDA_MAJOR.to_string(), nfd_cuda_major.clone());
        }
    }

    // NFD version tracking: feature.node.kubernetes.io/nfd.version -> basilica.ai/nfd-version
    if let Some(nfd_ver) = labels.get(nfd::NFD_VERSION) {
        if labels.get(basilica::NFD_VERSION).is_none() {
            normalized.insert(basilica::NFD_VERSION.to_string(), nfd_ver.clone());
        }
    }

    // Node type: if we have NFD GPU labels, set node-type to "gpu"
    if labels.get(basilica::NODE_TYPE).is_none()
        && (labels.contains_key(nfd::GPU_PRODUCT) || labels.contains_key(nfd::PCI_NVIDIA_PRESENT))
    {
        normalized.insert(basilica::NODE_TYPE.to_string(), "gpu".to_string());
    }

    normalized
}

/// Merge existing labels with NFD-derived labels for validation.
///
/// This creates a merged view of labels that includes both the original node labels
/// and any NFD-derived Basilica labels. The merged map is used for validation
/// to ensure all required labels are present (either from original or NFD source).
///
/// Original labels always take precedence over NFD-derived labels.
pub fn merge_with_nfd_labels(labels: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut merged = labels.clone();
    let nfd_labels = extract_nfd_labels(labels);
    for (key, value) in nfd_labels {
        merged.entry(key).or_insert(value);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_short_gpu_model_datacenter() {
        assert_eq!(
            extract_short_gpu_model("Tesla-A100-SXM4-80GB").as_ref(),
            "A100"
        );
        assert_eq!(
            extract_short_gpu_model("NVIDIA-A100-PCIE-40GB").as_ref(),
            "A100"
        );
        assert_eq!(
            extract_short_gpu_model("NVIDIA-H100-80GB-HBM3").as_ref(),
            "H100"
        );
        assert_eq!(
            extract_short_gpu_model("Tesla-V100-PCIE-32GB").as_ref(),
            "V100"
        );
        assert_eq!(extract_short_gpu_model("NVIDIA-A10").as_ref(), "A10");
        assert_eq!(extract_short_gpu_model("NVIDIA-A10G").as_ref(), "A10G");
        assert_eq!(extract_short_gpu_model("Tesla-T4").as_ref(), "T4");
        assert_eq!(extract_short_gpu_model("NVIDIA-L4").as_ref(), "L4");
        assert_eq!(extract_short_gpu_model("NVIDIA-L40S").as_ref(), "L40S");
    }

    #[test]
    fn test_extract_short_gpu_model_consumer() {
        assert_eq!(
            extract_short_gpu_model("NVIDIA-GeForce-RTX-4090").as_ref(),
            "RTX-4090"
        );
        assert_eq!(
            extract_short_gpu_model("NVIDIA-GeForce-RTX-3090-Ti").as_ref(),
            "RTX-3090"
        );
        assert_eq!(
            extract_short_gpu_model("GeForce RTX 4080").as_ref(),
            "RTX-4080"
        );
    }

    #[test]
    fn test_extract_short_gpu_model_unknown() {
        // Unknown models return sanitized original
        assert_eq!(
            extract_short_gpu_model("Unknown GPU Model").as_ref(),
            "Unknown-GPU-Model"
        );
    }

    #[test]
    fn test_extract_short_gpu_model_empty() {
        assert_eq!(extract_short_gpu_model("").as_ref(), "unknown");
        assert_eq!(extract_short_gpu_model("   ").as_ref(), "unknown");
    }

    #[test]
    fn test_extract_short_gpu_model_whitespace() {
        assert_eq!(
            extract_short_gpu_model("  Tesla-A100-SXM4-80GB  ").as_ref(),
            "A100"
        );
    }

    #[test]
    fn test_extract_short_gpu_model_case_insensitive() {
        // Lowercase input should match uppercase patterns
        assert_eq!(extract_short_gpu_model("a100-40gb").as_ref(), "A100");
        assert_eq!(extract_short_gpu_model("h100-80gb").as_ref(), "H100");
        assert_eq!(extract_short_gpu_model("v100").as_ref(), "V100");
        assert_eq!(extract_short_gpu_model("rtx-4090").as_ref(), "RTX-4090");

        // Mixed case
        assert_eq!(
            extract_short_gpu_model("Tesla-a100-SXM4-80GB").as_ref(),
            "A100"
        );
        assert_eq!(
            extract_short_gpu_model("nvidia-H100-80gb-HBM3").as_ref(),
            "H100"
        );
    }

    #[test]
    fn test_normalize_gpu_model() {
        // Uppercase + alphanumeric only
        assert_eq!(normalize_gpu_model("A100-40GB"), "A10040GB");
        assert_eq!(normalize_gpu_model("H100"), "H100");
        assert_eq!(normalize_gpu_model("rtx-4090"), "RTX4090");
        assert_eq!(
            normalize_gpu_model("Tesla-A100-SXM4-80GB"),
            "TESLAA100SXM480GB"
        );
        assert_eq!(normalize_gpu_model("  a100  "), "A100");
        assert_eq!(normalize_gpu_model(""), "UNKNOWN");
        assert_eq!(normalize_gpu_model("   "), "UNKNOWN");
    }

    #[test]
    fn test_normalize_gpu_models() {
        // Models should be normalized (uppercase, alphanumeric only)
        let models = vec!["A100-40GB".to_string(), "H100-80GB".to_string()];
        assert_eq!(normalize_gpu_models(&models), vec!["A10040GB", "H10080GB"]);

        // Already normalized models should pass through
        let short = vec!["A100".to_string(), "H100".to_string()];
        assert_eq!(normalize_gpu_models(&short), vec!["A100", "H100"]);

        // Empty input returns empty output
        let empty: Vec<String> = vec![];
        assert_eq!(normalize_gpu_models(&empty), Vec::<String>::new());

        // Single model
        let single = vec!["RTX-4090".to_string()];
        assert_eq!(normalize_gpu_models(&single), vec!["RTX4090"]);

        // Mixed formats
        let mixed = vec![
            "Tesla-A100-SXM4-80GB".to_string(),
            "V100".to_string(),
            "NVIDIA-H100-80GB-HBM3".to_string(),
        ];
        assert_eq!(
            normalize_gpu_models(&mixed),
            vec!["TESLAA100SXM480GB", "V100", "NVIDIAH10080GBHBM3"]
        );

        // Deduplication: case insensitive
        let dupes = vec!["A100-40GB".to_string(), "a100-40gb".to_string()];
        assert_eq!(normalize_gpu_models(&dupes), vec!["A10040GB"]);

        // Deduplication preserves order (first occurrence wins)
        let order = vec![
            "H100-80GB".to_string(),
            "A100-40GB".to_string(),
            "H100-80GB".to_string(),
        ];
        assert_eq!(normalize_gpu_models(&order), vec!["H10080GB", "A10040GB"]);

        // Case insensitive deduplication
        let case_dupes = vec!["a100-40gb".to_string(), "A100-40GB".to_string()];
        assert_eq!(normalize_gpu_models(&case_dupes), vec!["A10040GB"]);
    }

    #[test]
    fn test_mib_to_gb() {
        assert_eq!(mib_to_gb("81920"), Some(80)); // 80 GB
        assert_eq!(mib_to_gb("40960"), Some(40)); // 40 GB
        assert_eq!(mib_to_gb("16384"), Some(16)); // 16 GB
        assert_eq!(mib_to_gb("24576"), Some(24)); // 24 GB
        assert_eq!(mib_to_gb("49152"), Some(48)); // 48 GB
    }

    #[test]
    fn test_mib_to_gb_rounding() {
        // Test rounding behavior
        assert_eq!(mib_to_gb("15360"), Some(15)); // 15 GB
        assert_eq!(mib_to_gb("12288"), Some(12)); // 12 GB
    }

    #[test]
    fn test_mib_to_gb_invalid() {
        assert_eq!(mib_to_gb("invalid"), None);
        assert_eq!(mib_to_gb(""), None);
        assert_eq!(mib_to_gb("   "), None);
        assert_eq!(mib_to_gb("-100"), None);
    }

    #[test]
    fn test_mib_to_gb_whitespace() {
        assert_eq!(mib_to_gb("  81920  "), Some(80));
    }

    #[test]
    fn test_mib_to_gb_large_value() {
        // Test very large values that might overflow u32
        // 4TB in MiB = 4 * 1024 * 1024 = 4194304 MiB = 4096 GB
        assert_eq!(mib_to_gb("4194304"), Some(4096));
        // Values > u32::MAX GB should return None
        let huge: u64 = (u32::MAX as u64 + 1) * 1024;
        assert_eq!(mib_to_gb(&huge.to_string()), None);
    }

    #[test]
    fn test_has_nfd_gpu_labels() {
        let mut labels = BTreeMap::new();
        assert!(!has_nfd_gpu_labels(&labels));

        labels.insert(nfd::GPU_PRODUCT.to_string(), "Tesla-A100".to_string());
        assert!(has_nfd_gpu_labels(&labels));

        labels.clear();
        labels.insert(nfd::GPU_COUNT.to_string(), "4".to_string());
        assert!(has_nfd_gpu_labels(&labels));

        labels.clear();
        labels.insert(nfd::PCI_NVIDIA_PRESENT.to_string(), "true".to_string());
        assert!(has_nfd_gpu_labels(&labels));

        labels.clear();
        labels.insert("other.label".to_string(), "value".to_string());
        assert!(!has_nfd_gpu_labels(&labels));
    }

    #[test]
    fn test_get_nfd_version() {
        let mut labels = BTreeMap::new();
        assert_eq!(get_nfd_version(&labels), None);

        labels.insert(nfd::NFD_VERSION.to_string(), "0.14.2".to_string());
        assert_eq!(get_nfd_version(&labels), Some("0.14.2"));
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
        labels.insert(nfd::GPU_FAMILY.to_string(), "ampere".to_string());
        labels.insert(nfd::GPU_COMPUTE_MAJOR.to_string(), "8".to_string());
        labels.insert(nfd::GPU_COMPUTE_MINOR.to_string(), "0".to_string());
        labels.insert(nfd::CUDA_DRIVER_FULL.to_string(), "535.104.05".to_string());
        labels.insert(nfd::CUDA_RUNTIME_FULL.to_string(), "12.2".to_string());
        labels.insert(nfd::CUDA_RUNTIME_MAJOR.to_string(), "12".to_string());
        labels.insert(nfd::NFD_VERSION.to_string(), "0.14.2".to_string());

        let result = extract_nfd_labels(&labels);

        assert_eq!(
            result.get(basilica::GPU_MODEL),
            Some(&"TESLAA100SXM480GB".to_string())
        );
        assert_eq!(result.get(basilica::GPU_COUNT), Some(&"8".to_string()));
        assert_eq!(result.get(basilica::GPU_MEMORY_GB), Some(&"80".to_string()));
        assert_eq!(
            result.get(basilica::GPU_FAMILY),
            Some(&"ampere".to_string())
        );
        assert_eq!(
            result.get(basilica::GPU_COMPUTE_CAPABILITY),
            Some(&"8.0".to_string())
        );
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
        assert_eq!(
            result.get(basilica::NFD_VERSION),
            Some(&"0.14.2".to_string())
        );
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

        let result = extract_nfd_labels(&labels);

        // NFD extraction should skip fields that already have Basilica labels
        assert!(!result.contains_key(basilica::GPU_MODEL));
        assert!(!result.contains_key(basilica::GPU_COUNT));
    }

    #[test]
    fn test_extract_nfd_labels_validates_gpu_count() {
        let mut labels = BTreeMap::new();
        labels.insert(nfd::GPU_COUNT.to_string(), "invalid".to_string());

        let result = extract_nfd_labels(&labels);

        // Invalid GPU count should not be included
        assert!(!result.contains_key(basilica::GPU_COUNT));
    }

    #[test]
    fn test_extract_nfd_labels_partial_nfd() {
        // Test with only some NFD labels present
        let mut labels = BTreeMap::new();
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        // No other NFD labels

        let result = extract_nfd_labels(&labels);

        assert_eq!(
            result.get(basilica::GPU_MODEL),
            Some(&"TESLAA100SXM480GB".to_string())
        );
        assert_eq!(result.get(basilica::NODE_TYPE), Some(&"gpu".to_string()));
        // Other labels should not be present
        assert!(!result.contains_key(basilica::GPU_COUNT));
        assert!(!result.contains_key(basilica::GPU_MEMORY_GB));
    }

    #[test]
    fn test_extract_nfd_labels_pci_only() {
        // Test with only PCI presence indicator
        let mut labels = BTreeMap::new();
        labels.insert(nfd::PCI_NVIDIA_PRESENT.to_string(), "true".to_string());

        let result = extract_nfd_labels(&labels);

        // Should set node-type to gpu based on PCI presence
        assert_eq!(result.get(basilica::NODE_TYPE), Some(&"gpu".to_string()));
    }

    #[test]
    fn test_merge_with_nfd_labels() {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::DATACENTER.to_string(), "dc-1".to_string());
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "NVIDIA-H100-80GB-HBM3".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "8".to_string());

        let merged = merge_with_nfd_labels(&labels);

        // Original labels preserved
        assert_eq!(merged.get(basilica::DATACENTER), Some(&"dc-1".to_string()));
        // NFD-derived labels added
        assert_eq!(
            merged.get(basilica::GPU_MODEL),
            Some(&"NVIDIAH10080GBHBM3".to_string())
        );
        assert_eq!(merged.get(basilica::GPU_COUNT), Some(&"8".to_string()));
        // Original NFD labels still present
        assert_eq!(
            merged.get(nfd::GPU_PRODUCT),
            Some(&"NVIDIA-H100-80GB-HBM3".to_string())
        );
    }

    #[test]
    fn test_merge_preserves_existing_basilica_over_nfd() {
        let mut labels = BTreeMap::new();
        labels.insert(basilica::GPU_MODEL.to_string(), "H100".to_string());
        labels.insert(basilica::GPU_COUNT.to_string(), "4".to_string());
        labels.insert(
            nfd::GPU_PRODUCT.to_string(),
            "Tesla-A100-SXM4-80GB".to_string(),
        );
        labels.insert(nfd::GPU_COUNT.to_string(), "8".to_string());

        let merged = merge_with_nfd_labels(&labels);

        // Original Basilica labels should be preserved, not NFD values
        assert_eq!(merged.get(basilica::GPU_MODEL), Some(&"H100".to_string()));
        assert_eq!(merged.get(basilica::GPU_COUNT), Some(&"4".to_string()));
    }
}
