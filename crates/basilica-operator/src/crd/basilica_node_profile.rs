use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "basilica.ai",
    version = "v1",
    kind = "BasilicaNodeProfile",
    namespaced
)]
#[kube(status = "BasilicaNodeProfileStatus")]
pub struct BasilicaNodeProfileSpec {
    pub provider: String,
    pub region: String,
    pub gpu: NodeGpu,
    pub cpu: NodeCpu,
    pub memory_gb: u32,
    pub storage_gb: u32,
    pub network_gbps: u32,
    /// TEE (Trusted Execution Environment) capabilities
    #[serde(default)]
    pub tee: Option<NodeTee>,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
pub struct NodeGpu {
    pub model: String,
    pub count: u32,
    pub memory_gb: u32,
    /// Whether GPU supports Confidential Compute mode
    #[serde(default)]
    pub cc_capable: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
pub struct NodeCpu {
    pub model: String,
    pub cores: u32,
}

/// TEE (Trusted Execution Environment) capabilities of a node
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct NodeTee {
    /// Whether Intel TDX is available
    #[serde(default)]
    pub tdx_available: bool,
    /// Whether GPU Confidential Compute mode is enabled
    #[serde(default)]
    pub gpu_cc_enabled: bool,
    /// MRTD measurement (hex encoded) - build-time TD measurement
    #[serde(default)]
    pub mrtd_hex: Option<String>,
    /// Last TEE verification timestamp
    #[serde(default)]
    pub last_verified: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct BasilicaNodeProfileStatus {
    #[serde(default)]
    pub last_validated: Option<String>,
    #[serde(default)]
    pub kube_node_name: Option<String>,
    #[serde(default)]
    pub health: Option<String>,
    /// TEE verification status
    #[serde(default)]
    pub tee_verified: bool,
    /// TEE verification error (if any)
    #[serde(default)]
    pub tee_error: Option<String>,
}
