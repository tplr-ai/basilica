//! Python type exposures for Basilica SDK responses
//!
//! This module provides PyO3 bindings for response types, enabling
//! direct attribute access with full IDE autocomplete support.

// pyo3-stub-gen uses deprecated PyO3 APIs internally, we need to allow them
#![cfg_attr(feature = "stub-gen", allow(deprecated))]

use basilica_sdk::types::{
    AvailabilityInfo as SdkAvailabilityInfo, AvailableNode as SdkAvailableNode,
    CpuOffering as SdkCpuOffering, CpuSpec as SdkCpuSpec, GpuOffering as SdkGpuOffering,
    GpuPriceQuery as SdkGpuPriceQuery, GpuRequirements as SdkGpuRequirements,
    GpuSpec as SdkGpuSpec, HealthCheckConfig as SdkHealthCheckConfig,
    ListAvailableNodesQuery as SdkListAvailableNodesQuery, ListRentalsQuery as SdkListRentalsQuery,
    ListSecureCloudRentalsResponse as SdkListSecureCloudRentalsResponse,
    NodeDetails as SdkNodeDetails, PortMappingRequest as SdkPortMappingRequest,
    ProbeConfig as SdkProbeConfig, RentalState, RentalStatus as SdkRentalStatus,
    RentalStatusWithSshResponse as SdkRentalStatusWithSshResponse,
    ResourceRequirementsRequest as SdkResourceRequirementsRequest,
    SecureCloudRentalListItem as SdkSecureCloudRentalListItem,
    SecureCloudRentalResponse as SdkSecureCloudRentalResponse, SshAccess as SdkSshAccess,
    SshKeyResponse as SdkSshKeyResponse, StartRentalApiRequest as SdkStartRentalApiRequest,
    StartSecureCloudRentalRequest as SdkStartSecureCloudRentalRequest,
    StopSecureCloudRentalResponse as SdkStopSecureCloudRentalResponse,
    VolumeMountRequest as SdkVolumeMountRequest,
};
use basilica_validator::rental::RentalResponse as SdkRentalResponse;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::collections::HashMap;

/// SSH access information for a rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SshAccess {
    #[pyo3(get)]
    pub host: String,
    #[pyo3(get)]
    pub port: u16,
    #[pyo3(get)]
    pub user: String,
}

impl From<SdkSshAccess> for SshAccess {
    fn from(ssh: SdkSshAccess) -> Self {
        Self {
            host: ssh.host,
            port: ssh.port,
            user: ssh.username,
        }
    }
}

/// GPU specification details
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GpuSpec {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub memory_gb: u32,
    #[pyo3(get)]
    pub compute_capability: String,
}

impl From<SdkGpuSpec> for GpuSpec {
    fn from(spec: SdkGpuSpec) -> Self {
        Self {
            name: spec.name,
            memory_gb: spec.memory_gb,
            compute_capability: spec.compute_capability,
        }
    }
}

/// CPU specification details
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CpuSpec {
    #[pyo3(get)]
    pub cores: u32,
    #[pyo3(get)]
    pub model: String,
    #[pyo3(get)]
    pub memory_gb: u32,
}

impl From<SdkCpuSpec> for CpuSpec {
    fn from(spec: SdkCpuSpec) -> Self {
        Self {
            cores: spec.cores,
            model: spec.model,
            memory_gb: spec.memory_gb,
        }
    }
}

/// Node details including GPU and CPU specifications
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct NodeDetails {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub gpu_specs: Vec<GpuSpec>,
    #[pyo3(get)]
    pub cpu_specs: CpuSpec,
    #[pyo3(get)]
    pub location: Option<String>,
}

impl From<SdkNodeDetails> for NodeDetails {
    fn from(details: SdkNodeDetails) -> Self {
        Self {
            id: details.id,
            gpu_specs: details.gpu_specs.into_iter().map(Into::into).collect(),
            cpu_specs: details.cpu_specs.into(),
            location: details.location,
        }
    }
}

/// Rental status enumeration
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct RentalStatus {
    #[pyo3(get)]
    pub state: String,
    #[pyo3(get)]
    pub message: Option<String>,
}

impl From<SdkRentalStatus> for RentalStatus {
    fn from(status: SdkRentalStatus) -> Self {
        let state = match status {
            SdkRentalStatus::Pending => "Pending",
            SdkRentalStatus::Active => "Active",
            SdkRentalStatus::Terminated => "Terminated",
            SdkRentalStatus::Failed => "Failed",
        };

        Self {
            state: state.to_string(),
            message: None,
        }
    }
}

/// Response from starting a rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct RentalResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub ssh_credentials: Option<String>,
    #[pyo3(get)]
    pub container_id: String,
    #[pyo3(get)]
    pub container_name: String,
    #[pyo3(get)]
    pub status: String,
}

impl From<SdkRentalResponse> for RentalResponse {
    fn from(response: SdkRentalResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            ssh_credentials: response.ssh_credentials,
            container_id: response.container_info.container_id,
            container_name: response.container_info.container_name,
            status: response.container_info.status,
        }
    }
}

/// Full rental status response with SSH credentials (matches API response)
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct RentalStatusWithSshResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub status: RentalStatus,
    #[pyo3(get)]
    pub node: NodeDetails,
    #[pyo3(get)]
    pub ssh_credentials: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

impl From<SdkRentalStatusWithSshResponse> for RentalStatusWithSshResponse {
    fn from(response: SdkRentalStatusWithSshResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status.into(),
            node: response.node.into(),
            ssh_credentials: response.ssh_credentials,
            created_at: response.created_at.to_rfc3339(),
            updated_at: response.updated_at.to_rfc3339(),
        }
    }
}

/// Availability information for an node
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct AvailabilityInfo {
    #[pyo3(get)]
    pub available_until: Option<String>,
}

impl From<SdkAvailabilityInfo> for AvailabilityInfo {
    fn from(info: SdkAvailabilityInfo) -> Self {
        Self {
            available_until: info.available_until.map(|dt| dt.to_rfc3339()),
        }
    }
}

/// Available node with details and availability info
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct AvailableNode {
    #[pyo3(get)]
    pub node: NodeDetails,
    #[pyo3(get)]
    pub availability: AvailabilityInfo,
}

impl From<SdkAvailableNode> for AvailableNode {
    fn from(node: SdkAvailableNode) -> Self {
        Self {
            node: node.node.into(),
            availability: node.availability.into(),
        }
    }
}

/// Health check response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct HealthCheckResponse {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub version: String,
    #[pyo3(get)]
    pub timestamp: String,
    #[pyo3(get)]
    pub healthy_validators: usize,
    #[pyo3(get)]
    pub total_validators: usize,
}

impl From<basilica_sdk::types::HealthCheckResponse> for HealthCheckResponse {
    fn from(response: basilica_sdk::types::HealthCheckResponse) -> Self {
        Self {
            status: response.status,
            version: response.version,
            timestamp: response.timestamp.to_rfc3339(),
            healthy_validators: response.healthy_validators,
            total_validators: response.total_validators,
        }
    }
}

// Request types for Python bindings

/// GPU requirements for node selection
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GpuRequirements {
    #[pyo3(get, set)]
    pub gpu_count: u32,
    #[pyo3(get, set)]
    pub gpu_type: Option<String>,
    #[pyo3(get, set)]
    pub min_memory_gb: u32,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl GpuRequirements {
    #[new]
    #[pyo3(signature = (gpu_count, min_memory_gb, gpu_type=None))]
    fn new(gpu_count: u32, min_memory_gb: u32, gpu_type: Option<String>) -> Self {
        Self {
            gpu_count,
            gpu_type,
            min_memory_gb,
        }
    }
}

impl From<GpuRequirements> for SdkGpuRequirements {
    fn from(req: GpuRequirements) -> Self {
        Self {
            gpu_count: req.gpu_count,
            gpu_type: req.gpu_type,
            min_memory_gb: req.min_memory_gb,
        }
    }
}

/// Port mapping request
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct PortMappingRequest {
    #[pyo3(get, set)]
    pub container_port: u32,
    #[pyo3(get, set)]
    pub host_port: u32,
    #[pyo3(get, set)]
    pub protocol: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PortMappingRequest {
    #[new]
    #[pyo3(signature = (container_port, host_port, protocol=None))]
    fn new(container_port: u32, host_port: u32, protocol: Option<String>) -> Self {
        Self {
            container_port,
            host_port,
            protocol: protocol.unwrap_or_else(|| "tcp".to_string()),
        }
    }
}

impl From<PortMappingRequest> for SdkPortMappingRequest {
    fn from(port: PortMappingRequest) -> Self {
        Self {
            container_port: port.container_port,
            host_port: port.host_port,
            protocol: port.protocol,
        }
    }
}

/// Resource requirements request
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ResourceRequirementsRequest {
    #[pyo3(get, set)]
    pub cpu_cores: f64,
    #[pyo3(get, set)]
    pub memory_mb: i64,
    #[pyo3(get, set)]
    pub storage_mb: i64,
    #[pyo3(get, set)]
    pub gpu_count: u32,
    #[pyo3(get, set)]
    pub gpu_types: Vec<String>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ResourceRequirementsRequest {
    #[new]
    #[pyo3(signature = (cpu_cores=1.0, memory_mb=1024, storage_mb=10240, gpu_count=0, gpu_types=None))]
    fn new(
        cpu_cores: f64,
        memory_mb: i64,
        storage_mb: i64,
        gpu_count: u32,
        gpu_types: Option<Vec<String>>,
    ) -> Self {
        Self {
            cpu_cores,
            memory_mb,
            storage_mb,
            gpu_count,
            gpu_types: gpu_types.unwrap_or_default(),
        }
    }
}

impl From<ResourceRequirementsRequest> for SdkResourceRequirementsRequest {
    fn from(req: ResourceRequirementsRequest) -> Self {
        Self {
            cpu_cores: req.cpu_cores,
            memory_mb: req.memory_mb,
            storage_mb: req.storage_mb,
            gpu_count: req.gpu_count,
            gpu_types: req.gpu_types,
        }
    }
}

impl Default for ResourceRequirementsRequest {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_mb: 1024,
            storage_mb: 10240,
            gpu_count: 0,
            gpu_types: vec![],
        }
    }
}

/// Volume mount request
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct VolumeMountRequest {
    #[pyo3(get, set)]
    pub host_path: String,
    #[pyo3(get, set)]
    pub container_path: String,
    #[pyo3(get, set)]
    pub read_only: bool,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl VolumeMountRequest {
    #[new]
    #[pyo3(signature = (host_path, container_path, read_only=false))]
    fn new(host_path: String, container_path: String, read_only: bool) -> Self {
        Self {
            host_path,
            container_path,
            read_only,
        }
    }
}

impl From<VolumeMountRequest> for SdkVolumeMountRequest {
    fn from(vol: VolumeMountRequest) -> Self {
        Self {
            host_path: vol.host_path,
            container_path: vol.container_path,
            read_only: vol.read_only,
        }
    }
}

/// Start rental API request with GPU-based selection
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StartRentalApiRequest {
    #[pyo3(get, set)]
    pub gpu_category: String,
    #[pyo3(get, set)]
    pub gpu_count: u32,
    #[pyo3(get, set)]
    pub min_memory_gb: Option<u32>,
    #[pyo3(get, set)]
    pub max_hourly_rate: f64,
    #[pyo3(get, set)]
    pub container_image: String,
    #[pyo3(get, set)]
    pub ssh_public_key: String,
    #[pyo3(get, set)]
    pub environment: HashMap<String, String>,
    #[pyo3(get, set)]
    pub ports: Vec<PortMappingRequest>,
    #[pyo3(get, set)]
    pub resources: ResourceRequirementsRequest,
    #[pyo3(get, set)]
    pub command: Vec<String>,
    #[pyo3(get, set)]
    pub volumes: Vec<VolumeMountRequest>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl StartRentalApiRequest {
    #[new]
    #[pyo3(signature = (gpu_category, container_image, ssh_public_key, max_hourly_rate, gpu_count=1, min_memory_gb=None, environment=None, ports=None, resources=None, command=None, volumes=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu_category: String,
        container_image: String,
        ssh_public_key: String,
        max_hourly_rate: f64,
        gpu_count: u32,
        min_memory_gb: Option<u32>,
        environment: Option<HashMap<String, String>>,
        ports: Option<Vec<PortMappingRequest>>,
        resources: Option<ResourceRequirementsRequest>,
        command: Option<Vec<String>>,
        volumes: Option<Vec<VolumeMountRequest>>,
    ) -> Self {
        Self {
            gpu_category,
            gpu_count,
            min_memory_gb,
            max_hourly_rate,
            container_image,
            ssh_public_key,
            environment: environment.unwrap_or_default(),
            ports: ports.unwrap_or_default(),
            resources: resources.unwrap_or_default(),
            command: command.unwrap_or_default(),
            volumes: volumes.unwrap_or_default(),
        }
    }
}

fn usd_per_gpu_hour_to_cents(value: f64) -> Result<u32, String> {
    if !value.is_finite() {
        return Err("max_hourly_rate must be a finite number".to_string());
    }
    if value < 0.0 {
        return Err("max_hourly_rate must be non-negative".to_string());
    }

    let cents = (value * 100.0).round();
    if cents < 0.0 || cents > u32::MAX as f64 {
        return Err("max_hourly_rate is out of supported range".to_string());
    }

    Ok(cents as u32)
}

impl TryFrom<StartRentalApiRequest> for SdkStartRentalApiRequest {
    type Error = String;

    fn try_from(req: StartRentalApiRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            gpu_category: req.gpu_category,
            gpu_count: req.gpu_count,
            min_memory_gb: req.min_memory_gb,
            max_hourly_rate_cents: usd_per_gpu_hour_to_cents(req.max_hourly_rate)?,
            container_image: req.container_image,
            ssh_public_key: req.ssh_public_key,
            environment: req.environment,
            ports: req.ports.into_iter().map(Into::into).collect(),
            resources: req.resources.into(),
            command: req.command,
            volumes: req.volumes.into_iter().map(Into::into).collect(),
        })
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::usd_per_gpu_hour_to_cents;

    #[test]
    fn rounds_to_nearest_cent() {
        assert_eq!(usd_per_gpu_hour_to_cents(2.50).unwrap(), 250);
        assert_eq!(usd_per_gpu_hour_to_cents(1.234).unwrap(), 123);
        assert_eq!(usd_per_gpu_hour_to_cents(1.235).unwrap(), 124);
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(usd_per_gpu_hour_to_cents(f64::NAN).is_err());
        assert!(usd_per_gpu_hour_to_cents(f64::NEG_INFINITY).is_err());
        assert!(usd_per_gpu_hour_to_cents(-0.01).is_err());
    }
}

/// Query parameters for listing available nodes
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone, Default)]
pub struct ListAvailableNodesQuery {
    #[pyo3(get, set)]
    pub available: Option<bool>,
    #[pyo3(get, set)]
    pub min_gpu_memory: Option<u32>,
    #[pyo3(get, set)]
    pub gpu_type: Option<String>,
    #[pyo3(get, set)]
    pub min_gpu_count: Option<u32>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ListAvailableNodesQuery {
    #[new]
    #[pyo3(signature = (available=None, min_gpu_memory=None, gpu_type=None, min_gpu_count=None))]
    fn new(
        available: Option<bool>,
        min_gpu_memory: Option<u32>,
        gpu_type: Option<String>,
        min_gpu_count: Option<u32>,
    ) -> Self {
        Self {
            available,
            min_gpu_memory,
            gpu_type,
            min_gpu_count,
        }
    }
}

impl From<ListAvailableNodesQuery> for SdkListAvailableNodesQuery {
    fn from(query: ListAvailableNodesQuery) -> Self {
        Self {
            available: query.available,
            min_gpu_memory: query.min_gpu_memory,
            gpu_type: query.gpu_type,
            min_gpu_count: query.min_gpu_count,
            location: None, // Python SDK doesn't support location filtering yet
        }
    }
}

/// Query parameters for listing rentals
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone, Default)]
pub struct ListRentalsQuery {
    #[pyo3(get, set)]
    pub status: Option<String>, // We'll use String for the enum
    #[pyo3(get, set)]
    pub gpu_type: Option<String>,
    #[pyo3(get, set)]
    pub min_gpu_count: Option<u32>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ListRentalsQuery {
    #[new]
    #[pyo3(signature = (status=None, gpu_type=None, min_gpu_count=None))]
    fn new(status: Option<String>, gpu_type: Option<String>, min_gpu_count: Option<u32>) -> Self {
        Self {
            status,
            gpu_type,
            min_gpu_count,
        }
    }
}

impl From<ListRentalsQuery> for SdkListRentalsQuery {
    fn from(query: ListRentalsQuery) -> Self {
        let status = query.status.and_then(|s| match s.to_lowercase().as_str() {
            "provisioning" => Some(RentalState::Provisioning),
            "active" => Some(RentalState::Active),
            "stopping" => Some(RentalState::Stopping),
            "stopped" => Some(RentalState::Stopped),
            "failed" => Some(RentalState::Failed),
            _ => None,
        });

        Self {
            status,
            gpu_type: query.gpu_type,
            min_gpu_count: query.min_gpu_count,
        }
    }
}

// Deployment types

use basilica_sdk::types::{
    CreateDeploymentRequest as SdkCreateDeploymentRequest,
    DeleteDeploymentResponse as SdkDeleteDeploymentResponse,
    DeploymentListResponse as SdkDeploymentListResponse,
    DeploymentProgress as SdkDeploymentProgress, DeploymentResponse as SdkDeploymentResponse,
    DeploymentSummary as SdkDeploymentSummary, EnrollMetadataResponse as SdkEnrollMetadataResponse,
    EnvVar as SdkEnvVar, GpuRequirementsSpec as SdkGpuRequirementsSpec,
    PersistentStorageSpec as SdkPersistentStorageSpec, PodInfo as SdkPodInfo,
    ReplicaStatus as SdkReplicaStatus, ResourceRequirements as SdkResourceRequirements,
    SpreadMode as SdkSpreadMode, StorageBackend as SdkStorageBackend,
    StorageSpec as SdkStorageSpec, TopologySpreadConfig as SdkTopologySpreadConfig,
    WebSocketConfig as SdkWebSocketConfig,
};

/// Environment variable for container deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct EnvVar {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub value: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl EnvVar {
    #[new]
    fn new(name: String, value: String) -> Self {
        Self { name, value }
    }
}

impl From<EnvVar> for SdkEnvVar {
    fn from(env: EnvVar) -> Self {
        Self {
            name: env.name,
            value: env.value,
        }
    }
}

impl From<SdkEnvVar> for EnvVar {
    fn from(env: SdkEnvVar) -> Self {
        Self {
            name: env.name,
            value: env.value,
        }
    }
}

/// GPU requirements specification for container deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GpuRequirementsSpec {
    #[pyo3(get, set)]
    pub count: u32,
    #[pyo3(get, set)]
    pub model: Vec<String>,
    #[pyo3(get, set)]
    pub min_cuda_version: Option<String>,
    #[pyo3(get, set)]
    pub min_gpu_memory_gb: Option<u32>,
    #[pyo3(get, set)]
    pub interconnect: Option<String>,
    #[pyo3(get, set)]
    pub geo: Option<String>,
    #[pyo3(get, set)]
    pub spot: Option<bool>,
    #[pyo3(get, set)]
    pub infiniband: Option<bool>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl GpuRequirementsSpec {
    #[new]
    #[pyo3(signature = (count, model, min_cuda_version=None, min_gpu_memory_gb=None, interconnect=None, geo=None, spot=None, infiniband=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        count: u32,
        model: Vec<String>,
        min_cuda_version: Option<String>,
        min_gpu_memory_gb: Option<u32>,
        interconnect: Option<String>,
        geo: Option<String>,
        spot: Option<bool>,
        infiniband: Option<bool>,
    ) -> Self {
        Self {
            count,
            model,
            min_cuda_version,
            min_gpu_memory_gb,
            interconnect,
            geo,
            spot,
            infiniband,
        }
    }
}

impl From<GpuRequirementsSpec> for SdkGpuRequirementsSpec {
    fn from(gpu: GpuRequirementsSpec) -> Self {
        Self {
            count: gpu.count,
            model: gpu.model,
            min_cuda_version: gpu.min_cuda_version,
            min_gpu_memory_gb: gpu.min_gpu_memory_gb,
            interconnect: gpu.interconnect,
            geo: gpu.geo,
            spot: gpu.spot,
            infiniband: gpu.infiniband,
        }
    }
}

impl From<SdkGpuRequirementsSpec> for GpuRequirementsSpec {
    fn from(gpu: SdkGpuRequirementsSpec) -> Self {
        Self {
            count: gpu.count,
            model: gpu.model,
            min_cuda_version: gpu.min_cuda_version,
            min_gpu_memory_gb: gpu.min_gpu_memory_gb,
            interconnect: gpu.interconnect,
            geo: gpu.geo,
            spot: gpu.spot,
            infiniband: gpu.infiniband,
        }
    }
}

/// Resource requirements for container deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ResourceRequirements {
    #[pyo3(get, set)]
    pub cpu: String,
    #[pyo3(get, set)]
    pub memory: String,
    #[pyo3(get, set)]
    pub cpu_request: Option<String>,
    #[pyo3(get, set)]
    pub memory_request: Option<String>,
    #[pyo3(get, set)]
    pub gpus: Option<GpuRequirementsSpec>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ResourceRequirements {
    #[new]
    #[pyo3(signature = (cpu, memory, gpus=None, cpu_request=None, memory_request=None))]
    fn new(
        cpu: String,
        memory: String,
        gpus: Option<GpuRequirementsSpec>,
        cpu_request: Option<String>,
        memory_request: Option<String>,
    ) -> Self {
        Self {
            cpu,
            memory,
            gpus,
            cpu_request,
            memory_request,
        }
    }
}

impl From<ResourceRequirements> for SdkResourceRequirements {
    fn from(res: ResourceRequirements) -> Self {
        Self {
            cpu: res.cpu,
            memory: res.memory,
            cpu_request: res.cpu_request,
            memory_request: res.memory_request,
            gpus: res.gpus.map(|g| g.into()),
        }
    }
}

impl From<SdkResourceRequirements> for ResourceRequirements {
    fn from(res: SdkResourceRequirements) -> Self {
        Self {
            cpu: res.cpu,
            memory: res.memory,
            cpu_request: res.cpu_request,
            memory_request: res.memory_request,
            gpus: res.gpus.map(|g| g.into()),
        }
    }
}

/// Replica status for deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ReplicaStatus {
    #[pyo3(get)]
    pub desired: u32,
    #[pyo3(get)]
    pub ready: u32,
}

impl From<SdkReplicaStatus> for ReplicaStatus {
    fn from(status: SdkReplicaStatus) -> Self {
        Self {
            desired: status.desired,
            ready: status.ready,
        }
    }
}

/// Pod information
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct PodInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub node: Option<String>,
}

impl From<SdkPodInfo> for PodInfo {
    fn from(pod: SdkPodInfo) -> Self {
        Self {
            name: pod.name,
            status: pod.status,
            node: pod.node,
        }
    }
}

/// Pod spreading mode for controlling how pods are distributed across topology domains
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Default)]
pub enum SpreadMode {
    /// Best-effort spreading (ScheduleAnyway)
    #[default]
    Preferred,
    /// Strict spreading (DoNotSchedule)
    Required,
    /// One pod per node (pod anti-affinity)
    UniqueNodes,
}

impl From<SpreadMode> for SdkSpreadMode {
    fn from(mode: SpreadMode) -> Self {
        match mode {
            SpreadMode::Preferred => SdkSpreadMode::Preferred,
            SpreadMode::Required => SdkSpreadMode::Required,
            SpreadMode::UniqueNodes => SdkSpreadMode::UniqueNodes,
        }
    }
}

impl From<SdkSpreadMode> for SpreadMode {
    fn from(mode: SdkSpreadMode) -> Self {
        match mode {
            SdkSpreadMode::Preferred => SpreadMode::Preferred,
            SdkSpreadMode::Required => SpreadMode::Required,
            SdkSpreadMode::UniqueNodes => SpreadMode::UniqueNodes,
        }
    }
}

/// Configuration for pod topology spreading
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct TopologySpreadConfig {
    #[pyo3(get, set)]
    pub mode: SpreadMode,
    #[pyo3(get, set)]
    pub max_skew: i32,
    #[pyo3(get, set)]
    pub topology_key: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl TopologySpreadConfig {
    #[new]
    #[pyo3(signature = (mode=SpreadMode::Preferred, max_skew=1, topology_key="kubernetes.io/hostname"))]
    fn new(mode: SpreadMode, max_skew: i32, topology_key: &str) -> Self {
        Self {
            mode,
            max_skew,
            topology_key: topology_key.to_string(),
        }
    }

    /// Create config for unique nodes (one pod per node)
    #[staticmethod]
    fn unique_nodes() -> Self {
        Self {
            mode: SpreadMode::UniqueNodes,
            max_skew: 1,
            topology_key: "kubernetes.io/hostname".to_string(),
        }
    }
}

impl From<TopologySpreadConfig> for SdkTopologySpreadConfig {
    fn from(config: TopologySpreadConfig) -> Self {
        Self {
            mode: config.mode.into(),
            max_skew: config.max_skew,
            topology_key: config.topology_key,
        }
    }
}

impl From<SdkTopologySpreadConfig> for TopologySpreadConfig {
    fn from(config: SdkTopologySpreadConfig) -> Self {
        Self {
            mode: config.mode.into(),
            max_skew: config.max_skew,
            topology_key: config.topology_key,
        }
    }
}

/// WebSocket configuration for deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct WebSocketConfig {
    #[pyo3(get, set)]
    pub enabled: bool,
    #[pyo3(get, set)]
    pub idle_timeout_seconds: u32,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl WebSocketConfig {
    #[new]
    #[pyo3(signature = (enabled=true, idle_timeout_seconds=1800))]
    fn new(enabled: bool, idle_timeout_seconds: u32) -> Self {
        Self {
            enabled,
            idle_timeout_seconds,
        }
    }
}

impl From<WebSocketConfig> for SdkWebSocketConfig {
    fn from(config: WebSocketConfig) -> Self {
        Self {
            enabled: config.enabled,
            idle_timeout_seconds: config.idle_timeout_seconds,
        }
    }
}

impl From<SdkWebSocketConfig> for WebSocketConfig {
    fn from(config: SdkWebSocketConfig) -> Self {
        Self {
            enabled: config.enabled,
            idle_timeout_seconds: config.idle_timeout_seconds,
        }
    }
}

/// HTTP probe configuration for health checks
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ProbeConfig {
    #[pyo3(get, set)]
    pub path: String,
    #[pyo3(get, set)]
    pub port: Option<u16>,
    #[pyo3(get, set)]
    pub initial_delay_seconds: u32,
    #[pyo3(get, set)]
    pub period_seconds: u32,
    #[pyo3(get, set)]
    pub timeout_seconds: u32,
    #[pyo3(get, set)]
    pub failure_threshold: u32,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ProbeConfig {
    #[new]
    #[pyo3(signature = (path, port=None, initial_delay_seconds=30, period_seconds=10, timeout_seconds=5, failure_threshold=3))]
    fn new(
        path: String,
        port: Option<u16>,
        initial_delay_seconds: u32,
        period_seconds: u32,
        timeout_seconds: u32,
        failure_threshold: u32,
    ) -> Self {
        Self {
            path,
            port,
            initial_delay_seconds,
            period_seconds,
            timeout_seconds,
            failure_threshold,
        }
    }

    /// Create a probe config for vLLM large model loading (30 min startup timeout)
    #[staticmethod]
    fn vllm_large_model() -> Self {
        Self {
            path: "/health".to_string(),
            port: Some(8000),
            initial_delay_seconds: 0,
            period_seconds: 10,
            timeout_seconds: 5,
            failure_threshold: 180, // 30 minutes
        }
    }
}

impl From<ProbeConfig> for SdkProbeConfig {
    fn from(config: ProbeConfig) -> Self {
        Self {
            path: config.path,
            port: config.port,
            initial_delay_seconds: config.initial_delay_seconds,
            period_seconds: config.period_seconds,
            timeout_seconds: config.timeout_seconds,
            failure_threshold: config.failure_threshold,
        }
    }
}

/// Health check configuration for deployments
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct HealthCheckConfig {
    #[pyo3(get, set)]
    pub liveness: Option<ProbeConfig>,
    #[pyo3(get, set)]
    pub readiness: Option<ProbeConfig>,
    #[pyo3(get, set)]
    pub startup: Option<ProbeConfig>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl HealthCheckConfig {
    #[new]
    #[pyo3(signature = (liveness=None, readiness=None, startup=None))]
    fn new(
        liveness: Option<ProbeConfig>,
        readiness: Option<ProbeConfig>,
        startup: Option<ProbeConfig>,
    ) -> Self {
        Self {
            liveness,
            readiness,
            startup,
        }
    }

    /// Create health check config for vLLM large model (1T+ params)
    #[staticmethod]
    fn vllm_large_model() -> Self {
        Self {
            liveness: Some(ProbeConfig {
                path: "/health".to_string(),
                port: Some(8000),
                initial_delay_seconds: 60,
                period_seconds: 30,
                timeout_seconds: 10,
                failure_threshold: 3,
            }),
            readiness: Some(ProbeConfig {
                path: "/health".to_string(),
                port: Some(8000),
                initial_delay_seconds: 30,
                period_seconds: 10,
                timeout_seconds: 5,
                failure_threshold: 3,
            }),
            startup: Some(ProbeConfig {
                path: "/health".to_string(),
                port: Some(8000),
                initial_delay_seconds: 0,
                period_seconds: 10,
                timeout_seconds: 5,
                failure_threshold: 180, // 30 minutes for large model loading
            }),
        }
    }
}

impl From<HealthCheckConfig> for SdkHealthCheckConfig {
    fn from(config: HealthCheckConfig) -> Self {
        Self {
            liveness: config.liveness.map(Into::into),
            readiness: config.readiness.map(Into::into),
            startup: config.startup.map(Into::into),
        }
    }
}

/// Storage backend type
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum StorageBackend {
    R2,
    S3,
    GCS,
}

impl From<StorageBackend> for SdkStorageBackend {
    fn from(backend: StorageBackend) -> Self {
        match backend {
            StorageBackend::R2 => SdkStorageBackend::R2,
            StorageBackend::S3 => SdkStorageBackend::S3,
            StorageBackend::GCS => SdkStorageBackend::GCS,
        }
    }
}

/// Persistent storage specification
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct PersistentStorageSpec {
    #[pyo3(get, set)]
    pub enabled: bool,
    #[pyo3(get, set)]
    pub backend: StorageBackend,
    #[pyo3(get, set)]
    pub bucket: String,
    #[pyo3(get, set)]
    pub region: Option<String>,
    #[pyo3(get, set)]
    pub endpoint: Option<String>,
    #[pyo3(get, set)]
    pub credentials_secret: Option<String>,
    #[pyo3(get, set)]
    pub sync_interval_ms: u64,
    #[pyo3(get, set)]
    pub cache_size_mb: usize,
    #[pyo3(get, set)]
    pub mount_path: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PersistentStorageSpec {
    #[new]
    #[pyo3(signature = (enabled, backend, bucket="", region=None, endpoint=None, credentials_secret=None, sync_interval_ms=1000, cache_size_mb=1024, mount_path="/data"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        enabled: bool,
        backend: StorageBackend,
        bucket: &str,
        region: Option<String>,
        endpoint: Option<String>,
        credentials_secret: Option<String>,
        sync_interval_ms: u64,
        cache_size_mb: usize,
        mount_path: &str,
    ) -> Self {
        Self {
            enabled,
            backend,
            bucket: bucket.to_string(),
            region,
            endpoint,
            credentials_secret,
            sync_interval_ms,
            cache_size_mb,
            mount_path: mount_path.to_string(),
        }
    }
}

impl From<PersistentStorageSpec> for SdkPersistentStorageSpec {
    fn from(spec: PersistentStorageSpec) -> Self {
        Self {
            enabled: spec.enabled,
            backend: spec.backend.into(),
            bucket: spec.bucket,
            region: spec.region,
            endpoint: spec.endpoint,
            credentials_secret: spec.credentials_secret,
            sync_interval_ms: spec.sync_interval_ms,
            cache_size_mb: spec.cache_size_mb,
            mount_path: spec.mount_path,
        }
    }
}

/// Storage specification
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StorageSpec {
    #[pyo3(get, set)]
    pub persistent: Option<PersistentStorageSpec>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl StorageSpec {
    #[new]
    #[pyo3(signature = (persistent=None))]
    fn new(persistent: Option<PersistentStorageSpec>) -> Self {
        Self { persistent }
    }
}

impl From<StorageSpec> for SdkStorageSpec {
    fn from(spec: StorageSpec) -> Self {
        Self {
            persistent: spec.persistent.map(Into::into),
        }
    }
}

/// Create deployment request
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CreateDeploymentRequest {
    #[pyo3(get, set)]
    pub instance_name: String,
    #[pyo3(get, set)]
    pub image: String,
    #[pyo3(get, set)]
    pub replicas: u32,
    #[pyo3(get, set)]
    pub port: u32,
    #[pyo3(get, set)]
    pub command: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub args: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub env: Option<HashMap<String, String>>,
    #[pyo3(get, set)]
    pub resources: Option<ResourceRequirements>,
    #[pyo3(get, set)]
    pub ttl_seconds: Option<u32>,
    #[pyo3(get, set)]
    pub public: bool,
    #[pyo3(get, set)]
    pub storage: Option<StorageSpec>,
    #[pyo3(get, set)]
    pub topology_spread: Option<TopologySpreadConfig>,
    #[pyo3(get, set)]
    pub health_check: Option<HealthCheckConfig>,
    #[pyo3(get, set)]
    pub websocket: Option<WebSocketConfig>,
    #[pyo3(get, set)]
    pub public_metadata: bool,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl CreateDeploymentRequest {
    #[new]
    #[pyo3(signature = (instance_name, image, replicas, port, command=None, args=None, env=None, resources=None, ttl_seconds=None, public=true, storage=None, topology_spread=None, health_check=None, websocket=None, public_metadata=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        instance_name: String,
        image: String,
        replicas: u32,
        port: u32,
        command: Option<Vec<String>>,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        resources: Option<ResourceRequirements>,
        ttl_seconds: Option<u32>,
        public: bool,
        storage: Option<StorageSpec>,
        topology_spread: Option<TopologySpreadConfig>,
        health_check: Option<HealthCheckConfig>,
        websocket: Option<WebSocketConfig>,
        public_metadata: bool,
    ) -> Self {
        Self {
            instance_name,
            image,
            replicas,
            port,
            command,
            args,
            env,
            resources,
            ttl_seconds,
            public,
            storage,
            topology_spread,
            health_check,
            websocket,
            public_metadata,
        }
    }
}

impl From<CreateDeploymentRequest> for SdkCreateDeploymentRequest {
    fn from(req: CreateDeploymentRequest) -> Self {
        Self {
            instance_name: req.instance_name,
            image: req.image,
            replicas: req.replicas,
            port: req.port,
            command: req.command,
            args: req.args,
            env: req.env,
            resources: req.resources.map(Into::into),
            ttl_seconds: req.ttl_seconds,
            public: req.public,
            storage: req.storage.map(Into::into),
            health_check: req.health_check.map(Into::into),
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: req.topology_spread.map(Into::into),
            websocket: req.websocket.map(Into::into),
            public_metadata: req.public_metadata,
        }
    }
}

/// Deployment progress information
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeploymentProgress {
    #[pyo3(get)]
    pub current_step: String,
    #[pyo3(get)]
    pub percentage: Option<f64>,
    #[pyo3(get)]
    pub elapsed_seconds: u64,
}

impl From<SdkDeploymentProgress> for DeploymentProgress {
    fn from(progress: SdkDeploymentProgress) -> Self {
        Self {
            current_step: progress.current_step,
            percentage: progress.percentage,
            elapsed_seconds: progress.elapsed_seconds,
        }
    }
}

/// Deployment response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeploymentResponse {
    #[pyo3(get)]
    pub instance_name: String,
    #[pyo3(get)]
    pub user_id: String,
    #[pyo3(get)]
    pub namespace: String,
    #[pyo3(get)]
    pub state: String,
    #[pyo3(get)]
    pub url: String,
    #[pyo3(get)]
    pub replicas: ReplicaStatus,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: Option<String>,
    #[pyo3(get)]
    pub pods: Option<Vec<PodInfo>>,
    #[pyo3(get)]
    pub phase: Option<String>,
    #[pyo3(get)]
    pub message: Option<String>,
    #[pyo3(get)]
    pub progress: Option<DeploymentProgress>,
    /// Share token for private deployments (only returned on creation).
    #[pyo3(get)]
    pub share_token: Option<String>,
    /// Shareable URL with token query parameter for private deployments.
    #[pyo3(get)]
    pub share_url: Option<String>,
    #[pyo3(get)]
    pub websocket: Option<WebSocketConfig>,
    #[pyo3(get)]
    pub public_metadata: bool,
}

impl From<SdkDeploymentResponse> for DeploymentResponse {
    fn from(response: SdkDeploymentResponse) -> Self {
        Self {
            instance_name: response.instance_name,
            user_id: response.user_id,
            namespace: response.namespace,
            state: response.state,
            url: response.url,
            replicas: response.replicas.into(),
            created_at: response.created_at,
            updated_at: response.updated_at,
            pods: response
                .pods
                .map(|pods| pods.into_iter().map(Into::into).collect()),
            phase: response.phase,
            message: response.message,
            progress: response.progress.map(Into::into),
            share_token: response.share_token,
            share_url: response.share_url,
            websocket: response.websocket.map(Into::into),
            public_metadata: response.public_metadata,
        }
    }
}

/// Deployment summary for list responses
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeploymentSummary {
    #[pyo3(get)]
    pub instance_name: String,
    #[pyo3(get)]
    pub state: String,
    #[pyo3(get)]
    pub url: String,
    #[pyo3(get)]
    pub replicas: ReplicaStatus,
    #[pyo3(get)]
    pub created_at: String,
    /// Whether the deployment is publicly accessible.
    #[pyo3(get)]
    pub public: bool,
    #[pyo3(get)]
    pub websocket: Option<WebSocketConfig>,
    #[pyo3(get)]
    pub public_metadata: bool,
}

impl From<SdkDeploymentSummary> for DeploymentSummary {
    fn from(summary: SdkDeploymentSummary) -> Self {
        Self {
            instance_name: summary.instance_name,
            state: summary.state,
            url: summary.url,
            replicas: summary.replicas.into(),
            created_at: summary.created_at,
            public: summary.public,
            websocket: summary.websocket.map(Into::into),
            public_metadata: summary.public_metadata,
        }
    }
}

/// List deployments response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeploymentListResponse {
    #[pyo3(get)]
    pub deployments: Vec<DeploymentSummary>,
    #[pyo3(get)]
    pub total: usize,
}

impl From<SdkDeploymentListResponse> for DeploymentListResponse {
    fn from(response: SdkDeploymentListResponse) -> Self {
        Self {
            deployments: response.deployments.into_iter().map(Into::into).collect(),
            total: response.total,
        }
    }
}

/// Delete deployment response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeleteDeploymentResponse {
    #[pyo3(get)]
    pub instance_name: String,
    #[pyo3(get)]
    pub state: String,
    #[pyo3(get)]
    pub message: String,
}

impl From<SdkDeleteDeploymentResponse> for DeleteDeploymentResponse {
    fn from(response: SdkDeleteDeploymentResponse) -> Self {
        Self {
            instance_name: response.instance_name,
            state: response.state,
            message: response.message,
        }
    }
}

// ============================================================================
// SSH Key Management Types
// ============================================================================

/// SSH key response from registering or getting user's SSH key
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SshKeyResponse {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub user_id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub public_key: String,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

impl From<SdkSshKeyResponse> for SshKeyResponse {
    fn from(response: SdkSshKeyResponse) -> Self {
        Self {
            id: response.id,
            user_id: response.user_id,
            name: response.name,
            public_key: response.public_key,
            created_at: response.created_at.to_rfc3339(),
            updated_at: response.updated_at.to_rfc3339(),
        }
    }
}

// ============================================================================
// CPU Rental Types
// ============================================================================

/// CPU-only offering from secure cloud providers (no GPU)
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CpuOffering {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub vcpu_count: u32,
    #[pyo3(get)]
    pub system_memory_gb: u32,
    #[pyo3(get)]
    pub storage_gb: u32,
    #[pyo3(get)]
    pub region: String,
    #[pyo3(get)]
    pub hourly_rate: String,
    #[pyo3(get)]
    pub availability: bool,
    #[pyo3(get)]
    pub fetched_at: String,
}

impl From<SdkCpuOffering> for CpuOffering {
    fn from(offering: SdkCpuOffering) -> Self {
        Self {
            id: offering.id,
            provider: offering.provider,
            vcpu_count: offering.vcpu_count,
            system_memory_gb: offering.system_memory_gb,
            storage_gb: offering.storage_gb,
            region: offering.region,
            hourly_rate: offering.hourly_rate,
            availability: offering.availability,
            fetched_at: offering.fetched_at.to_rfc3339(),
        }
    }
}

/// Request to start a CPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StartCpuRentalRequest {
    #[pyo3(get, set)]
    pub offering_id: String,
    #[pyo3(get, set)]
    pub ssh_public_key_id: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl StartCpuRentalRequest {
    #[new]
    fn new(offering_id: String, ssh_public_key_id: String) -> Self {
        Self {
            offering_id,
            ssh_public_key_id,
        }
    }
}

impl From<StartCpuRentalRequest> for SdkStartSecureCloudRentalRequest {
    fn from(req: StartCpuRentalRequest) -> Self {
        Self {
            offering_id: req.offering_id,
            ssh_public_key_id: req.ssh_public_key_id,
        }
    }
}

/// Response from starting a CPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CpuRentalResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub deployment_id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub ip_address: Option<String>,
    #[pyo3(get)]
    pub ssh_command: Option<String>,
    #[pyo3(get)]
    pub hourly_cost: f64,
    #[pyo3(get)]
    pub is_spot: bool,
}

impl From<SdkSecureCloudRentalResponse> for CpuRentalResponse {
    fn from(response: SdkSecureCloudRentalResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            deployment_id: response.deployment_id,
            provider: response.provider,
            status: response.status,
            ip_address: response.ip_address,
            ssh_command: response.ssh_command,
            hourly_cost: response.hourly_cost,
            is_spot: response.is_spot,
        }
    }
}

/// Response from stopping a CPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StopCpuRentalResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub duration_hours: f64,
    #[pyo3(get)]
    pub total_cost: f64,
}

impl From<SdkStopSecureCloudRentalResponse> for StopCpuRentalResponse {
    fn from(response: SdkStopSecureCloudRentalResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status,
            duration_hours: response.duration_hours,
            total_cost: response.total_cost,
        }
    }
}

/// CPU rental list item
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CpuRentalListItem {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub provider_instance_id: Option<String>,
    #[pyo3(get)]
    pub gpu_type: String,
    #[pyo3(get)]
    pub gpu_count: u32,
    #[pyo3(get)]
    pub instance_type: String,
    #[pyo3(get)]
    pub location_code: Option<String>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub ip_address: Option<String>,
    #[pyo3(get)]
    pub hourly_cost: f64,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub stopped_at: Option<String>,
    #[pyo3(get)]
    pub ssh_command: Option<String>,
    #[pyo3(get)]
    pub vcpu_count: Option<u32>,
    #[pyo3(get)]
    pub system_memory_gb: Option<u32>,
    #[pyo3(get)]
    pub accumulated_cost: Option<String>,
    #[pyo3(get)]
    pub is_vip: bool,
    #[pyo3(get)]
    pub is_spot: bool,
}

impl From<SdkSecureCloudRentalListItem> for CpuRentalListItem {
    fn from(item: SdkSecureCloudRentalListItem) -> Self {
        Self {
            rental_id: item.rental_id,
            provider: item.provider,
            provider_instance_id: item.provider_instance_id,
            gpu_type: item.gpu_type,
            gpu_count: item.gpu_count,
            instance_type: item.instance_type,
            location_code: item.location_code,
            status: item.status,
            ip_address: item.ip_address,
            hourly_cost: item.hourly_cost,
            created_at: item.created_at.to_rfc3339(),
            stopped_at: item.stopped_at.map(|dt| dt.to_rfc3339()),
            ssh_command: item.ssh_command,
            vcpu_count: item.vcpu_count,
            system_memory_gb: item.system_memory_gb,
            accumulated_cost: item.accumulated_cost,
            is_vip: item.is_vip,
            is_spot: item.is_spot,
        }
    }
}

/// List CPU rentals response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ListCpuRentalsResponse {
    #[pyo3(get)]
    pub rentals: Vec<CpuRentalListItem>,
    #[pyo3(get)]
    pub total_count: usize,
}

impl From<SdkListSecureCloudRentalsResponse> for ListCpuRentalsResponse {
    fn from(response: SdkListSecureCloudRentalsResponse) -> Self {
        Self {
            rentals: response.rentals.into_iter().map(Into::into).collect(),
            total_count: response.total_count,
        }
    }
}

// ============================================================================
// GPU Rental Types (Secure Cloud)
// ============================================================================

/// Query parameters for filtering GPU price listings
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone, Default)]
pub struct GpuPriceQuery {
    #[pyo3(get, set)]
    pub interconnect: Option<String>,
    #[pyo3(get, set)]
    pub region: Option<String>,
    #[pyo3(get, set)]
    pub spot_only: Option<bool>,
    #[pyo3(get, set)]
    pub exclude_spot: Option<bool>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl GpuPriceQuery {
    #[new]
    #[pyo3(signature = (interconnect=None, region=None, spot_only=None, exclude_spot=None))]
    fn new(
        interconnect: Option<String>,
        region: Option<String>,
        spot_only: Option<bool>,
        exclude_spot: Option<bool>,
    ) -> Self {
        Self {
            interconnect,
            region,
            spot_only,
            exclude_spot,
        }
    }
}

impl From<GpuPriceQuery> for SdkGpuPriceQuery {
    fn from(q: GpuPriceQuery) -> Self {
        Self {
            interconnect: q.interconnect,
            region: q.region,
            spot_only: q.spot_only,
            exclude_spot: q.exclude_spot,
        }
    }
}

/// GPU offering from secure cloud providers
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GpuOffering {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub gpu_type: String,
    #[pyo3(get)]
    pub gpu_count: u32,
    #[pyo3(get)]
    pub vcpu_count: u32,
    #[pyo3(get)]
    pub system_memory_gb: u32,
    #[pyo3(get)]
    pub storage_gb: u32,
    #[pyo3(get)]
    pub region: String,
    #[pyo3(get)]
    pub hourly_rate: String,
    #[pyo3(get)]
    pub availability: bool,
    #[pyo3(get)]
    pub is_spot: bool,
    #[pyo3(get)]
    pub fetched_at: String,
    #[pyo3(get)]
    pub interconnect: Option<String>,
    #[pyo3(get)]
    pub gpu_memory_gb_per_gpu: Option<u32>,
}

impl From<SdkGpuOffering> for GpuOffering {
    fn from(offering: SdkGpuOffering) -> Self {
        Self {
            id: offering.id,
            provider: offering.provider.to_string(),
            gpu_type: offering.gpu_type.to_string(),
            gpu_count: offering.gpu_count,
            vcpu_count: offering.vcpu_count,
            system_memory_gb: offering.system_memory_gb,
            storage_gb: offering.storage.unwrap_or_default().parse().unwrap_or(0),
            region: offering.region,
            hourly_rate: offering.hourly_rate_per_gpu.to_string(),
            availability: offering.availability,
            is_spot: offering.is_spot,
            fetched_at: offering.fetched_at.to_rfc3339(),
            interconnect: offering.interconnect,
            gpu_memory_gb_per_gpu: offering.gpu_memory_gb_per_gpu,
        }
    }
}

/// Request to start a secure cloud GPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StartSecureCloudRentalRequest {
    #[pyo3(get, set)]
    pub offering_id: String,
    #[pyo3(get, set)]
    pub ssh_public_key_id: String,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl StartSecureCloudRentalRequest {
    #[new]
    fn new(offering_id: String, ssh_public_key_id: String) -> Self {
        Self {
            offering_id,
            ssh_public_key_id,
        }
    }
}

impl From<StartSecureCloudRentalRequest> for SdkStartSecureCloudRentalRequest {
    fn from(req: StartSecureCloudRentalRequest) -> Self {
        Self {
            offering_id: req.offering_id,
            ssh_public_key_id: req.ssh_public_key_id,
        }
    }
}

/// Response from starting a secure cloud GPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SecureCloudRentalResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub deployment_id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub ip_address: Option<String>,
    #[pyo3(get)]
    pub ssh_command: Option<String>,
    #[pyo3(get)]
    pub hourly_cost: f64,
    #[pyo3(get)]
    pub is_spot: bool,
}

impl From<SdkSecureCloudRentalResponse> for SecureCloudRentalResponse {
    fn from(response: SdkSecureCloudRentalResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            deployment_id: response.deployment_id,
            provider: response.provider,
            status: response.status,
            ip_address: response.ip_address,
            ssh_command: response.ssh_command,
            hourly_cost: response.hourly_cost,
            is_spot: response.is_spot,
        }
    }
}

/// Response from stopping a secure cloud GPU rental
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StopSecureCloudRentalResponse {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub duration_hours: f64,
    #[pyo3(get)]
    pub total_cost: f64,
}

impl From<SdkStopSecureCloudRentalResponse> for StopSecureCloudRentalResponse {
    fn from(response: SdkStopSecureCloudRentalResponse) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status,
            duration_hours: response.duration_hours,
            total_cost: response.total_cost,
        }
    }
}

/// Secure cloud GPU rental list item
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SecureCloudRentalListItem {
    #[pyo3(get)]
    pub rental_id: String,
    #[pyo3(get)]
    pub provider: String,
    #[pyo3(get)]
    pub provider_instance_id: Option<String>,
    #[pyo3(get)]
    pub gpu_type: String,
    #[pyo3(get)]
    pub gpu_count: u32,
    #[pyo3(get)]
    pub instance_type: String,
    #[pyo3(get)]
    pub location_code: Option<String>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub ip_address: Option<String>,
    #[pyo3(get)]
    pub hourly_cost: f64,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub stopped_at: Option<String>,
    #[pyo3(get)]
    pub ssh_command: Option<String>,
    #[pyo3(get)]
    pub vcpu_count: Option<u32>,
    #[pyo3(get)]
    pub system_memory_gb: Option<u32>,
    #[pyo3(get)]
    pub accumulated_cost: Option<String>,
    #[pyo3(get)]
    pub is_vip: bool,
    #[pyo3(get)]
    pub is_spot: bool,
}

impl From<SdkSecureCloudRentalListItem> for SecureCloudRentalListItem {
    fn from(item: SdkSecureCloudRentalListItem) -> Self {
        Self {
            rental_id: item.rental_id,
            provider: item.provider,
            provider_instance_id: item.provider_instance_id,
            gpu_type: item.gpu_type,
            gpu_count: item.gpu_count,
            instance_type: item.instance_type,
            location_code: item.location_code,
            status: item.status,
            ip_address: item.ip_address,
            hourly_cost: item.hourly_cost,
            created_at: item.created_at.to_rfc3339(),
            stopped_at: item.stopped_at.map(|dt| dt.to_rfc3339()),
            ssh_command: item.ssh_command,
            vcpu_count: item.vcpu_count,
            system_memory_gb: item.system_memory_gb,
            accumulated_cost: item.accumulated_cost,
            is_vip: item.is_vip,
            is_spot: item.is_spot,
        }
    }
}

/// List secure cloud GPU rentals response
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ListSecureCloudRentalsResponse {
    #[pyo3(get)]
    pub rentals: Vec<SecureCloudRentalListItem>,
    #[pyo3(get)]
    pub total_count: usize,
}

impl From<SdkListSecureCloudRentalsResponse> for ListSecureCloudRentalsResponse {
    fn from(response: SdkListSecureCloudRentalsResponse) -> Self {
        Self {
            rentals: response.rentals.into_iter().map(Into::into).collect(),
            total_count: response.total_count,
        }
    }
}

// ============================================================================
// Share Token Types
// ============================================================================

use basilica_sdk::types::{
    DeleteShareTokenResponse as SdkDeleteShareTokenResponse,
    RegenerateShareTokenResponse as SdkRegenerateShareTokenResponse,
    ShareTokenStatusResponse as SdkShareTokenStatusResponse,
};

/// Response from regenerating a share token for a private deployment
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct RegenerateShareTokenResponse {
    /// Raw token value. Only returned once, cannot be retrieved later.
    #[pyo3(get)]
    pub token: String,
    /// Full shareable URL with token as query parameter.
    #[pyo3(get)]
    pub share_url: String,
}

impl From<SdkRegenerateShareTokenResponse> for RegenerateShareTokenResponse {
    fn from(response: SdkRegenerateShareTokenResponse) -> Self {
        Self {
            token: response.token,
            share_url: response.share_url,
        }
    }
}

/// Response for checking share token status
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ShareTokenStatusResponse {
    /// Whether a share token exists for this deployment.
    #[pyo3(get)]
    pub exists: bool,
}

impl From<SdkShareTokenStatusResponse> for ShareTokenStatusResponse {
    fn from(response: SdkShareTokenStatusResponse) -> Self {
        Self {
            exists: response.exists,
        }
    }
}

/// Response from deleting/revoking a share token
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct DeleteShareTokenResponse {
    /// Whether a token was revoked.
    #[pyo3(get)]
    pub revoked: bool,
}

impl From<SdkDeleteShareTokenResponse> for DeleteShareTokenResponse {
    fn from(response: SdkDeleteShareTokenResponse) -> Self {
        Self {
            revoked: response.revoked,
        }
    }
}

// ============================================================================
// Public Deployment Metadata Types
// ============================================================================

use basilica_sdk::types::PublicDeploymentMetadataResponse as SdkPublicDeploymentMetadataResponse;

/// Response for metadata enrollment status
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct EnrollMetadataResponse {
    #[pyo3(get)]
    pub public_metadata: bool,
}

impl From<SdkEnrollMetadataResponse> for EnrollMetadataResponse {
    fn from(response: SdkEnrollMetadataResponse) -> Self {
        Self {
            public_metadata: response.public_metadata,
        }
    }
}

/// Public deployment metadata visible without authentication
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct PublicDeploymentMetadataResponse {
    #[pyo3(get)]
    pub instance_name: String,
    #[pyo3(get)]
    pub image: String,
    #[pyo3(get)]
    pub image_tag: String,
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub uptime_seconds: u64,
    #[pyo3(get)]
    pub replicas: ReplicaStatus,
    #[pyo3(get)]
    pub state: String,
}

impl From<SdkPublicDeploymentMetadataResponse> for PublicDeploymentMetadataResponse {
    fn from(response: SdkPublicDeploymentMetadataResponse) -> Self {
        Self {
            instance_name: response.instance_name,
            image: response.image,
            image_tag: response.image_tag,
            id: response.id,
            uptime_seconds: response.uptime_seconds,
            replicas: response.replicas.into(),
            state: response.state,
        }
    }
}
