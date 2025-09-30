//! Python type exposures for Basilica SDK responses
//!
//! This module provides PyO3 bindings for response types, enabling
//! direct attribute access with full IDE autocomplete support.

// pyo3-stub-gen uses deprecated PyO3 APIs internally, we need to allow them
#![cfg_attr(feature = "stub-gen", allow(deprecated))]

use basilica_sdk::types::{
    AvailabilityInfo as SdkAvailabilityInfo, AvailableExecutor as SdkAvailableExecutor,
    CpuSpec as SdkCpuSpec, ExecutorDetails as SdkExecutorDetails,
    ExecutorSelection as SdkExecutorSelection, GpuRequirements as SdkGpuRequirements,
    GpuSpec as SdkGpuSpec, ListAvailableExecutorsQuery as SdkListAvailableExecutorsQuery,
    ListRentalsQuery as SdkListRentalsQuery, PortMappingRequest as SdkPortMappingRequest,
    RentalState, RentalStatus as SdkRentalStatus,
    RentalStatusWithSshResponse as SdkRentalStatusWithSshResponse,
    ResourceRequirementsRequest as SdkResourceRequirementsRequest, SshAccess as SdkSshAccess,
    StartRentalApiRequest as SdkStartRentalApiRequest, VolumeMountRequest as SdkVolumeMountRequest,
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

/// Executor details including GPU and CPU specifications
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ExecutorDetails {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub gpu_specs: Vec<GpuSpec>,
    #[pyo3(get)]
    pub cpu_specs: CpuSpec,
    #[pyo3(get)]
    pub location: Option<String>,
}

impl From<SdkExecutorDetails> for ExecutorDetails {
    fn from(details: SdkExecutorDetails) -> Self {
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
    pub executor: ExecutorDetails,
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
            executor: response.executor.into(),
            ssh_credentials: response.ssh_credentials,
            created_at: response.created_at.to_rfc3339(),
            updated_at: response.updated_at.to_rfc3339(),
        }
    }
}

/// Availability information for an executor
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct AvailabilityInfo {
    #[pyo3(get)]
    pub available_until: Option<String>,
    #[pyo3(get)]
    pub verification_score: f64,
    #[pyo3(get)]
    pub uptime_percentage: f64,
}

impl From<SdkAvailabilityInfo> for AvailabilityInfo {
    fn from(info: SdkAvailabilityInfo) -> Self {
        Self {
            available_until: info.available_until.map(|dt| dt.to_rfc3339()),
            verification_score: info.verification_score,
            uptime_percentage: info.uptime_percentage,
        }
    }
}

/// Available executor with details and availability info
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct AvailableExecutor {
    #[pyo3(get)]
    pub executor: ExecutorDetails,
    #[pyo3(get)]
    pub availability: AvailabilityInfo,
}

impl From<SdkAvailableExecutor> for AvailableExecutor {
    fn from(executor: SdkAvailableExecutor) -> Self {
        Self {
            executor: executor.executor.into(),
            availability: executor.availability.into(),
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

/// GPU requirements for executor selection
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

/// Executor selection strategy
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass]
#[derive(Clone)]
pub enum ExecutorSelection {
    ExecutorId { executor_id: String },
    ExactGpuConfiguration { gpu_requirements: GpuRequirements },
}

impl From<ExecutorSelection> for SdkExecutorSelection {
    fn from(selection: ExecutorSelection) -> Self {
        match selection {
            ExecutorSelection::ExecutorId { executor_id } => {
                SdkExecutorSelection::ExecutorId { executor_id }
            }
            ExecutorSelection::ExactGpuConfiguration { gpu_requirements } => {
                SdkExecutorSelection::ExactGpuConfiguration {
                    gpu_requirements: gpu_requirements.into(),
                }
            }
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

/// Start rental API request
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct StartRentalApiRequest {
    #[pyo3(get, set)]
    pub executor_selection: ExecutorSelection,
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
    #[pyo3(get, set)]
    pub no_ssh: bool,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl StartRentalApiRequest {
    #[new]
    #[pyo3(signature = (executor_selection, container_image, ssh_public_key, environment=None, ports=None, resources=None, command=None, volumes=None, no_ssh=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        executor_selection: ExecutorSelection,
        container_image: String,
        ssh_public_key: String,
        environment: Option<HashMap<String, String>>,
        ports: Option<Vec<PortMappingRequest>>,
        resources: Option<ResourceRequirementsRequest>,
        command: Option<Vec<String>>,
        volumes: Option<Vec<VolumeMountRequest>>,
        no_ssh: bool,
    ) -> Self {
        Self {
            executor_selection,
            container_image,
            ssh_public_key,
            environment: environment.unwrap_or_default(),
            ports: ports.unwrap_or_default(),
            resources: resources.unwrap_or_default(),
            command: command.unwrap_or_default(),
            volumes: volumes.unwrap_or_default(),
            no_ssh,
        }
    }
}

impl From<StartRentalApiRequest> for SdkStartRentalApiRequest {
    fn from(req: StartRentalApiRequest) -> Self {
        Self {
            executor_selection: req.executor_selection.into(),
            container_image: req.container_image,
            ssh_public_key: req.ssh_public_key,
            environment: req.environment,
            ports: req.ports.into_iter().map(Into::into).collect(),
            resources: req.resources.into(),
            command: req.command,
            volumes: req.volumes.into_iter().map(Into::into).collect(),
            no_ssh: req.no_ssh,
        }
    }
}

/// Query parameters for listing available executors
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone, Default)]
pub struct ListAvailableExecutorsQuery {
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
impl ListAvailableExecutorsQuery {
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

impl From<ListAvailableExecutorsQuery> for SdkListAvailableExecutorsQuery {
    fn from(query: ListAvailableExecutorsQuery) -> Self {
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
