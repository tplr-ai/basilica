//! Python bindings for the Basilica SDK
#![allow(clippy::useless_conversion)]

mod types;

use basilica_sdk::{
    client::{DEFAULT_API_URL, DEFAULT_TIMEOUT_SECS},
    BasilicaClient as RustClient, ClientBuilder,
};
use pyo3::exceptions::{
    PyConnectionError, PyKeyError, PyPermissionError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::define_stub_info_gatherer;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction};
use pythonize::pythonize;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

use crate::types::{
    AvailableNode, CpuOffering, CpuRentalResponse, CreateDeploymentRequest,
    DeleteDeploymentResponse, DeleteShareTokenResponse, DeploymentListResponse, DeploymentResponse,
    EnrollMetadataResponse, HealthCheckResponse, ListAvailableNodesQuery, ListCpuRentalsResponse,
    ListRentalsQuery, PublicDeploymentMetadataResponse, RegenerateShareTokenResponse,
    RentalResponse, RentalStatusWithSshResponse, ShareTokenStatusResponse, SshKeyResponse,
    StartCpuRentalRequest, StartRentalApiRequest, StopCpuRentalResponse,
};

/// Python wrapper for BasilicaClient
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
struct BasilicaClient {
    inner: Arc<RustClient>,
    runtime: Runtime,
}

// Small helper to convert serializable Rust values into PyObject without
// re-wrapping PyErr (avoids clippy::useless_conversion).
fn to_pyobject<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<pyo3::PyAny>> {
    // `pythonize` already returns `PyResult<_>` so just propagate as-is.
    Ok(pythonize(py, value)?.unbind())
}

#[pymethods]
impl BasilicaClient {
    /// Create a new BasilicaClient
    ///
    /// Args:
    ///     base_url: The base URL of the Basilica API
    ///     api_key: Optional authentication token from 'basilica tokens create'
    ///
    /// Authentication priority:
    ///   1. Explicit `api_key` parameter
    ///   2. `BASILICA_API_TOKEN` environment variable
    ///   3. File-based auth (reads CLI tokens from `basilica login`)
    #[new]
    #[pyo3(signature = (base_url, api_key=None))]
    fn new(base_url: String, api_key: Option<String>) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let api_key = api_key.or_else(|| std::env::var("BASILICA_API_TOKEN").ok());

        let mut builder = ClientBuilder::default()
            .base_url(base_url)
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        if let Some(key) = api_key {
            builder = builder.with_api_key(&key);
        } else {
            builder = builder.with_file_auth();
        }

        let client = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create client: {}", e)))?;

        Ok(Self {
            inner: Arc::new(client),
            runtime,
        })
    }

    /// Check the health of the API
    fn health_check(&self, py: Python) -> PyResult<HealthCheckResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.health_check().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// List available nodes
    ///
    /// Args:
    ///     query: Optional query parameters
    #[pyo3(signature = (query=None))]
    fn list_nodes(
        &self,
        py: Python,
        query: Option<ListAvailableNodesQuery>,
    ) -> PyResult<Vec<AvailableNode>> {
        let client = Arc::clone(&self.inner);

        // Convert Python query to SDK query if provided
        let query = query.map(Into::into);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_available_nodes(query).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response
            .available_nodes
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Start a new rental
    ///
    /// Args:
    ///     request: Rental request parameters
    fn start_rental(&self, py: Python, request: StartRentalApiRequest) -> PyResult<RentalResponse> {
        let client = Arc::clone(&self.inner);

        // Convert Python request to SDK request
        let request = basilica_sdk::types::StartRentalApiRequest::try_from(request)
            .map_err(PyValueError::new_err)?;

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.start_rental(request).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get rental status
    ///
    /// Args:
    ///     rental_id: The rental ID
    fn get_rental(&self, py: Python, rental_id: String) -> PyResult<RentalStatusWithSshResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_rental_status(&rental_id).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Stop a rental
    ///
    /// Args:
    ///     rental_id: The rental ID
    fn stop_rental(&self, _py: Python, rental_id: String) -> PyResult<()> {
        let client = Arc::clone(&self.inner);

        self.runtime
            .block_on(async move { client.stop_rental(&rental_id).await })
            .map_err(|e| self.map_error_to_python(e))
    }

    /// List rentals
    ///
    /// Args:
    ///     query: Optional query parameters
    #[pyo3(signature = (query=None))]
    fn list_rentals(
        &self,
        py: Python,
        query: Option<ListRentalsQuery>,
    ) -> PyResult<Py<pyo3::PyAny>> {
        let client = Arc::clone(&self.inner);

        // Convert Python query to SDK query if provided
        let query = query.map(Into::into);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_rentals(query).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        // Keep list_rentals as PyObject for now since it returns a complex structure
        to_pyobject(py, &response)
    }

    /// Create a new deployment
    ///
    /// Args:
    ///     request: Deployment request parameters
    fn create_deployment(
        &self,
        py: Python,
        request: CreateDeploymentRequest,
    ) -> PyResult<DeploymentResponse> {
        let client = Arc::clone(&self.inner);

        let request = request.into();

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.create_deployment(request).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get deployment status by instance name
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn get_deployment(&self, py: Python, instance_name: String) -> PyResult<DeploymentResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_deployment(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Delete a deployment
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn delete_deployment(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<DeleteDeploymentResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.delete_deployment(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Restart a deployment (rolling restart)
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn restart_deployment(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<DeploymentResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.restart_deployment(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// List all deployments for the authenticated user
    fn list_deployments(&self, py: Python) -> PyResult<DeploymentListResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_deployments().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get deployment logs
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    ///     follow: Whether to follow logs (default: False)
    ///     tail: Optional number of lines to tail
    #[pyo3(signature = (instance_name, follow=false, tail=None))]
    fn get_deployment_logs(
        &self,
        py: Python,
        instance_name: String,
        follow: bool,
        tail: Option<u32>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client
                        .get_deployment_logs(&instance_name, follow, tail)
                        .await
                })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        let text = py
            .detach(|| self.runtime.block_on(async move { response.text().await }))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read response: {}", e)))?;

        Ok(text)
    }

    // ===== Share Token Management =====

    /// Regenerate a share token for a private deployment
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn regenerate_share_token(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<RegenerateShareTokenResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.regenerate_share_token(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get share token status for a private deployment
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn get_share_token_status(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<ShareTokenStatusResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_share_token_status(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Delete/revoke the share token for a private deployment
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn delete_share_token(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<DeleteShareTokenResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.delete_share_token(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    // ===== Public Deployment Metadata =====

    /// Enroll or unenroll a deployment in public metadata exposure
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    ///     enabled: Whether to enable or disable metadata enrollment
    fn enroll_metadata(
        &self,
        py: Python,
        instance_name: String,
        enabled: bool,
    ) -> PyResult<EnrollMetadataResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.enroll_metadata(&instance_name, enabled).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Check the current metadata enrollment status of a deployment
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn get_enrollment_status(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<EnrollMetadataResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_enrollment_status(&instance_name).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Fetch public metadata for a deployment (no authentication required)
    ///
    /// Args:
    ///     instance_name: The deployment instance name
    fn get_public_deployment_metadata(
        &self,
        py: Python,
        instance_name: String,
    ) -> PyResult<PublicDeploymentMetadataResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client.get_public_deployment_metadata(&instance_name).await
                })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get account balance
    fn get_balance(&self, py: Python) -> PyResult<Py<pyo3::PyAny>> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_balance().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        to_pyobject(py, &response)
    }

    /// List usage history
    ///
    /// Args:
    ///     limit: Maximum number of records (default: 50)
    ///     offset: Number of records to skip (default: 0)
    #[pyo3(signature = (limit=50, offset=0))]
    fn list_usage_history(&self, py: Python, limit: u32, offset: u32) -> PyResult<Py<pyo3::PyAny>> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client.list_usage_history(Some(limit), Some(offset)).await
                })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        to_pyobject(py, &response)
    }

    // ===== SSH Key Management =====

    /// Register an SSH key for the authenticated user
    ///
    /// Args:
    ///     name: A friendly name for the SSH key
    ///     public_key: The SSH public key content
    fn register_ssh_key(
        &self,
        py: Python,
        name: String,
        public_key: String,
    ) -> PyResult<SshKeyResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.register_ssh_key(&name, &public_key).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Get the authenticated user's registered SSH key
    fn get_ssh_key(&self, py: Python) -> PyResult<Option<SshKeyResponse>> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.get_ssh_key().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.map(Into::into))
    }

    /// Delete the authenticated user's SSH key
    fn delete_ssh_key(&self, _py: Python) -> PyResult<()> {
        let client = Arc::clone(&self.inner);

        self.runtime
            .block_on(async move { client.delete_ssh_key().await })
            .map_err(|e| self.map_error_to_python(e))
    }

    // ===== CPU Rental Methods =====

    /// List available CPU offerings from secure cloud providers
    fn list_cpu_offerings(&self, py: Python) -> PyResult<Vec<CpuOffering>> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_cpu_offerings().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into_iter().map(Into::into).collect())
    }

    /// List CPU rentals for the authenticated user
    fn list_cpu_rentals(&self, py: Python) -> PyResult<ListCpuRentalsResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_cpu_rentals().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Start a CPU rental
    ///
    /// Args:
    ///     request: CPU rental request parameters
    fn start_cpu_rental(
        &self,
        py: Python,
        request: StartCpuRentalRequest,
    ) -> PyResult<CpuRentalResponse> {
        let client = Arc::clone(&self.inner);
        let request = request.into();

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.start_cpu_rental(request).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Stop a CPU rental
    ///
    /// Args:
    ///     rental_id: The rental ID to stop
    fn stop_cpu_rental(&self, py: Python, rental_id: String) -> PyResult<StopCpuRentalResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.stop_cpu_rental(&rental_id).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    // ===== GPU Rental Methods =====

    /// List available GPU offerings from secure cloud providers
    ///
    /// Args:
    ///     query: Optional GpuPriceQuery to filter by interconnect, geo, spot
    #[pyo3(signature = (query=None))]
    fn list_secure_cloud_gpus(
        &self,
        py: Python,
        query: Option<types::GpuPriceQuery>,
    ) -> PyResult<Vec<types::GpuOffering>> {
        let client = Arc::clone(&self.inner);
        let sdk_query: basilica_sdk::types::GpuPriceQuery =
            query.map(Into::into).unwrap_or_default();

        let response = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client.list_secure_cloud_gpus_filtered(&sdk_query).await
                })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into_iter().map(Into::into).collect())
    }

    /// Start a secure cloud GPU rental
    ///
    /// Args:
    ///     request: GPU rental request parameters
    fn start_secure_cloud_rental(
        &self,
        py: Python,
        request: types::StartSecureCloudRentalRequest,
    ) -> PyResult<types::SecureCloudRentalResponse> {
        let client = Arc::clone(&self.inner);
        let request = request.into();

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.start_secure_cloud_rental(request).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// Stop a secure cloud GPU rental
    ///
    /// Args:
    ///     rental_id: The rental ID to stop
    fn stop_secure_cloud_rental(
        &self,
        py: Python,
        rental_id: String,
    ) -> PyResult<types::StopSecureCloudRentalResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.stop_secure_cloud_rental(&rental_id).await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }

    /// List secure cloud GPU rentals for the authenticated user
    fn list_secure_cloud_rentals(
        &self,
        py: Python,
    ) -> PyResult<types::ListSecureCloudRentalsResponse> {
        let client = Arc::clone(&self.inner);

        let response = py
            .detach(|| {
                self.runtime
                    .block_on(async move { client.list_secure_cloud_rentals().await })
            })
            .map_err(|e| self.map_error_to_python(e))?;

        Ok(response.into())
    }
}

impl BasilicaClient {
    /// Map Rust errors to appropriate Python exception types
    fn map_error_to_python(&self, error: basilica_sdk::ApiError) -> PyErr {
        use basilica_sdk::ApiError;

        match error {
            ApiError::InvalidRequest { message } => PyValueError::new_err(message),
            ApiError::NotFound { resource } => {
                PyKeyError::new_err(format!("Not found: {}", resource))
            }
            ApiError::Authentication { message } | ApiError::MissingAuthentication { message } => {
                PyPermissionError::new_err(format!("Authentication error: {}. Please provide a valid API key or set BASILICA_API_TOKEN environment variable.", message))
            }
            ApiError::Authorization { message } => PyPermissionError::new_err(message),
            ApiError::HttpClient(e) => PyConnectionError::new_err(e.to_string()),
            ApiError::BadRequest { message } => PyValueError::new_err(message),
            ApiError::Internal { message } => PyRuntimeError::new_err(message),
            _ => PyRuntimeError::new_err(error.to_string()),
        }
    }
}

/// Python module for Basilica SDK
#[pymodule]
fn _basilica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add constants
    m.add("DEFAULT_API_URL", DEFAULT_API_URL)?;
    m.add("DEFAULT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS)?;
    m.add(
        "DEFAULT_CONTAINER_IMAGE",
        "nvidia/cuda:12.2.0-base-ubuntu22.04",
    )?;
    m.add("DEFAULT_GPU_TYPE", "b200")?;
    m.add("DEFAULT_GPU_COUNT", 1)?;
    m.add("DEFAULT_GPU_MIN_MEMORY_GB", 0)?;
    m.add("DEFAULT_CPU_CORES", 0.0)?;
    m.add("DEFAULT_MEMORY_MB", 0)?;
    m.add("DEFAULT_STORAGE_MB", 0)?;
    m.add("DEFAULT_PORT_PROTOCOL", "tcp")?;
    m.add("DEFAULT_SSH_USER", "root")?;
    m.add("DEFAULT_SSH_PORT", 22)?;

    // Core client
    m.add_class::<BasilicaClient>()?;

    // Response types
    m.add_class::<types::HealthCheckResponse>()?;
    m.add_class::<types::RentalResponse>()?;
    m.add_class::<types::RentalStatusWithSshResponse>()?;
    m.add_class::<types::RentalStatus>()?;
    m.add_class::<types::SshAccess>()?;
    m.add_class::<types::NodeDetails>()?;
    m.add_class::<types::GpuSpec>()?;
    m.add_class::<types::CpuSpec>()?;
    m.add_class::<types::AvailableNode>()?;
    m.add_class::<types::AvailabilityInfo>()?;

    // Request types
    m.add_class::<types::StartRentalApiRequest>()?;
    m.add_class::<types::GpuRequirements>()?;
    m.add_class::<types::PortMappingRequest>()?;
    m.add_class::<types::ResourceRequirementsRequest>()?;
    m.add_class::<types::VolumeMountRequest>()?;
    m.add_class::<types::ListAvailableNodesQuery>()?;
    m.add_class::<types::ListRentalsQuery>()?;

    // Deployment types
    m.add_class::<types::EnvVar>()?;
    m.add_class::<types::GpuRequirementsSpec>()?;
    m.add_class::<types::ResourceRequirements>()?;
    m.add_class::<types::ReplicaStatus>()?;
    m.add_class::<types::PodInfo>()?;
    m.add_class::<types::SpreadMode>()?;
    m.add_class::<types::TopologySpreadConfig>()?;
    m.add_class::<types::StorageBackend>()?;
    m.add_class::<types::PersistentStorageSpec>()?;
    m.add_class::<types::StorageSpec>()?;
    m.add_class::<types::ProbeConfig>()?;
    m.add_class::<types::HealthCheckConfig>()?;
    m.add_class::<types::WebSocketConfig>()?;
    m.add_class::<types::CreateDeploymentRequest>()?;
    m.add_class::<types::DeploymentResponse>()?;
    m.add_class::<types::DeploymentSummary>()?;
    m.add_class::<types::DeploymentListResponse>()?;
    m.add_class::<types::DeleteDeploymentResponse>()?;

    // Share Token types
    m.add_class::<types::RegenerateShareTokenResponse>()?;
    m.add_class::<types::ShareTokenStatusResponse>()?;
    m.add_class::<types::DeleteShareTokenResponse>()?;

    // Public Metadata types
    m.add_class::<types::EnrollMetadataResponse>()?;
    m.add_class::<types::PublicDeploymentMetadataResponse>()?;

    // SSH Key types
    m.add_class::<types::SshKeyResponse>()?;

    // CPU Rental types
    m.add_class::<types::CpuOffering>()?;
    m.add_class::<types::StartCpuRentalRequest>()?;
    m.add_class::<types::CpuRentalResponse>()?;
    m.add_class::<types::StopCpuRentalResponse>()?;
    m.add_class::<types::CpuRentalListItem>()?;
    m.add_class::<types::ListCpuRentalsResponse>()?;

    // GPU Rental types (secure cloud)
    m.add_class::<types::GpuPriceQuery>()?;
    m.add_class::<types::GpuOffering>()?;
    m.add_class::<types::StartSecureCloudRentalRequest>()?;
    m.add_class::<types::SecureCloudRentalResponse>()?;
    m.add_class::<types::StopSecureCloudRentalResponse>()?;
    m.add_class::<types::SecureCloudRentalListItem>()?;
    m.add_class::<types::ListSecureCloudRentalsResponse>()?;

    Ok(())
}

// Define stub info gatherer for generating Python stub files
#[cfg(feature = "stub-gen")]
define_stub_info_gatherer!(stub_info);
