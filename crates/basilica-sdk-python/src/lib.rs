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
    AvailableNode, HealthCheckResponse, ListAvailableNodesQuery, ListRentalsQuery, RentalResponse,
    RentalStatusWithSshResponse, StartRentalApiRequest,
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
    #[new]
    #[pyo3(signature = (base_url, api_key=None))]
    fn new(base_url: String, api_key: Option<String>) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        // Check for API key - either provided directly or from BASILICA_API_TOKEN env var
        let api_key = api_key.or_else(|| std::env::var("BASILICA_API_TOKEN").ok());

        let api_key = api_key.ok_or_else(|| {
            PyRuntimeError::new_err(
                "No API key provided. Please provide an API key directly or set BASILICA_API_TOKEN environment variable. \
                Create a key using: basilica tokens create"
            )
        })?;

        let client = runtime
            .block_on(async {
                ClientBuilder::default()
                    .base_url(base_url)
                    .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
                    .with_api_key(&api_key)
                    .build()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create client: {}", e)))?;

        Ok(Self {
            inner: Arc::new(client),
            runtime,
        })
    }

    // Python SDK uses API key authentication
    // Users should create API keys via CLI: `basilica tokens create`
    // Then use the Python SDK with the key directly or via BASILICA_API_TOKEN environment variable

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
        let request = request.into();

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

/// Helper function to create node selection by ID
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
fn node_by_id(node_id: String) -> types::NodeSelection {
    types::NodeSelection::NodeId { node_id }
}

/// Helper function to create node selection by GPU requirements (exact count)
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
fn node_by_gpu(gpu_requirements: types::GpuRequirements) -> types::NodeSelection {
    types::NodeSelection::ExactGpuConfiguration { gpu_requirements }
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
    m.add_class::<types::NodeSelection>()?;
    m.add_class::<types::GpuRequirements>()?;
    m.add_class::<types::PortMappingRequest>()?;
    m.add_class::<types::ResourceRequirementsRequest>()?;
    m.add_class::<types::VolumeMountRequest>()?;
    m.add_class::<types::ListAvailableNodesQuery>()?;
    m.add_class::<types::ListRentalsQuery>()?;

    // Helper functions
    m.add_function(wrap_pyfunction!(node_by_id, m)?)?;
    m.add_function(wrap_pyfunction!(node_by_gpu, m)?)?;

    Ok(())
}

// Define stub info gatherer for generating Python stub files
#[cfg(feature = "stub-gen")]
define_stub_info_gatherer!(stub_info);
