//! Python bindings for sandbox types and operations
//!
//! Provides PyO3 wrappers around basilica_sdk::sandbox.

// pyo3-stub-gen uses deprecated PyO3 APIs internally, we need to allow them
#![cfg_attr(feature = "stub-gen", allow(deprecated))]

use basilica_sdk::sandbox as sdk;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

use crate::map_api_error;

// ============================================================================
// Config & Request Types
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass]
#[derive(Clone)]
pub enum NetworkIsolation {
    None,
    Egress,
    Full,
}

impl From<NetworkIsolation> for sdk::NetworkIsolation {
    fn from(value: NetworkIsolation) -> Self {
        match value {
            NetworkIsolation::None => sdk::NetworkIsolation::None,
            NetworkIsolation::Egress => sdk::NetworkIsolation::Egress,
            NetworkIsolation::Full => sdk::NetworkIsolation::Full,
        }
    }
}

impl From<sdk::NetworkIsolation> for NetworkIsolation {
    fn from(value: sdk::NetworkIsolation) -> Self {
        match value {
            sdk::NetworkIsolation::None => NetworkIsolation::None,
            sdk::NetworkIsolation::Egress => NetworkIsolation::Egress,
            sdk::NetworkIsolation::Full => NetworkIsolation::Full,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "SandboxGpuSpec")]
#[derive(Clone)]
pub struct GpuSpec {
    #[pyo3(get, set)]
    pub count: u32,
    #[pyo3(get, set)]
    pub model: Vec<String>,
    #[pyo3(get, set)]
    pub min_cuda_version: Option<String>,
    #[pyo3(get, set)]
    pub min_gpu_memory_gb: Option<u32>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl GpuSpec {
    #[new]
    #[pyo3(signature = (count, model=None, min_cuda_version=None, min_gpu_memory_gb=None))]
    fn new(
        count: u32,
        model: Option<Vec<String>>,
        min_cuda_version: Option<String>,
        min_gpu_memory_gb: Option<u32>,
    ) -> Self {
        Self {
            count,
            model: model.unwrap_or_default(),
            min_cuda_version,
            min_gpu_memory_gb,
        }
    }
}

impl From<GpuSpec> for sdk::GpuSpec {
    fn from(spec: GpuSpec) -> Self {
        Self {
            count: spec.count,
            model: spec.model,
            min_cuda_version: spec.min_cuda_version,
            min_gpu_memory_gb: spec.min_gpu_memory_gb,
        }
    }
}

impl From<sdk::GpuSpec> for GpuSpec {
    fn from(spec: sdk::GpuSpec) -> Self {
        Self {
            count: spec.count,
            model: spec.model,
            min_cuda_version: spec.min_cuda_version,
            min_gpu_memory_gb: spec.min_gpu_memory_gb,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "SandboxResourceSpec")]
#[derive(Clone)]
pub struct ResourceSpec {
    #[pyo3(get, set)]
    pub cpu: String,
    #[pyo3(get, set)]
    pub memory: String,
    #[pyo3(get, set)]
    pub gpus: Option<GpuSpec>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl ResourceSpec {
    #[new]
    #[pyo3(signature = (cpu="500m", memory="512Mi", gpus=None))]
    fn new(cpu: &str, memory: &str, gpus: Option<GpuSpec>) -> Self {
        Self {
            cpu: cpu.to_string(),
            memory: memory.to_string(),
            gpus,
        }
    }
}

impl From<ResourceSpec> for sdk::ResourceSpec {
    fn from(spec: ResourceSpec) -> Self {
        Self {
            cpu: spec.cpu,
            memory: spec.memory,
            gpus: spec.gpus.map(Into::into),
        }
    }
}

impl From<sdk::ResourceSpec> for ResourceSpec {
    fn from(spec: sdk::ResourceSpec) -> Self {
        Self {
            cpu: spec.cpu,
            memory: spec.memory,
            gpus: spec.gpus.map(Into::into),
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "SandboxEnvVar")]
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

impl From<EnvVar> for sdk::EnvVar {
    fn from(env: EnvVar) -> Self {
        Self {
            name: env.name,
            value: env.value,
        }
    }
}

impl From<sdk::EnvVar> for EnvVar {
    fn from(env: sdk::EnvVar) -> Self {
        Self {
            name: env.name,
            value: env.value,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SandboxConfig {
    #[pyo3(get, set)]
    pub language: String,
    #[pyo3(get, set)]
    pub runtime: String,
    #[pyo3(get, set)]
    pub image: Option<String>,
    #[pyo3(get, set)]
    pub resources: ResourceSpec,
    #[pyo3(get, set)]
    pub env: Vec<EnvVar>,
    #[pyo3(get, set)]
    pub timeout_seconds: u32,
    #[pyo3(get, set)]
    pub idle_timeout_seconds: u32,
    #[pyo3(get, set)]
    pub auto_snapshot: bool,
    #[pyo3(get, set)]
    pub restore_from: Option<String>,
    #[pyo3(get, set)]
    pub network_isolation: NetworkIsolation,
    #[pyo3(get, set)]
    pub namespace: Option<String>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl SandboxConfig {
    #[new]
    #[pyo3(signature = (
        language,
        runtime="firecracker",
        image=None,
        resources=None,
        env=None,
        timeout_seconds=3600,
        idle_timeout_seconds=600,
        auto_snapshot=false,
        restore_from=None,
        network_isolation=NetworkIsolation::None,
        namespace=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        language: String,
        runtime: &str,
        image: Option<String>,
        resources: Option<ResourceSpec>,
        env: Option<Vec<EnvVar>>,
        timeout_seconds: u32,
        idle_timeout_seconds: u32,
        auto_snapshot: bool,
        restore_from: Option<String>,
        network_isolation: NetworkIsolation,
        namespace: Option<String>,
    ) -> Self {
        Self {
            language,
            runtime: runtime.to_string(),
            image,
            resources: resources.unwrap_or_else(|| ResourceSpec::new("500m", "512Mi", None)),
            env: env.unwrap_or_default(),
            timeout_seconds,
            idle_timeout_seconds,
            auto_snapshot,
            restore_from,
            network_isolation,
            namespace,
        }
    }
}

impl From<SandboxConfig> for sdk::SandboxConfig {
    fn from(config: SandboxConfig) -> Self {
        Self {
            language: config.language,
            runtime: config.runtime,
            image: config.image,
            resources: config.resources.into(),
            env: config.env.into_iter().map(Into::into).collect(),
            timeout_seconds: config.timeout_seconds,
            idle_timeout_seconds: config.idle_timeout_seconds,
            auto_snapshot: config.auto_snapshot,
            restore_from: config.restore_from,
            network_isolation: config.network_isolation.into(),
            namespace: config.namespace,
        }
    }
}

impl From<sdk::SandboxConfig> for SandboxConfig {
    fn from(config: sdk::SandboxConfig) -> Self {
        Self {
            language: config.language,
            runtime: config.runtime,
            image: config.image,
            resources: config.resources.into(),
            env: config.env.into_iter().map(Into::into).collect(),
            timeout_seconds: config.timeout_seconds,
            idle_timeout_seconds: config.idle_timeout_seconds,
            auto_snapshot: config.auto_snapshot,
            restore_from: config.restore_from,
            network_isolation: config.network_isolation.into(),
            namespace: config.namespace,
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct ExecResult {
    #[pyo3(get)]
    pub stdout: String,
    #[pyo3(get)]
    pub stderr: String,
    #[pyo3(get)]
    pub exit_code: i32,
    #[pyo3(get)]
    pub duration_ms: u64,
}

impl From<sdk::ExecResult> for ExecResult {
    fn from(result: sdk::ExecResult) -> Self {
        Self {
            stdout: result.stdout,
            stderr: result.stderr,
            exit_code: result.exit_code,
            duration_ms: result.duration_ms,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct FileInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub is_dir: bool,
    #[pyo3(get)]
    pub size: u64,
    #[pyo3(get)]
    pub modified_at: Option<String>,
}

impl From<sdk::FileInfo> for FileInfo {
    fn from(info: sdk::FileInfo) -> Self {
        Self {
            name: info.name,
            path: info.path,
            is_dir: info.is_dir,
            size: info.size,
            modified_at: info.modified_at,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SnapshotInfo {
    #[pyo3(get)]
    pub snapshot_id: String,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub size_bytes: u64,
}

impl From<sdk::SnapshotInfo> for SnapshotInfo {
    fn from(info: sdk::SnapshotInfo) -> Self {
        Self {
            snapshot_id: info.snapshot_id,
            name: info.name,
            created_at: info.created_at,
            size_bytes: info.size_bytes,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct SandboxStatus {
    #[pyo3(get)]
    pub sandbox_id: String,
    #[pyo3(get)]
    pub state: SandboxState,
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub created_at: Option<String>,
    #[pyo3(get)]
    pub last_activity_at: Option<String>,
    #[pyo3(get)]
    pub pod_name: Option<String>,
    #[pyo3(get)]
    pub node_name: Option<String>,
    #[pyo3(get)]
    pub websocket_url: String,
    #[pyo3(get)]
    pub message: Option<String>,
    #[pyo3(get)]
    pub snapshot_id: Option<String>,
}

impl From<sdk::SandboxStatus> for SandboxStatus {
    fn from(status: sdk::SandboxStatus) -> Self {
        Self {
            sandbox_id: status.sandbox_id,
            state: status.state.into(),
            language: status.language,
            created_at: status.created_at,
            last_activity_at: status.last_activity_at,
            pod_name: status.pod_name,
            node_name: status.node_name,
            websocket_url: status.websocket_url,
            message: status.message,
            snapshot_id: status.snapshot_id,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass]
#[derive(Clone)]
pub enum SandboxState {
    Creating,
    Initializing,
    Ready,
    Executing,
    Snapshotting,
    Terminating,
    Terminated,
    Failed,
}

impl From<sdk::SandboxState> for SandboxState {
    fn from(state: sdk::SandboxState) -> Self {
        match state {
            sdk::SandboxState::Creating => SandboxState::Creating,
            sdk::SandboxState::Initializing => SandboxState::Initializing,
            sdk::SandboxState::Ready => SandboxState::Ready,
            sdk::SandboxState::Executing => SandboxState::Executing,
            sdk::SandboxState::Snapshotting => SandboxState::Snapshotting,
            sdk::SandboxState::Terminating => SandboxState::Terminating,
            sdk::SandboxState::Terminated => SandboxState::Terminated,
            sdk::SandboxState::Failed => SandboxState::Failed,
        }
    }
}

// ============================================================================
// Git Types
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GitCloneResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub branch: String,
    #[pyo3(get)]
    pub commit: String,
    #[pyo3(get)]
    pub error: Option<String>,
}

impl From<sdk::GitCloneResult> for GitCloneResult {
    fn from(result: sdk::GitCloneResult) -> Self {
        Self {
            success: result.success,
            path: result.path,
            branch: result.branch,
            commit: result.commit,
            error: result.error,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GitStatusResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub branch: String,
    #[pyo3(get)]
    pub clean: bool,
    #[pyo3(get)]
    pub staged: Vec<String>,
    #[pyo3(get)]
    pub modified: Vec<String>,
    #[pyo3(get)]
    pub untracked: Vec<String>,
    #[pyo3(get)]
    pub error: Option<String>,
}

impl From<sdk::GitStatusResult> for GitStatusResult {
    fn from(result: sdk::GitStatusResult) -> Self {
        Self {
            success: result.success,
            branch: result.branch,
            clean: result.clean,
            staged: result.staged,
            modified: result.modified,
            untracked: result.untracked,
            error: result.error,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GitCommitResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub commit_hash: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub error: Option<String>,
}

impl From<sdk::GitCommitResult> for GitCommitResult {
    fn from(result: sdk::GitCommitResult) -> Self {
        Self {
            success: result.success,
            commit_hash: result.commit_hash,
            message: result.message,
            error: result.error,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GitPushResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub remote: String,
    #[pyo3(get)]
    pub branch: String,
    #[pyo3(get)]
    pub error: Option<String>,
}

impl From<sdk::GitPushResult> for GitPushResult {
    fn from(result: sdk::GitPushResult) -> Self {
        Self {
            success: result.success,
            remote: result.remote,
            branch: result.branch,
            error: result.error,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct GitPullResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub remote: String,
    #[pyo3(get)]
    pub branch: String,
    #[pyo3(get)]
    pub commits_pulled: u32,
    #[pyo3(get)]
    pub error: Option<String>,
}

impl From<sdk::GitPullResult> for GitPullResult {
    fn from(result: sdk::GitPullResult) -> Self {
        Self {
            success: result.success,
            remote: result.remote,
            branch: result.branch,
            commits_pulled: result.commits_pulled,
            error: result.error,
        }
    }
}

// ============================================================================
// LSP Types
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct LspCapabilities {
    #[pyo3(get)]
    pub completion_provider: bool,
    #[pyo3(get)]
    pub hover_provider: bool,
    #[pyo3(get)]
    pub definition_provider: bool,
    #[pyo3(get)]
    pub references_provider: bool,
    #[pyo3(get)]
    pub document_symbol_provider: bool,
    #[pyo3(get)]
    pub raw: Option<String>,
}

impl From<sdk::LspCapabilities> for LspCapabilities {
    fn from(caps: sdk::LspCapabilities) -> Self {
        Self {
            completion_provider: caps.completion_provider,
            hover_provider: caps.hover_provider,
            definition_provider: caps.definition_provider,
            references_provider: caps.references_provider,
            document_symbol_provider: caps.document_symbol_provider,
            raw: caps.raw.map(|v| v.to_string()),
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct CompletionItem {
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub kind: Option<u32>,
    #[pyo3(get)]
    pub detail: Option<String>,
    #[pyo3(get)]
    pub documentation: Option<String>,
    #[pyo3(get)]
    pub insert_text: Option<String>,
    #[pyo3(get)]
    pub sort_text: Option<String>,
}

impl From<sdk::CompletionItem> for CompletionItem {
    fn from(item: sdk::CompletionItem) -> Self {
        Self {
            label: item.label,
            kind: item.kind,
            detail: item.detail,
            documentation: item.documentation,
            insert_text: item.insert_text,
            sort_text: item.sort_text,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct Position {
    #[pyo3(get)]
    pub line: u32,
    #[pyo3(get)]
    pub character: u32,
}

impl From<sdk::Position> for Position {
    fn from(pos: sdk::Position) -> Self {
        Self {
            line: pos.line,
            character: pos.character,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct HoverResult {
    #[pyo3(get)]
    pub contents: String,
    #[pyo3(get)]
    pub range_start: Option<Position>,
    #[pyo3(get)]
    pub range_end: Option<Position>,
}

impl From<sdk::HoverResult> for HoverResult {
    fn from(result: sdk::HoverResult) -> Self {
        Self {
            contents: result.contents,
            range_start: result.range_start.map(Into::into),
            range_end: result.range_end.map(Into::into),
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct Diagnostic {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub severity: u32,
    #[pyo3(get)]
    pub line: u32,
    #[pyo3(get)]
    pub character: u32,
    #[pyo3(get)]
    pub end_line: Option<u32>,
    #[pyo3(get)]
    pub end_character: Option<u32>,
    #[pyo3(get)]
    pub source: Option<String>,
    #[pyo3(get)]
    pub code: Option<String>,
}

impl From<sdk::Diagnostic> for Diagnostic {
    fn from(d: sdk::Diagnostic) -> Self {
        Self {
            message: d.message,
            severity: d.severity,
            line: d.line,
            character: d.character,
            end_line: d.end_line,
            end_character: d.end_character,
            source: d.source,
            code: d.code,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
#[derive(Clone)]
pub struct Location {
    #[pyo3(get)]
    pub uri: String,
    #[pyo3(get)]
    pub line: u32,
    #[pyo3(get)]
    pub character: u32,
}

impl From<sdk::Location> for Location {
    fn from(loc: sdk::Location) -> Self {
        Self {
            uri: loc.uri,
            line: loc.line,
            character: loc.character,
        }
    }
}

// ============================================================================
// Sandbox Wrapper
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(name = "Sandbox")]
pub struct PySandbox {
    inner: Arc<sdk::Sandbox>,
    runtime: Runtime,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PySandbox {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "Use Sandbox.create or Sandbox.create_async to construct a sandbox.",
        ))
    }

    #[classmethod]
    #[pyo3(signature = (base_url, api_key=None, config=None))]
    fn create(
        _cls: &Bound<'_, PyType>,
        base_url: String,
        api_key: Option<String>,
        config: Option<SandboxConfig>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let api_key = api_key.or_else(|| std::env::var("BASILICA_API_TOKEN").ok());
        let api_key = api_key.ok_or_else(|| {
            PyRuntimeError::new_err(
                "No API key provided. Please provide an API key directly or set BASILICA_API_TOKEN environment variable.",
            )
        })?;

        let config = config.unwrap_or_else(|| {
            SandboxConfig::new(
                "python".to_string(),
                "firecracker",
                None,
                None,
                None,
                3600,
                600,
                false,
                None,
                NetworkIsolation::None,
                None,
            )
        });
        let sdk_config: sdk::SandboxConfig = config.into();
        let sandbox = runtime
            .block_on(async { sdk::Sandbox::create(base_url, Some(api_key), sdk_config).await })
            .map_err(map_api_error)?;

        Ok(Self {
            inner: Arc::new(sandbox),
            runtime,
        })
    }

    #[classmethod]
    #[pyo3(signature = (base_url, api_key=None, config=None))]
    fn create_async<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        base_url: String,
        api_key: Option<String>,
        config: Option<SandboxConfig>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let api_key = api_key.or_else(|| std::env::var("BASILICA_API_TOKEN").ok());
        let api_key = api_key.ok_or_else(|| {
            PyRuntimeError::new_err(
                "No API key provided. Please provide an API key directly or set BASILICA_API_TOKEN environment variable.",
            )
        })?;

        let config = config.unwrap_or_else(|| {
            SandboxConfig::new(
                "python".to_string(),
                "firecracker",
                None,
                None,
                None,
                3600,
                600,
                false,
                None,
                NetworkIsolation::None,
                None,
            )
        });
        let sdk_config: sdk::SandboxConfig = config.into();

        pyo3_async_runtimes::tokio::future_into_py::<_, Py<PySandbox>>(py, async move {
            let runtime = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
            let sandbox = sdk::Sandbox::create(base_url, Some(api_key), sdk_config)
                .await
                .map_err(map_api_error)?;
            Python::attach(|py| {
                Py::new(
                    py,
                    PySandbox {
                        inner: Arc::new(sandbox),
                        runtime,
                    },
                )
            })
        })
    }

    #[classmethod]
    #[pyo3(signature = (base_url, api_key=None, sandbox_id=None))]
    fn get(
        _cls: &Bound<'_, PyType>,
        base_url: String,
        api_key: Option<String>,
        sandbox_id: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let api_key = api_key.or_else(|| std::env::var("BASILICA_API_TOKEN").ok());
        let api_key = api_key.ok_or_else(|| {
            PyRuntimeError::new_err(
                "No API key provided. Please provide an API key directly or set BASILICA_API_TOKEN environment variable.",
            )
        })?;
        let sandbox_id =
            sandbox_id.ok_or_else(|| PyRuntimeError::new_err("sandbox_id is required"))?;

        let temp = sdk::Sandbox::from_id(&base_url, Some(api_key.clone()), &sandbox_id);
        let status = runtime
            .block_on(async { temp.status().await })
            .map_err(map_api_error)?;

        let sandbox = sdk::Sandbox::from_id_with_language(
            base_url,
            Some(api_key),
            sandbox_id,
            status.language.clone(),
        );

        Ok(Self {
            inner: Arc::new(sandbox),
            runtime,
        })
    }

    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    fn language(&self) -> String {
        self.inner.language().to_string()
    }

    fn status(&self) -> PyResult<SandboxStatus> {
        let sandbox = Arc::clone(&self.inner);
        let status = self
            .runtime
            .block_on(async move { sandbox.status().await })
            .map_err(map_api_error)?;
        Ok(status.into())
    }

    fn status_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sandbox = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py::<_, SandboxStatus>(py, async move {
            let status = sandbox.status().await.map_err(map_api_error)?;
            Ok(status.into())
        })
    }

    fn wait_until_ready(&self, timeout_seconds: u64) -> PyResult<SandboxStatus> {
        let sandbox = Arc::clone(&self.inner);
        let status = self
            .runtime
            .block_on(async move {
                sandbox
                    .wait_until_ready(Duration::from_secs(timeout_seconds))
                    .await
            })
            .map_err(map_api_error)?;
        Ok(status.into())
    }

    fn wait_until_ready_async<'py>(
        &self,
        py: Python<'py>,
        timeout_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sandbox = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py::<_, SandboxStatus>(py, async move {
            let status = sandbox
                .wait_until_ready(Duration::from_secs(timeout_seconds))
                .await
                .map_err(map_api_error)?;
            Ok(status.into())
        })
    }

    #[pyo3(signature = (code, entrypoint=None, args=None, env=None, timeout_seconds=None))]
    fn run(
        &self,
        code: String,
        entrypoint: Option<String>,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        timeout_seconds: Option<u32>,
    ) -> PyResult<ExecResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                let args_ref = args
                    .as_ref()
                    .map(|a| a.iter().map(String::as_str).collect::<Vec<_>>());
                sandbox
                    .run_with_options(
                        &code,
                        entrypoint.as_deref(),
                        args_ref.as_ref().map(|v| v.as_slice()),
                        env.map(map_env),
                        timeout_seconds,
                    )
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (code, entrypoint=None, args=None, env=None, timeout_seconds=None))]
    fn run_async<'py>(
        &self,
        py: Python<'py>,
        code: String,
        entrypoint: Option<String>,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        timeout_seconds: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sandbox = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py::<_, ExecResult>(py, async move {
            let args_ref = args
                .as_ref()
                .map(|a| a.iter().map(String::as_str).collect::<Vec<_>>());
            let result = sandbox
                .run_with_options(
                    &code,
                    entrypoint.as_deref(),
                    args_ref.as_ref().map(|v| v.as_slice()),
                    env.map(map_env),
                    timeout_seconds,
                )
                .await
                .map_err(map_api_error)?;
            Ok(result.into())
        })
    }

    #[pyo3(signature = (command, workdir=None, stdin=None, env=None, timeout_seconds=None))]
    fn exec(
        &self,
        command: Vec<String>,
        workdir: Option<String>,
        stdin: Option<String>,
        env: Option<HashMap<String, String>>,
        timeout_seconds: Option<u32>,
    ) -> PyResult<ExecResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                let command_ref: Vec<&str> = command.iter().map(String::as_str).collect();
                sandbox
                    .exec_with_options(
                        &command_ref,
                        stdin.as_deref(),
                        workdir.as_deref(),
                        env.map(map_env),
                        timeout_seconds,
                    )
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (command, workdir=None, stdin=None, env=None, timeout_seconds=None))]
    fn exec_async<'py>(
        &self,
        py: Python<'py>,
        command: Vec<String>,
        workdir: Option<String>,
        stdin: Option<String>,
        env: Option<HashMap<String, String>>,
        timeout_seconds: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sandbox = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py::<_, ExecResult>(py, async move {
            let command_ref: Vec<&str> = command.iter().map(String::as_str).collect();
            let result = sandbox
                .exec_with_options(
                    &command_ref,
                    stdin.as_deref(),
                    workdir.as_deref(),
                    env.map(map_env),
                    timeout_seconds,
                )
                .await
                .map_err(map_api_error)?;
            Ok(result.into())
        })
    }

    #[pyo3(signature = (path, recursive=false))]
    fn list_files(&self, path: String, recursive: bool) -> PyResult<Vec<FileInfo>> {
        let sandbox = Arc::clone(&self.inner);
        let files = self
            .runtime
            .block_on(async move { sandbox.list_files_with_options(&path, recursive).await })
            .map_err(map_api_error)?;
        Ok(files.into_iter().map(Into::into).collect())
    }

    fn read_file(&self, path: String) -> PyResult<String> {
        let sandbox = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { sandbox.read_file(&path).await })
            .map_err(map_api_error)
    }

    fn write_file(&self, path: String, content: String) -> PyResult<()> {
        let sandbox = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { sandbox.write_file(&path, &content).await })
            .map_err(map_api_error)
    }

    fn create_snapshot(&self, name: Option<String>) -> PyResult<SnapshotInfo> {
        let sandbox = Arc::clone(&self.inner);
        let snapshot = self
            .runtime
            .block_on(async move { sandbox.create_snapshot(name.as_deref()).await })
            .map_err(map_api_error)?;
        Ok(snapshot.into())
    }

    fn delete(&self) -> PyResult<()> {
        let sandbox = self.inner.as_ref().clone();
        self.runtime
            .block_on(async move { sandbox.delete().await })
            .map_err(map_api_error)
    }

    fn websocket_url(&self) -> String {
        self.inner.websocket_url()
    }

    // Git operations
    #[pyo3(signature = (url, path=None, branch=None, depth=None))]
    fn git_clone(
        &self,
        url: String,
        path: Option<String>,
        branch: Option<String>,
        depth: Option<u32>,
    ) -> PyResult<GitCloneResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                sandbox
                    .git_clone(&url, path.as_deref(), branch.as_deref(), depth)
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (path=None))]
    fn git_status(&self, path: Option<String>) -> PyResult<GitStatusResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move { sandbox.git_status(path.as_deref()).await })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (message, path=None, author=None))]
    fn git_commit(
        &self,
        message: String,
        path: Option<String>,
        author: Option<String>,
    ) -> PyResult<GitCommitResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                sandbox
                    .git_commit(&message, path.as_deref(), author.as_deref())
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (path=None, remote=None, branch=None))]
    fn git_push(
        &self,
        path: Option<String>,
        remote: Option<String>,
        branch: Option<String>,
    ) -> PyResult<GitPushResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                sandbox
                    .git_push(path.as_deref(), remote.as_deref(), branch.as_deref())
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (path=None, remote=None, branch=None))]
    fn git_pull(
        &self,
        path: Option<String>,
        remote: Option<String>,
        branch: Option<String>,
    ) -> PyResult<GitPullResult> {
        let sandbox = Arc::clone(&self.inner);
        let result = self
            .runtime
            .block_on(async move {
                sandbox
                    .git_pull(path.as_deref(), remote.as_deref(), branch.as_deref())
                    .await
            })
            .map_err(map_api_error)?;
        Ok(result.into())
    }

    // LSP operations
    #[pyo3(signature = (language=None, root_path=None))]
    fn lsp_init(
        &self,
        language: Option<String>,
        root_path: Option<String>,
    ) -> PyResult<LspCapabilities> {
        let sandbox = Arc::clone(&self.inner);
        let root_path = root_path.unwrap_or_else(|| "/workspace".to_string());
        let caps = self
            .runtime
            .block_on(async move { sandbox.lsp_init(language.as_deref(), &root_path).await })
            .map_err(map_api_error)?;
        Ok(caps.into())
    }

    fn lsp_completion(
        &self,
        file: String,
        line: u32,
        character: u32,
    ) -> PyResult<Vec<CompletionItem>> {
        let sandbox = Arc::clone(&self.inner);
        let items = self
            .runtime
            .block_on(async move { sandbox.lsp_completion(&file, line, character).await })
            .map_err(map_api_error)?;
        Ok(items.into_iter().map(Into::into).collect())
    }

    fn lsp_hover(&self, file: String, line: u32, character: u32) -> PyResult<Option<HoverResult>> {
        let sandbox = Arc::clone(&self.inner);
        let hover = self
            .runtime
            .block_on(async move { sandbox.lsp_hover(&file, line, character).await })
            .map_err(map_api_error)?;
        Ok(hover.map(Into::into))
    }

    fn lsp_definition(&self, file: String, line: u32, character: u32) -> PyResult<Vec<Location>> {
        let sandbox = Arc::clone(&self.inner);
        let locations = self
            .runtime
            .block_on(async move { sandbox.lsp_definition(&file, line, character).await })
            .map_err(map_api_error)?;
        Ok(locations.into_iter().map(Into::into).collect())
    }

    fn lsp_did_open(&self, file: String, content: String) -> PyResult<()> {
        let sandbox = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { sandbox.lsp_did_open(&file, &content).await })
            .map_err(map_api_error)
    }

    fn lsp_did_change(&self, file: String, content: String) -> PyResult<()> {
        let sandbox = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { sandbox.lsp_did_change(&file, &content).await })
            .map_err(map_api_error)
    }

    fn lsp_shutdown(&self) -> PyResult<()> {
        let sandbox = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { sandbox.lsp_shutdown().await })
            .map_err(map_api_error)
    }
}

fn map_env(env: HashMap<String, String>) -> Vec<sdk::EnvVar> {
    env.into_iter()
        .map(|(name, value)| sdk::EnvVar { name, value })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_isolation_roundtrip() {
        let value = NetworkIsolation::Egress;
        let rust_value: sdk::NetworkIsolation = value.clone().into();
        assert_eq!(rust_value, sdk::NetworkIsolation::Egress);
        let back: NetworkIsolation = rust_value.into();
        assert_eq!(matches!(back, NetworkIsolation::Egress), true);
    }

    #[test]
    fn test_resource_spec_conversion() {
        let spec = ResourceSpec::new(
            "250m",
            "256Mi",
            Some(GpuSpec::new(1, Some(vec!["A100".to_string()]), None, None)),
        );
        let rust_spec: sdk::ResourceSpec = spec.into();
        assert_eq!(rust_spec.cpu, "250m");
        assert_eq!(rust_spec.memory, "256Mi");
        assert!(rust_spec.gpus.is_some());
    }

    #[test]
    fn test_env_map() {
        let mut env = HashMap::new();
        env.insert("KEY".to_string(), "value".to_string());
        let vars = map_env(env);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].name, "KEY");
        assert_eq!(vars[0].value, "value");
    }
}
