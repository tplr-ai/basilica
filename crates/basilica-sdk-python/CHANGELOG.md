# Changelog

All notable changes to the Basilica Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.25.2] - 2026-03-11

### Fixed
- Linux wheels now correctly tagged `cp310-abi3` instead of `cp38-cp38`
- Maturin builds now explicitly use Python 3.10 interpreter inside manylinux containers

## [0.25.1] - 2026-03-11

### Fixed
- Enable abi3 stable ABI for wheel builds so Linux wheels are compatible with Python 3.10+
- Add Linux aarch64 pre-built wheels
- Previously Linux x86_64 wheels were tagged cp38-only, forcing source compilation on Python 3.10+

### Added
- `health_check` parameter on `deploy()`, `deploy_async()`, `deploy_vllm()`, `deploy_sglang()`, `create_deployment()`, `create_deployment_async()`
- `HealthCheckConfig` and `ProbeConfig` types for configuring startup, liveness, and readiness probes
- Default health checks for `deploy_vllm()` (port 8000) and `deploy_sglang()` (port 30000) with 10-minute startup tolerance
- `health_check` support in `@deployment` decorator and `DeploymentSpec`
- `deploy_sglang_health_check.py` example demonstrating custom health probes for large model deployments

## [0.16.0] - 2026-02-02
### Added
- OpenClaw summon template support and provider preset defaults.

## [0.15.0] - 2026-01-30

### Added
- Share token management: `regenerate_share_token()`, `get_share_token_status()`, `revoke_share_token()`
- Health check binding: `health_check()`
- `is_spot` field on `GpuOffering`, `SecureCloudRentalResponse`, `CpuRentalResponse`, `SecureCloudRentalListItem`, `CpuRentalListItem`

## [0.14.0] - 2026-01-26

### Added
- Topology spread support for pod distribution across nodes via `topology_spread` parameter
- `TopologySpreadConfig` and `SpreadMode` types for configuring pod spread constraints
- Secure cloud GPU rental API: `list_secure_cloud_gpus()`, `start_secure_cloud_rental()`, `stop_secure_cloud_rental()`, `list_secure_cloud_rentals()`
- `GpuOffering`, `SecureCloudRentalResponse`, `SecureCloudRentalListItem` types for GPU rentals

### Changed
- Refactored SSH utilities to shared `rental_utils` module
- Reorganized type stubs and removed obsolete classes
- Removed `container_image`, `environment`, `ports` from secure cloud rental requests
- Made `estimated_hourly_cost` optional in offerings

### Fixed
- Topology spread now available in `create_deployment_async()` and all deployment methods

## [0.13.0] - 2026-01-20

### Changed
- Remove pre-flight node availability check from deploy methods
- SDK no longer calls `list_nodes` to auto-detect GPU models before deployment
- Deployments now rely on `min_gpu_memory_gb` for GPU scheduling instead of specific models
- Let the API/scheduler handle GPU selection and autoscaling

### Removed
- `_extract_gpu_model_id()` function (no longer needed)
- GPU model auto-detection logic that blocked deployments when no nodes were immediately available

### Fixed
- Deployments no longer fail with "No GPU nodes available" when cluster is empty
- Autoscaler can now provision nodes for pending GPU workloads

## [0.12.0] - 2026-01-13

### Added
- Async API methods: `deploy_async()`, `get_async()`, `list_async()` for concurrent operations
- Async low-level methods: `create_deployment_async()`, `get_deployment_async()`, `delete_deployment_async()`
- Async utility methods: `health_check_async()`, `list_nodes_async()`, `get_balance_async()`
- GPU model auto-detection from available nodes when `gpu_models` not specified
- `_extract_gpu_model_id()` for NVML name to K8s label conversion
- Callable source support in `deploy()` via `SourcePackager.from_function()`
- HTTP endpoint readiness verification in `wait_until_ready()`
- Async DNS resolution and HTTP readiness checks
- Comprehensive async test suite (`test_async_methods.py`)
- GPU model extraction test suite (`test_gpu_model_extraction.py`)
- Async concurrent deployment example (`21_async_concurrent.py`)

### Changed
- Rename `deployment.py` to `_deployment.py` (internal module)
- `deploy()` now accepts `Callable` source in addition to file paths and inline code
- Improved error handling in `list()` with warnings instead of silent failures
- Examples updated to use `min_gpu_memory_gb` instead of hardcoded `gpu_models`

## [0.11.0] - 2025-12-31

### Fixed
- Fix DNS propagation race condition in `wait_until_ready()` where the method could return before the deployment URL was DNS-resolvable
- Add `_is_dns_resolvable()` helper to verify DNS resolution before returning ready status

### Added
- Unit tests for DNS resolution verification in `wait_until_ready()`
- End-to-end integration test for DNS propagation fix

## [0.10.0] - 2025-12-19

### Added
- `deploy_vllm()` method for one-line vLLM inference server deployments
- `deploy_sglang()` method for one-line SGLang inference server deployments
- GPU requirements auto-detection based on model size via `templates/model_size.py`
- Support for all vLLM options: `tensor_parallel_size`, `dtype`, `quantization`, `gpu_memory_utilization`, etc.
- Support for all SGLang options: `context_length`, `mem_fraction_static`, etc.
- Auto-configured persistent storage for HuggingFace model caching
- Auto-generated deployment names from model identifiers

## [0.9.0] - 2025-12-09

### Added
- Progress callback support in `wait_until_ready()` with `on_progress` and `silent` parameters
- `wait_for_ready()` function with progress callback support in Rust SDK
- Deployment events, scaling, and health check types

### Changed
- Improved `wait_for_ready` state tracking and ready condition logic

## [0.8.0] - 2025-12-06

### Added
- Deployment progress and resource request bindings

## [0.7.0] - 2025-12-05

### Added
- Deployment progress tracking with `ProgressInfo` dataclass
- Progress callbacks in `wait_until_ready()` via `on_progress` parameter
- `DeploymentStatus.progress` field for tracking sync and startup progress

### Changed
- Improved deployment status reporting with detailed phase information

## [0.6.0] - 2025-11-13

### Added
- `@deployment` decorator for declarative function-based deployments
- `DeployedFunction` wrapper class for decorator API
- `DeploymentSpec` frozen dataclass for immutable deployment configuration
- High-level `deploy()` method on `BasilicaClient` for one-line deployments
- `Deployment` facade class with `url`, `logs()`, `delete()`, `status()` methods
- `DeploymentStatus` dataclass with `is_ready`, `is_failed`, `is_pending` properties
- Core facade modules: `deployment.py`, `decorators.py`, `spec.py`
- `Volume` class for persistent storage with `from_name()` factory method
- `SourcePackager` class for automatic source code packaging
- Framework auto-detection (FastAPI, Flask, Django) in `SourcePackager`
- GPU requirements support: `gpu_count`, `gpu_models`, `min_cuda_version`, `min_gpu_memory_gb`
- Storage support: `storage=True` or `storage="/path"` parameter

### Changed
- `BasilicaClient` now exposes both high-level (`deploy()`, `get()`, `list()`) and low-level APIs
- Bucket parameter is now optional in `PersistentStorageSpec`

## [0.5.0] - 2025-11-11

### Added
- Python SDK bindings for deployment operations via PyO3
- `create_deployment()`, `get_deployment()`, `delete_deployment()`, `list_deployments()` methods
- `get_deployment_logs()` for streaming container logs
- Deployment response types: `DeploymentResponse`, `DeploymentSummary`, `DeploymentListResponse`

## [0.4.0] - 2025-11-11

### Added
- GPU and storage types exposed to Python SDK
- `GpuRequirementsSpec` for GPU resource specifications
- `StorageSpec`, `PersistentStorageSpec`, `StorageBackend` types
- `ResourceRequirements` with GPU support

## [0.3.0] - 2025-11-11

### Added
- Public deployment parameter for creating public URLs
- Improved delete response serialization

### Fixed
- Environment variable serialization type handling

## [0.2.0] - 2025-11-11

### Added
- Comprehensive exception hierarchy with 12 exception types
- `BasilicaError`, `AuthenticationError`, `AuthorizationError`, `ValidationError`
- `DeploymentError`, `DeploymentNotFound`, `DeploymentTimeout`, `DeploymentFailed`
- `ResourceError`, `StorageError`, `NetworkError`, `RateLimitError`, `SourceError`

### Changed
- Migrated from JWT tokens to API key authentication
- Standardized API token environment variable naming to `BASILICA_API_TOKEN`

### Fixed
- PyO3 signature default value syntax for v0.26 compatibility
- pyo3-stub-gen build errors with extension-module feature

## [0.1.0] - 2025-10-10

### Added
- Initial release of Basilica Python SDK
- Support for GPU rental management via Basilica API
- Client authentication via API keys (environment variable or direct)
- Health check functionality for API monitoring
- Node listing and filtering with query parameters
- Rental lifecycle management:
  - Start rentals with flexible node selection
  - Get rental status with SSH access information
  - Stop active rentals
  - List all rentals with optional filtering
- SSH access utilities for easy connection to rental instances
- Auto-configuration from environment variables:
  - `BASILICA_API_URL` for API endpoint
  - `BASILICA_API_TOKEN` for authentication
- Type hints via `.pyi` stub files for IDE support
- PyO3-based Rust bindings for performance
- Cross-platform support (Linux, macOS, Windows)

### Documentation
- README with installation and usage instructions
- Inline API documentation
- Example code for common workflows

[Unreleased]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.14.0...HEAD
[0.14.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.13.0...basilica-sdk-python-v0.14.0
[0.13.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.12.0...basilica-sdk-python-v0.13.0
[0.12.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.11.0...basilica-sdk-python-v0.12.0
[0.11.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.10.0...basilica-sdk-python-v0.11.0
[0.10.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.9.0...basilica-sdk-python-v0.10.0
[0.9.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.8.0...basilica-sdk-python-v0.9.0
[0.8.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.7.0...basilica-sdk-python-v0.8.0
[0.7.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.6.0...basilica-sdk-python-v0.7.0
[0.6.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.5.0...basilica-sdk-python-v0.6.0
[0.5.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.4.0...basilica-sdk-python-v0.5.0
[0.4.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.3.0...basilica-sdk-python-v0.4.0
[0.3.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.2.0...basilica-sdk-python-v0.3.0
[0.2.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.1.0...basilica-sdk-python-v0.2.0
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/basilica-sdk-python-v0.1.0
