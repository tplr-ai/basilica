# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial crates.io release preparation
- Comprehensive documentation and examples

## [0.1.0] - 2025-01-01

### Added
- Kubernetes operator for GPU workload management
- UserDeployment CRD for declarative workloads
- GpuRental CRD for rental lifecycle
- GpuNode CRD for node registration
- Reconciliation controllers for all CRDs
- Node onboarding and lifecycle management
- GPU-aware scheduling
- Health monitoring and auto-recovery
- Rate limiting for API calls
- Prometheus metrics export
- Health check endpoints

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

