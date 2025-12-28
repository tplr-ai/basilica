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
- K3s GPU node autoscaling controller
- Demand-based scaling based on pending pods
- GPU-aware node provisioning
- Cost optimization through scale-down policies
- GpuScalingPolicy CRD for declarative configuration
- Cooldown periods to prevent thrashing
- Prometheus metrics export
- Health check endpoints
- Kubernetes native integration

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

