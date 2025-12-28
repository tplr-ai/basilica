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
- Bittensor subnet validator implementation
- SSH-based direct GPU hardware verification
- Two-tier validation: full validation (6 hours) and lightweight checks (10 minutes)
- GPU profiling and capability detection
- Miner scoring and weight calculation
- REST API for external access
- PostgreSQL and SQLite storage backends
- Ephemeral SSH key management
- Sr25519 signature-based authentication
- Prometheus metrics export
- Health check endpoints

### Features
- `client` - Enable HTTP client for external services (default)
- `test-utils` - Enable test utilities
- `cli` - Enable CLI support with clap derives

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

