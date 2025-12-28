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
- Core identity types: `Hotkey`, `NodeId`, `ValidatorUid`, `MinerUid`
- Cryptographic utilities: Blake3 hashing, Ed25519/Sr25519 signatures
- Configuration loading with TOML and environment variable support
- Persistence traits and repository patterns
- SSH key management abstractions
- Metrics collection interfaces
- Compute resource definitions
- GPU rental types and states
- Bittensor network integration utilities

### Features
- `sqlite` - SQLite persistence backend
- `postgres` - PostgreSQL persistence backend
- `crypto-extra` - Additional cryptographic utilities

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

