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
- HTTP gateway for Basilica validator network
- Load balancing across multiple validators
- Request aggregation for efficiency
- API key and JWT authentication
- Rate limiting with governor
- Response caching with moka
- WebSocket support for streaming
- Kubernetes integration for dynamic backends
- Prometheus metrics export
- Health check endpoints
- OpenAPI documentation generation

### Features
- `server` - Enable HTTP server functionality (default)
- `client` - Enable HTTP client functionality
- `utoipa` - Enable OpenAPI documentation generation
- `full` - Enable all features

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

