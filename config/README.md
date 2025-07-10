# Basilica Configuration Files

This directory contains all configuration files for Basilica components.

## Configuration Files

Each component has three types of configuration files:

- `{component}.correct.toml` - Production-ready reference configuration (the canonical example)
- `{component}.toml.example` - Template configuration with placeholders for easy setup
- `{component}.toml` - Actual configuration file (gitignored, must be created from example)

## Components

### Validator (`validator.toml`)

Bittensor neuron for verification and scoring. Key configuration sections:

- `[bittensor]` - Wallet and network settings (auto-detects chain endpoint)
- `[verification]` - Verification suite configuration with binary validation
- `[ssh_validation]` - SSH-based validation settings
- `[ssh_session]` - SSH session management and audit logging
- `[emission]` - GPU category emission allocation settings

**Requirements**: CUDA Toolkit 12.8 for GPU verification kernels

### Miner (`miner.toml`)

Bittensor neuron that manages executor fleets. Key configuration sections:

- `[bittensor]` - Wallet and network settings (auto-detects UID and chain endpoint)
- `[executor_management]` - Fleet management configuration with SSH access
- `[validator_comms]` - Communication with validators including rate limiting
- `[ssh_session]` - SSH session orchestration for validator access
- `[advertised_addresses]` - Service endpoint advertising

### Executor (`executor.toml`)

GPU machine agent for task execution. Key configuration sections:

- `[server]` - gRPC server configuration
- `[docker]` - Container runtime settings with GPU passthrough
- `[system]` - GPU and system monitoring configuration  
- `[validator]` - Validator access configuration with hotkey verification
- `[advertised_endpoint]` - Service endpoint advertising

**Requirements**: NVIDIA GPU with 8.7+ CUDA compute capability, CUDA Toolkit 12.8

### Public API (`public-api.toml`)

External HTTP API service. Key configuration sections:

- `[server]` - HTTP server settings
- `[discovery]` - Validator discovery configuration
- `[load_balancing]` - Load balancing strategies
- `[caching]` - Response caching with Redis support

### GPU Attestor (`gpu-attestor.toml`)

GPU verification tool configuration.

## Setup Instructions

### 1. Copy the Template Configuration

```bash
# For production (recommended)
cp config/validator.correct.toml config/validator.toml

# Or from example template
cp config/validator.toml.example config/validator.toml
```

### 2. Edit Configuration

```bash
vim config/validator.toml
```

### 3. Replace Placeholder Values

**Required placeholders to update:**

- `YOUR_WALLET_NAME` - Bittensor wallet name
- `YOUR_HOTKEY_NAME` - Bittensor hotkey name  
- `YOUR_PUBLIC_IP_HERE` - Server's public IP address
- `YOUR_MINER_HOTKEY_HERE` - Managing miner's hotkey (for executors)
- `YOUR_EXECUTOR_ID` - Unique executor identifier
- `YOUR_SSH_USERNAME` - SSH username for executor access
- `YOUR_SECURE_JWT_SECRET_HERE` - JWT secret for authentication

**Network Configuration:**

- Use `network = "finney"` and `netuid = 39` for mainnet
- Use `network = "test"` and `netuid = 387` for testnet
- Chain endpoints are auto-detected based on network type

## Key Features

### Dynamic Configuration

- **UID Discovery**: Services automatically discover their UID from Bittensor metagraph
- **Network Detection**: Chain endpoints auto-configured based on network type
- **Wallet Support**: Compatible with both JSON wallet files and raw seed phrases

### Production Features

- **SQLite Storage**: Persistent data storage with migrations
- **SSH Management**: Automated SSH session orchestration
- **Rate Limiting**: Configurable rate limits for validator/miner communication
- **Audit Logging**: SSH access audit trails
- **Binary Validation**: Cryptographic GPU verification

## Security Notes

- Configuration files contain sensitive information (hotkeys, IPs, secrets)
- Never commit actual `.toml` files to version control (they are gitignored)
- Use secure JWT secrets for production deployments
- Ensure proper file permissions (600) on production servers
- The `.correct.toml` files contain working examples but should be customized

## Environment-Specific Configurations

For different environments:

1. **Development**: Use `.toml.example` templates with local IPs
2. **Production**: Use `.correct.toml` as base with your specific values
3. **Multiple Environments**: Create separate config files (e.g., `validator.prod.toml`)
4. **CLI Override**: Use `--config` flag to specify which config file to use

## Monitoring Configuration

The `monitoring/` subdirectory contains:

- `prometheus.yml` - Prometheus configuration for metrics collection
- `grafana-datasources.yml` - Grafana datasource configuration
