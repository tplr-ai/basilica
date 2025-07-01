# Basilica Configuration Files

This directory contains all configuration files for Basilica components.

## Configuration Files

Each component has two files:

- `{component}.toml.example` - Example configuration with all available options documented
- `{component}.toml` - Actual configuration file (gitignored, must be created from example)

## Components

### Miner (`miner.toml`)

Bittensor neuron that manages executor fleets. Key configuration sections:

- `[bittensor]` - Wallet and network settings
- `[executor_management]` - Fleet management configuration
- `[validator_comms]` - Communication with validators

### Executor (`executor.toml`)

GPU machine agent for task execution. Key configuration sections:

- `[docker]` - Container runtime settings
- `[gpu_attestor]` - GPU verification configuration
- `[resources]` - Resource limits and allocation

### Validator (`validator.toml`)

Bittensor neuron for verification and scoring. Key configuration sections:

- `[bittensor]` - Wallet and network settings
- `[verification]` - Verification suite configuration
- `[ssh_validation]` - SSH-based validation settings

### Public API (`public-api.toml`)

External HTTP API service. Key configuration sections:

- `[server]` - HTTP server settings
- `[rate_limiting]` - Rate limit configuration
- `[redis]` - Cache backend settings

### GPU Attestor (`gpu-attestor.toml`)

GPU verification tool. Key configuration sections:

- `[attestor]` - Core attestation settings
- `[gpu]` - GPU device selection
- `[benchmarks]` - Performance test configuration

## Setup Instructions

1. Copy the example file:

   ```bash
   cp config/miner.toml.example config/miner.toml
   ```

2. Edit the configuration file with your settings:

   ```bash
   vim config/miner.toml
   ```

3. Replace placeholder values:
   - `YOUR_WALLET_NAME` - Bittensor wallet name
   - `YOUR_HOTKEY_NAME` - Bittensor hotkey name
   - `YOUR_PUBLIC_IP_HERE` - Server's public IP address
   - `YOUR_PASSWORD` - Database passwords
   - IP addresses for allowed miners/validators/executors

## Security Notes

- Configuration files contain sensitive information (keys, passwords, IPs)
- Never commit actual `.toml` files to version control
- Use environment variables for highly sensitive values
- Ensure proper file permissions (600) on production servers

## Environment-Specific Configurations

For different environments, you can:

1. Use environment variables to override config values
2. Create separate config files (e.g., `miner.dev.toml`, `miner.prod.toml`)
3. Use the `--config` flag to specify which config file to use
