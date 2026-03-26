# Basilica Configuration Files

This directory contains all configuration files for Basilica components.

## Configuration Files

Each component has configuration files following this pattern:

- `{component}.toml.example` - Template configuration with placeholders for easy setup
- `{component}.toml` - Actual configuration file (gitignored, must be created from example)

## Available Configuration Templates

| Component | Template File | Description |
|-----------|---------------|-------------|
| Validator | `validator.toml.example` | Bittensor neuron for verification and scoring |
| Miner | `miner.toml.example` | Bittensor neuron for GPU node orchestration |
| CLI | `cli.toml.example` | CLI tool configuration |
| GPU Attestor | `gpu-attestor.toml.example` | GPU verification tool |

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

Bittensor neuron that orchestrates GPU node access. Key configuration sections:

- `[bittensor]` - Wallet and network settings (auto-detects UID and chain endpoint)
- `[node_management]` - GPU node SSH endpoint configuration (gpu_category, gpu_count)
- `[bidding]` - GPU pricing per category (static prices in dollars per GPU-hour)
- `[ssh_session]` - SSH session orchestration for validator access
- `[advertised_addresses]` - Service endpoint advertising
- `[validator_assignment]` - Validator assignment strategy (automatic validator discovery)

### CLI (`cli.toml`)

CLI tool configuration. Key configuration sections:

- `[api]` - API endpoint settings
- `[ssh]` - SSH connection settings
- `[image]` - Default Docker image
- `[wallet]` - Bittensor wallet settings

### GPU Attestor (`gpu-attestor.toml`)

GPU verification tool configuration.

## Setup Instructions

### 1. Copy the Template Configuration

```bash
# For validator
cp config/validator.toml.example config/validator.toml

# For miner
cp config/miner.toml.example config/miner.toml

# For other components
cp config/cli.toml.example config/cli.toml
```

### 2. Edit Configuration

```bash
vim config/validator.toml
# or
vim config/miner.toml
```

### 3. Replace Placeholder Values

**Required placeholders to update:**

- `YOUR_WALLET_NAME` - Bittensor wallet name
- `YOUR_HOTKEY_NAME` - Bittensor hotkey name
- `YOUR_PUBLIC_IP_HERE` - Server's public IP address

**Network Configuration:**

- Use `network = "finney"` and `netuid = 39` for mainnet
- Use `network = "test"` and `netuid = 387` for testnet
- **Always specify `chain_endpoint`** to avoid metadata compatibility issues:
  - Finney: `chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"`
  - Test: `chain_endpoint = "wss://test.finney.opentensor.ai:443"`
  - Local: `chain_endpoint = "ws://127.0.0.1:9944"`

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

## Metadata Compatibility

**Important**: To avoid metadata compatibility errors, always:

1. **Regenerate metadata** before production deployments:

   ```bash
   ./scripts/generate-metadata.sh --network finney
   ```

2. **Specify chain endpoints** in configuration files to ensure runtime compatibility

3. **Rebuild services** after metadata updates to embed fresh metadata

## Security Notes

- Configuration files contain sensitive information (hotkeys, IPs, secrets)
- Never commit actual `.toml` files to version control (they are gitignored)
- Use secure JWT secrets for production deployments
- Ensure proper file permissions (600) on production servers

## Environment-Specific Configurations

For different environments:

1. **Development**: Use `.toml.example` templates with local IPs
2. **Production**: Copy `.toml.example` and customize with your values
3. **Multiple Environments**: Create separate config files (e.g., `validator.prod.toml`)
4. **CLI Override**: Use `--config` flag to specify which config file to use
