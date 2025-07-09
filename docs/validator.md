# Validator Guide

This guide covers running a Basilica validator node to verify and score GPU providers on the network.

## Overview

The validator component performs critical network functions:

- Verifies miner hardware capabilities through SSH-based challenges
- Scores miners based on performance and reliability
- Sets weights on the Bittensor network to reward quality providers
- Maintains network integrity through continuous verification

## Prerequisites

- Bittensor wallet with sufficient stake
- Linux system with stable internet connection
- SSH access for remote verification
- SQLite for verification history storage
- Validator public key file (for building from source)

## Quick Start

### 1. Set Up Your Wallet

Create and fund a validator wallet:

```bash
# Create wallet if needed (skip if you already have one)
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Register on the subnet (requires stake)
btcli subnet register --netuid 39 --wallet.name validator --wallet.hotkey default
```

**Note**: The wallet file format has changed. The validator now supports both JSON wallet files (new format) and raw seed phrases (old format). The JSON format includes fields like `secretPhrase`, `publicKey`, `accountId`, etc.

### 2. Configure the Validator

Create a `validator.toml` configuration file:

```toml
[server]
host = "0.0.0.0"
port = 8080

[database]
url = "sqlite:./data/validator.db"
max_connections = 5
min_connections = 1
run_migrations = true

[bittensor]
wallet_name = "validator"
hotkey_name = "default"
network = "finney"  # Options: "finney", "test", or "local"
netuid = 39  # Basilica subnet (use 387 for test network)
# chain_endpoint is auto-detected based on network if not specified
# finney: wss://entrypoint-finney.opentensor.ai:443
# test: wss://test.finney.opentensor.ai:443  
# local: ws://127.0.0.1:9944
weight_interval_secs = 300
axon_port = 9091
# external_ip = "your.external.ip.here"  # Optional

[verification]
verification_interval = { secs = 600, nanos = 0 }
max_concurrent_verifications = 50
challenge_timeout = { secs = 120, nanos = 0 }
min_score_threshold = 0.1
min_stake_threshold = 1.0
max_miners_per_round = 20
min_verification_interval = { secs = 1800, nanos = 0 }
netuid = 39  # Should match bittensor.netuid

[storage]
data_dir = "./data"

[api]
bind_address = "0.0.0.0:8080"
max_body_size = 1048576
# api_key = "your-api-key-here"  # Optional

[logging]
level = "info"
format = "pretty"

[metrics]
enabled = true
[metrics.prometheus]
enabled = true
port = 9090
```

### 3. Production Deployment (Recommended)

The easiest way to run a validator in production is using Docker Compose:

```bash
# Navigate to validator scripts directory
cd scripts/validator

# Copy and customize the production config
cp ../../config/validator.correct.toml ../../config/validator.toml
# Edit validator.toml with your specific settings:
# - Update external_ip and advertised_host to your public IP
# - Set your wallet_name and hotkey_name
# - Choose network: "finney" for mainnet or "test" for testnet

# Ensure your Bittensor wallet exists
ls ~/.bittensor/wallets/your_validator_wallet/hotkeys/

# Create required directories
mkdir -p /var/log/basilica

# Deploy with Docker Compose (includes auto-updates and monitoring)
docker compose -f compose.prod.yml up -d

# Check status
docker compose -f compose.prod.yml ps
docker logs basilica-validator
```

This production setup includes:
- **Automatic updates** via Watchtower
- **Health monitoring** with automatic restarts  
- **Persistent data storage** with named volumes
- **Proper logging** to `/var/log/basilica`
- **Network isolation** with dedicated Docker network

### 4. Alternative Deployment Methods

#### Using Build Script and Remote Deployment

```bash
# Build and deploy to remote server (see BASILICA-DEPLOYMENT-GUIDE.md)
./scripts/deploy.dev.sh -s validator -v user@your-server:port

# Deploy with wallet sync and health checks
./scripts/deploy.dev.sh -s validator -v user@your-server:port -w -c
```

#### Building from Source

```bash
# Build the validator using the build script
./scripts/validator/build.sh

# Or build manually
cargo build --release -p validator
```

#### Running with Docker Directly

```bash
# Build Docker image
docker build -f scripts/validator/Dockerfile -t basilica/validator .

# Run container
docker run -d \
  --name basilica-validator \
  --restart unless-stopped \
  -v ~/.bittensor:/home/basilica/.bittensor \
  -v ./config/validator.toml:/app/validator.toml:ro \
  -v validator-data:/app/data \
  -v ~/.ssh:/app/keys:ro \
  -p 50053:50053 \
  -p 3000:3000 \
  -p 8080:8080 \
  basilica/validator:latest --config /app/validator.toml
```

**Important Notes**:

- The validator automatically discovers its UID from the Bittensor metagraph based on its hotkey
- UIDs are no longer hardcoded in configuration files
- Chain endpoint is auto-detected based on network type if not explicitly specified
- Ensure proper firewall configuration for ports 50053 (gRPC), 3000 (API), and 8080 (metrics)
- For production, use the compose.prod.yml for automatic updates and monitoring

## Verification Process

### How Verification Works

1. **Discovery**: Validator queries the Bittensor metagraph for active miners
2. **Selection**: Chooses miners based on stake and last verification time
3. **Challenge**: Sends computational challenges via SSH to executor machines
4. **Validation**: Verifies hardware attestations and computational results
5. **Scoring**: Updates miner scores based on performance
6. **Weight Setting**: Submits weights to Bittensor network

### Verification Types

**Hardware Verification**

- Validates GPU specifications through attestation
- Checks cryptographic signatures (P256 ECDSA)
- Verifies hardware capabilities match claims

The hardware attestation process:

```bash
# Validators connect to executors via SSH and verify attestations
# The attestation includes:
# - GPU specifications (model, memory, compute capability)
# - System information (CPU, RAM, OS)
# - Cryptographic signature (P256 ECDSA)
# - Timestamp and validity period

# Validators verify:
# 1. Attestation signature is valid
# 2. Hardware matches claimed specifications
# 3. Attestation is recent (not expired)
# 4. GPU is actually accessible and functional
```

**Compute Verification**

- Runs benchmark tasks on miner GPUs
- Measures performance and accuracy
- Validates task completion times

**Availability Verification**

- Monitors miner uptime and responsiveness
- Tracks connection reliability
- Measures API response times

## Scoring Algorithm

Miners are scored on multiple factors:

```
Total Score = (
    Hardware Score * 0.3 +
    Performance Score * 0.4 +
    Availability Score * 0.2 +
    Reliability Score * 0.1
)
```

### Score Components

1. **Hardware Score**: Based on GPU capabilities and attestation validity
   - Valid attestation signature: Required for any score
   - GPU specifications: Higher capability GPUs score better
   - Attestation freshness: Recent attestations score higher
   - Hardware consistency: Stable hardware configuration preferred

2. **Performance Score**: Computational benchmark results
3. **Availability Score**: Uptime and response time metrics
4. **Reliability Score**: Historical performance consistency

## Advanced Configuration

### SSH Verification Setup

Configure SSH access for remote verification:

```toml
[verification.ssh]
key_path = "~/.ssh/validator_key"
known_hosts_path = "~/.ssh/known_hosts"
connection_timeout = { secs = 30 }
max_retries = 3
```

### Database Management

The validator stores verification history in SQLite:

```bash
# View verification history
sqlite3 /opt/basilica/data/validator.db "SELECT * FROM verifications ORDER BY timestamp DESC LIMIT 10;"

# Check miner scores
sqlite3 /opt/basilica/data/validator.db "SELECT hotkey, score, last_verified FROM miners;"

# Backup database
cp /opt/basilica/data/validator.db /backup/validator_$(date +%Y%m%d).db
```

### Weight Setting Strategy

Configure weight setting behavior:

```toml
[bittensor]
weight_interval_secs = 300  # How often to set weights
min_score_for_weight = 0.1  # Minimum score to receive weight
max_weight_miners = 256     # Maximum miners to assign weight

[verification]
# Only verify miners with minimum stake
min_stake_threshold = 1.0

# Spread verifications to avoid overload
max_miners_per_round = 20
```

## Monitoring and Maintenance

### Health Monitoring

```bash
# Check validator health
curl http://localhost:8080/health

# View current metrics (Prometheus format)
curl http://localhost:9090/metrics

# Get verification statistics
curl http://localhost:8080/api/v1/stats
```

### Performance Metrics

Monitor key validator metrics:

- Verification success rate
- Average verification time
- Weight setting frequency
- Network sync status

### Log Analysis

```bash
# View recent logs
tail -f /opt/basilica/logs/validator.log

# Check for verification errors
grep ERROR /opt/basilica/logs/validator.log | tail -20

# Monitor weight updates
grep "weights set" /opt/basilica/logs/validator.log
```

## Security Best Practices

1. **Wallet Security**
   - Use hardware wallets for cold keys
   - Secure hotkey with proper permissions (600)
   - Regular key rotation for SSH access

2. **Network Security**
   - Firewall configuration for API access
   - VPN for sensitive operations
   - Rate limiting on API endpoints

3. **Operational Security**
   - Regular backups of verification database
   - Monitoring for anomalous behavior
   - Incident response procedures

## Troubleshooting

### Common Issues

**Registration Failed**

```
Error: Validator registration failed: insufficient stake
```

- Ensure wallet has minimum stake requirement
- Check network connection to Bittensor
- Verify correct subnet ID

**Verification Timeout**

```
Error: SSH verification timeout for miner UID 123
```

- Check network connectivity to miner
- Verify SSH credentials are correct
- Increase timeout in configuration

**Database Errors**

```
Error: unable to open database file
```

- Ensure the data directory exists (e.g., `mkdir -p data`)
- Check file permissions on the data directory
- Verify the database URL in config uses proper format: `sqlite:./data/validator.db`
- The validator now correctly handles both relative and absolute paths
- SQLite connection mode `?mode=rwc` is automatically added for read-write-create

**Wallet Loading Error**

```
Error: Failed to load hotkey: Invalid format
```

- Ensure wallet file exists at `~/.bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}`
- Check if the wallet is in the correct format (JSON with secretPhrase field or raw seed phrase)
- Verify file permissions allow reading
- You can copy an existing wallet: `cp ~/.bittensor/wallets/miner/hotkeys/default ~/.bittensor/wallets/validator/hotkeys/default`

**Build Error - Missing Validator Key**
The gpu-attestor binary no longer requires a validator public key to build.

### Debug Mode

Enable detailed logging for troubleshooting:

```toml
[logging]
level = "debug"
format = "json"
include_location = true
```

## Best Practices

1. **Consistent Verification**
   - Run validator continuously for accurate scoring
   - Avoid long downtime periods
   - Monitor verification queue depth

2. **Fair Weight Distribution**
   - Verify all eligible miners regularly
   - Update weights based on recent data
   - Avoid bias in miner selection

3. **Resource Management**
   - Monitor CPU/memory usage
   - Tune concurrent verification limits
   - Implement proper error handling

## Next Steps

- Review the [Architecture Guide](architecture.md) to understand system design
- Read the [Miner Guide](miner.md) to understand what you're validating
- Join the Basilica validator community for support
