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

## Quick Start

### 1. Set Up Your Wallet

Create and fund a validator wallet:

```bash
# Create wallet if needed (skip if you already have one)
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey validator

# Register on the subnet (requires stake)
btcli subnet register --netuid 39 --wallet.name validator --wallet.hotkey validator
```

### 2. Configure the Validator

Create a `validator.toml` configuration file:

```toml
[database]
url = "sqlite:///opt/basilica/data/validator.db?mode=rwc"
max_connections = 10
min_connections = 2

[server]
host = "0.0.0.0"
port = 8081

[bittensor]
wallet_name = "validator"
hotkey_name = "validator"
network = "finney"  # or "test" for testnet
netuid = 39  # Basilica subnet
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
weight_interval_secs = 300
axon_port = 9091

[verification]
verification_interval = { secs = 600 }
max_concurrent_verifications = 50
challenge_timeout = { secs = 120 }
min_score_threshold = 0.1
min_stake_threshold = 1.0
max_miners_per_round = 20
min_verification_interval = { secs = 1800 }

[storage]
data_dir = "/opt/basilica/data"

[api]
bind_address = "0.0.0.0:8081"
max_body_size = 1048576

[logging]
level = "info"
format = "pretty"
```

### 3. Start the Validator

```bash
# Using the binary
./validator start --config validator.toml

# Or using Docker
docker run -d \
  -v ~/.bittensor:/root/.bittensor \
  -v ./validator.toml:/config/validator.toml \
  -v ./data:/opt/basilica/data \
  -p 8081:8081 \
  -p 9091:9091 \
  basilica/validator:latest
```

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
curl http://localhost:8081/health

# View current metrics
curl http://localhost:8081/metrics

# Get verification statistics
curl http://localhost:8081/api/v1/stats
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
Error: database is locked
```
- Check file permissions on database
- Ensure single validator instance
- Consider database maintenance

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