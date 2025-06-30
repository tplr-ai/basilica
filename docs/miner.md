# Miner Guide

This guide covers running a Basilica miner node with GPU executors to provide compute resources to the network.

## Overview

The miner component manages a fleet of GPU executor machines, handling:
- Registration on the Bittensor network
- Executor fleet management via gRPC
- Task distribution and monitoring
- Serving compute requests through the Axon server

## Prerequisites

- Bittensor wallet with TAO tokens
- Linux system with Docker support
- One or more GPU machines for executors
- Network connectivity between miner and executors

## Quick Start

### 1. Set Up Your Wallet

Ensure you have a Bittensor wallet configured:

```bash
# Create wallet if needed (skip if you already have one)
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### 2. Configure the Miner

Create a `miner.toml` configuration file:

```toml
[server]
host = "0.0.0.0"
port = 8080

[bittensor]
wallet_name = "miner"
hotkey_name = "default"
network = "finney"  # or "test" for testnet
netuid = 39  # Basilica subnet
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
weight_interval_secs = 300
axon_port = 9090

[executor]
deployment_mode = "ssh"
ssh_key_path = "~/.ssh/id_rsa"
max_concurrent_deployments = 5
health_check_interval = { secs = 30 }
reconnect_interval = { secs = 60 }

# Define your executor machines
[[executor.configs]]
id = "gpu-1"
grpc_address = "executor1.example.com:50051"
name = "GPU Executor 1"

[[executor.configs]]
id = "gpu-2"
grpc_address = "executor2.example.com:50051"
name = "GPU Executor 2"

[storage]
data_dir = "./data"

[logging]
level = "info"
format = "pretty"
```

### 3. Set Up Executors

On each GPU machine, run the executor:

```bash
# Download and run executor
./executor --server --config executor.toml
```

Executor configuration (`executor.toml`):

```toml
[server]
grpc_port = 50051
health_port = 8082

[bittensor]
enable_auth = true
jwt_secret = "your-secure-secret-key"

[system]
container_runtime = "docker"
gpu_enabled = true

[storage]
data_dir = "/opt/basilica/data"
attestation_dir = "/opt/basilica/attestations"

[logging]
level = "info"
```

### 4. Start the Miner

```bash
# Using the binary
./miner --config miner.toml

# Or using Docker
docker run -d \
  -v ~/.bittensor:/root/.bittensor \
  -v ./miner.toml:/config/miner.toml \
  -p 8080:8080 \
  -p 9090:9090 \
  basilica/miner:latest
```

## Advanced Configuration

### Fleet Management

The miner supports multiple deployment modes:

- **SSH Mode**: Direct SSH deployment to executor machines
- **Manual Mode**: Pre-deployed executors managed externally
- **Kubernetes Mode**: Kubernetes-based executor orchestration


### Monitoring

Monitor your miner's health and performance:

```bash
# Check miner health
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check executor status
curl http://localhost:8080/api/v1/executors
```

### Security Best Practices

1. **Secure Communication**
   - Use TLS for gRPC connections between miner and executors
   - Configure JWT authentication for executor access
   - Restrict network access with firewalls

2. **Key Management**
   - Keep hotkey secure with proper file permissions
   - Use separate wallets for different miners
   - Regularly rotate JWT secrets

3. **Resource Limits**
   - Set appropriate container resource limits
   - Monitor GPU memory usage
   - Configure task timeouts

## Troubleshooting

### Common Issues

**Executor Connection Failed**
```
Error: Failed to connect to executor at gpu-1:50051
```
- Verify executor is running and accessible
- Check firewall rules allow port 50051
- Ensure gRPC address is correct in config

**Registration Failed**
```
Error: Failed to serve axon on network
```
- Ensure wallet has sufficient TAO for registration
- Verify you're connected to the correct network
- Check if hotkey is already registered

**Hardware Attestation Failed**
```
Error: GPU attestation failed: No NVIDIA driver found
```
- Install NVIDIA drivers on executor machine
- Run executor with `--privileged` if using Docker
- Verify GPU is properly detected with `nvidia-smi`

### Logs and Debugging

Enable debug logging for detailed troubleshooting:

```toml
[logging]
level = "debug"
format = "json"
```

View logs:
```bash
# Miner logs
tail -f ./logs/miner.log

# Executor logs (on executor machine)
tail -f /opt/basilica/logs/executor.log
```

## Performance Optimization

1. **Network Optimization**
   - Place miner geographically close to executors
   - Use dedicated network connections
   - Enable gRPC compression for large payloads

2. **Resource Allocation**
   - Balance executor count with miner capacity
   - Monitor CPU/memory usage on miner
   - Tune verification intervals based on load

3. **Scaling Considerations**
   - Use load balancing for multiple miners
   - Implement executor pooling for efficiency
   - Consider horizontal scaling for large fleets

## Next Steps

- Review the [Architecture Guide](architecture.md) to understand the system design
- Check the [Validator Guide](validator.md) to understand how your miner is evaluated
- Join the Basilica community for support and updates