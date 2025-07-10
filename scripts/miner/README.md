# Basilica Miner

Bittensor neuron that manages executor fleets and handles validator communications.

## Quick Start

```bash
# Build locally
./build.sh

# Deploy with binary mode (default)
./deploy.sh -s root@miner.example.com:46088

# Deploy with systemd
./deploy.sh -s root@miner.example.com:46088 -d systemd

# Deploy with docker
./deploy.sh -s root@miner.example.com:46088 -d docker

# Run locally for development
docker-compose -f compose.dev.yml up
```

## Deployment Modes

The miner supports three deployment modes:

### Binary Mode (Default)
Deploys the miner binary and runs it with nohup:
```bash
./deploy.sh -s root@miner.example.com:46088
./deploy.sh -s root@miner.example.com:46088 -d binary
```

### Systemd Mode
Deploys the miner binary and manages it as a systemd service:
```bash
./deploy.sh -s root@miner.example.com:46088 -d systemd
```

### Docker Mode
Deploys using docker-compose with public registry images:
```bash
./deploy.sh -s root@miner.example.com:46088 -d docker
```

## Files

- `Dockerfile` - Container image definition
- `build.sh` - Build script for Docker image and binary
- `deploy.sh` - Multi-mode deployment script
- `systemd/basilica-miner.service` - Systemd service definition
- `compose.prod.yml` - Production docker-compose with watchtower
- `compose.dev.yml` - Development docker-compose with local build

## Configuration

The miner uses configuration files from the `config/` directory. Default is `config/miner.correct.toml`.

Specify custom config:
```bash
./deploy.sh -s root@miner.example.com:46088 -c config/miner.prod.toml
```

Key settings:
- `[bittensor]` - Wallet name, hotkey, network (finney/test/local), netuid 387
- `[server]` - Port 8080 for internal server, advertised_port 8080
- `[executor_management]` - Configure executor fleet connections
- `[database]` - SQLite database path: `/opt/basilica/data/miner.db`

## Ports

- `8080` - HTTP server (advertised on external IP)
- `55960` - Bittensor Axon port
- `9090` - Prometheus metrics endpoint

## Deployment Options

```bash
Usage: ./deploy.sh [OPTIONS]

OPTIONS:
    -s, --server USER@HOST:PORT      Server connection
    -d, --deploy-mode MODE           Deployment mode: binary, systemd, docker (default: binary)
    -c, --config FILE                Config file path (default: config/miner.correct.toml)
    -w, --sync-wallets               Sync local wallets to remote server
    -f, --follow-logs                Stream logs after deployment
    --health-check                   Perform health checks on service endpoints
    -t, --timeout SECONDS           SSH timeout (default: 60)
    -h, --help                       Show this help
```

## Examples

```bash
# Deploy with wallet sync and health checks
./deploy.sh -s root@miner.example.com:46088 -w --health-check

# Deploy with custom config and follow logs
./deploy.sh -s root@miner.example.com:46088 -c config/miner.staging.toml -f

# Deploy with systemd and health monitoring
./deploy.sh -s root@miner.example.com:46088 -d systemd --health-check -f
```

## Service Management

### Binary Mode
```bash
# Check status
ssh root@miner.example.com -p 46088 "pgrep -f miner"

# View logs
ssh root@miner.example.com -p 46088 "tail -f /opt/basilica/miner.log"

# Stop service
ssh root@miner.example.com -p 46088 "pkill -f miner"
```

### Systemd Mode
```bash
# Check status
ssh root@miner.example.com -p 46088 "systemctl status basilica-miner"

# View logs
ssh root@miner.example.com -p 46088 "journalctl -u basilica-miner -f"

# Restart service
ssh root@miner.example.com -p 46088 "systemctl restart basilica-miner"
```

### Docker Mode
```bash
# Check status
ssh root@miner.example.com -p 46088 "cd /opt/basilica && docker-compose ps"

# View logs
ssh root@miner.example.com -p 46088 "cd /opt/basilica && docker-compose logs -f"

# Restart containers
ssh root@miner.example.com -p 46088 "cd /opt/basilica && docker-compose restart"
```

## SSH Key Management

The miner automatically generates SSH keys for executor communication:
- SSH key location: `/root/.ssh/miner_executor_key`
- Public key: `/root/.ssh/miner_executor_key.pub`

The public key is automatically deployed to executor machines for miner-executor communication.