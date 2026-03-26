---
name: basilica-localnet-debug
description: Debug Basilica localnet services using existing scripts and Docker Compose commands
---

# Basilica Localnet Debugging

## Architecture Reference

### Services

| Service | Container Name | IP Address | Ports |
|---------|---------------|------------|-------|
| Subtensor | basilica-subtensor | 172.28.0.10 | 9944 (RPC), 30334 (P2P) |
| Validator | basilica-validator | 172.28.0.20 | 8080 (API), 9090 (Metrics) |
| Miner | basilica-miner | 172.28.0.30 | 8092 (gRPC), 8091 (Axon), 9091 (Metrics) |
| Prometheus | basilica-prometheus | 172.28.0.40 | 9099 |
| Grafana | basilica-grafana | 172.28.0.41 | 3000 |

### Database Architecture

> **IMPORTANT:** Validator and Miner use **SQLite**, NOT PostgreSQL.

| Service | Database | File Path |
|---------|----------|-----------|
| Validator | SQLite | `/opt/basilica/data/validator.db` |
| Miner | SQLite | `/var/lib/basilica/miner/data/miner.db` |

### Docker Compose Profiles

| Profile | Services Included |
|---------|------------------|
| `network` | subtensor |
| `validator` | subtensor, validator |
| `miner` | subtensor, validator, miner |
| `monitoring` | subtensor, validator, miner, prometheus, grafana |
| (none/all) | All services |

### Config Files

| File | Purpose |
|------|---------|
| `scripts/localnet/configs/validator.toml` | Validator configuration |
| `scripts/localnet/configs/miner.toml` | Miner configuration |
| `scripts/localnet/configs/prometheus.yml` | Prometheus scrape targets |

### Wallet Locations

- **Local wallets**: `scripts/localnet/wallets/`
- **Validator wallet**: `scripts/localnet/wallets/validator/`
- **Miner wallet**: `scripts/localnet/wallets/miner_1/`
- **Alice (funder)**: `scripts/localnet/wallets/alice/`

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `start.sh` | Start services with profiles | `./start.sh [profile] [--build]` |
| `stop.sh` | Stop services | `./stop.sh [--clean]` |
| `test.sh` | Health checks for all services | `./test.sh` |
| `init-subnet.sh` | Create wallets and register neurons | `./init-subnet.sh` |

### start.sh Profiles

```bash
./start.sh network     # Subtensor only
./start.sh validator   # Subtensor + Validator
./start.sh miner       # Above + Miner
./start.sh monitoring  # All + Prometheus + Grafana
./start.sh all         # Everything (default)
./start.sh --build     # Rebuild images before starting
```

### stop.sh Options

```bash
./stop.sh              # Stop containers, preserve data
./stop.sh --clean      # Remove containers, volumes, and network for fresh start (wallets in scripts/localnet/wallets/ are preserved)
```

## Docker Compose Commands

Run these from `scripts/localnet/` directory.

### View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs validator
docker compose logs miner
docker compose logs subtensor

# Follow logs in real-time
docker compose logs -f validator

# Tail last N lines
docker compose logs --tail=100 validator

# Filter for errors/warnings
docker compose logs validator 2>&1 | grep -iE "error|warn|fail"

# Combine follow + filter
docker compose logs -f validator 2>&1 | grep -iE "error|warn"
```

### Service Management

```bash
# Restart a service
docker compose restart validator

# Stop single service
docker compose stop validator

# Start single service
docker compose up -d validator

# Rebuild single service (no cache)
docker compose build --no-cache validator

# Rebuild and restart
docker compose up -d --build validator
```

### Container Access

```bash
# Shell into container
docker exec -it basilica-validator /bin/sh
docker exec -it basilica-miner /bin/sh

# Run command in container
docker exec basilica-validator cat /app/validator.toml
docker exec basilica-validator ls -la /opt/basilica/data/  # Check validator data
```

### Volume Management

```bash
# List volumes
docker volume ls | grep localnet

# Inspect volume
docker volume inspect localnet_validator-data

# Remove all project volumes (destructive)
docker compose down -v
```

## Debugging Workflows

### Service Won't Start

1. Check if container exists:
   ```bash
   docker compose ps -a
   ```

2. Check logs for errors:
   ```bash
   docker compose logs validator 2>&1 | tail -50
   ```

3. Check dependencies are running:
   ```bash
   docker compose ps subtensor  # Validator depends on this
   docker compose ps validator  # Miner depends on this being healthy
   ```

4. Check health endpoint:
   ```bash
   curl -s http://localhost:8080/health  # Validator
   curl -s http://localhost:9091/metrics  # Miner
   curl -s http://localhost:9944/health  # Subtensor
   ```

### Quick Rebuild Cycle

```bash
docker compose stop validator && \
docker compose build --no-cache validator && \
docker compose up -d validator && \
docker compose logs -f validator
```

### Database Issues (SQLite)

Validator and miner use SQLite databases stored in their data volumes.

```bash
# Check validator database exists
docker exec basilica-validator ls -la /opt/basilica/data/validator.db

# Check miner database exists
docker exec basilica-miner ls -la /var/lib/basilica/miner/data/miner.db

# Reset validator database (will be recreated on startup)
docker compose stop validator
docker volume rm localnet_validator-data
docker compose up -d validator

# Reset miner database
docker compose stop miner
docker volume rm localnet_miner-data
docker compose up -d miner
```

### Network Connectivity

```bash
# Check if services can reach each other
docker exec basilica-validator curl -s http://subtensor:9944/health
docker exec basilica-miner curl -s http://validator:8080/health

# Check network exists
docker network ls | grep basilica

# Inspect network
docker network inspect localnet_basilica-localnet
```

### Wallet Issues

```bash
# Check wallets exist
ls -la scripts/localnet/wallets/

# Check wallet balance
uvx --from bittensor-cli btcli wallet balance \
    --wallet-path scripts/localnet/wallets \
    --wallet-name validator \
    --network local

# Re-initialize wallets (requires subtensor running)
./init-subnet.sh
```

### Full Reset

```bash
./stop.sh --clean    # Removes containers, volumes, and network (scripts/localnet/wallets/ is preserved)
./start.sh miner     # Fresh start with wallet re-registration
```

## Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused ws://subtensor:9944` | Subtensor not ready | Wait for subtensor, check `docker compose logs subtensor` |
| `Wallet not found` | Wallets not initialized | Run `./init-subnet.sh` after subtensor is up |
| `Registration failed` | Insufficient balance | Check wallet balance, re-run `./init-subnet.sh` |
| `Health check failed` | Service crashed/starting | Check logs, wait for startup (60s start_period) |
| `Database connection failed` | SQLite file corrupted or missing | Reset service volume: `docker volume rm localnet_validator-data` and restart |
| `Port already in use` | Another service on port | Stop conflicting service or change port in docker-compose.yml |
| `Cannot connect to Docker daemon` | Docker not running | Start Docker Desktop |
| `No space left on device` | Docker storage full | `docker system prune -a` |

## Endpoints Reference

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Subtensor | `ws://localhost:9944` | Chain RPC |
| Subtensor | `http://localhost:9944/health` | Health check |
| Validator API | `http://localhost:8080` | REST API |
| Validator Health | `http://localhost:8080/health` | Health check |
| Validator Metrics | `http://localhost:9090/metrics` | Prometheus metrics |
| Miner gRPC | `localhost:8092` | Validator communications |
| Miner Axon | `localhost:8091` | Bittensor axon |
| Miner Metrics | `http://localhost:9091/metrics` | Prometheus metrics |
| Prometheus | `http://localhost:9099` | Metrics UI |
| Grafana | `http://localhost:3000` | Dashboards (admin/admin) |

## Files Reference

### Docker Volumes

| Volume | Mount Point | Purpose |
|--------|------------|---------|
| `subtensor-data` | `/tmp/alice` | Chain state |
| `validator-data` | `/opt/basilica/data` | Validator SQLite DB + state |
| `validator-ssh` | `/opt/basilica/data/ssh_keys` | SSH keys |
| `miner-data` | `/var/lib/basilica/miner/data` | Miner SQLite DB + state |
| `miner-ssh` | `/var/lib/basilica/miner/.ssh` | SSH keys |
| `prometheus-data` | `/prometheus` | Metrics storage |
| `grafana-data` | `/var/lib/grafana` | Dashboards |

### Config File Mounts

| Host Path | Container Path | Service |
|-----------|---------------|---------|
| `configs/validator.toml` | `/app/validator.toml` | validator |
| `configs/miner.toml` | `/app/miner.toml` | miner |
| `configs/prometheus.yml` | `/etc/prometheus/prometheus.yml` | prometheus |
| `wallets/` | `/root/.bittensor/wallets` | validator |
| `wallets/` | `/var/lib/basilica/miner/.bittensor/wallets` | miner |
