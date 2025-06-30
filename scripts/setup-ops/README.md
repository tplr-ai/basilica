# Remote Server Setup Scripts

Scripts for setting up remote servers for Basilica development and testing.

## Usage

### Complete Automated Setup (Recommended)

1. **Configure your servers** by editing `scripts/setup-ops/servers.conf`:
```bash
# Edit the configuration file with your server details
VALIDATOR=user@your-validator-host:port
MINER=user@your-miner-host:port  
EXECUTOR=user@your-executor-host:port
```

2. **Run the complete setup**:
```bash
# Setup everything with one command
./scripts/setup-ops/setup-all.sh
```

This will:
1. Setup all three servers (validator, miner, executor)
2. Automatically distribute SSH keys between all machines
3. Test cross-machine connectivity
4. Verify everything is working

### Individual Setup (Advanced)

```bash
# Setup individual servers
./scripts/setup-ops/remote.sh validator user@host -p port
./scripts/setup-ops/remote.sh miner user@host -p port
./scripts/setup-ops/remote.sh executor user@host -p port

# Then setup SSH connectivity
./scripts/setup-ops/setup-ssh.sh
```

## What the Script Does

### All Servers
- ✅ Installs basic dependencies (curl, wget, git, build-essential, pkg-config, libssl-dev, rsync)
- ✅ Installs Rust toolchain (latest stable)
- ✅ Installs Docker and Docker Compose
- ✅ Generates SSH keys for cross-machine access
- ✅ Creates `/opt/basilica` workspace directory
- ✅ Verifies all installations

### Executor Specific
- ✅ Installs NVIDIA Container Toolkit
- ✅ Configures Docker for GPU access
- ✅ Tests GPU availability

## Scripts Overview

### `setup-all.sh`
Complete automation - runs everything in sequence and tests connectivity.

### `remote.sh` 
Sets up individual servers with all dependencies and generates SSH keys.

### `setup-ssh.sh`
Automates SSH key distribution and tests cross-machine connectivity.

## SSH Key Management

The scripts automatically handle SSH key generation and distribution:

1. Each server generates its own `~/.ssh/basilica` key pair
2. `setup-ssh.sh` collects all public keys and distributes them to all machines
3. Connectivity testing ensures everything works properly

No manual key copying required!

## Features

- **Idempotent**: Safe to run multiple times
- **Simple**: Minimal, maintainable code
- **Role-aware**: Specific setup for executor GPU requirements
- **Verified**: Comprehensive final verification step

## Configuration File

### `servers.conf` Format

```bash
# Server configuration format: user@host:port
VALIDATOR=root@validator-host:22
MINER=root@miner-host:22  
EXECUTOR=root@executor-host:22
```

### Server Roles

- **Validator**: Validates hardware attestations and manages consensus
- **Miner**: Manages executor fleet and communicates with validators  
- **Executor**: Provides GPU compute resources (requires NVIDIA Docker setup)

## Troubleshooting

### Docker Issues
```bash
# Restart Docker if needed
service docker restart

# Check Docker status
docker info
```

### SSH Issues
```bash
# Test SSH connectivity
ssh -i ~/.ssh/basilica target_host

# Check SSH key permissions
chmod 600 ~/.ssh/basilica
chmod 644 ~/.ssh/basilica.pub
```

### NVIDIA Docker Issues (Executor only)
```bash
# Test GPU access
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all ubuntu:22.04 ls /dev/nvidia*
```
