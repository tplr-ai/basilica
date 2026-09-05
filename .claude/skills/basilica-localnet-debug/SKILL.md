---
name: basilica-localnet-debug
description: Diagnose the public repository's local Subtensor, validator and miner Compose profiles using read-only checks before repairs.
---

# Basilica localnet debugging

This is contributor-only local development guidance. Resolve paths from the
public repository root; run Compose commands in `scripts/localnet/`.
Read [localnet setup and verification](../../../scripts/localnet/README.md).
Do not confuse local chain test funds with customer platform funding.

## Supported service inventory

| Profile | Required services |
| --- | --- |
| `network` | Subtensor |
| `validator` | Subtensor, validator |
| `miner` | Subtensor, validator, miner |
| `all` | Same three services as `miner` |

Compose requires an explicit profile. `start.sh` defaults to all services.
There is no monitoring profile, Prometheus/Grafana service or PostgreSQL service
in this stack. Subtensor serves chain RPC on container port 9944. Validator serves
HTTP 8080, metrics 9090 and bidding gRPC 50052. Miner uses gRPC/axon on **the same
container port 50051**, published by default as host 8092, plus metrics on container
9090 (host 9091). There is no separate 8091 listener in the configured miner.

Validator and miner persist SQLite files configured in their mounted TOML:

| Service | Default file |
| --- | --- |
| Validator | `/opt/basilica/data/validator.db` |
| Miner | `/var/lib/basilica/miner/data/miner.db` |

## Read-only diagnosis

1. Identify the intended profile and Compose project. Available credentials,
   local wallets or Docker access do not authorize unrelated cleanup or cloud work.
2. From the repository root run the matching check:

```bash
./scripts/localnet/test.sh network
./scripts/localnet/test.sh validator
./scripts/localnet/test.sh miner
```

The checker requires Python 3.11+ and Docker Compose. It derives required services,
container IDs and host ports from the selected Compose profile/project; it respects
`COMPOSE_FILE` and `COMPOSE_PROJECT_NAME`. Ambient `COMPOSE_PROFILES` cannot widen
the selected profile. Exit 0 means selected probes passed; 1 means unhealthy or
unverifiable; 2 means invalid arguments. Check stderr/prerequisites before retrying.

Checks cover chain RPC readiness, validator HTTP health, service metrics, miner
TCP reachability and initialized SQLite headers. SQLite checks read the file only;
they do not prove database integrity. TCP reachability does not prove authenticated
RPC, registration, GPU execution or end-to-end rental behavior.

3. Inspect only the selected services, from `scripts/localnet/`:

```bash
docker compose --profile network ps -a
docker compose --profile network logs --tail=50 subtensor
# For the validator/miner profiles:
docker compose --profile miner logs --tail=50 validator miner
```

4. Compare config with `configs/validator.example.toml` and
   `configs/miner.example.toml`. Do not dump private wallet or SSH-key contents.
   A missing database can mean startup failed before migrations; inspect logs
   before proposing a reset. HTTP `/health` alone does not query SQLite.

## Repairs and lifecycle

Start/rebuild/restart and data deletion are separate from read-only diagnosis.
For an authorized local stack change, `start.sh [profile] [--build]` copies missing
example configs, generates a local miner SSH key, initializes local chain wallets
and registration, and starts selected services. Its initial config-review prompt
requires an interactive terminal. See the README for direct Compose network setup
when only the chain is required.

`stop.sh` preserves volumes by default; `stop.sh --clean` deletes local chain and
service data while preserving wallet files. Review the project and affected volumes
before an authorized reset. Never use global Docker prune as a localnet repair.
Record cleanup intent before starting an isolated test stack, and remove only that
test's project containers, volumes and network afterward.

## Verification of checker changes

```bash
python3 -m unittest discover -s scripts/localnet/tests -v
for script in scripts/localnet/test.sh scripts/localnet/start.sh; do bash -n "$script"; done
shellcheck scripts/localnet/test.sh scripts/localnet/start.sh
```

These unit tests isolate Docker/HTTP boundaries and do not start services. A real
network-profile check is a separate local Docker validation with cleanup, as
documented in the README. Do not claim full validator/miner runtime verification
from hermetic checker tests or a Subtensor-only check.
