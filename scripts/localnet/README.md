# Localnet profiles and health checks

The supported Compose stack contains Subtensor, a SQLite-backed validator and a
SQLite-backed miner. It does not deploy PostgreSQL, Prometheus or Grafana.

| Profile | Services | Default host endpoints |
| --- | --- | --- |
| `network` | subtensor | Chain RPC 9944 |
| `validator` | subtensor, validator | Plus validator API 8080, metrics 9090, bidding gRPC 50052 |
| `miner` / `all` | subtensor, validator, miner | Plus miner gRPC/axon 8092, metrics 9091 |

Every Compose service has a profile: use `--profile network`, `--profile validator`,
`--profile miner` or `--profile all` with direct Compose commands. `start.sh`
defaults to the complete stack. There is no monitoring profile or separate miner
8091 listener; miner's configured axon and gRPC endpoint are container port 50051.

## Start and inspect

Run from this directory. `start.sh [profile] [--build]` sets up the shared network,
copies missing example configs, generates a local SSH key and initializes local
wallets/subnet registration. The initial config-review prompt is interactive.
This workflow needs Docker Compose, curl, nc, SSH tooling, uv/uvx and jq. Building
validator/miner images also needs access to their pinned Rust dependencies.

For an explicitly requested **chain-only** local check that does not need wallet
initialization or service builds:

```bash
docker network create basilica-localnet --subnet 172.28.0.0/16
docker compose --profile network up -d
./test.sh network
```

Create the named network only if absent; inspect an existing network instead of
replacing it. Stop the owned project when finished. Normal `stop.sh` preserves
volumes; `stop.sh --clean` resets chain/service data. Do not remove unrelated
containers, volumes or shared networks. An isolated review can supply a separate
`COMPOSE_PROJECT_NAME`, a `COMPOSE_FILE` override for container/network names and
loopback dynamic ports, and clean up that exact project afterward.

## Read-only health contract

`test.sh [profile]` requires Python 3.11+ and Docker Compose. The default is `all`.
The checker asks Compose for selected services, container IDs and published ports,
so an isolated project or overridden host port works. `COMPOSE_PROFILES` is ignored
by the checker to prevent ambient profiles from widening the explicit selection.
Docker must be local to the machine running the checker; remote Docker engines
need their own reachable endpoint routing.

- Subtensor: JSON-RPC `system_health` must respond and report `isSyncing=false`.
- Validator: HTTP health must report healthy; metrics must contain its service prefix.
- Miner: published gRPC TCP port accepts connections; metrics contain its service prefix.
- Included validator/miner services: read the SQLite URL from the mounted TOML and
  check that the container's file begins with the initialized SQLite header.

All included containers must be running and any Docker health status must be ready.
Missing services, unavailable probes, unsupported service/configuration or invalid
SQLite headers fail the check. Nothing is created or modified by `test.sh`.

Exit status is 0 for all selected probes passing, 1 for unhealthy/unverifiable,
and 2 for invalid CLI arguments. A successful probe is **not** a database integrity
check, authenticated gRPC call, neuron registration or rental/training test.
The validator health endpoint does not itself query the database.

## Validate changes

```bash
python3 -m unittest discover -s scripts/localnet/tests -v
for script in scripts/localnet/test.sh scripts/localnet/start.sh; do bash -n "$script"; done
shellcheck scripts/localnet/test.sh scripts/localnet/start.sh
```

Run the commands above from the repository root. Unit tests use isolated Docker/HTTP
test doubles and a real SQLite fixture file; they create no live services. Separately
validate Compose inventory for every supported profile with
`docker compose --profile PROFILE config --services` from this directory. For live
validation, launch the selected local stack, record actual checker results and the
image/source revision, and remove only the resources created for that test. State
which profiles were actually launched; a network-profile pass does not prove full
validator/miner runtime readiness.
