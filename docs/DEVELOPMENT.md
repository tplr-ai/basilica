# Developing Basilica

Paths and commands below are relative to this repository root unless stated
otherwise. Root [AGENTS.md](../AGENTS.md) is the shared agent policy;
[CLAUDE.md](../CLAUDE.md) links to the same body.

## Prerequisites

Install Rust through rustup; `rust-toolchain.toml` selects the repository version.
Add rustfmt and clippy with `rustup component add rustfmt clippy`. Builds require
Protobuf `protoc` (CI uses 25.9), a C/C++ compiler, pkg-config and OpenSSL development
headers. On Ubuntu the CI packages include `pkg-config libssl-dev xxd clang`;
CI additionally uses mold for linking. On macOS use Xcode command line tools and
Homebrew `protobuf pkg-config openssl`. Python SDK work needs Python 3.10–3.13
and uv 0.11.32 (the version pinned by CI; older uv cannot parse the
relative `exclude-newer` setting). Check `.github/workflows/ci.yml` for the current platform matrix.

Dependencies are locked in `Cargo.lock` and the Python SDK's `uv.lock`. Do not
update either lockfile as a side effect of a fix. Public standalone builds do
not require the backend checkout. To test changes across both repositories,
follow the backend development guide's explicit sibling dependency mode.

## Task-to-check map

Begin with `cargo fmt --all -- --check`; `cargo fmt` changes source. Use the
narrowest relevant package and test filter, then the CI lane for changed code.
These checks do not create platform resources or use a Kubernetes context.

| Work area | Source entry point | Validation |
| --- | --- | --- |
| CLI, interactive/agent input | `crates/basilica-cli/src/cli/` and `src/interactive/` | `cargo test --locked -p basilica-cli --lib`; `cargo clippy --locked -p basilica-cli --all-targets --all-features -- -D warnings` |
| Rust SDK | `crates/basilica-sdk/src/` | `cargo test --locked -p basilica-sdk --lib`; `cargo clippy --locked -p basilica-sdk --all-targets --all-features -- -D warnings` |
| Common types / protocol | `crates/basilica-common/src/`, `crates/basilica-protocol/src/` | `cargo check --locked -p basilica-common -p basilica-protocol`; `cargo clippy --locked -p basilica-common -p basilica-protocol --all-targets -- -D warnings` |
| Miner / validator | `crates/basilica-miner/src/`, `crates/basilica-validator/src/` | `cargo clippy --locked -p basilica-miner -p basilica-validator --all-targets --all-features -- -D warnings`; select deterministic unit tests before running integration tests |
| Python SDK / managed training | `crates/basilica-sdk-python/python/basilica/`, `crates/basilica-sdk-python/src/` | Locked Python workflow below; `cargo clippy --locked -p basilica-sdk-python --all-targets -- -D warnings` |
| Localnet and its diagnostics | `scripts/localnet/` | `for script in scripts/localnet/*.sh; do bash -n "$script" || exit; done`; inspect selected Compose profile; runtime checks require an isolated localnet |
| Agent skills and examples | `.claude/skills/`, `docs/agent-cloud-ops.md` | `python3 scripts/ci/check-agent-instructions.py`; `python3 scripts/ci/test-agent-instructions.py`; `python3 scripts/ci/test-agent-installer.py`; `cargo test --locked -p basilica-cli --lib agent_rental_playbook` (parser only, no handlers) |

For changes to the CI workflow, install actionlint and run
`actionlint .github/workflows/ci.yml`. The checked-in `.github/actionlint.yaml`
declares the Blacksmith runner label used by this repository.

## Python SDK: clean CI-equivalent environment

Run from `crates/basilica-sdk-python/`:

```bash
uv sync --locked --extra dev --no-install-project
uv run --no-sync maturin develop --release --locked
uv run --no-sync python -m pytest tests/ --collect-only -q \
  --ignore=tests/test_dns_propagation_e2e.py \
  --ignore=tests/test_gpu_flavour_preferences.py
uv run --no-sync python -m pytest tests/ -v \
  --ignore=tests/test_dns_propagation_e2e.py \
  --ignore=tests/test_gpu_flavour_preferences.py
```

The dev extra includes httpx, pytest, pytest-asyncio, maturin and tooling needed
for collection. Installing pytest alone is insufficient. The two excluded files
are live tests requiring credentials and potentially chargeable deployments;
only run them when the user explicitly authorizes that work and cleanup.

## Integration, database and localnet checks

Inspect each test's prerequisites before running it. Unit tests use isolated
fixtures/test doubles; this does not permit stubbed production implementations.
Do not infer authorization to contact real services from a reachable endpoint,
existing credential, wallet or kubeconfig. Localnet startup changes local
containers, networks, wallets and volumes; use an isolated development stack
when authorized, and specify the selected profile in health checks. This repo's
minimal localnet uses SQLite inside its validator/miner, not an external SQL
service. Backend database/integration tests belong to the backend guide.

Report the exact check, result and skipped coverage. A missing compiler, image,
chain, database or test dependency is an unmet prerequisite, not a passing test.
