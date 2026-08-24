# Basilica Scripts

Tooling scripts for building, installing, and testing Basilica.

## Directories

- **lib/** — shared shell helpers (color output, SSH wrapper, test utilities).
- **test/** — workspace test runner, verifier, and statistics scripts.
- **web/** — the user-facing CLI installer (`install.sh`) and agent skills installer.

## Installing the CLI

Use the hosted installer — this is what users run:

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

The source for this installer lives in [`web/install.sh`](web/install.sh).

## Running tests

```bash
just test                  # cargo test --workspace
just test-run              # scripted runner with filters
just test-verify           # verifies test structure
just test-stats            # prints test counts per crate
```
