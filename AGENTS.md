# Basilica contributor and agent guidance

This repository owns the public CLI, Python/Rust SDKs, common/protocol crates,
miner, validator and customer examples. Internal APIs, billing, orchestration,
cluster-manager Python and IaC belong to the separate `basilica-backend`
repository; read its own `AGENTS.md` before working there. Standalone public
builds do not require a particular parent directory or sibling checkout.

`AGENTS.md` is authoritative; `CLAUDE.md` links to this same body. All paths below
resolve from this repository root. Scoped `AGENTS.md` files add subsystem rules;
any scoped `CLAUDE.md` must link to the same body rather than duplicate policy.

## Develop Basilica

Start with [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for prerequisites, locked
build/test commands and separation of ordinary checks from live operations.

| Task | Source / scoped guidance |
| --- | --- |
| CLI syntax, JSON and noninteractive behavior | [CLI rules](crates/basilica-cli/AGENTS.md), [CLI source](crates/basilica-cli/src/cli/) |
| Python bindings, decorators, managed distributed training | [Python SDK rules](crates/basilica-sdk-python/AGENTS.md), [Python source](crates/basilica-sdk-python/python/basilica/) |
| Rust SDK | [SDK source](crates/basilica-sdk/src/), [SDK README](crates/basilica-sdk/README.md) |
| Common types / protocol | [common](crates/basilica-common/src/), [protocol](crates/basilica-protocol/src/) |
| Miner / validator | [miner](crates/basilica-miner/src/), [validator](crates/basilica-validator/src/) |
| Examples and user docs | [examples](examples/README.md), [documentation index](docs/README.md) |
| Local development stack | [localnet skill](.claude/skills/basilica-localnet-debug/SKILL.md), [scripts](scripts/localnet/) |
| Installed customer agent guidance | [distribution contract](docs/AGENT-SKILLS.md), [CLI manifest](crates/basilica-cli/src/cli/handlers/skills/bundle.json) |

## Work and verification contract

- Complete the requested scope, including ordinary implementation and validation
  needed to make it reviewable. Do not create unrelated changes or resources.
- Track multi-step work in a task checklist. A checklist is allowed; an
  unimplemented production stub or TODO-as-implementation is not. Isolated test
  doubles are allowed for deterministic unit/regression tests.
- Read source and current references before historical prompts/transcripts.
  Prefer narrow package checks and checking commands (`cargo fmt --all -- --check`)
  before mutating commands. Report failures, skipped tests and missing prerequisites.
- Do not infer permission to use live services from available credentials or a
  reachable environment. Never read or commit secrets to troubleshoot instructions.
- Commits, pushes, PRs, issues and deployments follow the user's requested scope.
  Conventional scoped commits describe the actual change. No automatic issue,
  PR, ADR or subagent requirement applies to every small fix. Use subagents when
  requested and available; otherwise perform and review the work directly.
- Keep production code complete, simple and robust. Preserve existing changes by
  other contributors and verify behavior before reporting success.

## Operate Basilica

Use the installed **use-basilica** customer skill for CLI/SDK operations. These
checkout adapters route to its pinned authoritative source when it is absent:

- [account ops](.claude/skills/basilica-account-ops/SKILL.md): auth, credits, deposits
- [rentals ops](.claude/skills/basilica-rentals-ops/SKILL.md): machines, SSH, volumes
- [serverless ops](.claude/skills/basilica-serverless-ops/SKILL.md): HTTP deploys, inference, OpenClaw/Tau
- [SDK ops](.claude/skills/basilica-sdk-ops/SKILL.md): Python automation and managed training
- [cloud operator router](.claude/skills/basilica-cloud-operator/SKILL.md): broad tasks

For end-to-end user documentation, read [the operator playbook](docs/agent-cloud-ops.md).
Chargeable actions require user intent. Include TTL/cleanup plans and remove
created resources when finished unless the user requested persistence.

Managed multi-rank PyTorch/NCCL (DDP, DiLoCo, FSDP) uses
`@basilica.distributed` or `basilica.distributed(command=[...])`; see the
[SDK rules](crates/basilica-sdk-python/AGENTS.md). Choose direct rentals for
manual host/SSH control, custom system setup or an unsupported runtime, rather
than routing all distributed workloads to rentals.
