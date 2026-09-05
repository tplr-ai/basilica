---
name: basilica-cloud-operator
description: Use when the user broadly wants to operate Basilica as a cloud platform customer, including account setup, funding, rentals, deployments, inference, OpenClaw, Tau, or SDK automation.
---

# Basilica Cloud Operator

This is a checkout routing adapter. The authoritative installed customer skill
is **use-basilica** from **one-covenant/basilica-skills**. The adjacent entries
route task categories to that single maintained command reference.

Use it when the user says things like:

- "use Basilica"
- "deploy this on Basilica"
- "rent a GPU"
- "fund my account"
- "check my balance"
- "spin up OpenClaw"
- "do this from Python with Basilica"

## Routing

Open the narrower skill that matches the task. Links resolve relative to this
skill directory, including when the bundle is installed outside a checkout:

- [basilica-account-ops](../basilica-account-ops/SKILL.md)
  - auth, tokens, balance, funding, deposits
- [basilica-rentals-ops](../basilica-rentals-ops/SKILL.md)
  - direct machines, SSH, volumes, file copy, teardown
- [basilica-serverless-ops](../basilica-serverless-ops/SKILL.md)
  - managed deploys, inference endpoints, OpenClaw, Tau, logs, scale, delete
- [basilica-sdk-ops](../basilica-sdk-ops/SKILL.md)
  - Python automation, scripts, notebooks, SDK caveats

## First Decision

Pick one of these control planes first:

- balance/funding/account question -> account ops
- wants an SSH box -> rentals ops
- wants a URL or HTTP API -> serverless ops
- wants Python code or automation -> SDK ops

## Guardrails

- treat `basilica up`, `basilica deploy`, `basilica summon`, and SDK create calls as chargeable actions
- prefer read-only inspection first: balance, ls, ps, deploy status, logs
- when creating a deployment, set a TTL unless the user explicitly wants persistence
- when creating a rental, tear it down after the task unless the user explicitly wants to keep it

When the adjacent skills are unavailable, use the installed `use-basilica` skill
or the [pinned customer skill](https://github.com/one-covenant/basilica-skills/blob/7368f8d4f8156e46a3067501afa7e2f09785eb5b/skills/use-basilica/SKILL.md).

## Canonical Source Paths

The following are contributor references relative to a Basilica checkout root.
Without a checkout, use the [source repository](https://github.com/one-covenant/basilica)
or the pinned customer skill above; do not assume these paths exist locally.

- `AGENTS.md`
- `crates/basilica-cli/src/cli/`
- `crates/basilica-sdk-python/`
- `examples/`

## Publication

Installed customer guidance changes in **basilica-skills**, followed by a public
CLI manifest update and release. See the
[distribution guide](https://github.com/one-covenant/basilica/blob/main/docs/AGENT-SKILLS.md).
