---
name: basilica-account-ops
description: Use when the user wants to log into Basilica, create API tokens, check balance, fund credits with TAO, or inspect deposit history.
---

# Basilica Account Ops

Use this skill for the account-level Basilica operator surface:

- CLI install and login
- device-code auth for remote/headless shells
- API token creation/revocation
- checking credit balance
- funding via TAO deposit address
- reviewing deposit history

## Canonical CLI Setup

Install the CLI:

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

Authenticate:

```bash
basilica login
```

For headless terminals:

```bash
basilica login --device-code
```

Log out:

```bash
basilica logout
```

## API Tokens

Use tokens for SDK/programmatic work, CI, and scripts.

Create:

```bash
basilica tokens create my-agent-token
```

List:

```bash
basilica tokens list
```

Revoke:

```bash
basilica tokens revoke my-agent-token --yes
```

Export for SDK usage:

```bash
export BASILICA_API_TOKEN="basilica_..."
```

## Balance And Funding

Check current balance:

```bash
basilica balance
```

Get or create the user deposit address:

```bash
basilica fund
```

List deposit history:

```bash
basilica fund list --limit 100 --offset 0
```

## Funding Flow Agents Should Follow

If the user says "top up", "fund", "add credits", or "what address do I send TAO to", the operational flow is:

1. run `basilica fund`
2. return the deposit address
3. tell the user to send TAO to that address
4. use `basilica fund list` to confirm deposit arrival
5. use `basilica balance` to confirm credits updated

Important details currently exposed by the CLI implementation:

- funding method is TAO
- minimum deposit is `0.1 TAO`
- funds settle after `12` confirmations

There is no separate `top-up` command. `basilica fund` is the top-up entrypoint.

## What To Prefer

- Prefer `basilica login` for CLI work.
- Prefer API tokens only when the task needs SDK calls or automation.
- Prefer the CLI over the Python SDK for deposit-address creation and deposit-history inspection.

## Safe Read-Only Commands

These are safe to run without creating billable resources:

```bash
basilica balance
basilica fund
basilica fund list --limit 20 --offset 0
basilica tokens list
```

## Common Agent Responses

### Check Whether The User Is Ready To Spend

```bash
basilica balance
```

If the balance is too low for rentals or deploys, direct the user to:

```bash
basilica fund
```

### Prepare A Script Environment

```bash
basilica tokens create my-script-token
export BASILICA_API_TOKEN="basilica_..."
```

### Rotate A Leaked Or Old Token

```bash
basilica tokens list
basilica tokens revoke old-token-name --yes
basilica tokens create replacement-token
```

## Caveats

- The CLI is the authoritative operator surface for deposits.
- The public Python `BasilicaClient` currently exposes `get_balance()` and `list_usage_history()`, but not a clean first-class deposit-account workflow.
- If the user asks for billing/spend analysis, combine CLI balance checks with SDK usage-history access.

## File Pointers

- `crates/basilica-cli/src/cli/handlers/auth.rs`
- `crates/basilica-cli/src/cli/handlers/tokens.rs`
- `crates/basilica-cli/src/cli/handlers/fund.rs`
- `crates/basilica-cli/src/cli/handlers/balance.rs`
- `crates/basilica-sdk-python/README.md`

## TODOs

- add a repo-local example that shows `basilica fund` output and the expected deposit-history shape
- document whether future CLI releases expose spend history or per-resource usage directly
