# Release environment protection

The tag-triggered release workflows are gated by GitHub deployment environments.
Jobs that publish public artifacts declare an `environment:` in YAML; the
environment's **protection rules** are configured in the repository settings
and are what actually enforces human approval, wait timers, or branch
restrictions.

## Environments in use

| Environment     | Used by                                              | Purpose                                                      |
|-----------------|------------------------------------------------------|--------------------------------------------------------------|
| `cli-release`   | `.github/workflows/release-cli.yml` — `create-release` job | Gate for the whole CLI release pipeline (binaries depend on this job) |
| `auth0-variables` | `.github/workflows/release-cli.yml` — `build-binaries` matrix | Provides Auth0 build-time variables (secrets)                |
| `pypi`          | `.github/workflows/release-python-sdk.yml` — `publish-to-pypi` job | PyPI trusted-publishing OIDC audience + gate before upload   |

## Required GitHub-side setup

Go to **Settings → Environments** in the `one-covenant/basilica` repo and for
each of `cli-release` and `pypi`:

1. **Add protection rule: Required reviewers.** Add the maintainers who should
   approve each release. With a single approver the release blocks on one
   click; with multiple the first approver unblocks it.
2. **Restrict deployment branches/tags to `main` only.** Under "Deployment
   branches and tags" pick "Selected branches and tags" and add only `main`.
   This prevents a pushed tag on a feature branch from triggering a release.
3. Leave wait timer at 0 unless you want a forced cooling-off period.

For `pypi` specifically, the existing trusted-publishing configuration (OIDC
audience, PyPI project name) must remain in place — only add protection rules,
do not change the publisher configuration.

`auth0-variables` does **not** need a reviewer gate: its sole job is to feed
build-time variables into the binary build matrix, and the pipeline is already
blocked upstream by `cli-release`.

## Verifying the gate works

After configuring `cli-release`:

1. Push a throwaway tag `basilica-cli-v0.26.1-test` (or use `workflow_dispatch`).
2. Workflow starts, `create-release` job enters "Waiting for review" status.
3. No binary build job runs until the review is approved.
4. Approve via the Actions UI; pipeline continues.

For `pypi`:

1. Push a throwaway tag `basilica-sdk-python-v0.26.1-test`.
2. All wheel build jobs run (they are not gated — they only produce artifacts
   in the runner, no external side effects).
3. `publish-to-pypi` enters "Waiting for review" before any upload happens.
4. Approve → PyPI receives the upload.
