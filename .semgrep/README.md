# Semgrep SAST

The `Semgrep SAST` workflow has two independent jobs:

- `semgrep/custom-required` validates and tests Basilica-owned rules, then
  blocks matches in production workflows.
- `semgrep/advisory` runs pinned community security rules over Rust, Python,
  shell, Dockerfile, Docker Compose, and GitHub Actions surfaces.

Community findings use Semgrep's `--no-error` finding policy, while scanner
crashes and malformed or missing configuration remain fatal. The workflow does
not use `continue-on-error`, `|| true`, or `pull_request_target`.

The initial full baseline ran 142 rules over 479 tracked targets and produced
49 advisory findings. Parser warnings remain visible for embedded Actions
expressions and one complex shell script; they are not suppressed or presented
as complete coverage.

The repository-owned rules enforce two narrow CI invariants: workflows may not
use `pull_request_target`, and may not grant `packages: write`. Adjacent
positive and negative fixtures are checked with `semgrep --test` before the
blocking scan.

SARIF is retained as a short-lived artifact. Upload to GitHub code scanning is
limited to trusted events and same-repository pull requests. The CE scan does
not require a Semgrep account or token.

To update Semgrep, review the engine and rule changes, update the immutable
values in `upstream-rules.lock.yml` and the workflow, then repeat validation,
fixture tests, the blocking scan, and the advisory baseline scan.
