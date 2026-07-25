# Semgrep SAST

The `Semgrep SAST` workflow runs the pinned Semgrep CE engine with the pinned
community Rust rules. There are currently no Basilica-owned Semgrep rules.
Repository-specific rules should only be added for a concrete invariant or
regression, with adjacent positive and negative fixtures.

Pull request and branch runs are invoked by the main CI workflow so scanner,
configuration, artifact, and trusted code-scanning upload failures are included
in the stable `ci-success` aggregate. Weekly and manual runs remain available
directly from this workflow.

Rust findings use Semgrep's `--no-error` finding policy, while scanner
crashes and malformed or missing configuration remain fatal. The workflow does
not use `continue-on-error`, `|| true`, or `pull_request_target`.

SARIF is retained as a short-lived artifact. Upload to GitHub code scanning is
limited to trusted events and same-repository pull requests. The CE scan does
not require a Semgrep account or token.

To update Semgrep, review the engine and rule changes, update the immutable
values in `upstream-rules.lock.yml` and the workflow, then repeat workflow
validation and the advisory Rust baseline scan.
