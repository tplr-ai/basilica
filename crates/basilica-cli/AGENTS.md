# CLI contributor guidance

Read [the root policy](../../AGENTS.md) and
[the contributor checks](../../docs/DEVELOPMENT.md).

- `src/cli/commands.rs` owns clap syntax; `src/cli/handlers/` owns behavior.
  Validate example commands with `Args::try_parse_from`, never by invoking an
  authenticated handler that may create or delete resources.
- Noninteractive behavior is an API contract. Preserve structured `MissingInput`
  errors/recovery hints, explicit offering selection, and `src/interactive/gate.rs`.
  New flags need both parser coverage and the relevant noninteractive gate check.
- Keep JSON output machine-readable and propagate remote process exit codes.
  Diagnostics belong to stderr; do not insert human status text into JSON output.
- Rental creation with `--offering-id` accepts `--name` and `--detach`; offering
  selection filters conflict because the selected offering already fixes them.
- Customer skill installation is owned by the versioned manifest and storage
  module under `src/cli/handlers/skills/`. Read the
  [distribution contract](../../docs/AGENT-SKILLS.md) before changing installation,
  migration or guidance publication; never overwrite user-modified skill content.

Run `cargo test --locked -p basilica-cli --lib` from the repository root for unit
coverage, then `cargo clippy --locked -p basilica-cli --all-targets --all-features -- -D warnings`.
Live account/rental/deployment tests require separate explicit authorization.
