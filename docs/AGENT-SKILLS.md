# Agent skill distribution

The authoritative customer skill is **use-basilica**, maintained in
[one-covenant/basilica-skills](https://github.com/one-covenant/basilica-skills).
This repository's `.claude/skills/basilica-*-ops` and `basilica-cloud-operator`
entries are contributor routing adapters. They do not publish a second customer
command catalog. `basilica-localnet-debug` is contributor-only local development
guidance. [The operator playbook](agent-cloud-ops.md) remains user documentation.

## Installation and bundle identity

```bash
basilica skills install -y
basilica skills list
basilica skills uninstall -y
```

Both the CLI and the legacy `https://basilica.ai/agents/install.sh` entry point
use the CLI's versioned manifest in
[skills/bundle.json](../crates/basilica-cli/src/cli/handlers/skills/bundle.json).
The shell entry point delegates to the CLI and requires its version-aware
installer; it no longer fetches separate website files. Old `--cursor-only`,
`--codex-only` and `--claude-only` flags remain supported. `--base-url` and
`BASILICA_AGENT_BASE_URL` now fail with migration instructions instead of
selecting another bundle.

The manifest pins an immutable source commit and SHA-256 hashes of every
installed file. Installation rejects downloads whose extracted contents differ.
`BASILICA_SKILLS_TARBALL_URL` can mirror that exact bundle, not select arbitrary
unversioned content. Installation and listing print source and version; each
installed skill records them with owned-file hashes in `.basilica-install.json`.
Reports should include both the CLI version and bundle version. Updating the CLI
then reinstalling selects the bundle pinned by that release; reinstallation with
the same CLI is reproducible.

Universal tools (Codex, Cursor, OpenCode, Amp, Gemini) use `~/.agents/skills`.
Claude Code uses `~/.claude/skills`. Use `--agent universal` or
`--agent claude-code` to choose destinations explicitly. The shell compatibility
entry point selects both by default, matching its previous multi-tool intent.

## Migration and preservation

Installation/upgrade first checks every selected canonical destination. Only an
unchanged receipt-owned skill, or an exact recognized historical snapshot, may
be replaced. Unknown files, user edits, extra files or empty directories, invalid receipts, symlinks
and unrelated skills are preserved; conflicting canonical destinations stop
installation with a path to back up manually. New files are staged before an
existing version is renamed; failed replacement restores the previous version.

After successful installation, known unchanged legacy names are removed from
selected tool roots. Universal migration also visits `~/.cursor/skills` and
`~/.codex/skills`; it does not migrate a non-selected Claude installation.
Recognized legacy names are `basilica-cloud-operator`, `basilica-account-ops`,
`basilica-rentals-ops`, `basilica-serverless-ops`, and `basilica-sdk-ops`.
The matching legacy `BASILICA-CLOUD-OPS.md` is removed too. Unrecognized or edited
legacy copies are reported and retained for manual reconciliation. Uninstall
uses the same ownership checks across canonical and historical directories.
It never assumes ownership from a name alone.

Historical fingerprints and the compressed migration fixture come from public
commit `f3749c72`; the fixture contains the original five skills and playbook.
The known unreceipted `use-basilica` fingerprint and `pinned-bundle.tar.gz`
fixture come from the pinned source revision. Older unknown snapshots are intentionally preserved. Add a reviewed
historical fingerprint only with the corresponding original fixture/evidence.

## Publishing guidance changes

1. For installed customer behavior, edit and validate `skills/use-basilica/` in
   **basilica-skills**, then publish that source change through its normal review
   process. Editing a routing adapter here does not update installed customers.
2. Update this CLI's manifest to that exact reviewed revision and its complete
   file-hash map. Keep prior recognized fingerprints for safe migration. Update
   the pinned source links in contributor routing adapters at the same time.
3. Run the CLI skill tests, parser/instruction checks and temporary-directory
   install/upgrade/uninstall checks. Check the downloaded pinned archive against
   the manifest without installing into a real user's directories.
4. Release the CLI normally. Both supported entry points then select the same
   released manifest; no separate website skill publishing step remains.

This workflow does not authorize live resource creation to validate examples.
