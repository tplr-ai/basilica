# Gitleaks policy

`Gitleaks` scans the exact commit range introduced by a pull request or push.
The workflow pins the scanner version, verifies the selected Linux archive
against the repository-owned checksum list, fetches full history, and redacts
findings from logs.

The required baseline review on `476ef9d9` produced 28 findings:

| Classification | Files | Findings | Decision |
| --- | --- | ---: | --- |
| Public Auth0 client identifier | `crates/basilica-common/build.rs` | 1 | Allow only the named identifier line. Client IDs are not credentials. |
| Generated PEM framing assertions | Four named crypto implementation/test files | 8 | Allow only private-key rule matches on the named files and framing expressions. No key material is stored there. |
| Fixed cryptography and SDK examples | `aead.rs`, `types.rs` | 3 | Allow only the two fixed test-value patterns in those files. |
| Documentation token placeholder | `docs/GETTING-STARTED.md` | 5 | Allow only the exact public placeholder token. |
| Deterministic localnet identities | Six exact SSH/wallet fixture files | 11 | Allow the exact public fixture files required by localnet. No directory is blanket-allowlisted. |

All 28 findings are intentional public examples or localnet fixtures. The
review found no real credential, so no rotation is required. Any new finding
outside these narrow exceptions blocks the workflow and must be classified
before changing this policy.
