---
name: basilica-account-ops
description: Route Basilica account setup, authentication, credits, deposits and funding tasks to the authoritative customer skill.
---

# Basilica Account and funding

This checkout entry is a contributor routing adapter. For account setup, authentication, credits, deposits and funding,
open the installed **use-basilica** skill. If it is unavailable, read the
[pinned customer skill](https://github.com/one-covenant/basilica-skills/blob/7368f8d4f8156e46a3067501afa7e2f09785eb5b/skills/use-basilica/SKILL.md)
and its **Account and funding** section. The URL is usable without a checkout or installation.

The customer skill owns command syntax, authentication, authorization and cleanup
procedures. Do not infer authorization to create chargeable resources from this
routing entry. For an overview of other tasks, open the adjacent
[cloud operator router](../basilica-cloud-operator/SKILL.md).

For contributor changes to installed guidance, edit `skills/use-basilica/` in
the **basilica-skills** source repository, then update the public CLI bundle pin
and release it. The [distribution guide](https://github.com/one-covenant/basilica/blob/main/docs/AGENT-SKILLS.md)
describes versioning, publication and migration; editing this adapter alone does
not change what customers install.
