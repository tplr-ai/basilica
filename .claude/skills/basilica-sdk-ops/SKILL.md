---
name: basilica-sdk-ops
description: Route Basilica Python automation, notebooks, SDK integration and managed distributed training tasks to the authoritative customer skill.
---

# Basilica Python SDK and distributed training

This checkout entry is a contributor routing adapter. For Python automation, notebooks, SDK integration and managed distributed training,
open the installed **use-basilica** skill. If it is unavailable, read the
[pinned customer skill](https://github.com/one-covenant/basilica-skills/blob/7368f8d4f8156e46a3067501afa7e2f09785eb5b/skills/use-basilica/SKILL.md)
and its **SDK Utilities** and **Distributed Training** sections. The URL is usable without a checkout or installation.

The customer skill owns command syntax, authentication, authorization and cleanup
procedures. Do not infer authorization to create chargeable resources from this
routing entry. For an overview of other tasks, open the adjacent
[cloud operator router](../basilica-cloud-operator/SKILL.md).

For contributor changes to installed guidance, edit `skills/use-basilica/` in
the **basilica-skills** source repository, then update the public CLI bundle pin
and release it. The [distribution guide](https://github.com/one-covenant/basilica/blob/main/docs/AGENT-SKILLS.md)
describes versioning, publication and migration; editing this adapter alone does
not change what customers install.

Managed multi-rank PyTorch/NCCL (DDP, DiLoCo, FSDP) uses
`@basilica.distributed` or `basilica.distributed(command=[...])`. Choose rentals
when the task specifically requires manual host/SSH control or an unsupported
launcher/runtime, rather than selecting rentals for all distributed training.
