---
name: basilica-rentals-ops
description: Use when the user wants direct machine access on Basilica via rentals, including discovery, SSH keys, secure-cloud/community-cloud selection, volumes, SSH, exec, file copy, and teardown.
---

# Basilica Rentals Ops

Use this skill when the user wants:

- a GPU or CPU machine with shell access
- direct SSH access instead of an HTTP deployment
- custom training environments
- long-running jobs that are easier to manage on a box
- huge models that may be awkward as serverless deployments

## When To Choose Rentals Instead Of Deploys

Prefer rentals when the task needs:

- SSH
- custom package or system setup
- distributed training
- manual process control
- large multi-GPU inference where health-check timing is likely to be painful

If the user wants a public URL, an HTTP API, or a managed inference endpoint, use the serverless skill instead.

## Canonical Preflight

Authenticate:

```bash
basilica login
```

Check balance before creating cost-bearing resources:

```bash
basilica balance
```

Make sure an SSH key exists in Basilica:

```bash
basilica ssh-keys list
basilica ssh-keys add
```

## Discover Available Compute

List everything:

```bash
basilica ls
```

Filter by GPU type:

```bash
basilica ls h100
basilica ls h200
```

Filter by budget, location, and compute pool:

```bash
basilica ls --price-max 5 --country US
basilica ls --compute secure-cloud
basilica ls --compute community-cloud
```

Useful filters supported by the CLI include:

- positional GPU type like `h100`
- `--gpu-min`
- `--gpu-max`
- `--price-max`
- `--memory-min`
- `--country`
- `--compute`

## Start A Rental

Example: single-machine rental

```bash
basilica up h100 --gpu-count 1
```

Example: large multi-GPU box

```bash
basilica up h200 --gpu-count 8 --compute secure-cloud
```

The compute selector currently maps to:

- `citadel`, `secure-cloud`, `secure`
- `bourse`, `community-cloud`, `community`

Prefer being explicit with `--compute` when the user cares about the environment.

## Inspect Running Or Previous Rentals

List active rentals:

```bash
basilica ps
```

Include history:

```bash
basilica ps --history
```

Inspect one rental:

```bash
basilica status <rental-id>
```

## Operate A Rental

SSH into the machine:

```bash
basilica ssh <rental-id>
```

Run a one-off command:

```bash
basilica exec "nvidia-smi" --target <rental-id>
```

Stream logs:

```bash
basilica logs <rental-id>
```

Restart:

```bash
basilica restart <rental-id>
```

Copy files:

```bash
basilica cp ./local.txt <rental-id>:/workspace/local.txt
basilica cp <rental-id>:/workspace/output.txt ./output.txt
```

## Volumes

For secure-cloud rentals that need persistent storage:

Create a volume:

```bash
basilica volumes create --name cache --size 100 --provider hyperstack --region US-1
```

List volumes:

```bash
basilica volumes list
```

Attach to a rental:

```bash
basilica volumes attach cache --rental <rental-id>
```

Detach:

```bash
basilica volumes detach cache --yes
```

Delete:

```bash
basilica volumes delete cache --yes
```

Current volume constraints exposed in the CLI:

- provider is `hyperstack`
- regions include `US-1`, `CANADA-1`, `NORWAY-1`

## Teardown

Stop one rental:

```bash
basilica down <rental-id>
```

Stop everything:

```bash
basilica down --all
```

Agents should tear down rentals at the end of a task unless the user explicitly wants them preserved.

## Operational Notes

- `basilica up` ensures an SSH key is present before provisioning.
- Secure-cloud log streaming is limited; if logs are thin or missing, use `basilica ssh <rental-id>` and inspect the machine directly.
- If the user needs CPU-only secure-cloud resources, the Python SDK currently exposes that surface more directly than the CLI.

## Good Agent Defaults

- start with `basilica balance`
- then `basilica ls ...`
- then create the rental
- immediately return the rental ID and connection method
- explicitly ask whether to keep or tear down the machine when the work is done

## File Pointers

- `crates/basilica-cli/src/cli/handlers/gpu_rental.rs`
- `crates/basilica-cli/src/cli/handlers/ssh_keys.rs`
- `crates/basilica-cli/src/cli/handlers/volumes.rs`
- `crates/basilica-sdk-python/examples/start_secure_cloud_gpu_rental.py`
- `crates/basilica-sdk-python/examples/start_cpu_rental.py`

## TODOs

- add a tested example for secure-cloud CPU rentals once the CLI gets a cleaner CPU-first workflow
- add a repo-local transcript showing `basilica up` -> `basilica ssh` -> `basilica down`
