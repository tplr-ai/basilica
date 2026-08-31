"""RL training namespace: GRPO post-training on the Basilica RL Training API.

    >>> from basilica import BasilicaClient
    >>> client = BasilicaClient()               # BASILICA_API_TOKEN / CLI login
    >>> client.rl.create_cluster(
    ...     name="my-pool",
    ...     base_model="Qwen/Qwen2.5-7B-Instruct",
    ...     gpu_model="H100",
    ... )
    >>> client.rl.wait_cluster("my-pool")
    >>> job = client.rl.create_job(
    ...     cluster="my-pool", max_steps=50,
    ...     reward_name="my-reward", reward_source=REWARD_PY,
    ...     dataset_name="my-data", dataset_repo="openai/gsm8k",
    ...     dataset_config="main", dataset_split="train",
    ...     prompt_column="question", answer_column="answer",
    ... )
    >>> final = client.rl.wait_job(job["name"])  # {phase, step, metrics, artifactURI}

THIN WRAPPER over the compiled core: this module builds the ergonomic
kwargs into wire dicts and hands them to the Rust binding's ``rl_*``
methods, which serde-validate against the core's typed DTOs
(``basilica_sdk::rl`` — the compile-time-shared contract with the server)
and send through the core transport. That inherits the full auth chain
(explicit key, BASILICA_API_TOKEN, and the CLI-login token fallback) and
the core's error mapping: non-2xx surfaces as ValueError (bad request),
PermissionError (authz), ConnectionError (transport), KeyError
(not found), or RuntimeError (server error), each carrying the server's
message verbatim.

The ``body=`` escape hatch on every create call sends a raw dict; unknown
fields survive the typed round-trip verbatim (serde-flatten catch-alls in
the core DTOs), so server-side schema additions never strand you on an SDK
release.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

_TERMINAL_JOB_PHASES = frozenset({"Succeeded", "Failed", "TimedOut"})
# Degraded is deliberately NOT here: a cluster degrades on transient fleet
# unhealth (pod restart, node blip) and can recover to Ready; only
# Terminating can never become Ready again.
_DEAD_CLUSTER_PHASES = frozenset({"Terminating"})
# A single LB 502 or connection reset must not abort a multi-hour wait;
# this many CONSECUTIVE poll failures (reset on any success) give up.
_POLL_FAILURE_BUDGET = 5


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class RlNamespace:
    """The ``client.rl`` surface. Constructed by BasilicaClient; hold no
    credentials here — the compiled core owns auth and transport."""

    def __init__(self, core: Any):
        self._core = core

    # -- clusters ----------------------------------------------------------

    def create_cluster(
        self,
        *,
        base_model: Optional[str] = None,
        gpu_model: Optional[str] = None,
        trainer_gpus: int = 4,
        rollout_gpus: int = 4,
        name: Optional[str] = None,
        min_memory_gb: Optional[int] = None,
        idle_ttl: Optional[str] = None,
        relay: Optional[dict] = None,
        body: Optional[dict] = None,
    ) -> dict:
        """POST /rl/clusters — a warm trainer+rollout GPU pool.

        Certified shapes: 4+4 (H100 for <16B models, H200-class for >=16B —
        admission rejects bad pairings with an actionable message).

        ``relay`` selects bring-your-own storage (wire-shaped dict; the same
        dict works inside a manifest's ``cluster`` block)::

            relay={
                "mode": "byo",
                "endpoint": "https://<acct>.r2.cloudflarestorage.com",
                "bucket": "my-weights",
                "basePrefix": "teams/rl/",          # optional
                "accessKeyId": "...",               # write-only: becomes a
                "secretAccessKey": "...",           #   secret, never echoed
                # or instead of the pair: "credentialsSecret": "my-secret"
            }

        The response then carries ``effectivePrefix`` — the uid-scoped key
        prefix all cluster data lands under; tighten your IAM grant to it.
        Omit ``relay`` for platform-managed storage (today's behavior).
        ``body`` replaces the built request entirely (raw wire dict, escape
        hatch) — no other kwargs are consulted when it is given.
        """
        if body is None:
            if base_model is None or gpu_model is None:
                raise ValueError(
                    "base_model and gpu_model are required (unless a raw body= is given)"
                )

            def fleet(count: int) -> dict:
                return {
                    "replicas": 1,
                    "gpu": _drop_none(
                        {"model": gpu_model, "count": count, "minMemoryGb": min_memory_gb}
                    ),
                }

            body = _drop_none(
                {
                    "name": name,
                    "baseModel": base_model,
                    "trainer": fleet(trainer_gpus),
                    "rollout": fleet(rollout_gpus),
                    "idleTtl": idle_ttl,
                    "relay": relay,
                }
            )
        return json.loads(self._core.rl_create_cluster(json.dumps(body)))

    def rotate_credentials(
        self, name: str, *, access_key_id: str, secret_access_key: str
    ) -> dict:
        """POST /rl/clusters/{name}/credentials — rotate a BYO cluster's
        storage credentials (only clusters created with the inline pair;
        clusters using ``credentialsSecret`` update their own secret).

        Sequencing: create the NEW key at your provider first (both keys
        valid), call this, then revoke the OLD key after the returned
        ``rotatedAt`` plus a few seconds — the relay daemon restarts onto
        the new material in that window; in-flight transfers finish on the
        old key. Revoking first fails the running job with
        ``RelayAuthFailed`` (recoverable, but wastes the retry window)."""
        req = {"accessKeyId": access_key_id, "secretAccessKey": secret_access_key}
        return json.loads(
            self._core.rl_rotate_cluster_credentials(name, json.dumps(req))
        )

    def get_cluster(self, name: str) -> dict:
        return json.loads(self._core.rl_get_cluster(name))

    def delete_cluster(self, name: str) -> dict:
        """Delete a cluster. Refused (ValueError) while a job is active —
        the error names the blocking job; delete the job first (that IS the
        cancel path). Deleting the namespace's last cluster also tears down
        its RL prerequisites server-side."""
        return json.loads(self._core.rl_delete_cluster(name))

    def delete_job(self, name: str) -> dict:
        """Delete a job — valid in any phase. Deleting a running job IS the
        cancel path (pods are torn down by the operator's stop ladder)."""
        return json.loads(self._core.rl_delete_job(name))

    def wait_cluster(
        self, name: str, timeout_s: float = 1800.0, poll_s: float = 15.0
    ) -> dict:
        """Poll until phase == Ready. Raises RuntimeError immediately on
        Terminating (it can never become Ready — waiting out the full
        timeout would hide the failure), TimeoutError on the deadline.
        Degraded keeps polling: fleets recover from transient unhealth."""
        deadline = time.monotonic() + timeout_s
        failures = 0
        while True:
            try:
                cluster = self.get_cluster(name)
            except (ConnectionError, RuntimeError):
                failures += 1
                if failures >= _POLL_FAILURE_BUDGET:
                    raise
                time.sleep(poll_s)
                continue
            failures = 0
            phase = cluster.get("phase")
            if phase == "Ready":
                return cluster
            if phase in _DEAD_CLUSTER_PHASES:
                raise RuntimeError(f"cluster {name!r} entered {phase}: {cluster}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"cluster {name!r} not Ready after {timeout_s}s "
                    f"(last phase: {phase!r})"
                )
            time.sleep(poll_s)

    # -- jobs --------------------------------------------------------------

    def create_job(
        self,
        *,
        cluster: Optional[str] = None,
        max_steps: Optional[int] = None,
        name: Optional[str] = None,
        algorithm: str = "grpo",
        # custom reward (user:<name> + inline source); omit for the builtin
        reward_name: Optional[str] = None,
        reward_source: Optional[str] = None,
        judge: bool = False,
        judge_model: Optional[str] = None,
        # custom dataset (public HF repo + column mapping); omit for builtin
        dataset_name: Optional[str] = None,
        dataset_repo: Optional[str] = None,
        dataset_split: Optional[str] = None,
        dataset_config: Optional[str] = None,
        prompt_column: Optional[str] = None,
        answer_column: Optional[str] = None,
        lr: Optional[str] = None,
        body: Optional[dict] = None,
    ) -> dict:
        """POST /rl/jobs — a GRPO training job on a Ready cluster.

        The reward is any deterministic stdlib-Python
        ``reward(prompt, completion, **ctx) -> float``; it runs in an
        isolated credential-free pod. ``judge=True`` exposes
        ``ctx["judge"](prompt)`` backed by an in-cluster judge model
        (requires a custom reward). ``body`` replaces the built request
        entirely — no other kwargs are consulted when it is given.

        Orphan kwargs raise: a ``reward_source`` without ``reward_name`` (or
        dataset fields without ``dataset_name``) would otherwise be silently
        dropped and the BUILTIN reward/dataset would run on a paid GPU job.
        """
        if body is None:
            if cluster is None or max_steps is None:
                raise ValueError(
                    "cluster and max_steps are required (unless a raw body= is given)"
                )
            reward = None
            if reward_name is not None:
                if reward_source is None:
                    raise ValueError("reward_source is required with reward_name")
                reward = {"ref": f"user:{reward_name}", "source": reward_source}
                if judge or judge_model:
                    reward["judge"] = _drop_none({"model": judge_model})
            elif reward_source is not None:
                raise ValueError("reward_name is required with reward_source")
            elif judge or judge_model:
                raise ValueError(
                    "judge requires a custom reward (it is called from your reward code)"
                )
            dataset = None
            if dataset_name is not None:
                dataset = {
                    "ref": f"user:{dataset_name}",
                    "hf": _drop_none(
                        {
                            "repo": dataset_repo,
                            "config": dataset_config,
                            "split": dataset_split,
                            "promptColumn": prompt_column,
                            "answerColumn": answer_column,
                        }
                    ),
                }
            elif any(
                v is not None
                for v in (
                    dataset_repo,
                    dataset_config,
                    dataset_split,
                    prompt_column,
                    answer_column,
                )
            ):
                raise ValueError("dataset_name is required with dataset fields")
            body = _drop_none(
                {
                    "clusterRef": cluster,
                    "name": name,
                    "algorithm": algorithm,
                    "maxSteps": max_steps,
                    "reward": reward,
                    "dataset": dataset,
                    "lr": lr,
                }
            )
        return json.loads(self._core.rl_create_job(json.dumps(body)))

    def get_job(self, name: str) -> dict:
        return json.loads(self._core.rl_get_job(name))

    def wait_job(
        self, name: str, timeout_s: float = 6 * 3600.0, poll_s: float = 30.0
    ) -> dict:
        """Poll until the job is terminal (Succeeded/Failed/TimedOut) and
        return the final document either way — check ``phase`` yourself;
        raising on Failed would hide the failure detail behind an
        exception. Transient poll errors are tolerated up to
        ``_POLL_FAILURE_BUDGET`` consecutive failures — a single LB blip
        must not abort a multi-hour wait."""
        deadline = time.monotonic() + timeout_s
        failures = 0
        while True:
            try:
                job = self.get_job(name)
            except (ConnectionError, RuntimeError):
                failures += 1
                if failures >= _POLL_FAILURE_BUDGET:
                    raise
                time.sleep(poll_s)
                continue
            failures = 0
            if job.get("phase") in _TERMINAL_JOB_PHASES:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {name!r} not terminal after {timeout_s}s "
                    f"(last phase: {job.get('phase')!r})"
                )
            time.sleep(poll_s)

    # -- manifest (declarative: one document -> cluster and/or job) --------

    def submit_manifest(self, manifest: dict) -> dict:
        return json.loads(self._core.rl_submit_manifest(json.dumps(manifest)))
