"""
Distributed-training SDK surface (SDK arch § 4 / § 6 / § 8 / § 9).

User-facing types and the `DistributedTraining` facade returned by
`client.deploy_distributed(...)` and `@basilica.distributed`. Researchers
import these from `basilica` directly (re-exported in __init__.py).

Tenancy invariant (SDK arch § 7): bench data is per-UD. There is NO
`client.preflight(...)` and NO `client.nccl_baseline(...)` standalone
helper -- those would imply a shared cluster-wide cache, which violates
the platform's tenancy contract. Bench results live on
`DistributedTraining.bench` and are owned by the user's namespace.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from .exceptions import (
    BelowMinimumWorld,
    UDTerminalState,
    WorldSizeOutOfBounds,
)

# Phase 5b (#445): operator phases that pin the UD as terminal. See the
# operator's `DeploymentPhase::is_terminal` for the source of truth. The
# Python side reads `status.phase` as a snake_case string straight from
# the JSON the API returns, so the comparison is string-based.
TERMINAL_PHASES: frozenset = frozenset({"succeeded", "failed", "cancelled"})

if TYPE_CHECKING:
    from . import BasilicaClient


# =============================================================================
# Type model (SDK arch § 8).
# =============================================================================


@dataclass(frozen=True)
class WorldSize:
    """
    World-size triple. `1 <= min <= target <= max`.

    `min` is the floor below which the run pauses (torchelastic's
    MIN_NODES). `target` is the StatefulSet replica count. `max` is the
    hard cap; `scale()` rejects requests above it.
    """

    min: int
    target: int
    max: int

    def __post_init__(self) -> None:
        if self.min < 1:
            raise WorldSizeOutOfBounds(
                f"WorldSize.min must be >= 1, got {self.min}",
                requested=self.min,
                min=1,
                max=self.max if self.max >= 1 else 1,
            )
        if self.target < self.min:
            raise WorldSizeOutOfBounds(
                f"WorldSize.target ({self.target}) must be >= "
                f"WorldSize.min ({self.min})",
                requested=self.target,
                min=self.min,
                max=self.max,
            )
        if self.max < self.target:
            raise WorldSizeOutOfBounds(
                f"WorldSize.max ({self.max}) must be >= "
                f"WorldSize.target ({self.target})",
                requested=self.target,
                min=self.min,
                max=self.max,
            )


@dataclass(frozen=True)
class ProviderFilter:
    """
    Provider include/exclude lists for worker scheduling.

    Empty `include` = any provider. Match is on the `basilica.ai/provider`
    node label (`hyperstack`, `verda`, `masscompute`, `shadeform`).
    `exclude` is subtractive after `include`.
    """

    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorldStatus:
    """
    Snapshot of `status.distributed.worldSize` written by the operator.

    `below_minimum=True` means `ready < min` and the run is paused (or
    the workload's own logic should pause). The operator emits a K8s
    Event `WorldShrunkBelowMin` on transition.
    """

    ready: int
    target: int
    min: int
    max: int
    below_minimum: bool


@dataclass(frozen=True)
class RankStatus:
    """
    Per-rank pod observation. Read-only.

    `provider` and `region` come from node labels; `None` when the pod
    has not been scheduled or the labels are absent.
    """

    rank: int
    pod_name: str
    node: Optional[str]
    provider: Optional[str]
    region: Optional[str]
    phase: str
    restarts: int


@dataclass(frozen=True)
class RankExit:
    """
    Phase 5b (#445) per-rank exit diagnostics.

    Populated by the operator on `status.distributed.rankExits` when the
    UD reaches a terminal state (or on the deferred reconcile that fires
    when workers complete before bench is terminal). Empty while the UD
    is non-terminal -- the SDK distinguishes "in flight" from "finished"
    by absence-vs-presence of `rank_exits`.

    Mirrors the K8s container `terminated` block so consumers can read
    per-rank exit codes after the operator has scaled the worker
    StatefulSet to `replicas: 0` and the pods are no longer queryable.
    """

    rank: int
    exit_code: int
    termination_reason: Optional[str]
    restart_count: int


@dataclass(frozen=True)
class BenchResult:
    """
    Per-UD bench probe result (SDK arch § 7 / § 8).

    Measured on this UD's actual nodes; always fresh by construction
    (no freshness tier). `None`-valued fields mean the probe could not
    measure them (typically a partial sweep).

    Bandwidth fields are GB/s (1 GB = 10^9 bytes), matching the NCCL
    paper-canonical units. Latency is microseconds at the smallest
    swept message size.
    """

    measured_at: datetime
    busbw_gbps_p10: Optional[float]
    busbw_gbps_p50: Optional[float]
    busbw_gbps_p90: Optional[float]
    algbw_gbps_p50: Optional[float]
    latency_us_at_1mib: Optional[float]
    size_bytes_swept: List[int]
    probe_node_a: str
    probe_node_b: str

    @classmethod
    def from_status_dict(cls, raw: Dict[str, Any]) -> "BenchResult":
        """Construct from the operator's `status.distributed.bench.result` JSON."""
        ts = raw.get("measuredAt") or raw.get("measured_at")
        return cls(
            measured_at=_parse_rfc3339(ts) if ts else datetime.fromtimestamp(0),
            busbw_gbps_p10=raw.get("busbwGbpsP10"),
            busbw_gbps_p50=raw.get("busbwGbpsP50"),
            busbw_gbps_p90=raw.get("busbwGbpsP90"),
            algbw_gbps_p50=raw.get("algbwGbpsP50"),
            latency_us_at_1mib=raw.get("latencyUsAt1mib"),
            size_bytes_swept=list(raw.get("sizeBytesSwept", [])),
            probe_node_a=raw.get("probeNodeA", ""),
            probe_node_b=raw.get("probeNodeB", ""),
        )


@dataclass(frozen=True)
class DistributedMetrics:
    """
    Snapshot of platform-side metrics for this UD (SDK arch § 6).

    Sourced from Prometheus via the API gateway. Read-only. Time-series
    data not exposed here; use Grafana for trend views.
    """

    world_ready: int
    world_target: int
    rendezvous_restarts_total: int
    rank_restarts_total: int
    time_to_first_rendezvous_seconds: Optional[float]
    last_resize_at: Optional[datetime]


def _parse_rfc3339(s: str) -> datetime:
    """Parse RFC 3339 timestamp; tolerates trailing Z."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# =============================================================================
# DistributedTraining facade (SDK arch § 6).
# =============================================================================


class DistributedTraining:
    """
    Handle for a distributed-training UserDeployment.

    Returned by `client.deploy_distributed(...)` and `@basilica.distributed`.
    Methods read the live operator status (refreshed on each call) and
    issue control-plane requests (`scale`, `delete`, `wait_*`). Every
    method has an `_async` counterpart per SDK arch § 9.

    Attributes:
        name: The deployment instance name.
        namespace: The user's K8s namespace (`u-<sanitized-user-id>`).
        rendezvous_endpoint: `<ud>-rendezvous.<ns>.svc.cluster.local:2379`
            for `etcd-v2` backend; in-cluster only.
    """

    name: str
    namespace: str
    rendezvous_endpoint: str

    def __init__(self, client: "BasilicaClient", name: str):
        self._client = client
        self.name = name
        self._cached_status: Optional[Dict[str, Any]] = None
        # Issue #495: target requested by the most recent `scale()` call,
        # cleared once the operator's `status.worldSize.target` reflects it.
        # `wait_until_target_world` consults this to avoid short-circuiting
        # against a stale `status.worldSize.target` (the API patches `spec`
        # only; the operator mirrors `spec → status` on its next reconcile).
        self._pending_scale_target: Optional[int] = None

    # -------------------------------------------------------------------------
    # Status / observation
    # -------------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload status from the API. Subsequent property reads use the new snapshot."""
        deployment = self._client.get(self.name)
        self._cached_status = _coerce_to_dict(deployment)
        self.namespace = self._cached_status.get("namespace", "")
        self.rendezvous_endpoint = self._derive_rendezvous_endpoint()

    async def refresh_async(self) -> None:
        """Async variant of `refresh`."""
        await asyncio.get_event_loop().run_in_executor(None, self.refresh)

    @property
    def world(self) -> WorldStatus:
        """Most recent world-size observation. Calls `refresh()` lazily."""
        if self._cached_status is None:
            self.refresh()
        ws = (self._cached_status or {}).get("distributed", {}).get("worldSize") or {}
        return WorldStatus(
            ready=int(ws.get("ready", 0)),
            target=int(ws.get("target", 0)),
            min=int(ws.get("min", 0)),
            max=int(ws.get("max", 0)),
            below_minimum=bool(ws.get("belowMinimum", False)),
        )

    @property
    def ranks(self) -> List[RankStatus]:
        """Per-rank pod observations from `status.distributed.ranks`."""
        if self._cached_status is None:
            self.refresh()
        ranks_raw = (self._cached_status or {}).get("distributed", {}).get("ranks") or []
        return [
            RankStatus(
                rank=int(r.get("rank", 0)),
                pod_name=r.get("podName", ""),
                node=r.get("nodeName"),
                provider=r.get("provider"),
                region=r.get("region"),
                phase=r.get("phase", "Unknown"),
                restarts=int(r.get("restarts", 0)),
            )
            for r in ranks_raw
        ]

    @property
    def phase(self) -> str:
        """
        Phase 5b (#445): operator-driven lifecycle phase.

        One of `pending` | `scheduling` | `pulling` | `initializing` |
        `storage_sync` | `starting` | `health_check` | `ready` |
        `degraded` | `failed` | `suspended` | `terminating` |
        `succeeded` | `cancelled`. The terminal triple
        (`succeeded`, `failed`, `cancelled`) is sticky — once set the
        operator does not regress to a non-terminal phase.

        Empty string when the operator has not yet written a status
        block (just-created UD before the first reconcile).
        """
        if self._cached_status is None:
            self.refresh()
        # `phase` is a top-level field on DeploymentResponse (snake_case
        # on the PyO3 binding -> camelCase'd by `_coerce_to_dict`).
        phase = (self._cached_status or {}).get("phase")
        if phase is None:
            return ""
        return str(phase)

    @property
    def is_terminal(self) -> bool:
        """
        Phase 5b (#445): convenience for callers that just want to know
        "is this UD done?" — `phase in {succeeded, failed, cancelled}`.
        """
        return self.phase in TERMINAL_PHASES

    @property
    def rank_exits(self) -> List[RankExit]:
        """
        Phase 5b (#445): per-rank exit diagnostics, populated when the
        UD reaches a terminal state (or on the deferred reconcile when
        workers complete before bench finishes). Empty while the UD is
        running; absence vs. presence is the SDK's signal for
        "training in flight" vs. "training finished".
        """
        if self._cached_status is None:
            self.refresh()
        exits_raw = (
            (self._cached_status or {}).get("distributed", {}).get("rankExits") or []
        )
        return [
            RankExit(
                rank=int(e.get("rank", 0)),
                exit_code=int(e.get("exitCode", 0)),
                termination_reason=e.get("terminationReason"),
                restart_count=int(e.get("restartCount", 0)),
            )
            for e in exits_raw
        ]

    @property
    def bench(self) -> Optional[BenchResult]:
        """
        Per-UD bench probe result, or `None` if `bench.mode != on-start`
        or the probe has not yet completed its first attempt.

        SDK arch § 7: never a cross-tenant aggregate; always measured
        on this UD's own nodes, billed against this user's GPU minutes.
        """
        if self._cached_status is None:
            self.refresh()
        bench_status = (
            (self._cached_status or {}).get("distributed", {}).get("bench")
        )
        if not bench_status:
            return None
        result = bench_status.get("result")
        if not result:
            return None
        return BenchResult.from_status_dict(result)

    def metrics(self) -> DistributedMetrics:
        """Platform-side metric snapshot for this UD (SDK arch § 6)."""
        if self._cached_status is None:
            self.refresh()
        # Phase 5b ships the metric surface read-only on the status; full
        # Prometheus scraping is on the platform-internal collector
        # (architecture doc § 12). Phase 6 widens this if needed.
        ws = self.world
        return DistributedMetrics(
            world_ready=ws.ready,
            world_target=ws.target,
            rendezvous_restarts_total=0,
            rank_restarts_total=sum(r.restarts for r in self.ranks),
            time_to_first_rendezvous_seconds=None,
            last_resize_at=None,
        )

    async def metrics_async(self) -> DistributedMetrics:
        """Async variant of `metrics`."""
        return await asyncio.get_event_loop().run_in_executor(None, self.metrics)

    def events(self, since: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        K8s Events for this UD's namespace, scoped by `involvedObject` to
        this UD. SDK arch § 6. Returns a list of `{event_type, reason,
        message, count, last_timestamp}` dicts. The `since` filter is
        accepted for forward-compat (Phase 6 server-side filter); current
        behavior is "all available events, most recent first".
        """
        response = self._client.get_deployment_events(self.name, since=since)
        # The PyO3 binding returns a dict like
        # `{events: [...], total: N}` (pythonize of DeploymentEventsResponse).
        if isinstance(response, dict):
            return list(response.get("events", []))
        return []

    async def events_async(self, since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Async variant of `events`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.events(since=since)
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def scale(self, target: int) -> WorldStatus:
        """
        Patch `worldSize.target` on the CR. Returns immediately with the
        intended new state; ranks join/drain asynchronously. Use
        `wait_until_target_world(timeout=...)` to block.

        Validates locally before the network call:
        - `target >= 1` (immediate rejection)
        - `target` falls within the deployment's current
          `[worldSize.min, worldSize.max]` (refresh first to guard
          against stale cached bounds)

        Raises:
            WorldSizeOutOfBounds: target < 1, or target outside `[min, max]`.
        """
        if target < 1:
            raise WorldSizeOutOfBounds(
                f"scale: target must be >= 1, got {target}",
                requested=target,
                min=1,
                max=1,
            )
        # Refresh so the bounds check uses the live `worldSize.{min,max}`,
        # not whatever was cached from earlier (the operator can mutate
        # min/max only at create-time, but a stale cache from before
        # creation would show min=max=0).
        self.refresh()
        # Phase 5b (#445): refuse to scale a UD that has already reached
        # a terminal state. The operator's defense in depth (renderer
        # uses `effective_worker_replicas` and the reconciler emits a
        # `UDTerminalState` Warning Event on attempted mutations) catches
        # `kubectl edit` mutations that bypass the SDK; this guard is the
        # primary user-facing rejection.
        current_phase = self.phase
        if current_phase in TERMINAL_PHASES:
            raise UDTerminalState(
                f"scale: UD '{self.name}' is in terminal phase "
                f"'{current_phase}'; recreate the UD to scale",
                phase=current_phase,
                requested_target=target,
            )
        ws = self.world
        # ws.min == 0 means we have not yet observed the operator's first
        # status write; in that case skip the bounds check and let the
        # API gateway return its 400 with the canonical message.
        if ws.min > 0 and (target < ws.min or target > ws.max):
            raise WorldSizeOutOfBounds(
                f"scale: target {target} out of bounds "
                f"[{ws.min}, {ws.max}]",
                requested=target,
                min=ws.min,
                max=ws.max,
            )
        # Issue the patch via the new POST /deployments/{name}/scale-distributed
        # endpoint shipped in the Phase 5b precursor (basilica-backend #421).
        self._client._client.scale_distributed_deployment(self.name, target)
        # Issue #495: record the requested target before the refresh races
        # the operator. `wait_until_target_world` keys off this to ignore a
        # stale `status.worldSize.target` until the operator reconciles
        # `spec → status` and reflects the new target.
        self._pending_scale_target = target
        self.refresh()
        # Issue #495: return a snapshot reflecting the requested target so
        # callers that print or inspect the return value see the truth.
        # `status.worldSize.target` may still carry the pre-patch value
        # because the API patches `spec` only; the operator mirrors to
        # status on its next reconcile.
        observed = self.world
        return WorldStatus(
            ready=observed.ready,
            target=target,
            min=observed.min,
            max=observed.max,
            below_minimum=observed.below_minimum,
        )

    async def scale_async(self, target: int) -> WorldStatus:
        """Async variant of `scale`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.scale(target)
        )

    def wait_until_min_world(self, timeout: int = 300) -> None:
        """
        Block until `ready >= min`, or raise `BelowMinimumWorld` on
        timeout. Polls every 5s. SDK arch § 11.
        """
        deadline = time.monotonic() + max(timeout, 0)
        while time.monotonic() < deadline:
            self.refresh()
            ws = self.world
            if ws.ready >= ws.min and ws.min > 0:
                return
            if timeout == 0:
                # Honor a 0-timeout as "raise immediately if not ready".
                break
            time.sleep(min(5, max(timeout // 10, 1)))
        # One last refresh so the exception carries the up-to-date count.
        self.refresh()
        ws = self.world
        if ws.ready < ws.min or ws.min == 0:
            raise BelowMinimumWorld(
                f"wait_until_min_world timed out after {timeout}s "
                f"(ready={ws.ready}, required_min={ws.min})",
                ready=ws.ready,
                required_min=ws.min,
                timeout=timeout,
            )

    async def wait_until_min_world_async(self, timeout: int = 300) -> None:
        """Async variant of `wait_until_min_world`."""
        deadline = asyncio.get_event_loop().time() + max(timeout, 0)
        while asyncio.get_event_loop().time() < deadline:
            await self.refresh_async()
            ws = self.world
            if ws.ready >= ws.min and ws.min > 0:
                return
            if timeout == 0:
                break
            await asyncio.sleep(min(5, max(timeout // 10, 1)))
        await self.refresh_async()
        ws = self.world
        if ws.ready < ws.min or ws.min == 0:
            raise BelowMinimumWorld(
                f"wait_until_min_world_async timed out after {timeout}s "
                f"(ready={ws.ready}, required_min={ws.min})",
                ready=ws.ready,
                required_min=ws.min,
                timeout=timeout,
            )

    def wait_until_target_world(self, timeout: int = 600) -> None:
        """
        Block until `ready >= target`, or raise `BelowMinimumWorld` on timeout.

        Phase 5b (#445): returns cleanly when the UD has already reached
        `succeeded` -- by definition the world was assembled (the
        operator only transitions to `succeeded` after every rank exited
        with `exit_code == 0`), so the contract `ready >= target` was
        met at some point. `failed` / `cancelled` raise `BelowMinimumWorld`
        because the contract was NOT met.

        Issue #495: when called immediately after `scale(target=N)`, the
        operator's `status.worldSize.target` may still carry the
        pre-patch value (the API patches `spec` only; status mirroring
        happens on the next reconcile). The success condition therefore
        also requires `status.worldSize.target >= pending_scale_target`
        when a scale is pending — otherwise the loop short-circuits
        against the stale target.
        """
        deadline = time.monotonic() + max(timeout, 0)
        while time.monotonic() < deadline:
            self.refresh()
            ws = self.world
            if self.phase == "succeeded":
                self._pending_scale_target = None
                return
            if self.phase in ("failed", "cancelled"):
                raise BelowMinimumWorld(
                    f"wait_until_target_world: UD '{self.name}' reached "
                    f"terminal phase '{self.phase}' before world reached target",
                    ready=ws.ready,
                    required_min=self._effective_required_target(ws),
                    timeout=timeout,
                )
            if self._target_world_reached(ws):
                self._pending_scale_target = None
                return
            if timeout == 0:
                break
            time.sleep(min(5, max(timeout // 10, 1)))
        self.refresh()
        ws = self.world
        if self.phase == "succeeded":
            self._pending_scale_target = None
            return
        if not self._target_world_reached(ws):
            required = self._effective_required_target(ws)
            raise BelowMinimumWorld(
                f"wait_until_target_world timed out after {timeout}s "
                f"(ready={ws.ready}, required_min={required})",
                ready=ws.ready,
                required_min=required,
                timeout=timeout,
            )
        self._pending_scale_target = None

    def _target_world_reached(self, ws: WorldStatus) -> bool:
        """
        Issue #495: success predicate for `wait_until_target_world`. A
        pending scale request gates the predicate so the caller does not
        observe `ready >= stale_target` while the operator has not yet
        mirrored the patched `spec.worldSize.target` into status.
        """
        pending = self._pending_scale_target
        if pending is not None and pending > 0:
            if ws.target < pending:
                return False
            return ws.ready >= pending
        return ws.ready >= ws.target and ws.target > 0

    def _effective_required_target(self, ws: WorldStatus) -> int:
        """Required target reported by exception messages (max of pending vs. observed)."""
        pending = self._pending_scale_target or 0
        return max(pending, ws.target)

    async def wait_until_target_world_async(self, timeout: int = 600) -> None:
        """Async variant of `wait_until_target_world`. Same Phase 5b semantics
        and the same #495 pending-scale gate."""
        deadline = asyncio.get_event_loop().time() + max(timeout, 0)
        while asyncio.get_event_loop().time() < deadline:
            await self.refresh_async()
            ws = self.world
            if self.phase == "succeeded":
                self._pending_scale_target = None
                return
            if self.phase in ("failed", "cancelled"):
                raise BelowMinimumWorld(
                    f"wait_until_target_world_async: UD '{self.name}' reached "
                    f"terminal phase '{self.phase}' before world reached target",
                    ready=ws.ready,
                    required_min=self._effective_required_target(ws),
                    timeout=timeout,
                )
            if self._target_world_reached(ws):
                self._pending_scale_target = None
                return
            if timeout == 0:
                break
            await asyncio.sleep(min(5, max(timeout // 10, 1)))
        await self.refresh_async()
        ws = self.world
        if self.phase == "succeeded":
            self._pending_scale_target = None
            return
        if not self._target_world_reached(ws):
            required = self._effective_required_target(ws)
            raise BelowMinimumWorld(
                f"wait_until_target_world_async timed out after {timeout}s "
                f"(ready={ws.ready}, required_min={required})",
                ready=ws.ready,
                required_min=required,
                timeout=timeout,
            )
        self._pending_scale_target = None

    def wait_until_complete(self, timeout: int = 1800) -> WorldStatus:
        """
        Phase 5b (#445): block until the UD reaches any terminal state.

        Returns the final `WorldStatus` on `succeeded`. Raises
        `BelowMinimumWorld` on `failed` (with the per-rank exit
        diagnostics from `rank_exits` summarised in the message). Raises
        `UDTerminalState` if the UD is ALREADY terminal at the time of
        the call -- so the caller distinguishes "I waited and it
        completed" from "I called this on an already-completed UD".

        Polls every 10s. The default 30-minute timeout assumes a typical
        bench Job (~5 min) plus reasonable training time; long-running
        runs should pass an explicit timeout.

        Raises:
            UDTerminalState: UD was already terminal at entry.
            BelowMinimumWorld: UD reached `failed` / `cancelled` mid-wait
                or the timeout elapsed without a terminal transition.
        """
        # Refuse to block on something that already reached a terminal
        # state. Caller can read `t.world` / `t.bench` / `t.rank_exits`
        # directly instead.
        self.refresh()
        if self.is_terminal:
            raise UDTerminalState(
                f"wait_until_complete: UD '{self.name}' is already in "
                f"terminal phase '{self.phase}'",
                phase=self.phase,
                requested_target=None,
            )

        deadline = time.monotonic() + max(timeout, 0)
        while time.monotonic() < deadline:
            self.refresh()
            current_phase = self.phase
            if current_phase == "succeeded":
                return self.world
            if current_phase in ("failed", "cancelled"):
                ws = self.world
                exits = self.rank_exits
                # Summarise non-zero exits for the error message so the
                # caller has actionable context without an extra fetch.
                bad = [
                    f"rank {e.rank}: exit_code={e.exit_code}"
                    + (f" reason={e.termination_reason}" if e.termination_reason else "")
                    for e in exits
                    if e.exit_code != 0
                ]
                detail = "; ".join(bad) if bad else "no per-rank exits recorded"
                raise BelowMinimumWorld(
                    f"wait_until_complete: UD '{self.name}' reached "
                    f"terminal phase '{current_phase}' ({detail})",
                    ready=ws.ready,
                    required_min=ws.min,
                    timeout=timeout,
                )
            if timeout == 0:
                break
            time.sleep(min(10, max(timeout // 30, 1)))
        # Timeout: surface as `BelowMinimumWorld` with the live shape.
        self.refresh()
        ws = self.world
        if self.phase == "succeeded":
            return ws
        raise BelowMinimumWorld(
            f"wait_until_complete: timed out after {timeout}s "
            f"(phase={self.phase}, ready={ws.ready}, target={ws.target})",
            ready=ws.ready,
            required_min=ws.min,
            timeout=timeout,
        )

    async def wait_until_complete_async(self, timeout: int = 1800) -> WorldStatus:
        """Async variant of `wait_until_complete`. Same Phase 5b semantics."""
        await self.refresh_async()
        if self.is_terminal:
            raise UDTerminalState(
                f"wait_until_complete_async: UD '{self.name}' is already in "
                f"terminal phase '{self.phase}'",
                phase=self.phase,
                requested_target=None,
            )

        deadline = asyncio.get_event_loop().time() + max(timeout, 0)
        while asyncio.get_event_loop().time() < deadline:
            await self.refresh_async()
            current_phase = self.phase
            if current_phase == "succeeded":
                return self.world
            if current_phase in ("failed", "cancelled"):
                ws = self.world
                exits = self.rank_exits
                bad = [
                    f"rank {e.rank}: exit_code={e.exit_code}"
                    + (f" reason={e.termination_reason}" if e.termination_reason else "")
                    for e in exits
                    if e.exit_code != 0
                ]
                detail = "; ".join(bad) if bad else "no per-rank exits recorded"
                raise BelowMinimumWorld(
                    f"wait_until_complete_async: UD '{self.name}' reached "
                    f"terminal phase '{current_phase}' ({detail})",
                    ready=ws.ready,
                    required_min=ws.min,
                    timeout=timeout,
                )
            if timeout == 0:
                break
            await asyncio.sleep(min(10, max(timeout // 30, 1)))
        await self.refresh_async()
        ws = self.world
        if self.phase == "succeeded":
            return ws
        raise BelowMinimumWorld(
            f"wait_until_complete_async: timed out after {timeout}s "
            f"(phase={self.phase}, ready={ws.ready}, target={ws.target})",
            ready=ws.ready,
            required_min=ws.min,
            timeout=timeout,
        )

    def delete(self) -> None:
        """
        Delete the UD. Owner-references cascade to StatefulSet,
        rendezvous Deployment, bench Job, ConfigMap, and NetworkPolicies.

        Returns immediately; the operator handles deletion asynchronously.
        Researchers needing clean shutdown should `dist.barrier()` +
        checkpoint inside their training loop before calling this.
        """
        self._client.delete_deployment(self.name)
        self._cached_status = None

    async def delete_async(self) -> None:
        """Async variant of `delete`."""
        await asyncio.get_event_loop().run_in_executor(None, self.delete)

    # -------------------------------------------------------------------------
    # Logs and debug
    # -------------------------------------------------------------------------

    def logs(
        self,
        tail: Optional[int] = None,
        follow: bool = False,
        rank: Optional[int] = None,
    ) -> str:
        """
        Get logs for the deployment. SDK arch § 6.

        Phase 5b: the underlying `/deployments/{name}/logs` endpoint
        does not support per-rank filtering server-side, so the `rank`
        argument is accepted for forward-compat but ignored. All ranks'
        logs are returned merged. Per-rank filtering is a Phase 6
        backend enhancement (basilica-backend follow-up).
        """
        if rank is not None and rank > 0:
            import warnings as _warnings
            _warnings.warn(
                "DistributedTraining.logs(rank=...) is accepted for "
                "forward-compat but ignored in Phase 5b: the API does "
                "not yet support per-rank log filtering. All ranks' "
                "logs returned merged.",
                stacklevel=2,
            )
        return self._client.get_deployment_logs(
            self.name, follow=follow, tail=tail
        )

    async def logs_async(
        self,
        tail: Optional[int] = None,
        follow: bool = False,
        rank: Optional[int] = None,
    ) -> str:
        """Async variant of `logs`. Same Phase 5b rank-filter limitation."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.logs(tail=tail, follow=follow, rank=rank)
        )

    def stream_logs(
        self,
        ranks: Optional[List[int]] = None,
    ) -> Iterator[Tuple[int, str]]:
        """
        Yield `(rank, line)` tuples for each log line.

        Phase 5b: the API returns merged logs without rank tags, so
        every line is yielded as `(0, line)`. The `ranks` filter is
        accepted for forward-compat but ignored. SDK arch § 6 / § 13
        per-rank multiplexing is a Phase 6 backend enhancement.
        """
        if ranks is not None:
            import warnings as _warnings
            _warnings.warn(
                "DistributedTraining.stream_logs(ranks=...) is accepted "
                "but ignored in Phase 5b: per-rank streaming is a "
                "Phase 6 backend enhancement.",
                stacklevel=2,
            )
        for line in self.logs(follow=False).splitlines():
            yield (0, line)

    async def stream_logs_async(
        self,
        ranks: Optional[List[int]] = None,
    ) -> AsyncIterator[Tuple[int, str]]:
        """Async variant of `stream_logs`. Same Phase 5b limitations."""
        if ranks is not None:
            import warnings as _warnings
            _warnings.warn(
                "DistributedTraining.stream_logs_async(ranks=...) is "
                "accepted but ignored in Phase 5b.",
                stacklevel=2,
            )
        text = await self.logs_async(follow=False)
        for line in text.splitlines():
            yield (0, line)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _derive_rendezvous_endpoint(self) -> str:
        """`<ud>-rendezvous.<ns>.svc.cluster.local:2379` -- in-cluster only."""
        ns = self.namespace or "u-unknown"
        return f"{self.name}-rendezvous.{ns}.svc.cluster.local:2379"


def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
    """
    PyO3 DeploymentResponse -> Python dict for status field access.
    Tolerates dict input (already-converted) and PyO3 wrappers.

    Issue #449: the dict produced here MUST include the `distributed`
    block end-to-end. Earlier SDK builds dropped it because the PyO3
    binding had no `distributed` getter; the read properties on
    `DistributedTraining` walked an absent key and returned zeros.

    The PyO3 `DeploymentResponse` exposes attributes in snake_case
    (Rust convention); this helper maps them to the camelCase keys
    the operator emits on the wire. `distributed` is already a
    camelCase Python dict on the binding side (built via
    `pythonize` from the strongly-typed `DistributedStatus` JSON), so
    it is copied through verbatim.
    """
    if isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    # Scalar / Optional[str] / Optional[bool] fields. Each maps to its
    # camelCase wire key.
    for attr in (
        "instance_name",
        "user_id",
        "namespace",
        "image",
        "state",
        "url",
        "created_at",
        "updated_at",
        "phase",
        "message",
        "share_token",
        "share_url",
        "public_metadata",
    ):
        if hasattr(obj, attr):
            out[_to_camel(attr)] = getattr(obj, attr)
    # Pass-through dict (from PyO3 `pythonize` on the Rust side). The
    # operator emits camelCase, so this dict is already in the shape the
    # `world` / `ranks` / `bench` / `metrics` properties expect. Type-check
    # against `dict` so MagicMock auto-attrs (which are truthy but not
    # dicts) do not silently leak into the output and shadow the
    # `_distributed_status` test-mock fallback below.
    if hasattr(obj, "distributed"):
        d = getattr(obj, "distributed", None)
        if isinstance(d, dict):
            out["distributed"] = d
    # Test-mock back-compat: callers that build a MagicMock with a
    # `_distributed_status` attribute (the pre-#449 mock pattern) keep
    # working without rewrites.
    if "distributed" not in out and hasattr(obj, "_distributed_status"):
        out["distributed"] = obj._distributed_status  # type: ignore[attr-defined]
    return out


def _to_camel(snake: str) -> str:
    """`instance_name` -> `instanceName`. Used by the dict-coercion helper."""
    parts = snake.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])
