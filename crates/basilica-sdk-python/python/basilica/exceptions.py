"""
Basilica SDK Exception Hierarchy

This module provides a comprehensive exception hierarchy for the Basilica SDK,
offering clear and actionable error messages for common failure scenarios.

Exception Hierarchy:
    BasilicaError (base)
    ├── AuthenticationError     - Token/API key issues
    ├── AuthorizationError      - Permission denied
    ├── ValidationError         - Invalid input parameters
    ├── DeploymentError         - Deployment lifecycle errors
    │   ├── DeploymentNotFound  - Deployment doesn't exist
    │   ├── DeploymentTimeout   - Deployment didn't become ready
    │   └── DeploymentFailed    - Deployment entered failed state
    ├── ResourceError           - GPU/node availability issues
    ├── StorageError            - Storage configuration errors
    ├── NetworkError            - Connection/API communication issues
    └── InferenceError          - Managed Inference gateway errors
        ├── InferenceAuthenticationError - 401 bad/expired API key
        ├── InsufficientCreditsError     - 402 balance below floor
        ├── InferenceQuotaExceededError  - 429 quota cap tripped (cap, retry_after)
        ├── InferenceModelNotFoundError  - 404 unknown/unowned model
        └── InferenceUnavailableError    - 503 pool saturated/unavailable

Example:
    >>> from basilica.exceptions import DeploymentTimeout
    >>> try:
    ...     deployment = client.deploy("my-app", source="app.py")
    ... except DeploymentTimeout as e:
    ...     print(f"Deployment timed out after {e.timeout_seconds}s")
    ...     print(f"Last state: {e.last_state}")
"""

from typing import Optional


class BasilicaError(Exception):
    """
    Base exception for all Basilica SDK errors.

    All Basilica-specific exceptions inherit from this class, making it easy
    to catch all SDK errors with a single except clause.

    Attributes:
        message: Human-readable error description
        code: Optional error code from the API
        retryable: Whether the operation might succeed if retried

    Example:
        >>> try:
        ...     client.deploy(...)
        ... except BasilicaError as e:
        ...     print(f"Basilica error: {e}")
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        retryable: bool = False
    ):
        self.message = message
        self.code = code
        self.retryable = retryable
        super().__init__(message)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class AuthenticationError(BasilicaError):
    """
    Raised when API authentication fails.

    This typically occurs when:
    - No API token is provided
    - The API token is invalid or expired
    - The BASILICA_API_TOKEN environment variable is not set

    Example:
        >>> # No token set
        >>> client = BasilicaClient()
        AuthenticationError: No API token provided. Set BASILICA_API_TOKEN or pass api_key parameter.

    Resolution:
        Create a token using: basilica tokens create
        Then either:
        - Set environment variable: export BASILICA_API_TOKEN="basilica_..."
        - Pass directly: BasilicaClient(api_key="basilica_...")
    """

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            code="AUTH_FAILED",
            retryable=False
        )


class AuthorizationError(BasilicaError):
    """
    Raised when the authenticated user lacks permission for an operation.

    This occurs when:
    - Attempting to access another user's deployment
    - API token lacks required scopes
    - Account has been suspended

    Example:
        >>> client.get_deployment("someone-elses-deployment")
        AuthorizationError: Access denied to deployment 'someone-elses-deployment'
    """

    def __init__(self, message: str = "Permission denied", resource: Optional[str] = None):
        self.resource = resource
        if resource and "denied" not in message.lower():
            message = f"Access denied to {resource}: {message}"
        super().__init__(
            message=message,
            code="FORBIDDEN",
            retryable=False
        )


class ValidationError(BasilicaError):
    """
    Raised when input parameters fail validation.

    This occurs when:
    - Instance name contains invalid characters
    - Port number is out of range
    - Resource values are invalid (e.g., negative CPU)
    - Required parameters are missing

    Attributes:
        field: The field that failed validation (if known)
        value: The invalid value that was provided

    Example:
        >>> client.deploy(name="My App!")  # Invalid characters
        ValidationError: Instance name 'My App!' is invalid. Use lowercase letters, numbers, and hyphens only.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[str] = None
    ):
        self.field = field
        self.value = value
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            retryable=False
        )


class DeploymentError(BasilicaError):
    """
    Base exception for deployment-related errors.

    This is the parent class for all deployment lifecycle errors.
    Catch this to handle any deployment issue.
    """

    def __init__(
        self,
        message: str,
        instance_name: Optional[str] = None,
        code: str = "DEPLOYMENT_ERROR",
        retryable: bool = False
    ):
        self.instance_name = instance_name
        super().__init__(message=message, code=code, retryable=retryable)


class DeploymentNotFound(DeploymentError):
    """
    Raised when a deployment cannot be found.

    This occurs when:
    - The deployment was deleted
    - The instance name is incorrect
    - The deployment belongs to another user

    Example:
        >>> client.get_deployment("nonexistent")
        DeploymentNotFound: Deployment 'nonexistent' not found
    """

    def __init__(self, instance_name: str):
        super().__init__(
            message=f"Deployment '{instance_name}' not found",
            instance_name=instance_name,
            code="NOT_FOUND",
            retryable=False
        )


class DeploymentTimeout(DeploymentError):
    """
    Raised when a deployment fails to become ready within the timeout.

    Attributes:
        timeout_seconds: The timeout that was exceeded
        last_state: The last observed deployment state
        replicas_ready: Number of replicas that were ready
        replicas_desired: Total number of replicas desired

    Example:
        >>> client.deploy("my-app", source="app.py", timeout=60)
        DeploymentTimeout: Deployment 'my-app' not ready after 60s (state: Pending, replicas: 0/1)
    """

    def __init__(
        self,
        instance_name: str,
        timeout_seconds: int,
        last_state: str = "Unknown",
        replicas_ready: int = 0,
        replicas_desired: int = 1
    ):
        self.timeout_seconds = timeout_seconds
        self.last_state = last_state
        self.replicas_ready = replicas_ready
        self.replicas_desired = replicas_desired

        super().__init__(
            message=(
                f"Deployment '{instance_name}' not ready after {timeout_seconds}s "
                f"(state: {last_state}, replicas: {replicas_ready}/{replicas_desired})"
            ),
            instance_name=instance_name,
            code="TIMEOUT",
            retryable=True
        )


class DeploymentFailed(DeploymentError):
    """
    Raised when a deployment enters a failed state.

    This occurs when:
    - Container image cannot be pulled
    - Container crashes on startup
    - Resource limits are exceeded
    - Health checks fail

    Attributes:
        reason: The reason for failure (if available)

    Example:
        >>> client.deploy("my-app", image="nonexistent:image")
        DeploymentFailed: Deployment 'my-app' failed: ImagePullBackOff
    """

    def __init__(self, instance_name: str, reason: Optional[str] = None):
        self.reason = reason
        message = f"Deployment '{instance_name}' failed"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(
            message=message,
            instance_name=instance_name,
            code="FAILED",
            retryable=False
        )


class ResourceError(BasilicaError):
    """
    Raised when requested resources are unavailable.

    This occurs when:
    - No GPU nodes match the requirements
    - Cluster capacity is exhausted
    - Requested GPU model is not available

    Attributes:
        resource_type: The type of resource that's unavailable (e.g., "GPU", "node")

    Example:
        >>> client.deploy("my-app", gpu_count=8, gpu_models=["H100"])
        ResourceError: No nodes available with 8x H100 GPUs
    """

    def __init__(self, message: str, resource_type: Optional[str] = None):
        self.resource_type = resource_type
        super().__init__(
            message=message,
            code="RESOURCE_UNAVAILABLE",
            retryable=True
        )


class StorageError(BasilicaError):
    """
    Raised when storage configuration or operations fail.

    This occurs when:
    - Invalid storage backend specified
    - Storage credentials are invalid
    - Mount path is not allowed

    Example:
        >>> client.deploy("my-app", storage="/etc/passwd")
        StorageError: Mount path '/etc/passwd' is not allowed
    """

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            retryable=False
        )


class NetworkError(BasilicaError):
    """
    Raised when API communication fails.

    This occurs when:
    - API server is unreachable
    - Request times out
    - Network connection is lost

    Attributes:
        original_error: The underlying network error

    Example:
        >>> client.deploy("my-app", source="app.py")
        NetworkError: Failed to connect to api.basilica.ai: Connection refused
    """

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.original_error = original_error
        super().__init__(
            message=message,
            code="NETWORK_ERROR",
            retryable=True
        )


class RateLimitError(BasilicaError):
    """
    Raised when API rate limits are exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)

    Example:
        >>> for i in range(1000):
        ...     client.list_deployments()
        RateLimitError: Rate limit exceeded. Retry after 60 seconds.
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        self.retry_after = retry_after
        if retry_after:
            message = f"{message}. Retry after {retry_after} seconds."
        super().__init__(
            message=message,
            code="RATE_LIMITED",
            retryable=True
        )


class SourceError(BasilicaError):
    """
    Raised when source code handling fails.

    This occurs when:
    - Source file does not exist
    - File cannot be read
    - Source code is empty

    Attributes:
        source_path: The path that was provided (if any)

    Example:
        >>> client.deploy("my-app", source="nonexistent.py")
        SourceError: Source file 'nonexistent.py' not found
    """

    def __init__(self, message: str, source_path: Optional[str] = None):
        self.source_path = source_path
        super().__init__(
            message=message,
            code="SOURCE_ERROR",
            retryable=False
        )


# =============================================================================
# Distributed-training exceptions (SDK arch § 8).
#
# Specific subclasses for the operational shapes a researcher iterating on
# `world_size` will hit, with the actionable numeric context surfaced as
# attributes (current/limit, ready/required_min, requested/min/max). Generic
# ValidationError or RuntimeError would force the caller to re-parse the
# message string -- these structured exceptions don't.
# =============================================================================


class DistributedError(BasilicaError):
    """
    Base class for distributed-training-specific errors.

    All exceptions raised by `@basilica.distributed` (canonical surface)
    and methods on the `DistributedTraining` facade derive from this
    class. Catch this to handle any distributed-training failure mode
    generically.
    """

    def __init__(
        self,
        message: str,
        code: str = "DISTRIBUTED_ERROR",
        retryable: bool = False,
    ):
        super().__init__(message=message, code=code, retryable=retryable)


class QuotaExceeded(DistributedError):
    """
    Raised when a distributed deployment would exceed the namespace's rank
    budget.

    The platform enforces a per-namespace cap on concurrent distributed
    ranks (default 10; override via `basilica.ai/distributed-rank-budget`
    annotation on the namespace). When `bench.mode = on-start`, the bench
    probe (2 ranks) counts against the same budget.

    Attributes:
        current: Ranks currently in use across the namespace.
        requested: Ranks the new (or scaled) deployment would add.
        limit: Hard cap from the namespace annotation.

    Example:
        >>> # Namespace already at 8/10, requesting 4-rank UD with bench
        >>> @basilica.distributed(...)
        ... def train(): ...
        >>> training = train()
        QuotaExceeded: namespace rank budget exceeded:
            current=8, requested=worker(4)+bench(2)=6, limit=10
    """

    def __init__(
        self,
        message: str,
        current: int,
        requested: int,
        limit: int,
    ):
        self.current = current
        self.requested = requested
        self.limit = limit
        super().__init__(
            message=message,
            code="DISTRIBUTED_QUOTA_EXCEEDED",
            retryable=False,
        )


class BelowMinimumWorld(DistributedError):
    """
    Raised by `wait_until_min_world(timeout=...)` when the timeout expires
    before `min` ranks are ready.

    The UD is NOT auto-deleted; the caller decides whether to keep waiting
    (capacity may still arrive) or `delete()`. SDK arch § 11.

    Attributes:
        ready: Ranks that did become ready.
        required_min: `worldSize.min` from the spec.
        timeout: The timeout value (seconds) the wait used. `None` when
            the exception is raised outside a wait context.

    Example:
        >>> @basilica.distributed(..., timeout=300)
        ... def train(): ...
        >>> training = train()
        BelowMinimumWorld: ready=2, required_min=4, timeout=300s
    """

    def __init__(
        self,
        message: str,
        ready: int,
        required_min: int,
        timeout: Optional[int] = None,
    ):
        self.ready = ready
        self.required_min = required_min
        self.timeout = timeout
        super().__init__(
            message=message,
            code="DISTRIBUTED_BELOW_MINIMUM_WORLD",
            retryable=True,
        )


class RendezvousUnavailable(DistributedError):
    """
    Raised when the per-UD rendezvous Pod fails to start within a bounded
    retry window. The UD is NOT auto-deleted; investigation is the
    researcher's call (likely to require platform-team support).

    Common causes:
    - Operator error rendering the rendezvous Deployment.
    - Image pull failure on the rendezvous (etcd) image.
    - NetworkPolicy misconfiguration blocking workers from reaching
      port 2379 on the rendezvous Pod.
    """

    def __init__(self, message: str = "Rendezvous Pod unavailable"):
        super().__init__(
            message=message,
            code="DISTRIBUTED_RENDEZVOUS_UNAVAILABLE",
            retryable=True,
        )


class WorldSizeOutOfBounds(DistributedError):
    """
    Raised when a `WorldSize` triple violates `1 <= min <= target <= max`,
    or when `scale(target)` is called with a target outside
    `[worldSize.min, worldSize.max]`.

    The dataclass `WorldSize.__post_init__` raises this for construction-
    time violations (`min=0`, `min > target`, `target > max`); the API
    raises it for `scale()` calls that pass dataclass validation but
    violate the live bounds.

    Attributes:
        requested: The target / value that failed validation.
        min: `worldSize.min` (or 1 if unknown).
        max: `worldSize.max` (or large sentinel if unknown).

    Example:
        >>> WorldSize(min=4, target=2, max=8)  # target < min
        WorldSizeOutOfBounds: requested=2, min=4, max=8

        >>> training.scale(target=12)  # if max=8
        WorldSizeOutOfBounds: requested=12, min=4, max=8
    """

    def __init__(
        self,
        message: str,
        requested: int,
        min: int,
        max: int,
    ):
        self.requested = requested
        self.min = min
        self.max = max
        super().__init__(
            message=message,
            code="DISTRIBUTED_WORLD_SIZE_OUT_OF_BOUNDS",
            retryable=False,
        )


class UDTerminalState(DistributedError):
    """
    Phase 5b (#445): raised when the user attempts to mutate a
    `DistributedTraining` UD that has already reached a terminal state
    (`succeeded`, `failed`, or `cancelled`).

    Two paths raise this:

    1. `t.scale(target=N)` against a terminal UD. The operator's
       defense in depth catches `kubectl edit` mutations that bypass the
       SDK with a `UDTerminalState` Warning Event, but `t.scale` is the
       primary user-facing rejection.
    2. `t.wait_until_complete()` against a UD that is ALREADY terminal at
       the time of the call. Distinguishes "I waited and it completed"
       from "I called this on an already-completed UD" -- the caller
       should read `t.world` / `t.bench` / `t.rank_exits` directly and
       call `t.delete()` when done.

    Attributes:
        phase: The terminal phase observed (`succeeded` | `failed` |
            `cancelled`).
        requested_target: The `target` the caller passed to `scale()`,
            or `None` for the `wait_until_complete()` path.

    Example:
        >>> training.refresh()
        >>> training.phase
        'succeeded'
        >>> training.scale(target=4)
        UDTerminalState: phase='succeeded', requested_target=4

        >>> training.wait_until_complete()
        UDTerminalState: phase='succeeded', requested_target=None

    The user's only entry point for cleanup is `t.delete()`.
    """

    def __init__(
        self,
        message: str,
        phase: str,
        requested_target: Optional[int] = None,
    ):
        self.phase = phase
        self.requested_target = requested_target
        super().__init__(
            message=message,
            code="DISTRIBUTED_UD_TERMINAL_STATE",
            retryable=False,
        )


# =============================================================================
# Managed Inference exceptions (MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE §4).
#
# The inference gateway (https://inference.basilica.ai) answers failures with
# OpenAI-style error JSON:
#
#     {"error": {"message": "...", "type": "...", "code": "...", "cap": "..."}}
#
# and maps them onto a small status vocabulary: 401 bad key, 402 insufficient
# credits, 404 unknown/unowned model, 429 quota (naming the tripped cap:
# rpm|tpm|concurrency|budget, plus Retry-After), 503 pool saturated. The
# classes below mirror that contract 1:1 so operator automation can branch on
# the typed exception and read structured attributes (notably `.cap` and
# `.retry_after` on InferenceQuotaExceededError) instead of parsing strings.
# =============================================================================


class InferenceError(BasilicaError):
    """
    Base exception for Managed Inference gateway errors.

    All exceptions raised by `client.inference` derive from this class; catch
    it to handle any inference failure mode generically. (Like every SDK
    error it also derives from `BasilicaError`.)

    Attributes:
        status_code: HTTP status returned by the gateway (401/402/404/429/503),
            or None when the failure happened before/without an HTTP response.
        error_type: The OpenAI `error.type` string from the gateway, if any.

    Example:
        >>> try:
        ...     models = client.inference.list_models()
        ... except InferenceError as e:
        ...     print(f"Inference error (status={e.status_code}): {e}")
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        code: str = "INFERENCE_ERROR",
        retryable: bool = False,
    ):
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message=message, code=code, retryable=retryable)


class InferenceAuthenticationError(InferenceError):
    """
    Raised when the inference gateway rejects the API key (HTTP 401).

    The gateway takes `Authorization: Bearer basilica_...` keys. This error
    means the key is missing, malformed, expired, or revoked.

    Resolution:
        Create a key with `basilica tokens create`, then either set
        BASILICA_API_TOKEN or pass `api_key=` to BasilicaClient.
    """

    def __init__(
        self,
        message: str = "Inference authentication failed",
        error_type: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            status_code=401,
            error_type=error_type,
            code="INFERENCE_AUTH_FAILED",
            retryable=False,
        )


class InsufficientCreditsError(InferenceError):
    """
    Raised when the account balance is below the gateway's floor (HTTP 402).

    The gateway enforces an affordability gate before any engine work; a 402
    means the request was rejected without consuming credits. Top up the
    account balance, then retry.

    Attributes:
        balance: Current balance as reported by the gateway, if included in
            the error payload; None otherwise.
    """

    def __init__(
        self,
        message: str = "Insufficient credits",
        error_type: Optional[str] = None,
        balance: Optional[str] = None,
    ):
        self.balance = balance
        super().__init__(
            message=message,
            status_code=402,
            error_type=error_type,
            code="INFERENCE_INSUFFICIENT_CREDITS",
            retryable=False,
        )


class InferenceQuotaExceededError(InferenceError):
    """
    Raised when an admission quota cap trips (HTTP 429).

    This is the operator-automation contract: the gateway rejects floods at
    admission, naming which per-tenant cap tripped and how long to back off.
    Both attributes are first-class here so callers can automate retries and
    alerting without parsing the message string.

    Attributes:
        cap: Which cap tripped -- one of "rpm", "tpm", "concurrency",
            "budget" -- or None if the gateway did not name one.
        retry_after: Seconds to wait before retrying, from the gateway's
            Retry-After header (or error body); None if not provided.

    Example:
        >>> try:
        ...     client.inference.get_model("llama-3.1-70b-instruct")
        ... except InferenceQuotaExceededError as e:
        ...     time.sleep(e.retry_after or 1.0)
        ...     alert(f"inference {e.cap} cap tripped")
    """

    def __init__(
        self,
        message: str = "Inference quota exceeded",
        cap: Optional[str] = None,
        retry_after: Optional[float] = None,
        error_type: Optional[str] = None,
    ):
        self.cap = cap
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_type=error_type,
            code="INFERENCE_QUOTA_EXCEEDED",
            retryable=True,
        )


class InferenceModelNotFoundError(InferenceError):
    """
    Raised when the requested model is unknown or unowned (HTTP 404).

    The gateway returns 404 (not 403) for models the caller does not own --
    e.g. another tenant's adapter -- to avoid confirming private names.

    Attributes:
        model: The model id that was requested, if known.
    """

    def __init__(
        self,
        message: str = "Inference model not found",
        model: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        self.model = model
        super().__init__(
            message=message,
            status_code=404,
            error_type=error_type,
            code="INFERENCE_MODEL_NOT_FOUND",
            retryable=False,
        )


class InferenceUnavailableError(InferenceError):
    """
    Raised when the serving pool is saturated or unavailable (HTTP 503).

    The gateway sheds load before engine queues grow, so a 503 is a signal to
    back off and retry, not a hard failure.

    Attributes:
        retry_after: Seconds to wait before retrying, from the gateway's
            Retry-After header; None if not provided.
    """

    def __init__(
        self,
        message: str = "Inference pool unavailable",
        retry_after: Optional[float] = None,
        error_type: Optional[str] = None,
    ):
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=503,
            error_type=error_type,
            code="INFERENCE_UNAVAILABLE",
            retryable=True,
        )
