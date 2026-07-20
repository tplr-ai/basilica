"""
Managed Inference: model discovery, usage rollups, and OpenAI wiring.

The Basilica inference gateway (https://inference.basilica.ai) is an
OpenAI-compatible endpoint: it speaks the OpenAI wire format for
``POST /v1/chat/completions`` and ``POST /v1/completions`` and renders the
tenant-visible model catalog at ``GET /v1/models`` (see
MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE §4).

This module deliberately does NOT re-wrap chat completions: the wire contract
is OpenAI's, so the canonical inference client is the stock ``openai`` SDK --
use :meth:`InferenceClient.openai_client_args` to point it at the gateway.
What Basilica owns here is the layer around that contract:

- discovery:  :meth:`InferenceClient.list_models` / ``get_model``
- usage:      :meth:`InferenceClient.usage` (billing rollups, Decimal charges)
- wiring:     :meth:`InferenceClient.openai_client_args`
- errors:     typed exceptions mirroring the gateway's OpenAI error JSON
  (401 auth, 402 insufficient credits, 429 quota with ``cap``/``retry_after``,
  404 model, 503 pool) -- see ``basilica.exceptions``.

All HTTP, wire parsing, and error mapping are owned by the Rust core
(``basilica_sdk::inference``) and reached through the ``_basilica`` PyO3
extension; this module keeps the Python-native sugar on top: frozen
dataclasses, ``datetime.date`` / ``decimal.Decimal`` values, lazy auth
resolution, and async wrappers.

Authentication is the user's ``basilica_`` API key as a Bearer token, resolved
in the same priority order as ``BasilicaClient``: explicit ``api_key``
parameter, then the ``BASILICA_API_TOKEN`` environment variable, then the
CLI login token store (``basilica login``) on a best-effort basis.

Example:
    >>> from basilica import BasilicaClient
    >>> client = BasilicaClient()
    >>> for model in client.inference.list_models():
    ...     print(model.id)
    >>> rows = client.inference.usage(from_date="2026-07-01", model="llama-3.1-70b-instruct")
    >>> for row in rows:
    ...     print(row.date, row.model, row.charge_credits)
"""

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from basilica._basilica import InferenceClient as _RustInferenceClient

from .exceptions import AuthenticationError

# Default gateway origin. The Rust core appends the OpenAI-style "/v1/..."
# paths; BASILICA_INFERENCE_ENDPOINT overrides the origin for staging/local
# gateways (explicit flag > env var > this default, matching the Rust core).
DEFAULT_INFERENCE_ENDPOINT = "https://inference.basilica.ai"
INFERENCE_ENDPOINT_ENV_VAR = "BASILICA_INFERENCE_ENDPOINT"

# Discovery/usage are lightweight JSON GETs; they do not need the deploy-scale
# DEFAULT_TIMEOUT_SECS the Rust core uses.
DEFAULT_INFERENCE_TIMEOUT_SECS = 30


@dataclass(frozen=True)
class InferenceModel:
    """
    A model entry from the gateway's tenant-visible catalog (``GET /v1/models``).

    Attributes:
        id: Registry model name used in completion requests
            (e.g. "llama-3.1-70b-instruct").
        created: Unix timestamp the entry was created, if provided.
        owned_by: Owning organization string (OpenAI wire field), if provided.
        raw: The model object as decoded by the Rust core (the gateway fields
            the Rust client types), for access to fields this Python SDK
            version does not surface as attributes yet.
    """

    id: str
    created: Optional[int] = None
    owned_by: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_json(cls, payload: Dict[str, Any]) -> "InferenceModel":
        return cls(
            id=str(payload.get("id", "")),
            created=payload.get("created"),
            owned_by=payload.get("owned_by"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class InferenceUsageRow:
    """
    One row of an inference usage rollup (``GET /v1/inference/usage/summary``).

    Attributes:
        date: UTC day the usage belongs to.
        model: Model the tokens were billed against.
        tenant_id: Tenant the usage belongs to.
        kid: API key id the usage is attributed to, when the rollup is
            per-key; None when the row is key-aggregated.
        prompt_tokens: Billed prompt (input) tokens.
        completion_tokens: Billed completion (output) tokens.
        cached_tokens: Cached-input tokens (billed at the cached rate).
        charge_credits: Total charge in credits, parsed as Decimal -- never
            float, so summing a column of rows does not drift.

    Example:
        >>> rows = client.inference.usage(from_date="2026-07-01")
        >>> total = sum((r.charge_credits for r in rows), Decimal("0"))
    """

    date: date
    model: str
    tenant_id: str
    kid: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    charge_credits: Decimal
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_json(cls, payload: Dict[str, Any]) -> "InferenceUsageRow":
        raw_date = payload.get("date")
        parsed_date: date
        if isinstance(raw_date, date):
            # The binding already hands over a datetime.date.
            parsed_date = raw_date
        else:
            # Rollup dates are ISO "YYYY-MM-DD"; fromisoformat also tolerates
            # a full datetime string by taking its date part via split.
            parsed_date = date.fromisoformat(str(raw_date).split("T")[0])

        # charge_credits is a decimal-string on the wire (e.g. "12.340000")
        # and arrives from the binding as a Decimal. Decimal(str) preserves
        # it exactly; a missing/blank value is 0.
        raw_charge = payload.get("charge_credits")
        charge = Decimal(str(raw_charge)) if raw_charge not in (None, "") else Decimal("0")

        return cls(
            date=parsed_date,
            model=str(payload.get("model", "")),
            tenant_id=str(payload.get("tenant_id", "")),
            kid=payload.get("kid"),
            prompt_tokens=int(payload.get("prompt_tokens") or 0),
            completion_tokens=int(payload.get("completion_tokens") or 0),
            cached_tokens=int(payload.get("cached_tokens") or 0),
            charge_credits=charge,
            raw=dict(payload),
        )


def _cli_token_store_paths() -> List[Path]:
    """
    Candidate locations of the CLI login token store (auth.json).

    Mirrors the Rust SDK's etcetera-based data dir
    (`<platform-data-dir>/basilica/auth.json`): XDG on Linux/BSD, Application
    Support on macOS, %APPDATA% on Windows.
    """
    candidates: List[Path] = []
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        candidates.append(Path(xdg_data_home) / "basilica" / "auth.json")
    home = Path.home()
    candidates.append(home / ".local" / "share" / "basilica" / "auth.json")
    candidates.append(home / "Library" / "Application Support" / "basilica" / "auth.json")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "basilica" / "auth.json")
    return candidates


def _jwt_is_expired(token: str, now: Optional[float] = None) -> bool:
    """Best-effort JWT `exp` check; tokens without a parsable exp are kept."""
    parts = token.split(".")
    if len(parts) != 3:
        return False
    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
    except Exception:
        return False
    if exp is None:
        return False
    return float(exp) <= (now if now is not None else time.time())


def _read_cli_access_token() -> Optional[str]:
    """
    Best-effort read of the `basilica login` token store.

    The gateway accepts the same Auth0 JWTs as basilica-api (spec §4.2), so a
    stored, unexpired access token works as a Bearer credential. Refresh is
    intentionally out of scope for the Python surface -- an expired token
    falls through to AuthenticationError telling the user to re-login or use
    an API key.
    """
    for path in _cli_token_store_paths():
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            token = data.get("access_token")
            if isinstance(token, str) and token and not _jwt_is_expired(token):
                return token
        except Exception:
            continue
    return None


class InferenceClient:
    """
    Discovery/usage surface for Basilica Managed Inference.

    Obtain one via ``BasilicaClient().inference`` rather than constructing it
    directly; the property wires the client's API key and API base URL
    through.

    Attributes:
        endpoint: The gateway origin (default https://inference.basilica.ai,
            overridable via BASILICA_INFERENCE_ENDPOINT).
        api_base: The Basilica API origin the usage rollups live on.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = DEFAULT_INFERENCE_TIMEOUT_SECS,
    ):
        """
        Initialize the inference surface.

        Args:
            api_key: Explicit API key. Falls back to BASILICA_API_TOKEN, then
                the CLI login token store.
            api_base: Basilica API origin hosting the usage rollups
                (default: BASILICA_API_URL env var or https://api.basilica.ai).
            endpoint: Gateway origin (default: BASILICA_INFERENCE_ENDPOINT env
                var or https://inference.basilica.ai).
            timeout: HTTP timeout in seconds for discovery/usage calls.
        """
        self._api_key = api_key
        if api_base is None:
            api_base = os.environ.get("BASILICA_API_URL", "https://api.basilica.ai")
        if endpoint is None or not str(endpoint).strip():
            # Mirror the Rust core's resolution: blank values count as unset.
            env_endpoint = os.environ.get(INFERENCE_ENDPOINT_ENV_VAR)
            endpoint = (
                env_endpoint
                if env_endpoint and env_endpoint.strip()
                else DEFAULT_INFERENCE_ENDPOINT
            )
        self._api_base = api_base.rstrip("/")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._rust_client: Optional[_RustInferenceClient] = None

    @property
    def endpoint(self) -> str:
        """The inference gateway origin."""
        return self._endpoint

    @property
    def api_base(self) -> str:
        """The Basilica API origin the usage rollups are served from."""
        return self._api_base

    def _resolve_api_key(self) -> str:
        """
        Resolve the Bearer credential: explicit key, then BASILICA_API_TOKEN,
        then the CLI login token store (best-effort).

        Raises:
            AuthenticationError: If no credential is available.
        """
        if self._api_key:
            return self._api_key
        env_key = os.environ.get("BASILICA_API_TOKEN")
        if env_key:
            return env_key
        cli_token = _read_cli_access_token()
        if cli_token:
            return cli_token
        raise AuthenticationError(
            "No API key available for inference. Set BASILICA_API_TOKEN, pass "
            "api_key= to BasilicaClient, or run 'basilica login' "
            "(create a key with 'basilica tokens create')."
        )

    def _rust(self) -> _RustInferenceClient:
        """
        The low-level Rust-core client (``basilica_sdk::inference`` via PyO3),
        built lazily on first use so auth resolution stays lazy and
        constructing the surface without a credential keeps working.
        """
        if self._rust_client is None:
            self._rust_client = _RustInferenceClient(
                self._resolve_api_key(),
                api_base=self._api_base,
                endpoint=self._endpoint,
                timeout_secs=self._timeout,
            )
        return self._rust_client

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_models(self) -> List[InferenceModel]:
        """
        List the tenant-visible model catalog (``GET /v1/models``).

        Returns:
            List of InferenceModel entries from the gateway registry.

        Raises:
            InferenceAuthenticationError: Bad or missing API key (401).
            InferenceQuotaExceededError: Quota cap tripped (429); see `.cap`
                and `.retry_after`.
            InferenceUnavailableError: Serving pool unavailable (503).
            NetworkError: Gateway unreachable.

        Example:
            >>> for model in client.inference.list_models():
            ...     print(model.id)
        """
        return [InferenceModel._from_json(entry) for entry in self._rust().list_models()]

    def get_model(self, name: str) -> InferenceModel:
        """
        Get a single model's metadata (``GET /v1/models/{name}``).

        Args:
            name: Registry model name, e.g. "llama-3.1-70b-instruct".

        Returns:
            The InferenceModel entry.

        Raises:
            InferenceModelNotFoundError: Unknown or unowned model (404). The
                gateway returns 404 rather than 403 for models owned by other
                tenants, to avoid confirming private names.

        Example:
            >>> model = client.inference.get_model("llama-3.1-70b-instruct")
        """
        return InferenceModel._from_json(self._rust().get_model(name))

    # ------------------------------------------------------------------
    # Usage rollups
    # ------------------------------------------------------------------

    def usage(
        self,
        from_date: Optional[Union[str, date]] = None,
        to_date: Optional[Union[str, date]] = None,
        model: Optional[str] = None,
        kid: Optional[str] = None,
    ) -> List[InferenceUsageRow]:
        """
        Fetch inference usage rollups (``GET {api_base}/v1/inference/usage/summary``).

        Args:
            from_date: Start of the window, inclusive -- ``datetime.date`` or
                ISO "YYYY-MM-DD" string. None = server default window.
            to_date: End of the window, inclusive -- same formats.
            model: Restrict to one model's rows.
            kid: Restrict to one API key id's rows.

        Returns:
            List of InferenceUsageRow with `charge_credits` as Decimal.

        Raises:
            InferenceAuthenticationError: Bad or missing API key (401).
            InferenceQuotaExceededError: Quota cap tripped (429).
            NetworkError: API unreachable.

        Example:
            >>> rows = client.inference.usage("2026-07-01", "2026-07-31")
            >>> for row in rows:
            ...     print(row.date, row.model, row.charge_credits)
        """
        rows = self._rust().usage(
            from_date=self._format_date(from_date) if from_date is not None else None,
            to_date=self._format_date(to_date) if to_date is not None else None,
            model=model,
            kid=kid,
        )
        return [InferenceUsageRow._from_json(row) for row in rows]

    @staticmethod
    def _format_date(value: Union[str, date]) -> str:
        """Serialize a date filter to the API's ISO "YYYY-MM-DD" format."""
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    # ------------------------------------------------------------------
    # OpenAI SDK wiring
    # ------------------------------------------------------------------

    def openai_client_args(self) -> Dict[str, str]:
        """
        Constructor kwargs pointing the stock ``openai`` SDK at the gateway.

        Why a dict and not wrapped ``chat.completions`` methods on this
        client: the gateway's wire contract is OpenAI's own -- chat
        completions, SSE streaming, tool calls, the whole schema -- so the
        canonical inference client is the official ``openai`` package, which
        tracks that schema as it evolves. Basilica's SDK surface is the layer
        around the contract (model discovery, usage rollups, typed admission
        errors); re-wrapping completions here would only lag upstream.

        Returns:
            Dict with ``base_url`` (``<endpoint>/v1``) and ``api_key``,
            ready to splat into the OpenAI constructor.

        Example:
            >>> from openai import OpenAI
            >>> from basilica import BasilicaClient
            >>> basilica = BasilicaClient()
            >>> openai = OpenAI(**basilica.inference.openai_client_args())
            >>> resp = openai.chat.completions.create(
            ...     model="llama-3.1-70b-instruct",
            ...     messages=[{"role": "user", "content": "hello"}],
            ... )
        """
        api_key = self._resolve_api_key()
        return {
            "base_url": self._rust().openai_base_url(),
            "api_key": api_key,
        }

    # ------------------------------------------------------------------
    # Async variants (same run_in_executor pattern as BasilicaClient)
    # ------------------------------------------------------------------

    async def list_models_async(self) -> List[InferenceModel]:
        """List the model catalog asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.list_models)

    async def get_model_async(self, name: str) -> InferenceModel:
        """Get a single model's metadata asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_model, name)

    async def usage_async(
        self,
        from_date: Optional[Union[str, date]] = None,
        to_date: Optional[Union[str, date]] = None,
        model: Optional[str] = None,
        kid: Optional[str] = None,
    ) -> List[InferenceUsageRow]:
        """Fetch inference usage rollups asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.usage(from_date, to_date, model, kid)
        )
