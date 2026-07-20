"""
Unit tests for the Managed Inference surface (client.inference).

Coverage (MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE §4):
- Model catalog parsing: list_models / get_model from the gateway's
  OpenAI-style /v1/models payloads.
- Usage rollup parsing: Decimal charge_credits (never float), date handling,
  query-param construction (from/to/model/kid), envelope tolerance.
- Error mapping from the gateway's OpenAI error JSON: 401, 402, 404, 503,
  and the 429 operator-automation contract (`.cap` and `.retry_after`
  attributes, Retry-After header precedence over the body).
- openai_client_args shape + BASILICA_INFERENCE_ENDPOINT override.
- Authorization header construction (Bearer <key>) and key resolution
  priority (explicit > BASILICA_API_TOKEN > CLI token store).
- Async variants (run_in_executor pattern, matching BasilicaClient).

The wire logic lives in the Rust core (basilica_sdk::inference) behind the
``_basilica`` PyO3 extension, so urlopen-level patching no longer reaches it.
Instead these tests stand up a threaded in-process HTTP server on the
loopback and point the client at it: every request is made by the real Rust
client (reqwest) and every response parsed by the real Rust core, so the
tests exercise the exact production wire path. The mock server records
requests for URL/query/header assertions. Error-message text is owned by the
Rust core (single source of truth); the tests assert the typed exception,
its status_code, and the structured attributes the Python surface documents.
"""

import http.server
import json
import socket
import threading
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

import pytest

from basilica import BasilicaClient
from basilica.exceptions import (
    AuthenticationError,
    InferenceAuthenticationError,
    InferenceError,
    InferenceModelNotFoundError,
    InferenceQuotaExceededError,
    InferenceUnavailableError,
    InsufficientCreditsError,
    NetworkError,
)
from basilica.inference import (
    DEFAULT_INFERENCE_ENDPOINT,
    InferenceClient,
    InferenceModel,
    InferenceUsageRow,
)

TEST_API_KEY = "basilica_testkey123"
TEST_ENDPOINT = "https://inference.test.local"
TEST_API_BASE = "https://api.test.local"

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "llama-3.1-70b-instruct",
            "object": "model",
            "created": 1750000000,
            "owned_by": "basilica",
        },
        {
            "id": "qwen2.5-14b-instruct",
            "object": "model",
            "created": 1750000100,
            "owned_by": "basilica",
        },
    ],
}

USAGE_PAYLOAD = {
    "rows": [
        {
            "date": "2026-07-18",
            "model": "llama-3.1-70b-instruct",
            "tenant_id": "t-123",
            "kid": "key-abc",
            "prompt_tokens": 1500,
            "completion_tokens": 3200,
            "cached_tokens": 400,
            "charge_credits": "12.340000",
        },
        {
            "date": "2026-07-19",
            "model": "qwen2.5-14b-instruct",
            "tenant_id": "t-123",
            "kid": None,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "cached_tokens": 0,
            "charge_credits": "0.000010",
        },
    ]
}


class _RecordedRequest:
    """One request the mock server received (raw, undecoded path)."""

    def __init__(self, method: str, path: str, headers):
        self.method = method
        self.path = path
        self.headers = headers


class _MockServer:
    """
    A threaded in-process HTTP server standing in for the inference gateway
    or the management API. Routes are registered per test; every request the
    Rust client makes is recorded for assertion.
    """

    def __init__(self):
        self._routes: List[tuple] = []  # (path, status, body bytes, headers)
        self.requests: List[_RecordedRequest] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(_RecordedRequest("GET", self.path, self.headers))
                route = outer._find_route(self.path)
                if route is None:
                    status, body, headers = (
                        404,
                        b'{"error": {"message": "no route registered"}}',
                        {},
                    )
                else:
                    status, body, headers = route
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep test output clean
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def add_json(self, path: str, payload, status: int = 200, headers: Optional[dict] = None):
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        if isinstance(body, str):
            body = body.encode()
        self._routes.append((path, status, body, dict(headers or {})))

    def _find_route(self, request_path: str):
        """Longest-prefix match; a route also matches with a query string."""
        best = None
        for path, status, body, headers in self._routes:
            if (
                request_path == path
                or request_path.startswith(path + "?")
                or request_path.startswith(path + "/")
            ):
                if best is None or len(path) > len(best[0]):
                    best = (path, status, body, headers)
        if best is None:
            return None
        _, status, body, headers = best
        return status, body, headers

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def mock_servers():
    """Factory for mock servers; every created server is stopped on teardown."""
    servers: List[_MockServer] = []

    def make_server() -> _MockServer:
        server = _MockServer()
        servers.append(server)
        return server

    yield make_server

    for server in servers:
        server.stop()


def _make_client(**kwargs) -> InferenceClient:
    """An InferenceClient pinned to test origins with an explicit key."""
    kwargs.setdefault("api_key", TEST_API_KEY)
    kwargs.setdefault("api_base", TEST_API_BASE)
    kwargs.setdefault("endpoint", TEST_ENDPOINT)
    return InferenceClient(**kwargs)


def _dead_port_url() -> str:
    """A loopback URL whose port is closed (connection refused)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Keep the developer's real env/login out of every test."""
    monkeypatch.delenv("BASILICA_API_TOKEN", raising=False)
    monkeypatch.delenv("BASILICA_INFERENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("BASILICA_API_URL", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-empty"))
    monkeypatch.setattr(
        "basilica.inference._cli_token_store_paths", lambda: [tmp_path / "auth.json"]
    )


class TestListModels:
    def test_parses_catalog_entries(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", MODELS_PAYLOAD)
        models = _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()

        assert len(models) == 2
        first = models[0]
        assert isinstance(first, InferenceModel)
        assert first.id == "llama-3.1-70b-instruct"
        assert first.created == 1750000000
        assert first.owned_by == "basilica"
        # raw echoes the gateway model object (fields beyond the dataclass)
        assert first.raw["object"] == "model"

    def test_requests_gateway_models_url(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", MODELS_PAYLOAD)
        _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()

        assert len(gateway.requests) == 1
        request = gateway.requests[0]
        assert request.path == "/v1/models"
        assert request.method == "GET"

    def test_tolerates_bare_list_payload(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", MODELS_PAYLOAD["data"])
        models = _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert [m.id for m in models] == [
            "llama-3.1-70b-instruct",
            "qwen2.5-14b-instruct",
        ]

    def test_rejects_unexpected_payload(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", {"data": {"nope": 1}})
        with pytest.raises(InferenceError):
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()


class TestGetModel:
    def test_parses_single_model(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json(
            "/v1/models/llama-3.1-70b-instruct", MODELS_PAYLOAD["data"][0]
        )
        model = _make_client(endpoint=gateway.url, api_base=gateway.url).get_model(
            "llama-3.1-70b-instruct"
        )

        assert model.id == "llama-3.1-70b-instruct"
        assert gateway.requests[0].path == "/v1/models/llama-3.1-70b-instruct"

    def test_url_quotes_model_name(self, mock_servers):
        gateway = mock_servers()
        payload = {"id": "org/model:adapter", "object": "model"}
        gateway.add_json("/v1/models/org%2Fmodel%3Aadapter", payload)
        _make_client(endpoint=gateway.url, api_base=gateway.url).get_model(
            "org/model:adapter"
        )

        assert gateway.requests[0].path.endswith("/v1/models/org%2Fmodel%3Aadapter")

    def test_404_raises_model_not_found(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json(
            "/v1/models/nope",
            {"error": {"message": "unknown model", "type": "invalid_request_error"}},
            status=404,
        )
        with pytest.raises(InferenceModelNotFoundError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).get_model("nope")

        assert exc_info.value.status_code == 404
        # The Rust core reports the requested model name, not the body text.
        assert exc_info.value.model == "nope"
        assert "nope" in str(exc_info.value)


class TestUsage:
    def test_parses_rows_with_decimal_charges_and_dates(self, mock_servers):
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", USAGE_PAYLOAD)
        rows = _make_client(endpoint=api.url, api_base=api.url).usage()

        assert len(rows) == 2
        row = rows[0]
        assert isinstance(row, InferenceUsageRow)
        assert row.date == date(2026, 7, 18)
        assert row.model == "llama-3.1-70b-instruct"
        assert row.tenant_id == "t-123"
        assert row.kid == "key-abc"
        assert row.prompt_tokens == 1500
        assert row.completion_tokens == 3200
        assert row.cached_tokens == 400
        # The money contract: Decimal, exact, never float.
        assert isinstance(row.charge_credits, Decimal)
        assert row.charge_credits == Decimal("12.340000")
        assert rows[1].charge_credits == Decimal("0.000010")
        assert rows[1].kid is None
        total = sum((r.charge_credits for r in rows), Decimal("0"))
        assert total == Decimal("12.340010")

    def test_usage_hits_api_base_not_gateway_and_builds_query(self, mock_servers):
        gateway = mock_servers()
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", USAGE_PAYLOAD)
        client = _make_client(endpoint=gateway.url, api_base=api.url)
        client.usage(
            from_date=date(2026, 7, 1),
            to_date="2026-07-31",
            model="llama-3.1-70b-instruct",
            kid="key-abc",
        )

        # The management plane got the request; the gateway saw nothing.
        assert gateway.requests == []
        assert len(api.requests) == 1
        request = api.requests[0]
        assert request.path.startswith("/v1/inference/usage/summary?")
        query = request.path.split("?", 1)[1]
        for expected in (
            "from=2026-07-01",
            "to=2026-07-31",
            "model=llama-3.1-70b-instruct",
            "kid=key-abc",
        ):
            assert expected in query

    def test_usage_without_filters_sends_no_query(self, mock_servers):
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", USAGE_PAYLOAD)
        _make_client(endpoint=api.url, api_base=api.url).usage()

        assert api.requests[0].path == "/v1/inference/usage/summary"

    def test_usage_tolerates_bare_list_and_rows_envelope(self, mock_servers):
        # The Rust core's tolerance contract: a bare array or {"rows": [...]}.
        for payload in (USAGE_PAYLOAD["rows"], {"rows": USAGE_PAYLOAD["rows"]}):
            api = mock_servers()
            api.add_json("/v1/inference/usage/summary", payload)
            rows = _make_client(endpoint=api.url, api_base=api.url).usage()
            assert len(rows) == 2

    def test_usage_rejects_unexpected_payload(self, mock_servers):
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", {"nope": []})
        with pytest.raises(InferenceError):
            _make_client(endpoint=api.url, api_base=api.url).usage()


class TestErrorMapping:
    @pytest.mark.parametrize(
        "status,payload,exc_type",
        [
            (
                401,
                {"error": {"message": "bad key", "type": "authentication_error"}},
                InferenceAuthenticationError,
            ),
            (
                402,
                {
                    "error": {
                        "message": "insufficient credits",
                        "type": "billing_error",
                        "balance": "0.5",
                    }
                },
                InsufficientCreditsError,
            ),
            (
                404,
                {"error": {"message": "unknown model", "type": "invalid_request_error"}},
                InferenceModelNotFoundError,
            ),
            (
                503,
                {"error": {"message": "pool saturated", "type": "server_error"}},
                InferenceUnavailableError,
            ),
        ],
    )
    def test_status_maps_to_typed_exception(self, mock_servers, status, payload, exc_type):
        gateway = mock_servers()
        gateway.add_json("/v1/models", payload, status=status)
        gateway.add_json("/v1/models/unknown-model", payload, status=status)
        client = _make_client(endpoint=gateway.url, api_base=gateway.url)
        with pytest.raises(exc_type) as exc_info:
            if status == 404:
                # The Rust core maps 404 to ModelNotFound only on the
                # single-model route; a catalog-endpoint 404 is a generic
                # invalid-resource error there.
                client.get_model("unknown-model")
            else:
                client.list_models()

        assert exc_info.value.status_code == status

    def test_402_raises_insufficient_credits(self, mock_servers):
        # The Rust core owns the 402 mapping and does not surface the
        # gateway's balance field; the Python attribute stays, unset.
        gateway = mock_servers()
        gateway.add_json(
            "/v1/models",
            {"error": {"message": "no credits", "balance": "0.5"}},
            status=402,
        )
        with pytest.raises(InsufficientCreditsError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert exc_info.value.status_code == 402
        assert exc_info.value.balance is None

    def test_429_exposes_cap_and_retry_after_from_header(self, mock_servers):
        """The operator-automation contract: structured .cap / .retry_after."""
        gateway = mock_servers()
        payload = {
            "error": {
                "message": "rpm cap exceeded",
                "type": "rate_limit_exceeded",
                "cap": "rpm",
            }
        }
        gateway.add_json("/v1/models", payload, status=429, headers={"Retry-After": "17"})
        with pytest.raises(InferenceQuotaExceededError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()

        err = exc_info.value
        assert err.cap == "rpm"
        assert err.retry_after == 17.0
        assert err.status_code == 429
        assert err.retryable is True

    @pytest.mark.parametrize("cap", ["rpm", "tpm", "concurrency", "budget"])
    def test_429_cap_names_from_body(self, mock_servers, cap):
        gateway = mock_servers()
        payload = {"error": {"message": "quota", "cap": cap}}
        gateway.add_json("/v1/models", payload, status=429, headers={"Retry-After": "3"})
        with pytest.raises(InferenceQuotaExceededError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()

        assert exc_info.value.cap == cap
        assert exc_info.value.retry_after == 3.0

    def test_429_header_retry_after_wins_over_body(self, mock_servers):
        gateway = mock_servers()
        payload = {"error": {"message": "quota", "cap": "tpm", "retry_after": 99}}
        gateway.add_json("/v1/models", payload, status=429, headers={"Retry-After": "5"})
        with pytest.raises(InferenceQuotaExceededError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert exc_info.value.retry_after == 5.0

    def test_429_missing_cap_is_generic_inference_error(self, mock_servers):
        # The Rust core's 429 contract: the cap name comes from `error.cap`
        # only -- a 429 without one is a generic (non-quota) InferenceError,
        # not a guessed QuotaExceeded.
        gateway = mock_servers()
        payload = {"error": {"message": "quota", "code": "concurrency"}}
        gateway.add_json("/v1/models", payload, status=429)
        with pytest.raises(InferenceError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert type(exc_info.value) is InferenceError
        assert "concurrency" in str(exc_info.value)

    def test_503_exposes_retry_after(self, mock_servers):
        gateway = mock_servers()
        payload = {"error": {"message": "pool saturated"}}
        gateway.add_json("/v1/models", payload, status=503, headers={"Retry-After": "30"})
        with pytest.raises(InferenceUnavailableError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.retryable is True

    def test_unmapped_status_raises_generic_inference_error(self, mock_servers):
        gateway = mock_servers()
        payload = {"error": {"message": "weird", "type": "teapot"}}
        gateway.add_json("/v1/models", payload, status=418)
        with pytest.raises(InferenceError) as exc_info:
            _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()
        assert type(exc_info.value) is InferenceError
        # The Rust core's Invalid variant carries the status and body excerpt.
        assert "418" in str(exc_info.value)
        assert "weird" in str(exc_info.value)

    def test_network_failure_raises_network_error(self):
        url = _dead_port_url()
        client = _make_client(endpoint=url, api_base=url, timeout=5)
        with pytest.raises(NetworkError):
            client.list_models()


class TestOpenAIClientArgs:
    def test_shape_and_values(self):
        client = _make_client()
        args = client.openai_client_args()
        assert args == {
            "base_url": f"{TEST_ENDPOINT}/v1",
            "api_key": TEST_API_KEY,
        }

    def test_default_endpoint_and_env_override(self, monkeypatch):
        # Default: the production gateway.
        client = InferenceClient(api_key=TEST_API_KEY, api_base=TEST_API_BASE)
        assert client.openai_client_args()["base_url"] == (
            f"{DEFAULT_INFERENCE_ENDPOINT}/v1"
        )
        assert DEFAULT_INFERENCE_ENDPOINT == "https://inference.basilica.ai"

        # Env override points the whole surface at another gateway.
        monkeypatch.setenv("BASILICA_INFERENCE_ENDPOINT", "http://localhost:8080/")
        client = InferenceClient(api_key=TEST_API_KEY, api_base=TEST_API_BASE)
        args = client.openai_client_args()
        assert args["base_url"] == "http://localhost:8080/v1"
        assert client.endpoint == "http://localhost:8080"

    def test_docstring_shows_canonical_openai_wiring(self):
        assert "OpenAI(**" in InferenceClient.openai_client_args.__doc__
        assert "openai_client_args())" in InferenceClient.openai_client_args.__doc__


class TestAuth:
    def test_authorization_header_uses_bearer_key(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", MODELS_PAYLOAD)
        _make_client(endpoint=gateway.url, api_base=gateway.url).list_models()

        assert gateway.requests[0].headers.get("Authorization") == (
            f"Bearer {TEST_API_KEY}"
        )
        # The Accept header the old urlopen implementation sent is gone --
        # header construction is owned by the Rust core (reqwest) now.

    def test_key_resolution_priority_explicit_over_env(self, monkeypatch):
        monkeypatch.setenv("BASILICA_API_TOKEN", "basilica_envkey")
        client = _make_client()
        assert client._resolve_api_key() == TEST_API_KEY

    def test_key_resolution_env_fallback(self, monkeypatch):
        monkeypatch.setenv("BASILICA_API_TOKEN", "basilica_envkey")
        client = _make_client(api_key=None)
        assert client._resolve_api_key() == "basilica_envkey"

    def test_key_resolution_cli_token_store(self, tmp_path):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps({"access_token": "cli-access-token", "refresh_token": "r"})
        )
        client = _make_client(api_key=None)
        assert client._resolve_api_key() == "cli-access-token"

    def test_key_resolution_missing_raises_authentication_error(self):
        client = _make_client(api_key=None)
        with pytest.raises(AuthenticationError):
            client._resolve_api_key()

    def test_openai_client_args_resolves_key_lazily(self, monkeypatch):
        client = _make_client(api_key=None)
        with pytest.raises(AuthenticationError):
            client.openai_client_args()
        monkeypatch.setenv("BASILICA_API_TOKEN", "basilica_envkey")
        assert client.openai_client_args()["api_key"] == "basilica_envkey"


class TestBasilicaClientIntegration:
    def _bare_client(self, api_key=TEST_API_KEY) -> BasilicaClient:
        """BasilicaClient without constructing the Rust core (no auth needed)."""
        client = BasilicaClient.__new__(BasilicaClient)
        client._base_url = TEST_API_BASE
        client._api_key = api_key
        client._inference_client = None
        return client

    def test_inference_property_wires_key_and_base_url(self):
        client = self._bare_client()
        inference = client.inference
        assert isinstance(inference, InferenceClient)
        assert inference.api_base == TEST_API_BASE
        assert inference._resolve_api_key() == TEST_API_KEY

    def test_inference_property_is_cached(self):
        client = self._bare_client()
        assert client.inference is client.inference

    def test_inference_honors_base_url_env_for_usage(self, monkeypatch):
        monkeypatch.setenv("BASILICA_API_URL", "https://api.staging.local")
        client = BasilicaClient.__new__(BasilicaClient)
        client._base_url = "https://api.staging.local"
        client._api_key = TEST_API_KEY
        client._inference_client = None
        assert client.inference.api_base == "https://api.staging.local"


class TestAsyncVariants:
    @pytest.mark.asyncio
    async def test_list_models_async(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", MODELS_PAYLOAD)
        client = _make_client(endpoint=gateway.url, api_base=gateway.url)
        models = await client.list_models_async()
        assert [m.id for m in models] == [
            "llama-3.1-70b-instruct",
            "qwen2.5-14b-instruct",
        ]

    @pytest.mark.asyncio
    async def test_get_model_async(self, mock_servers):
        gateway = mock_servers()
        payload = MODELS_PAYLOAD["data"][0]
        gateway.add_json("/v1/models/llama-3.1-70b-instruct", payload)
        client = _make_client(endpoint=gateway.url, api_base=gateway.url)
        model = await client.get_model_async("llama-3.1-70b-instruct")
        assert model.id == "llama-3.1-70b-instruct"

    @pytest.mark.asyncio
    async def test_usage_async(self, mock_servers):
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", USAGE_PAYLOAD)
        client = _make_client(endpoint=api.url, api_base=api.url)
        rows = await client.usage_async(from_date="2026-07-01")
        assert rows[0].charge_credits == Decimal("12.340000")
        assert rows[0].date == date(2026, 7, 18)

    @pytest.mark.asyncio
    async def test_async_error_mapping(self, mock_servers):
        gateway = mock_servers()
        payload = {"error": {"message": "quota", "cap": "budget"}}
        gateway.add_json("/v1/models", payload, status=429)
        client = _make_client(endpoint=gateway.url, api_base=gateway.url)
        with pytest.raises(InferenceQuotaExceededError) as exc_info:
            await client.list_models_async()
        assert exc_info.value.cap == "budget"


class TestCliTokenStoreHelpers:
    def test_jwt_expiry_check(self):
        from basilica.inference import _jwt_is_expired

        def make_jwt(exp):
            import base64 as b64

            payload = b64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
            payload = payload.rstrip(b"=").decode()
            return f"header.{payload}.sig"

        assert _jwt_is_expired(make_jwt(1), now=1000) is True
        assert _jwt_is_expired(make_jwt(2000), now=1000) is False
        # Non-JWT tokens (e.g. basilica_ API keys) are never "expired".
        assert _jwt_is_expired("basilica_whatever", now=1000) is False

    def test_expired_cli_token_is_skipped(self, tmp_path):
        from unittest.mock import patch

        from basilica.inference import _read_cli_access_token

        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps({"access_token": "expired-token", "refresh_token": "r"})
        )
        with patch("basilica.inference._jwt_is_expired", return_value=True):
            with patch(
                "basilica.inference._cli_token_store_paths",
                lambda: [tmp_path / "auth.json"],
            ):
                assert _read_cli_access_token() is None


class TestRustBinding:
    """Tests pinned to the PyO3 binding surface (_basilica.InferenceClient)."""

    def _binding(self, url: str, **kwargs):
        from basilica._basilica import InferenceClient as RustInferenceClient

        kwargs.setdefault("api_base", url)
        kwargs.setdefault("endpoint", url)
        kwargs.setdefault("timeout_secs", 5)
        return RustInferenceClient(TEST_API_KEY, **kwargs)

    def test_binding_resolves_urls_and_strips_slashes(self, mock_servers):
        gateway = mock_servers()
        binding = self._binding(gateway.url + "/")
        assert binding.endpoint() == gateway.url
        assert binding.api_base() == gateway.url
        assert binding.openai_base_url() == f"{gateway.url}/v1"

    def test_binding_usage_returns_native_decimal_and_date(self, mock_servers):
        api = mock_servers()
        api.add_json("/v1/inference/usage/summary", USAGE_PAYLOAD)
        rows = self._binding(api.url).usage()

        row = rows[0]
        # rust_decimal -> decimal.Decimal, exact (scale preserved, never float)
        assert isinstance(row["charge_credits"], Decimal)
        assert str(row["charge_credits"]) == "12.340000"
        assert str(rows[1]["charge_credits"]) == "0.000010"
        # chrono NaiveDate -> datetime.date
        assert type(row["date"]) is date
        assert not isinstance(row["date"], datetime)
        assert row["date"] == date(2026, 7, 18)
        assert row["prompt_tokens"] == 1500
        # Optional fields absent on the wire are omitted from the dict
        assert "kid" not in rows[1] or rows[1].get("kid") is None

    def test_binding_model_dict_omits_absent_optionals(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json("/v1/models", [{"id": "llama-3.1-70b-instruct"}])
        models = self._binding(gateway.url).list_models()
        assert models == [{"id": "llama-3.1-70b-instruct"}]

    def test_binding_quota_error_attributes(self, mock_servers):
        gateway = mock_servers()
        payload = {"error": {"message": "quota", "cap": "tpm"}}
        gateway.add_json("/v1/models", payload, status=429, headers={"Retry-After": "7"})
        with pytest.raises(InferenceQuotaExceededError) as exc_info:
            self._binding(gateway.url).list_models()
        err = exc_info.value
        assert err.cap == "tpm"
        assert err.retry_after == 7.0
        assert err.status_code == 429
        assert err.retryable is True

    def test_binding_unavailable_error_retry_after(self, mock_servers):
        gateway = mock_servers()
        gateway.add_json(
            "/v1/models",
            {"error": {"message": "pool saturated"}},
            status=503,
            headers={"Retry-After": "9"},
        )
        with pytest.raises(InferenceUnavailableError) as exc_info:
            self._binding(gateway.url).list_models()
        assert exc_info.value.retry_after == 9.0
        assert exc_info.value.retryable is True

    def test_binding_transport_maps_to_network_error(self):
        binding = self._binding(_dead_port_url())
        with pytest.raises(NetworkError) as exc_info:
            binding.list_models()
        assert exc_info.value.retryable is True
