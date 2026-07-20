"""FR-1 conformance: unmodified OpenAI SDK clients vs. the Basilica managed-inference gateway.

Spec: basilica-backend/docs/architecture/MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE.md
  - FR-1 / §4.3   two-class field policy (reserved -> 400; unknown -> drop + header)
  - §20.2         field policy matrix (allowlist, fanout rules)
  - §4.3/§4.2     OpenAI-style error JSON (400/401/402/404/429/503)
  - §5.6/§9.4     SSE relay: data: frames + [DONE], usage stripping, mid-stream errors

Contract sources mirrored by the mock (keep in sync — a drift here is a bug):
  - basilica-backend/crates/basilica-inference/src/openai.rs    (allowlist, reserved, fanout)
  - basilica-backend/crates/basilica-inference/src/admission.rs (error shapes, Retry-After)
  - basilica-backend/crates/basilica-inference/src/auth.rs      (401 shape)
  - basilica-backend/crates/basilica-inference/src/registry.rs  (/v1/models shapes, 404)
  - basilica-backend/crates/basilica-inference/src/relay.rs     (SSE framing, error event)

Run modes:
  - Mock (default): the official `openai` SDK client talks to an in-process
    mock gateway (httpx.MockTransport) that implements the wire contract
    faithfully, including gateway-internal observability (what the engine
    would have received) so drop/rebuild semantics can be asserted.
  - Live (Phase 10): set INFERENCE_LIVE_URL and INFERENCE_LIVE_KEY to point
    the same client at a real gateway. Tests that require fault injection or
    mock introspection skip automatically; every client-visible assertion
    (field policy, error mapping, streaming, usage, models) still runs.
"""

from __future__ import annotations

import gc
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import httpx
import openai
import pytest

# NFR-8: the conformance suite pins one SDK version (requirements.txt). If you
# bump it, re-verify the suite and update this constant in the same change.
OPENAI_SDK_PIN = "2.46.0"

MOCK_BASE_URL = "http://conformance-mock.local/v1"
MOCK_VALID_KEY = "basilica_conformance_mock_key"
DEFAULT_MODEL = "llama-3.1-70b-instruct"
SECOND_MODEL = "qwen2.5-7b"

# ---------------------------------------------------------------------------
# §20.2 field policy, mirrored from basilica-inference/src/openai.rs
# ---------------------------------------------------------------------------

ALLOWLISTED_FIELDS = [
    "model",
    "messages",
    "prompt",
    "temperature",
    "top_p",
    "stop",
    "stream",
    "stream_options",
    "max_tokens",
    "n",
    "best_of",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "logprobs",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
]

DEFAULT_MAX_N = 8
IGNORED_FIELDS_HEADER = "x-basilica-ignored-fields"

_LORA_DETAIL = (
    "LoRA adapters are addressed via the `model` field (`model:adapter`); "
    "dedicated lora/adapter selector fields are reserved"
)
_RESERVED_DETAILS = {
    "cachesalt": (
        "cache_salt is server-minted from the tenant trust group (I1); "
        "client-supplied salts are never forwarded to the engine"
    ),
    "priority": "engine scheduling priority is server-controlled",
    "requestid": (
        "request_id is server-minted (UUIDv7); a client X-Request-Id is "
        "accepted only as a trace tag"
    ),
    "promptcachekey": (
        "prompt_cache_key attempts to influence cache routing; reserved "
        "in v0 (may map into the salt domain at P4, D-semcache)"
    ),
    "lora": _LORA_DETAIL,
    "loraid": _LORA_DETAIL,
    "loraname": _LORA_DETAIL,
    "loraadapter": _LORA_DETAIL,
    "loramodules": _LORA_DETAIL,
    "adapter": _LORA_DETAIL,
    "adapterid": _LORA_DETAIL,
    "adaptername": _LORA_DETAIL,
}


def _reserved_detail(name: str) -> Optional[str]:
    """Reserved-name lookup on the normalized form (lowercase ASCII, `_`/`-`
    stripped) — openai.rs `reserved_detail`."""
    normalized = name.replace("_", "").replace("-", "").lower()
    return _RESERVED_DETAILS.get(normalized)


class Reject(Exception):
    """openai.rs `RejectReason` — every variant maps to HTTP 400."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def error_body(self) -> dict[str, Any]:
        # RejectReason::to_error_body: message/type/code only (no `param`).
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": self.code,
            }
        }


def _reject_reserved(path: str) -> "Reject":
    leaf = path.rsplit(".", 1)[-1]
    # Nested paths index into arrays (`a.b[0].c`); classify the leaf key.
    leaf = leaf.rsplit("[", 1)[0] if "[" in leaf else leaf
    detail = _reserved_detail(leaf)
    assert detail is not None, f"{path} is not actually reserved"
    return Reject(f"field `{path}` is reserved and server-controlled: {detail}", "reserved_field")


def _scan_nested_reserved(path: str, value: Any) -> None:
    """Reject if a reserved name appears anywhere inside an unknown field's
    value (the extra_body smuggling shape) — openai.rs `scan_nested_reserved`."""
    if isinstance(value, dict):
        for key, inner in value.items():
            child = f"{path}.{key}"
            if _reserved_detail(key) is not None:
                raise _reject_reserved(child)
            _scan_nested_reserved(child, inner)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_nested_reserved(f"{path}[{index}]", item)


def _validate_fanout(n: Any, best_of: Any, stream: Any, max_n: int = DEFAULT_MAX_N) -> None:
    """§9.2 fan-out validation — openai.rs `validate_fanout`."""
    n_value = n if n is not None else 1
    if n_value > max_n:
        raise Reject(
            f"`n` value {n_value} exceeds the per-request fan-out cap of {max_n} (max_n)",
            "fanout_cap_exceeded",
        )
    if best_of is not None:
        if best_of > max_n:
            raise Reject(
                f"`best_of` value {best_of} exceeds the per-request fan-out cap of {max_n} (max_n)",
                "fanout_cap_exceeded",
            )
        if best_of < n_value:
            raise Reject(
                f"`best_of` ({best_of}) must be greater than or equal to `n` ({n_value})",
                "best_of_less_than_n",
            )
        if best_of > 1 and stream is True:
            raise Reject(
                "`best_of` > 1 cannot be used with `stream: true`: the engine must "
                "finish all candidates before ranking (§9.2)",
                "best_of_cannot_stream",
            )


def validate_and_rebuild(body: dict[str, Any], endpoint: str) -> tuple[dict[str, Any], list[str]]:
    """The §4.3 allowlist reconstruction — openai.rs `validate_and_rebuild_fields`.

    Returns (engine_request, ignored_fields). Raises Reject on reserved fields
    and fan-out violations. Declared fields per endpoint: chat declares
    `messages`, completions declares `prompt`; the other one is technically
    allowlisted-but-undeclared and therefore takes the drop path, exactly like
    the Rust code (Allowlisted verdict in the extras map is dropped + named).
    """
    declared = {"messages"} if endpoint == "chat" else {"prompt"}
    ignored: list[str] = []

    engine: dict[str, Any] = {}
    for key, value in body.items():
        if key in declared or (key in ALLOWLISTED_FIELDS and key not in ("messages", "prompt")):
            engine[key] = value
            continue
        # Anything else lands in `extra` and is classified.
        detail = _reserved_detail(key)
        if detail is not None:
            raise Reject(
                f"field `{key}` is reserved and server-controlled: {detail}",
                "reserved_field",
            )
        _scan_nested_reserved(key, value)
        ignored.append(key)

    # The same policy for unknown keys inside `stream_options`.
    client_options = body.get("stream_options")
    engine_stream_options: Optional[dict[str, Any]] = None
    if isinstance(client_options, dict):
        engine_stream_options = {}
        for key in ("include_usage", "continuous_usage_stats"):
            if key in client_options:
                engine_stream_options[key] = client_options[key]
        for key, value in client_options.items():
            if key in ("include_usage", "continuous_usage_stats"):
                continue
            detail = _reserved_detail(key)
            if detail is not None:
                raise Reject(
                    f"field `stream_options.{key}` is reserved and server-controlled: {detail}",
                    "reserved_field",
                )
            _scan_nested_reserved(f"stream_options.{key}", value)
            ignored.append(f"stream_options.{key}")
        engine["stream_options"] = engine_stream_options

    _validate_fanout(body.get("n"), body.get("best_of"), body.get("stream"))

    return engine, sorted(set(ignored))


# ---------------------------------------------------------------------------
# §4.3 error shapes, mirrored from admission.rs / auth.rs / registry.rs
# ---------------------------------------------------------------------------

def _openai_error(
    status: int,
    message: str,
    kind: str,
    code: Optional[str],
    *,
    param: Any = None,
    retry_after: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> httpx.Response:
    """admission.rs `openai_error`: {"error": {message, type, param, code}}."""
    error: dict[str, Any] = {"message": message, "type": kind, "param": param, "code": code}
    if extra:
        error.update(extra)
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = str(max(retry_after, 1))
    return httpx.Response(status, json={"error": error}, headers=headers)


def _quota_trip_response(cap: str, current: int, limit: int, retry_after: Optional[int]) -> httpx.Response:
    """admission.rs `quota_trip_response`: 429 naming the tripped cap."""
    error = {
        "message": f"quota exceeded: {cap} cap tripped ({current} > {limit})",
        "type": "rate_limit_error",
        "param": None,
        "code": "quota_exceeded",
        "cap": cap,
    }
    headers = {"retry-after": str(max(retry_after if retry_after is not None else 1, 1))}
    return httpx.Response(429, json={"error": error}, headers=headers)


def _model_not_found_response() -> httpx.Response:
    """registry.rs `render_model(None)` — byte-identical for unknown/retired."""
    return httpx.Response(
        404,
        json={
            "error": {
                "message": "The model does not exist or is not available.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


# ---------------------------------------------------------------------------
# The mock gateway
# ---------------------------------------------------------------------------

POOL_EMPTY_RETRY_AFTER_SECS = 5  # admission.rs POOL_EMPTY_RETRY_AFTER_SECS

CHAT_CONTENT = "The mock gateway answers."
CHAT_PIECES = ["The ", "mock ", "gateway ", "answers."]
CHAT_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
NONSTREAM_CHAT_USAGE = {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20}
COMPLETION_TEXT = "Mock completion text."
COMPLETION_PIECES = ["Mock ", "completion ", "text."]
COMPLETION_USAGE = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
NONSTREAM_COMPLETION_USAGE = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


@dataclass
class RecordedRequest:
    """What the mock observed, for drop/rebuild assertions."""

    path: str
    raw_body: dict[str, Any]
    engine_body: Optional[dict[str, Any]] = None
    ignored_fields: list[str] = field(default_factory=list)
    stream_frames_pulled: int = 0
    stream_finalized: bool = False


class MockGateway:
    """An in-process gateway implementing the FR-1 wire contract faithfully.

    Request order mirrors the §5.2 admission pipeline (auth -> quota ->
    balance -> model resolve -> field policy -> pool -> relay) so precedence
    between error classes matches the real gateway.
    """

    def __init__(self) -> None:
        self.models = {DEFAULT_MODEL: 1_700_000_000, SECOND_MODEL: 1_700_000_100}
        # Fault injection arms (tests set these, per-test instance):
        self.quota_trip: Optional[tuple[str, int, int, Optional[int]]] = None
        self.balance_reject = False
        self.pool_empty = False
        # Stream behavior knobs:
        self.stream_framing = "normal"  # normal | multiline (split data:, keepalives)
        self.stream_mid_error = False
        self.stream_pieces: Optional[int] = None  # None = the natural piece list
        self.records: list[RecordedRequest] = []

    @property
    def last_request(self) -> RecordedRequest:
        assert self.records, "no request recorded yet"
        return self.records[-1]

    # -- routing -----------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {MOCK_VALID_KEY}":
            # auth.rs: verified credential missing/invalid -> 401.
            return _openai_error(
                401, "Invalid API key or token", "authentication_error", "invalid_api_key"
            )

        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
            data = [
                {"id": name, "object": "model", "created": created, "owned_by": "basilica"}
                for name, created in sorted(self.models.items())
            ]
            return httpx.Response(200, json={"object": "list", "data": data})
        if request.method == "GET" and path.startswith("/v1/models/"):
            name = path.removeprefix("/v1/models/")
            if name not in self.models:
                return _model_not_found_response()
            return httpx.Response(
                200,
                json={
                    "id": name,
                    "object": "model",
                    "created": self.models[name],
                    "owned_by": "basilica",
                },
            )
        if request.method == "POST" and path in ("/v1/chat/completions", "/v1/completions"):
            return self._handle_inference(path, request)
        return _openai_error(404, "Unknown route", "invalid_request_error", "unknown_route")

    # -- the admission pipeline (simplified but order-faithful) ------------

    def _handle_inference(self, path: str, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        endpoint = "chat" if path.endswith("chat/completions") else "completions"
        record = RecordedRequest(path=path, raw_body=body)
        self.records.append(record)

        # Step 3: price-free quota (429 + Retry-After naming the cap).
        if self.quota_trip is not None:
            cap, current, limit, retry_after = self.quota_trip
            return _quota_trip_response(cap, current, limit, retry_after)

        # Step 4/6: balance gate (402 billing_error).
        if self.balance_reject:
            return _openai_error(
                402,
                "insufficient balance for this request: budget 0.10000000 credits "
                "is below the 0.50000000 credit minimum viable",
                "billing_error",
                "insufficient_balance",
            )

        # Step 5: model resolve (404).
        if body.get("model") not in self.models:
            return _model_not_found_response()

        # Step 7: stamp — the §4.3 two-class field policy + §9.2 fan-out.
        try:
            engine_body, ignored = validate_and_rebuild(body, endpoint)
        except Reject as reject:
            return httpx.Response(400, json=reject.error_body())
        record.engine_body = engine_body
        record.ignored_fields = ignored

        # Step 8: route (503 pool-empty + Retry-After).
        if self.pool_empty:
            return _openai_error(
                503,
                f"no ready endpoints in pool `pool-{body['model']}`",
                "server_error",
                "pool_unavailable",
                retry_after=POOL_EMPTY_RETRY_AFTER_SECS,
            )

        # Step 9: relay.
        headers = {"x-request-id": str(uuid.uuid4())}
        if ignored:
            headers[IGNORED_FIELDS_HEADER] = ", ".join(ignored)
        if body.get("stream") is True:
            headers["content-type"] = "text/event-stream"
            headers["cache-control"] = "no-cache"
            return httpx.Response(
                200, headers=headers, content=self._sse_stream(body, endpoint, record)
            )
        return httpx.Response(200, headers=headers, json=self._json_response(body, endpoint))

    # -- response synthesis -------------------------------------------------

    def _json_response(self, body: dict[str, Any], endpoint: str) -> dict[str, Any]:
        created = 1_700_000_500
        if endpoint == "chat":
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": created,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": CHAT_CONTENT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": NONSTREAM_CHAT_USAGE,
            }
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": created,
            "model": body["model"],
            "choices": [
                {"text": COMPLETION_TEXT, "index": 0, "logprobs": None, "finish_reason": "stop"}
            ],
            "usage": NONSTREAM_COMPLETION_USAGE,
        }

    def _sse_frames(self, body: dict[str, Any], endpoint: str) -> list[tuple[str, Optional[str]]]:
        """Build the (data, event) frame sequence, honoring §5.6 usage rules.

        Usage rides frames only when the client opted in via
        stream_options.include_usage; continuous_usage_stats puts running
        usage on every chunk, otherwise usage rides only the terminal
        choices-less chunk. A client that did not opt in sees no usage at all.
        """
        options = body.get("stream_options") or {}
        include_usage = bool(options.get("include_usage"))
        continuous = include_usage and bool(options.get("continuous_usage_stats"))
        n = body.get("n") or 1
        created = 1_700_000_500
        request_id = (
            f"chatcmpl-{uuid.uuid4().hex[:24]}" if endpoint == "chat" else f"cmpl-{uuid.uuid4().hex[:24]}"
        )
        pieces = CHAT_PIECES if endpoint == "chat" else COMPLETION_PIECES
        usage = CHAT_USAGE if endpoint == "chat" else COMPLETION_USAGE
        if self.stream_pieces is not None:
            pieces = [f"tok{i} " for i in range(self.stream_pieces)]

        def chunk(index: int, piece_index: int) -> dict[str, Any]:
            if endpoint == "chat":
                delta: dict[str, Any] = {"role": "assistant"} if piece_index == 0 else {}
                if piece_index < len(pieces):
                    delta["content"] = pieces[piece_index]
                payload: dict[str, Any] = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body["model"],
                    "choices": [{"index": index, "delta": delta, "finish_reason": None}],
                }
            else:
                payload = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": body["model"],
                    "choices": [{"text": pieces[piece_index], "index": index, "finish_reason": None}],
                }
            if continuous:
                done = min(piece_index + 1, usage["completion_tokens"])
                payload["usage"] = {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": done,
                    "total_tokens": usage["prompt_tokens"] + done,
                }
            return payload

        frames: list[tuple[str, Optional[str]]] = []
        emit_count = 2 if self.stream_mid_error else len(pieces)
        for piece_index in range(emit_count):
            for choice_index in range(n):
                frames.append((json.dumps(chunk(choice_index, piece_index)), None))

        if self.stream_mid_error:
            # §5.6: an upstream failure mid-stream surfaces as an SSE error
            # event, never a silent truncation (relay.rs error frame shape).
            error_payload = json.dumps(
                {
                    "error": {
                        "message": "upstream stream failed: engine connection reset",
                        "type": "upstream_error",
                        "code": 502,
                    }
                }
            )
            frames.append((error_payload, "error"))
            return frames

        for choice_index in range(n):
            finish: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk" if endpoint == "chat" else "text_completion",
                "created": created,
                "model": body["model"],
                "choices": [
                    (
                        {"index": choice_index, "delta": {}, "finish_reason": "stop"}
                        if endpoint == "chat"
                        else {"text": "", "index": choice_index, "finish_reason": "stop"}
                    )
                ],
            }
            if continuous:
                finish["usage"] = usage
            frames.append((json.dumps(finish), None))

        if include_usage:
            usage_frame: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk" if endpoint == "chat" else "text_completion",
                "created": created,
                "model": body["model"],
                "choices": [],
                "usage": usage,
            }
            frames.append((json.dumps(usage_frame), None))
        return frames

    def _encode_frame(self, data: str, event: Optional[str]) -> bytes:
        lines = ""
        if event is not None:
            lines += f"event: {event}\n"
        if self.stream_framing == "multiline":
            # §5.6 hardening: the relay (and the SDK) must tolerate multi-line
            # data: fields and comment keepalives. Split the JSON at its first
            # top-level comma (inside the object head — never inside a string).
            split_at = data.find(",")
            assert split_at != -1
            lines += f"data: {data[: split_at + 1]}\ndata: {data[split_at + 1 :]}\n\n"
        else:
            lines += f"data: {data}\n\n"
        return lines.encode()

    def _sse_stream(self, body: dict[str, Any], endpoint: str, record: RecordedRequest) -> Iterator[bytes]:
        """The byte-level SSE generator. Lazy per frame so a client that stops
        consuming is observable (frames pulled < emitted, finalization on close)."""
        frames = self._sse_frames(body, endpoint)
        try:
            if self.stream_framing == "multiline":
                yield b": gateway-keepalive\n\n"
            for data, event in frames:
                record.stream_frames_pulled += 1
                yield self._encode_frame(data, event)
                if self.stream_framing == "multiline":
                    yield b": keepalive\n\n"
            if not self.stream_mid_error:
                record.stream_frames_pulled += 1
                yield b"data: [DONE]\n\n"
        finally:
            record.stream_finalized = True


# ---------------------------------------------------------------------------
# Fixtures: mock by default; live gateway via INFERENCE_LIVE_URL/KEY
# ---------------------------------------------------------------------------

LIVE_URL = os.environ.get("INFERENCE_LIVE_URL")
LIVE_KEY = os.environ.get("INFERENCE_LIVE_KEY")


@dataclass
class Gateway:
    client: openai.OpenAI
    mock: Optional[MockGateway]
    model: str

    @property
    def is_live(self) -> bool:
        return self.mock is None


@pytest.fixture()
def gw() -> Iterator[Gateway]:
    if LIVE_URL and LIVE_KEY:
        client = openai.OpenAI(
            base_url=LIVE_URL,
            api_key=LIVE_KEY,
            max_retries=0,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        yield Gateway(client=client, mock=None, model=os.environ.get("INFERENCE_LIVE_MODEL", DEFAULT_MODEL))
        client.close()
        return
    mock = MockGateway()
    http_client = httpx.Client(transport=httpx.MockTransport(mock.handler))
    client = openai.OpenAI(
        base_url=MOCK_BASE_URL, api_key=MOCK_VALID_KEY, http_client=http_client, max_retries=0
    )
    yield Gateway(client=client, mock=mock, model=DEFAULT_MODEL)
    client.close()


def require_mock(gw: Gateway) -> MockGateway:
    """Tests needing fault injection or engine-side introspection only run
    against the mock (a live gateway cannot be fault-injected deterministically)."""
    if gw.mock is None:
        pytest.skip("requires the mock gateway (fault injection / introspection)")
    return gw.mock


def chat(gw: Gateway, **kwargs: Any):
    kwargs.setdefault("messages", [{"role": "user", "content": "Say hi."}])
    return gw.client.chat.completions.create(model=gw.model, **kwargs)


# ---------------------------------------------------------------------------
# §4.1 endpoints — happy paths
# ---------------------------------------------------------------------------

def test_sdk_version_is_the_pinned_one() -> None:
    # NFR-8: conformance pins a specific SDK version (requirements.txt).
    assert openai.__version__ == OPENAI_SDK_PIN


def test_chat_completions_non_stream_content_and_usage(gw: Gateway) -> None:
    response = chat(gw)
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    if gw.mock is not None:
        assert response.choices[0].message.content == CHAT_CONTENT
        assert response.usage.prompt_tokens == NONSTREAM_CHAT_USAGE["prompt_tokens"]
        assert response.usage.completion_tokens == NONSTREAM_CHAT_USAGE["completion_tokens"]


def test_chat_completions_streaming_chunks_and_done(gw: Gateway) -> None:
    with chat(gw, stream=True) as stream:
        chunks = list(stream)
    text = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices if choice.delta
    )
    assert text, "stream produced no content"
    # §5.6: a client that did NOT set stream_options.include_usage must see no
    # usage block on any chunk (the gateway strips it downstream).
    assert all(chunk.usage is None for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
    if gw.mock is not None:
        assert text == CHAT_CONTENT


def test_chat_completions_streaming_include_usage(gw: Gateway) -> None:
    with chat(gw, stream=True, stream_options={"include_usage": True}) as stream:
        chunks = list(stream)
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert usage_chunks, "include_usage=True but no chunk carried a usage block"
    final = usage_chunks[-1]
    assert final.choices == [], "the terminal usage chunk must be choices-less"
    assert final.usage is not None
    assert final.usage.total_tokens == final.usage.prompt_tokens + final.usage.completion_tokens
    # Intermediate content chunks carry usage only under continuous_usage_stats.
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    assert all(chunk.usage is None for chunk in content_chunks)


def test_chat_completions_streaming_continuous_usage_stats(gw: Gateway) -> None:
    # continuous_usage_stats is a vLLM extension the gateway forwards (§9.4);
    # the SDK does not type it, so it travels via extra_body.
    with chat(
        gw,
        stream=True,
        extra_body={"stream_options": {"include_usage": True, "continuous_usage_stats": True}},
    ) as stream:
        chunks = list(stream)
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == len(chunks), (
        "continuous_usage_stats: every chunk must carry running usage"
    )
    completion_counts = [chunk.usage.completion_tokens for chunk in usage_chunks if chunk.usage]
    assert completion_counts == sorted(completion_counts), "running usage must be monotonic"
    assert all((chunk.usage.prompt_tokens or 0) > 0 for chunk in usage_chunks)


def test_completions_legacy_non_stream(gw: Gateway) -> None:
    response = gw.client.completions.create(model=gw.model, prompt="Once upon a time")
    assert response.object == "text_completion"
    assert response.choices[0].text
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    if gw.mock is not None:
        assert response.choices[0].text == COMPLETION_TEXT


def test_completions_legacy_streaming(gw: Gateway) -> None:
    with gw.client.completions.create(
        model=gw.model, prompt="Once upon a time", stream=True
    ) as stream:
        chunks = list(stream)
    text = "".join(choice.text or "" for chunk in chunks for choice in chunk.choices)
    assert text
    assert all(chunk.usage is None for chunk in chunks)
    if gw.mock is not None:
        assert text == COMPLETION_TEXT


def test_models_list(gw: Gateway) -> None:
    page = gw.client.models.list()
    models = list(page)
    assert models, "catalog must not be empty"
    assert all(model.object == "model" for model in models)
    assert all(isinstance(model.id, str) and model.id for model in models)
    assert all(isinstance(model.created, int) for model in models)
    if gw.mock is not None:
        # registry.rs render_models: live-only, sorted by id, owned_by basilica.
        assert [model.id for model in models] == sorted(gw.mock.models)
        assert all(model.owned_by == "basilica" for model in models)


def test_models_retrieve(gw: Gateway) -> None:
    model = gw.client.models.retrieve(gw.model)
    assert model.id == gw.model
    assert model.object == "model"


def test_models_retrieve_unknown_is_openai_404(gw: Gateway) -> None:
    with pytest.raises(openai.NotFoundError) as excinfo:
        gw.client.models.retrieve(f"no-such-model-{uuid.uuid4().hex[:12]}")
    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.body["code"] == "model_not_found"
    assert exc.body["type"] == "invalid_request_error"
    assert exc.body["param"] == "model"


def test_success_responses_carry_server_request_id(gw: Gateway) -> None:
    # with_raw_response proves the header is on the wire for a plain call.
    raw = gw.client.chat.completions.with_raw_response.create(
        model=gw.model, messages=[{"role": "user", "content": "Say hi."}]
    )
    request_id = raw.headers.get("x-request-id")
    assert request_id, "x-request-id header missing (server-minted UUIDv7)"
    uuid.UUID(request_id)  # must parse as a UUID


# ---------------------------------------------------------------------------
# §4.3 field policy — reserved -> 400
# ---------------------------------------------------------------------------

RESERVED_ALIASES = [
    "cache_salt",
    "Cache_Salt",
    "cacheSalt",
    "CACHE-SALT",
    "cachesalt",
    "priority",
    "PRIORITY",
    "request_id",
    "requestId",
    "prompt_cache_key",
    "promptCacheKey",
    "lora",
    "lora_name",
    "loraName",
    "lora_modules",
    "adapter",
    "adapter_id",
    "adapterName",
]


@pytest.mark.parametrize("alias", RESERVED_ALIASES)
def test_reserved_fields_rejected_400(gw: Gateway, alias: str) -> None:
    """§4.3 class (a): reserved server-controlled fields are rejected 400 with
    an OpenAI-style body naming the field — case/separator aliases included."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, extra_body={alias: "evil"})
    exc = excinfo.value
    assert exc.status_code == 400
    assert exc.body["type"] == "invalid_request_error"
    assert exc.body["code"] == "reserved_field"
    assert alias in exc.body["message"], "the server message must name the field"
    assert alias in str(exc), "the SDK exception must surface the server message"


def test_reserved_field_nested_in_unknown_value_rejected(gw: Gateway) -> None:
    """The gateway scans values of unknown fields for smuggled reserved names
    (the extra_body nesting shape) and reports the dotted path."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, extra_body={"vendor_ext": {"cache_salt": "evil"}})
    exc = excinfo.value
    assert exc.body["code"] == "reserved_field"
    assert "vendor_ext.cache_salt" in exc.body["message"]


def test_reserved_field_deeply_nested_in_arrays_rejected(gw: Gateway) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, extra_body={"vendor_ext": {"inner": [{"list": [1, {"CacheSalt": "evil"}]}]}})
    exc = excinfo.value
    assert "vendor_ext.inner[0].list[1].CacheSalt" in exc.body["message"]


def test_reserved_field_inside_stream_options_rejected(gw: Gateway) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, stream=True, extra_body={"stream_options": {"priority": 0}})
    exc = excinfo.value
    assert exc.body["code"] == "reserved_field"
    assert "stream_options.priority" in exc.body["message"]


def test_reserved_names_inside_allowlisted_fields_are_user_content(gw: Gateway) -> None:
    """A tool's JSON Schema may legitimately declare a `cache_salt` property;
    message text may mention it. Contents of allowlisted fields are user
    content, not request-envelope fields — they pass through."""
    response = chat(
        gw,
        messages=[
            {"role": "user", "content": "What does cache_salt do? Use set_cache."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "set_cache",
                    "parameters": {
                        "type": "object",
                        "properties": {"cache_salt": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert response.choices[0].message.content


# ---------------------------------------------------------------------------
# §4.3 field policy — unknown-but-unreserved -> drop + header
# ---------------------------------------------------------------------------

def test_unknown_fields_dropped_with_warning_header(gw: Gateway) -> None:
    """§4.3 class (b): unknown fields keep newer SDKs working — the request
    succeeds, the field never reaches the engine, and the warning header
    names what was dropped."""
    raw = gw.client.chat.completions.with_raw_response.create(
        model=gw.model,
        messages=[{"role": "user", "content": "Say hi."}],
        extra_body={"future_sdk_field": "x", "vendor_ext": {"nested": 1}},
    )
    parsed = raw.parse()
    assert parsed.choices[0].message.content, "unknown fields must not break the request"
    header = raw.headers.get(IGNORED_FIELDS_HEADER)
    assert header is not None, "dropped fields must be named in the warning header"
    assert "future_sdk_field" in header
    assert "vendor_ext" in header
    if gw.mock is not None:
        record = gw.mock.last_request
        # The mock saw the fields arrive on the wire ...
        assert record.raw_body["future_sdk_field"] == "x"
        assert record.raw_body["vendor_ext"] == {"nested": 1}
        # ... and dropped them from the engine request.
        assert record.engine_body is not None
        assert "future_sdk_field" not in record.engine_body
        assert "vendor_ext" not in record.engine_body
        assert record.ignored_fields == ["future_sdk_field", "vendor_ext"]


def test_no_ignored_fields_header_on_clean_request(gw: Gateway) -> None:
    raw = gw.client.chat.completions.with_raw_response.create(
        model=gw.model, messages=[{"role": "user", "content": "Say hi."}]
    )
    assert IGNORED_FIELDS_HEADER not in raw.headers


def test_allowlisted_but_wrong_endpoint_field_is_dropped_not_forwarded(gw: Gateway) -> None:
    """`prompt` on chat completions is allowlisted-but-undeclared: the gateway
    drops it and names it, it is never forwarded to the engine."""
    mock = require_mock(gw)
    raw = gw.client.chat.completions.with_raw_response.create(
        model=gw.model,
        messages=[{"role": "user", "content": "Say hi."}],
        extra_body={"prompt": "smuggled prompt"},
    )
    assert raw.parse().choices[0].message.content
    assert "prompt" in (raw.headers.get(IGNORED_FIELDS_HEADER) or "")
    assert mock.last_request.engine_body is not None
    assert "prompt" not in mock.last_request.engine_body


def test_stream_options_unknown_inner_key_dropped(gw: Gateway) -> None:
    mock = require_mock(gw)
    with chat(
        gw,
        stream=True,
        extra_body={"stream_options": {"include_usage": True, "future_opt": 1}},
    ) as stream:
        chunks = list(stream)
    assert chunks
    assert mock.last_request.ignored_fields == ["stream_options.future_opt"]
    # The engine copy keeps only the two known booleans.
    assert mock.last_request.engine_body is not None
    assert mock.last_request.engine_body["stream_options"] == {"include_usage": True}


def test_allowlisted_fields_pass_through_value_identical(gw: Gateway) -> None:
    """§20.2 pass-through matrix: allowlisted fields reach the engine request
    value-identical (opaque for stop/response_format/tools/tool_choice/logit_bias)."""
    mock = require_mock(gw)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    chat(
        gw,
        temperature=0.5,
        top_p=0.9,
        seed=42,
        max_tokens=64,
        stop=["."],
        logit_bias={"42": -1},
        frequency_penalty=0.1,
        presence_penalty=-0.1,
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice="auto",
        logprobs=True,
        extra_body={"repetition_penalty": 1.1},
    )
    engine = mock.last_request.engine_body
    assert engine is not None
    assert engine["temperature"] == 0.5
    assert engine["top_p"] == 0.9
    assert engine["seed"] == 42
    assert engine["max_tokens"] == 64
    assert engine["stop"] == ["."]
    assert engine["logit_bias"] == {"42": -1}
    assert engine["frequency_penalty"] == 0.1
    assert engine["presence_penalty"] == -0.1
    assert engine["response_format"] == {"type": "json_object"}
    assert engine["tools"] == tools
    assert engine["tool_choice"] == "auto"
    assert engine["logprobs"] is True
    assert engine["repetition_penalty"] == 1.1
    # Nothing beyond the allowlist leaks: engine keys are exactly the declared set.
    assert set(engine) <= set(ALLOWLISTED_FIELDS)


# ---------------------------------------------------------------------------
# §9.2 fan-out rules
# ---------------------------------------------------------------------------

def test_best_of_less_than_n_rejected(gw: Gateway) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, n=2, extra_body={"best_of": 1})
    exc = excinfo.value
    assert exc.body["code"] == "best_of_less_than_n"
    assert "`best_of` (1) must be greater than or equal to `n` (2)" in exc.body["message"]


def test_best_of_above_cap_rejected(gw: Gateway) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, extra_body={"best_of": DEFAULT_MAX_N + 1})
    exc = excinfo.value
    assert exc.body["code"] == "fanout_cap_exceeded"
    assert "best_of" in exc.body["message"]


def test_n_above_cap_rejected(gw: Gateway) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, n=DEFAULT_MAX_N + 1)
    exc = excinfo.value
    assert exc.body["code"] == "fanout_cap_exceeded"
    assert "`n`" in exc.body["message"]


def test_best_of_cannot_stream_rejected(gw: Gateway) -> None:
    """Explicit best_of > 1 forces the non-streaming path (§9.2): the engine
    must finish all candidates before ranking."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        chat(gw, stream=True, extra_body={"best_of": 2})
    exc = excinfo.value
    assert exc.body["code"] == "best_of_cannot_stream"
    assert "cannot be used with `stream: true`" in exc.body["message"]


def test_n_greater_than_one_streams_fine(gw: Gateway) -> None:
    """An unset best_of defaults to n, so n > 1 neither invalidates nor blocks
    streaming (§9.2 corrected semantics)."""
    with chat(gw, stream=True, n=2) as stream:
        chunks = list(stream)
    indices = {choice.index for chunk in chunks for choice in chunk.choices}
    assert indices == {0, 1}


# ---------------------------------------------------------------------------
# §4.3 error shapes surfaced through the SDK
# ---------------------------------------------------------------------------

def test_401_bad_key_raises_authentication_error(gw: Gateway) -> None:
    """Runs in both modes: a bad credential against mock OR live gateway."""
    if gw.is_live:
        bad_client = openai.OpenAI(
            base_url=LIVE_URL, api_key="basilica_definitely_invalid_key", max_retries=0
        )
    else:
        assert gw.mock is not None
        bad_client = openai.OpenAI(
            base_url=MOCK_BASE_URL,
            api_key="basilica_definitely_invalid_key",
            http_client=httpx.Client(transport=httpx.MockTransport(gw.mock.handler)),
            max_retries=0,
        )
    try:
        with pytest.raises(openai.AuthenticationError) as excinfo:
            bad_client.chat.completions.create(
                model=gw.model, messages=[{"role": "user", "content": "hi"}]
            )
        exc = excinfo.value
        assert exc.status_code == 401
        assert exc.body["type"] == "authentication_error"
        assert exc.body["code"] == "invalid_api_key"
    finally:
        bad_client.close()


def test_429_rate_limit_surfaces_retry_after_and_cap(gw: Gateway) -> None:
    """§4.3: 429 names the tripped cap (rpm|tpm|concurrency|budget) and carries
    Retry-After; the SDK raises RateLimitError with both reachable."""
    mock = require_mock(gw)
    mock.quota_trip = ("rpm", 101, 100, 7)
    with pytest.raises(openai.RateLimitError) as excinfo:
        chat(gw)
    exc = excinfo.value
    assert exc.status_code == 429
    assert exc.response.headers["retry-after"] == "7"
    assert exc.body["type"] == "rate_limit_error"
    assert exc.body["code"] == "quota_exceeded"
    assert exc.body["cap"] == "rpm", "the tripped cap must be surfaced from the body"
    assert "rpm cap tripped" in exc.body["message"]


def test_429_concurrency_cap_supplies_default_retry_after(gw: Gateway) -> None:
    """The concurrency cap has no store-computed reset hint; the gateway
    supplies one (admission.rs CONCURRENCY_RETRY_AFTER_SECS)."""
    mock = require_mock(gw)
    mock.quota_trip = ("concurrency", 9, 8, None)
    with pytest.raises(openai.RateLimitError) as excinfo:
        chat(gw, stream=True)
    exc = excinfo.value
    assert exc.body["cap"] == "concurrency"
    assert int(exc.response.headers["retry-after"]) >= 1


def test_402_insufficient_balance_surfaces_as_status_error(gw: Gateway) -> None:
    """402 has no dedicated SDK class; it must surface as APIStatusError with
    the billing_error body intact."""
    mock = require_mock(gw)
    mock.balance_reject = True
    with pytest.raises(openai.APIStatusError) as excinfo:
        chat(gw)
    exc = excinfo.value
    assert type(exc) is openai.APIStatusError, "402 must not masquerade as another class"
    assert exc.status_code == 402
    assert exc.body["type"] == "billing_error"
    assert exc.body["code"] == "insufficient_balance"
    assert "insufficient balance" in exc.body["message"]


def test_503_pool_unavailable_carries_retry_after(gw: Gateway) -> None:
    mock = require_mock(gw)
    mock.pool_empty = True
    with pytest.raises(openai.InternalServerError) as excinfo:
        chat(gw)
    exc = excinfo.value
    assert exc.status_code == 503
    assert exc.response.headers["retry-after"] == str(POOL_EMPTY_RETRY_AFTER_SECS)
    assert exc.body["type"] == "server_error"
    assert exc.body["code"] == "pool_unavailable"


def test_404_unknown_model_on_invoke(gw: Gateway) -> None:
    """Unknown model on the invoke path: the same registry 404 shape as
    GET /v1/models/{id} (never distinguishable from unowned/retired)."""
    with pytest.raises(openai.NotFoundError) as excinfo:
        gw.client.chat.completions.create(
            model=f"no-such-model-{uuid.uuid4().hex[:12]}",
            messages=[{"role": "user", "content": "hi"}],
        )
    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.body["code"] == "model_not_found"
    assert exc.body["param"] == "model"


# ---------------------------------------------------------------------------
# §5.6 SSE wire behavior
# ---------------------------------------------------------------------------

def test_stream_response_content_type_headers(gw: Gateway) -> None:
    with gw.client.chat.completions.with_streaming_response.create(
        model=gw.model, messages=[{"role": "user", "content": "hi"}], stream=True
    ) as response:
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("text/event-stream")
        assert response.headers.get("cache-control") == "no-cache"
        chunks = list(response.parse())
    assert chunks


def test_sse_multiline_data_frames_and_keepalive_comments(gw: Gateway) -> None:
    """SSE framing edge cases: multi-line `data:` fields (joined per the SSE
    spec) and `:` keepalive comment frames must not disturb the SDK parser."""
    mock = require_mock(gw)
    mock.stream_framing = "multiline"
    with chat(gw, stream=True) as stream:
        chunks = list(stream)
    text = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices if choice.delta
    )
    assert text == CHAT_CONTENT
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_sse_midstream_error_event_raises_api_error(gw: Gateway) -> None:
    """§5.6: an upstream failure mid-stream surfaces as an SSE error event,
    never a silent truncation — the SDK raises APIError with the server
    message, after delivering the frames that preceded it."""
    mock = require_mock(gw)
    mock.stream_mid_error = True
    received: list[str] = []
    with pytest.raises(openai.APIError) as excinfo:
        with chat(gw, stream=True) as stream:
            for chunk in stream:
                received.extend(
                    choice.delta.content or "" for choice in chunk.choices if choice.delta
                )
    exc = excinfo.value
    assert "upstream stream failed" in str(exc)
    assert exc.body["type"] == "upstream_error"
    assert received == CHAT_PIECES[:2], "frames before the error must be delivered"


def test_stream_abort_client_disconnect_is_clean(gw: Gateway) -> None:
    """Client disconnect mid-stream: the SDK closes the response without an
    exception, and the producer is finalized instead of hanging or leaking.
    (The gateway-side propagation — engine cancel + KV free — is pinned by the
    Rust relay tests; here we pin the client-visible half.)"""
    mock = require_mock(gw)
    mock.stream_pieces = 50
    stream = chat(gw, stream=True)
    received: list[str] = []
    with stream:
        iterator = iter(stream)
        for _ in range(3):
            chunk = next(iterator)
            received.extend(choice.delta.content or "" for choice in chunk.choices if choice.delta)
    assert stream.response.is_closed, "closing the SDK stream must close the HTTP response"
    assert received, "content received before the abort must be intact"

    del stream, iterator
    gc.collect()
    record = mock.last_request
    assert record.stream_finalized, "the producer must be finalized after client disconnect"
    total_frames = 50 + 1 + 1  # pieces + finish + [DONE]
    assert record.stream_frames_pulled < total_frames, (
        "the client really stopped mid-stream (not a full drain)"
    )
