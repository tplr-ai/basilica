# FR-1 conformance — OpenAI SDK vs. the managed-inference gateway

Pins **FR-1** of the Managed Inference spec
(`basilica-backend/docs/architecture/MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE.md`):
*unmodified OpenAI SDK clients must work against the gateway.* The suite pins
the official `openai` Python SDK (NFR-8: one specific version, see
`requirements.txt`) and exercises the wire contract — the §20.2 field-policy
matrix, the §4.3 error shapes, and the §5.6/§9.4 streaming/usage behavior —
against a **mock gateway** that mirrors the Rust implementation byte-for-byte.
The live run against a real gateway is Phase 10 (see *Live mode* below).

Contract sources mirrored (drift here is a bug — update both sides together):

- `basilica-backend/crates/basilica-inference/src/openai.rs` — allowlist,
  reserved-field set + alias normalization, fan-out rules, drop/header policy
- `basilica-backend/crates/basilica-inference/src/admission.rs` — 400/402/404/
  429/503 error bodies, `Retry-After`, admission-stage ordering
- `basilica-backend/crates/basilica-inference/src/auth.rs` — 401 shape
- `basilica-backend/crates/basilica-inference/src/registry.rs` — `/v1/models`
  shapes and the unknown/retired 404
- `basilica-backend/crates/basilica-inference/src/relay.rs` — SSE framing,
  `event: error` mid-stream shape

## Running (mock mode, the default)

From `basilica/crates/basilica-sdk-python/`:

```bash
python3 -m venv conformance/.venv-conformance
conformance/.venv-conformance/bin/pip install -r conformance/requirements.txt
conformance/.venv-conformance/bin/python -m pytest conformance/ -q
```

The venv is self-contained (`conformance/.venv-conformance`, git-ignored) and
never touches system site-packages. No network, no services: the mock gateway
is an in-process `httpx.MockTransport` handler, so the whole suite runs in
about a second.

## Live mode (Phase 10)

Point the same unmodified SDK client at a real gateway:

```bash
export INFERENCE_LIVE_URL="https://inference.basilica.ai/v1"   # base URL incl. /v1
export INFERENCE_LIVE_KEY="basilica_..."                        # never committed
# optional: model name when the live catalog differs from the spec's example
export INFERENCE_LIVE_MODEL="llama-3.1-70b-instruct"

conformance/.venv-conformance/bin/python -m pytest conformance/ -q
```

When both env vars are set, the `gw` fixture builds a real `openai.OpenAI`
client against `$INFERENCE_LIVE_URL` instead of the mock. Every client-visible
assertion still runs (field policy, error mapping, streaming, usage rules,
models). Tests that need fault injection (429/402/503 on demand), synthetic
SSE framing, or engine-side introspection call `require_mock()` and **skip**
in live mode — a live gateway cannot be tripped deterministically and must not
be probed for its internals. No credentials live in the repo; the mock key is
a constant that only the mock accepts.

## Coverage matrix (FR-1)

| Spec | Behavior | Tests |
|---|---|---|
| §4.1 | `POST /v1/chat/completions` non-stream: content + `usage` totals | `test_chat_completions_non_stream_content_and_usage` |
| §4.1/§5.6 | chat streaming: `data:` chunks, `[DONE]`, no usage unless opted in | `test_chat_completions_streaming_chunks_and_done` |
| §5.6 | `stream_options.include_usage`: terminal choices-less usage chunk; absent otherwise | `test_chat_completions_streaming_include_usage` |
| §9.4 | `continuous_usage_stats`: running usage on every chunk, monotonic | `test_chat_completions_streaming_continuous_usage_stats` |
| §4.1 | `POST /v1/completions` legacy, non-stream + stream | `test_completions_legacy_non_stream`, `test_completions_legacy_streaming` |
| §4.1 | `GET /v1/models` list shape (sorted, live-only, `owned_by`) | `test_models_list` |
| §4.1/§4.3 | `GET /v1/models/{id}` retrieve + unknown → `NotFoundError` (`model_not_found`) | `test_models_retrieve`, `test_models_retrieve_unknown_is_openai_404` |
| §5.2 | `x-request-id` server-minted UUID on success responses | `test_success_responses_carry_server_request_id` |
| §4.3 (a) | reserved fields → 400 `reserved_field`, all case/separator aliases (`cache_salt`, `priority`, `request_id`, `prompt_cache_key`, `lora*`/`adapter*` families) | `test_reserved_fields_rejected_400` (18 aliases) |
| §4.3 (a) | smuggling via unknown-field nesting + arrays, dotted path in message | `test_reserved_field_nested_in_unknown_value_rejected`, `test_reserved_field_deeply_nested_in_arrays_rejected` |
| §4.3 (a) | reserved key inside `stream_options` | `test_reserved_field_inside_stream_options_rejected` |
| §4.3 | reserved names *inside allowlisted fields* (tool schema, message text) are user content — pass | `test_reserved_names_inside_allowlisted_fields_are_user_content` |
| §4.3 (b) | unknown fields → dropped + `x-basilica-ignored-fields`; engine never sees them | `test_unknown_fields_dropped_with_warning_header` |
| §4.3 (b) | clean request → no warning header | `test_no_ignored_fields_header_on_clean_request` |
| §4.3 (b) | allowlisted-but-undeclared (`prompt` on chat) → drop path, not forwarded | `test_allowlisted_but_wrong_endpoint_field_is_dropped_not_forwarded` |
| §4.3 (b) | unknown `stream_options` inner key → dropped as `stream_options.<name>` | `test_stream_options_unknown_inner_key_dropped` |
| §20.2 | pass-through fields reach the engine value-identical; nothing beyond the allowlist leaks | `test_allowlisted_fields_pass_through_value_identical` |
| §9.2 | `best_of < n` → 400 `best_of_less_than_n` | `test_best_of_less_than_n_rejected` |
| §9.2 | `n` / `best_of` cap (`max_n` = 8) → 400 `fanout_cap_exceeded` | `test_n_above_cap_rejected`, `test_best_of_above_cap_rejected` |
| §9.2 | explicit `best_of > 1` + `stream: true` → 400 `best_of_cannot_stream` | `test_best_of_cannot_stream_rejected` |
| §9.2 | unset `best_of` defaults to `n`: `n > 1` streams fine | `test_n_greater_than_one_streams_fine` |
| §4.2 | 401 → `AuthenticationError`, `authentication_error`/`invalid_api_key` | `test_401_bad_key_raises_authentication_error` |
| §4.3 | 429 → `RateLimitError` + `Retry-After` header + `cap` from body (`rpm`/`concurrency`) | `test_429_rate_limit_surfaces_retry_after_and_cap`, `test_429_concurrency_cap_supplies_default_retry_after` |
| §4.3 | 402 → exact `APIStatusError`, `billing_error`/`insufficient_balance` surfaced | `test_402_insufficient_balance_surfaces_as_status_error` |
| §4.3 | 503 pool-empty → `InternalServerError` + `Retry-After: 5`, `pool_unavailable` | `test_503_pool_unavailable_carries_retry_after` |
| §4.3 | unknown model on invoke → same registry 404 shape | `test_404_unknown_model_on_invoke` |
| §5.6 | stream response headers: `text/event-stream`, `cache-control: no-cache` | `test_stream_response_content_type_headers` |
| §5.6 | SSE edge cases: multi-line `data:` frames, `:` keepalive comments | `test_sse_multiline_data_frames_and_keepalive_comments` |
| §5.6 | mid-stream upstream error → SSE `event: error` → SDK `APIError` (never silent truncation); prior frames delivered | `test_sse_midstream_error_event_raises_api_error` |
| §5.6 | client disconnect mid-stream is clean: response closed, producer finalized, no hang/leak | `test_stream_abort_client_disconnect_is_clean` |
| NFR-8 | SDK version is the pinned one | `test_sdk_version_is_the_pinned_one` |

## Notes / scope

- The mock mirrors the §5.2 admission ordering (auth → quota → balance →
  model → field policy → pool → relay) so error-class precedence matches the
  real gateway.
- Stream-abort coverage pins the client-visible half (clean close, producer
  finalization). The gateway-internal propagation — engine cancel, KV free,
  prompt-inclusive settlement — is covered by the Rust relay/metering tests
  in `basilica-inference`, not by this SDK suite.
- Transitive dependencies are resolved by pip; only the conformance-relevant
  surface is pinned (`openai`, `httpx`, `pytest`).
