"""basilica.rl contract tests, run against the COMPILED core transport: a
real stdlib HTTP server receives what the Rust client actually sends,
asserting the exact wire shapes the RL API's deny-unknown-fields DTOs
enforce — key casing, the nested `ref` renames, auth header, escape-hatch
passthrough, poll loops, and the core's error mapping.

Requires the built extension (maturin develop / the installed wheel);
skipped cleanly where only the pure-python tree is on the path.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

basilica = pytest.importorskip("basilica")
pytest.importorskip("basilica._basilica")


class _Recorder(BaseHTTPRequestHandler):
    requests: list = []
    responses: list = []  # (status, body-dict) popped per request

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        _Recorder.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        status, resp = (
            _Recorder.responses.pop(0) if _Recorder.responses else (200, {})
        )
        payload = json.dumps(resp).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = _handle

    def log_message(self, *_):
        pass


def _api_error(message, code="BASILICA_API_BAD_REQUEST"):
    """The basilica-api error envelope (error.rs into_response)."""
    return {"error": {"code": code, "message": message,
                      "timestamp": "2026-08-24T00:00:00Z", "retryable": False}}


@pytest.fixture()
def server():
    _Recorder.requests = []
    _Recorder.responses = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Recorder
    httpd.shutdown()
    httpd.server_close()


def rl(base):
    return basilica.BasilicaClient(base_url=base, api_key="test-key").rl


def test_create_cluster_wire_shape(server):
    base, rec = server
    rec.responses = [(200, {"name": "my-pool", "uid": "u1", "phase": "Provisioning"})]
    rl(base).create_cluster(
        name="my-pool",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        gpu_model="H100",
        trainer_gpus=4,
        rollout_gpus=4,
        idle_ttl="30m",
    )
    (r,) = rec.requests
    assert (r["method"], r["path"]) == ("POST", "/rl/clusters")
    assert r["auth"] == "Bearer test-key"
    assert r["body"] == {
        "name": "my-pool",
        "baseModel": "Qwen/Qwen2.5-7B-Instruct",
        "trainer": {"replicas": 1, "gpu": {"model": "H100", "count": 4}},
        "rollout": {"replicas": 1, "gpu": {"model": "H100", "count": 4}},
        "idleTtl": "30m",
    }


def test_create_job_full_wire_shape(server):
    base, rec = server
    rec.responses = [(200, {"name": "j1", "uid": "u2", "phase": "Pending"})]
    rl(base).create_job(
        cluster="my-pool",
        max_steps=50,
        reward_name="my-reward",
        reward_source="def reward(prompt, completion, **ctx):\n    return 1.0\n",
        judge=True,
        dataset_name="my-data",
        dataset_repo="openai/gsm8k",
        dataset_config="main",
        dataset_split="train",
        prompt_column="question",
        answer_column="answer",
    )
    (r,) = rec.requests
    assert (r["method"], r["path"]) == ("POST", "/rl/jobs")
    body = r["body"]
    assert body["clusterRef"] == "my-pool"
    assert body["maxSteps"] == 50
    assert body["algorithm"] == "grpo"
    # the serde renames: nested identity fields are `ref`
    assert body["reward"]["ref"] == "user:my-reward"
    assert body["reward"]["judge"] == {}
    assert body["dataset"]["ref"] == "user:my-data"
    assert body["dataset"]["hf"] == {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "promptColumn": "question",
        "answerColumn": "answer",
    }
    # None-valued optionals must be ABSENT (deny_unknown_fields tolerates
    # absence; null in a non-Option server slot would 400)
    assert "name" not in body and "lr" not in body


def test_judge_without_custom_reward_is_a_client_error(server):
    base, _ = server
    with pytest.raises(ValueError, match="judge requires a custom reward"):
        rl(base).create_job(cluster="c", max_steps=3, judge=True)


def test_builtin_job_minimal_body(server):
    base, rec = server
    rec.responses = [(200, {"name": "j1", "uid": "u2", "phase": "Pending"})]
    rl(base).create_job(cluster="my-pool", max_steps=3)
    (r,) = rec.requests
    assert r["body"] == {"clusterRef": "my-pool", "algorithm": "grpo", "maxSteps": 3}


def test_raw_body_escape_hatch_preserves_unknown_fields(server):
    # serde-flatten catch-alls in the core DTOs: a field this SDK version
    # doesn't know must SURVIVE the typed round-trip verbatim. body= needs
    # no placeholder kwargs — it replaces the built request entirely.
    base, rec = server
    raw = {"clusterRef": "x", "algorithm": "grpo", "maxSteps": 1, "futureField": True}
    rec.responses = [(200, {"name": "j1", "uid": "u2", "phase": "Pending"})]
    rl(base).create_job(body=raw)
    (r,) = rec.requests
    assert r["body"]["futureField"] is True
    assert r["body"]["maxSteps"] == 1
    assert r["body"]["clusterRef"] == "x"


def test_orphan_kwargs_raise_instead_of_silently_dropping(server):
    # reward_source without reward_name (or dataset fields without
    # dataset_name) would otherwise silently run the BUILTIN reward/dataset
    # on a paid GPU job.
    base, rec = server
    with pytest.raises(ValueError, match="reward_name is required"):
        rl(base).create_job(cluster="c", max_steps=3, reward_source="def reward(): ...")
    with pytest.raises(ValueError, match="dataset_name is required"):
        rl(base).create_job(cluster="c", max_steps=3, dataset_repo="openai/gsm8k")
    with pytest.raises(ValueError, match="cluster and max_steps are required"):
        rl(base).create_job(reward_name="r", reward_source="...")
    assert rec.requests == []


def test_wait_job_polls_to_terminal(server):
    base, rec = server
    rec.responses = [
        (200, {"phase": "Running"}),
        (200, {"phase": "Running"}),
        (200, {"phase": "Succeeded", "artifactURI": "s3://x/uid"}),
    ]
    final = rl(base).wait_job("j1", timeout_s=30, poll_s=0.01)
    assert final["phase"] == "Succeeded"
    assert final["artifactURI"] == "s3://x/uid"
    assert len(rec.requests) == 3
    assert all(r["path"] == "/rl/jobs/j1" for r in rec.requests)


def test_wait_job_returns_failed_rather_than_raising(server):
    base, rec = server
    rec.responses = [(200, {"phase": "Failed"})]
    final = rl(base).wait_job("j2", timeout_s=5, poll_s=0.01)
    assert final["phase"] == "Failed"


def test_api_error_surfaces_server_message(server):
    # 400 + the basilica-api envelope -> the core maps BadRequest ->
    # PyValueError carrying the server's message verbatim.
    base, rec = server
    rec.responses = [(400, _api_error("trainer fleet totals 7 GPUs; the GRPO train batch ..."))]
    with pytest.raises(ValueError, match="totals 7 GPUs"):
        rl(base).create_cluster(
            base_model="Qwen/Qwen2.5-7B-Instruct", gpu_model="H100", trainer_gpus=7
        )


def test_invalid_request_json_rejected_client_side(server):
    # the binding serde-validates BEFORE any HTTP: a body that cannot parse
    # into the typed DTO raises without touching the network.
    base, rec = server
    with pytest.raises(ValueError, match="invalid RL job request"):
        rl(base).create_job(cluster="c", max_steps=1, body={"maxSteps": "not-a-number"})
    assert rec.requests == []


def test_manifest_posts_verbatim(server):
    base, rec = server
    doc = {"cluster": {"baseModel": "m"}, "job": {"maxSteps": 3}}
    rl(base).submit_manifest(doc)
    (r,) = rec.requests
    assert (r["method"], r["path"], r["body"]) == ("POST", "/rl/manifest", doc)


def test_wait_cluster_polls_to_ready(server):
    base, rec = server
    rec.responses = [
        (200, {"name": "c1", "uid": "u1", "phase": "Provisioning"}),
        (200, {"name": "c1", "uid": "u1", "phase": "Warming"}),
        (200, {"name": "c1", "uid": "u1", "phase": "Ready"}),
    ]
    final = rl(base).wait_cluster("c1", timeout_s=30, poll_s=0.01)
    assert final["phase"] == "Ready"
    assert len(rec.requests) == 3
    assert all(r["path"] == "/rl/clusters/c1" for r in rec.requests)


def test_wait_cluster_raises_immediately_on_terminating(server):
    # Terminating can never become Ready: waiting out the full timeout would
    # hide the failure. (Degraded, by contrast, keeps polling — it recovers.)
    base, rec = server
    rec.responses = [(200, {"name": "c1", "uid": "u1", "phase": "Terminating"})]
    with pytest.raises(RuntimeError, match="entered Terminating"):
        rl(base).wait_cluster("c1", timeout_s=30, poll_s=0.01)
    assert len(rec.requests) == 1


def test_wait_job_survives_transient_poll_errors(server):
    # A single LB 502 mid-wait must not abort a multi-hour poll; only
    # CONSECUTIVE failures beyond the budget give up.
    base, rec = server
    rec.responses = [
        (200, {"phase": "Running"}),
        (500, _api_error("upstream blip", code="BASILICA_API_INTERNAL_ERROR")),
        (500, _api_error("upstream blip", code="BASILICA_API_INTERNAL_ERROR")),
        (200, {"phase": "Succeeded"}),
    ]
    final = rl(base).wait_job("j1", timeout_s=30, poll_s=0.01)
    assert final["phase"] == "Succeeded"
    assert len(rec.requests) == 4


def test_invalid_name_rejected_client_side(server):
    # names become URL path segments + k8s object names: DNS-1035 is checked
    # in the core before any HTTP
    base, rec = server
    with pytest.raises(ValueError, match="DNS-1035"):
        rl(base).get_job("Bad/Name")
    with pytest.raises(ValueError, match="DNS-1035"):
        rl(base).get_cluster("-leading-dash")
    assert rec.requests == []


# ----- delete surface error mapping (#561, the Talos LOW from #559) --------


def test_delete_cluster_refusal_maps_to_valueerror_naming_the_job(server):
    """The active-job 400 refusal surfaces as ValueError carrying the
    server's message verbatim — including the blocking job's name, which is
    the part callers act on (delete that job first)."""
    base, rec = server
    rec.responses = [
        (400, _api_error("cluster my-pool has an active job (my-job); delete the job first"))
    ]
    with pytest.raises(ValueError, match=r"my-job"):
        rl(base).delete_cluster("my-pool")
    (r,) = rec.requests
    assert (r["method"], r["path"]) == ("DELETE", "/rl/clusters/my-pool")
    assert r["auth"] == "Bearer test-key"


def test_delete_job_wire_shape(server):
    base, rec = server
    rec.responses = [(200, {"name": "my-job"})]
    assert rl(base).delete_job("my-job") == {"name": "my-job"}
    (r,) = rec.requests
    assert (r["method"], r["path"]) == ("DELETE", "/rl/jobs/my-job")
    assert r["auth"] == "Bearer test-key"


def test_delete_cluster_not_found_maps_per_the_error_table(server):
    """404 maps to the not-found exception class (pinned to the observed
    live behavior of the published wheel)."""
    base, rec = server
    rec.responses = [
        (404, _api_error("rl cluster not found: absent", code="BASILICA_API_NOT_FOUND"))
    ]
    with pytest.raises(KeyError, match=r"not found"):
        rl(base).delete_cluster("absent")


def test_create_cluster_relay_wire_shape(server):
    # #1578: the relay dict passes through verbatim (already wire-shaped),
    # and the response's effectivePrefix reaches the caller.
    base, rec = server
    rec.responses = [(200, {
        "name": "byo-pool", "uid": "u1", "phase": "Provisioning",
        "effectivePrefix": "teams/rl/u1/",
    })]
    out = rl(base).create_cluster(
        name="byo-pool",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        gpu_model="H100",
        relay={
            "mode": "byo",
            "endpoint": "https://acc.r2.cloudflarestorage.com",
            "bucket": "my-weights",
            "basePrefix": "teams/rl/",
            "accessKeyId": "AK",
            "secretAccessKey": "SK",
        },
    )
    (r,) = rec.requests
    assert r["body"]["relay"] == {
        "mode": "byo",
        "endpoint": "https://acc.r2.cloudflarestorage.com",
        "bucket": "my-weights",
        "basePrefix": "teams/rl/",
        "accessKeyId": "AK",
        "secretAccessKey": "SK",
    }
    assert out["effectivePrefix"] == "teams/rl/u1/"


def test_create_cluster_without_relay_is_wire_identical(server):
    # Omitting relay must serialize NO relay key at all (deny_unknown_fields
    # servers predating BYO would 400 on an unexpected null).
    base, rec = server
    rec.responses = [(200, {"name": "p", "uid": "u", "phase": "Provisioning"})]
    rl(base).create_cluster(base_model="m/m", gpu_model="H100")
    (r,) = rec.requests
    assert "relay" not in r["body"]


def test_rotate_credentials_wire_shape(server):
    base, rec = server
    rec.responses = [(200, {"name": "byo-pool",
                            "rotatedAt": "2026-08-31T12:00:00+00:00"})]
    out = rl(base).rotate_credentials(
        "byo-pool", access_key_id="NEWAK", secret_access_key="NEWSK"
    )
    (r,) = rec.requests
    assert (r["method"], r["path"]) == ("POST", "/rl/clusters/byo-pool/credentials")
    assert r["body"] == {"accessKeyId": "NEWAK", "secretAccessKey": "NEWSK"}
    assert out["rotatedAt"].startswith("2026-08-31")


def test_rotate_credentials_maps_errors(server):
    # A platform-mode cluster refuses rotation with an actionable 400 ->
    # ValueError carrying the server message.
    base, rec = server
    rec.responses = [(400, _api_error(
        "RelayModeInvalid: cluster plat uses platform-managed storage"))]
    with pytest.raises(ValueError, match="RelayModeInvalid"):
        rl(base).rotate_credentials("plat", access_key_id="A", secret_access_key="B")
