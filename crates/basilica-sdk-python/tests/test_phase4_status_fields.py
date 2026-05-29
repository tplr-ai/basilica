"""
Phase 4 (UD Status Honesty) regression tests for the Python SDK.

PR #497 added `container_statuses` and `phase_progress` to the API
DeploymentResponse but the change never reached the Python bindings.
v0.31.0 shipped without them, so SDK consumers could not read either
field. This test guards the PyO3 binding + Python facade together.
"""

from unittest.mock import MagicMock

from basilica import ContainerStatusInfo, Deployment, DeploymentStatus


def _make_deployment() -> Deployment:
    return Deployment(
        client=MagicMock(),
        instance_name="phase4-test",
        url="https://phase4.deployments.basilica.ai",
        namespace="u-test",
        user_id="test-user",
        state="Active",
        created_at="2026-05-27T00:00:00Z",
        replicas_ready=1,
        replicas_desired=1,
    )


def test_parse_status_response_surfaces_container_statuses() -> None:
    deployment = _make_deployment()

    snap = MagicMock()
    snap.pod_name = "pod-0"
    snap.container_name = "main"
    snap.state = "running"
    snap.reason = None
    snap.message = None
    snap.restart_count = 0

    response = MagicMock()
    response.state = "Active"
    response.replicas.ready = 1
    response.replicas.desired = 1
    response.phase = "ready"
    response.progress = None
    response.container_statuses = [snap]
    response.phase_progress = 7

    status = deployment._parse_status_response(response)

    assert isinstance(status, DeploymentStatus)
    assert status.phase_progress == 7
    assert len(status.container_statuses) == 1
    cs = status.container_statuses[0]
    assert isinstance(cs, ContainerStatusInfo)
    assert cs.pod_name == "pod-0"
    assert cs.container_name == "main"
    assert cs.state == "running"
    assert cs.restart_count == 0
    assert cs.reason is None


def test_parse_status_response_handles_missing_phase4_fields() -> None:
    deployment = _make_deployment()

    response = MagicMock(spec=["state", "replicas", "phase", "progress"])
    response.state = "Pending"
    response.replicas.ready = 0
    response.replicas.desired = 1
    response.phase = "pending"
    response.progress = None

    status = deployment._parse_status_response(response)

    assert status.phase_progress == 0
    assert status.container_statuses == []


def test_pyo3_deployment_response_exposes_phase4_fields() -> None:
    """The compiled PyO3 module must expose the new fields on DeploymentResponse."""
    from basilica import _basilica

    assert hasattr(_basilica, "ContainerStatusSnapshot")
    cls = _basilica.DeploymentResponse
    sig = cls.__doc__ or ""
    assert "container_statuses" in sig or hasattr(cls, "container_statuses")
    assert "phase_progress" in sig or hasattr(cls, "phase_progress")
