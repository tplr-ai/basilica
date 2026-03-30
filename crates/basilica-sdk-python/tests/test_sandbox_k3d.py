"""SDK integration tests against a live K3d cluster.

These tests require a running K3d cluster with basilica-api (dev mode) and the
sandbox operator deployed. Run with:

    pytest tests/test_sandbox_k3d.py -v -m k3d

Or via the runner script:

    scripts/localtest/sandbox-k3d-e2e.sh sdk-test
"""

import os
import signal
import subprocess
import time

import pytest

from basilica import BasilicaClient
from basilica.sandbox import (
    CreateSandboxRequest,
    Sandbox,
    SandboxClient,
    SandboxDetail,
    SandboxSummary,
)

API_URL = os.environ.get("BASILICA_API_URL", "http://localhost:18082")
SANDBOX_IMAGE = os.environ.get(
    "SANDBOX_IMAGE",
    "k3d-basilica-registry:5050/basilica-exec-agent:latest",
)
NAMESPACE = "u-test-user"

# Mark all tests in this file as k3d (skipped by default)
pytestmark = pytest.mark.k3d


@pytest.fixture(scope="module")
def client():
    """Create a BasilicaClient connected to the K3d test API."""
    return BasilicaClient(base_url=API_URL, api_key="test-token")


@pytest.fixture(scope="module")
def sandbox_client(client):
    """Create a SandboxClient."""
    return SandboxClient(client._client)


@pytest.fixture(scope="module")
def sandbox_lifecycle(sandbox_client):
    """Create a sandbox, wait for Running, set up port-forward, yield, cleanup.

    This fixture is module-scoped so sandbox setup/teardown is shared across tests.
    """
    # Create sandbox
    sandbox = sandbox_client.create(
        image=SANDBOX_IMAGE,
        cpu="1",
        memory="512Mi",
        ttl_seconds=600,
    )
    sandbox_id = sandbox.sandbox_id
    print(f"\nCreated sandbox: {sandbox_id} (domain: {sandbox.domain})")

    # Wait for Running
    max_wait = 120
    start = time.time()
    while time.time() - start < max_wait:
        detail = sandbox_client.get(sandbox_id)
        if detail.status == "Running":
            break
        time.sleep(2)
    else:
        pytest.fail(f"Sandbox {sandbox_id} did not reach Running state in {max_wait}s")

    print(f"Sandbox {sandbox_id} is Running")

    # Set up port-forward
    pod_name = f"sandbox-{sandbox_id}"

    # Wait for pod ready
    subprocess.run(
        [
            "kubectl", "wait", "--for=condition=Ready",
            f"pod/{pod_name}", "-n", NAMESPACE, "--timeout=90s",
        ],
        check=True,
        capture_output=True,
    )

    local_port = 20000 + (os.getpid() % 10000)
    pf_proc = subprocess.Popen(
        [
            "kubectl", "port-forward", "-n", NAMESPACE,
            f"pod/{pod_name}", f"{local_port}:9999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    assert pf_proc.poll() is None, "Port-forward process died"

    data_plane_url = f"http://localhost:{local_port}"
    print(f"Data-plane URL: {data_plane_url}")

    # Override the sandbox's data-plane URL
    sandbox.with_data_plane_url(data_plane_url)

    yield {
        "sandbox": sandbox,
        "sandbox_id": sandbox_id,
        "sandbox_client": sandbox_client,
        "data_plane_url": data_plane_url,
    }

    # Cleanup
    pf_proc.terminate()
    try:
        pf_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pf_proc.kill()

    try:
        sandbox_client.delete(sandbox_id)
        print(f"\nDeleted sandbox: {sandbox_id}")
    except Exception as e:
        print(f"\nFailed to delete sandbox {sandbox_id}: {e}")


# ============================================================================
# Control-plane tests
# ============================================================================


class TestControlPlane:
    """Tests for sandbox control-plane operations (create, list, get, delete)."""

    def test_create_sandbox(self, sandbox_lifecycle):
        """Create sandbox with valid image, verify response fields."""
        sandbox = sandbox_lifecycle["sandbox"]
        assert sandbox.sandbox_id, "sandbox_id should not be empty"
        assert sandbox.domain, "domain should not be empty"
        assert sandbox.exec_agent_secret, "exec_agent_secret should be present"
        assert sandbox.status in ("Pending", "Running"), f"unexpected status: {sandbox.status}"

    def test_create_sandbox_invalid_image(self, sandbox_client):
        """Create sandbox with invalid image, verify error."""
        with pytest.raises(Exception) as exc_info:
            sandbox_client.create(
                image="nonexistent-registry.invalid/no-such-image:latest",
            )
        print(f"Expected error for invalid image: {exc_info.value}")

    def test_list_sandboxes(self, sandbox_lifecycle):
        """List sandboxes, verify created sandbox appears."""
        sandbox_client = sandbox_lifecycle["sandbox_client"]
        sandbox_id = sandbox_lifecycle["sandbox_id"]
        sandboxes = sandbox_client.list()
        assert isinstance(sandboxes, list)
        ids = [s.sandbox_id for s in sandboxes]
        assert sandbox_id in ids, f"Sandbox {sandbox_id} should appear in list, got: {ids}"

    def test_get_sandbox(self, sandbox_lifecycle):
        """Get sandbox by ID, verify detail fields."""
        sandbox_client = sandbox_lifecycle["sandbox_client"]
        sandbox_id = sandbox_lifecycle["sandbox_id"]
        detail = sandbox_client.get(sandbox_id)
        assert isinstance(detail, SandboxDetail)
        assert detail.sandbox_id == sandbox_id
        assert detail.image == SANDBOX_IMAGE
        assert detail.cpu
        assert detail.memory
        assert detail.status == "Running"

    def test_get_nonexistent_sandbox(self, sandbox_client):
        """Get nonexistent sandbox, verify error."""
        with pytest.raises(Exception) as exc_info:
            sandbox_client.get("sb-nonexistent-00000000")
        print(f"Expected error for nonexistent: {exc_info.value}")


# ============================================================================
# Data-plane tests
# ============================================================================


class TestDataPlane:
    """Tests for sandbox data-plane operations (exec, run, files)."""

    def test_exec_command(self, sandbox_lifecycle):
        """Exec command on running sandbox, verify stdout and exit code."""
        sandbox = sandbox_lifecycle["sandbox"]
        result = sandbox.exec(["echo", "hello-python-sdk"])
        assert result["exitCode"] == 0, f"exit code should be 0, got: {result['exitCode']}"
        assert "hello-python-sdk" in result["stdout"], f"stdout: {result['stdout']}"

    def test_run_code(self, sandbox_lifecycle):
        """Run code on running sandbox, verify output."""
        sandbox = sandbox_lifecycle["sandbox"]
        result = sandbox.run("print('run-output-99')")
        assert result["exitCode"] == 0, f"exit code should be 0, got: {result['exitCode']}"
        assert "run-output-99" in result["stdout"], f"stdout: {result['stdout']}"

    def test_file_write_and_read(self, sandbox_lifecycle):
        """Write file, read it back, verify content matches."""
        sandbox = sandbox_lifecycle["sandbox"]
        test_content = "Python SDK integration test\nLine 2"

        write_result = sandbox.files.write("/tmp/py-sdk-test.txt", test_content)
        assert write_result["path"] == "/tmp/py-sdk-test.txt"

        read_result = sandbox.files.read("/tmp/py-sdk-test.txt")
        assert read_result["content"] == test_content
        assert read_result["path"] == "/tmp/py-sdk-test.txt"

    def test_file_list(self, sandbox_lifecycle):
        """List files, verify written file appears."""
        sandbox = sandbox_lifecycle["sandbox"]
        # Ensure file exists
        sandbox.files.write("/tmp/py-list-test.txt", "list test")

        result = sandbox.files.list("/tmp")
        names = [f["name"] for f in result["files"]]
        assert "py-list-test.txt" in names, f"File should appear in listing, got: {names}"


# ============================================================================
# URL helpers tests
# ============================================================================


class TestURLHelpers:
    """Tests for sandbox URL helper properties."""

    def test_data_plane_url(self, sandbox_lifecycle):
        """Verify data_plane_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.data_plane_url
        assert url.startswith("https://"), f"should start with https://, got: {url}"
        assert sandbox.domain in url

    def test_ws_url(self, sandbox_lifecycle):
        """Verify ws_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.ws_url
        assert url.startswith("wss://"), f"should start with wss://, got: {url}"
        assert url.endswith("/ws"), f"should end with /ws, got: {url}"

    def test_exec_url(self, sandbox_lifecycle):
        """Verify exec_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.exec_url
        assert url.startswith("https://"), f"should start with https://, got: {url}"
        assert url.endswith("/exec"), f"should end with /exec, got: {url}"


# ============================================================================
# Deletion tests (run last)
# ============================================================================


class TestDeletion:
    """Tests for sandbox deletion. These use their own sandboxes."""

    def test_delete_sandbox(self, sandbox_client):
        """Delete sandbox, verify it's gone."""
        # Create a temporary sandbox for deletion test
        sandbox = sandbox_client.create(image=SANDBOX_IMAGE, ttl_seconds=60)
        sandbox_id = sandbox.sandbox_id

        sandbox_client.delete(sandbox_id)
        time.sleep(2)

        # Verify it's gone or terminating
        sandboxes = sandbox_client.list()
        active = [
            s for s in sandboxes
            if s.sandbox_id == sandbox_id and s.status != "Terminating"
        ]
        assert len(active) == 0, f"Deleted sandbox should not appear as active"

    def test_delete_nonexistent_sandbox(self, sandbox_client):
        """Delete nonexistent sandbox, verify error."""
        with pytest.raises(Exception) as exc_info:
            sandbox_client.delete("sb-nonexistent-00000000")
        print(f"Expected error for deleting nonexistent: {exc_info.value}")
