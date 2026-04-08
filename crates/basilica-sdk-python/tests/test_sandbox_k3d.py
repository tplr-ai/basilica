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
NAMESPACE = "u-test-user"


def sandbox_image() -> str:
    image = os.environ.get("SANDBOX_IMAGE")
    if image:
        return image

    tag_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scripts",
            "localtest",
            ".sandbox-image-tag",
        )
    )
    if os.path.exists(tag_file):
        with open(tag_file, "r", encoding="utf-8") as f:
            tag = f.read().strip()
        if tag:
            return f"k3d-basilica-registry:5050/basilica-exec-agent:{tag}"

    return "k3d-basilica-registry:5050/basilica-exec-agent:latest"

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
        image=sandbox_image(),
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
        listed = next(s for s in sandboxes if s.sandbox_id == sandbox_id)
        assert listed.ttl_seconds == 600
        assert listed.network_isolation == "egress"
        assert listed.from_warm_pool is False

    def test_get_sandbox(self, sandbox_lifecycle):
        """Get sandbox by ID, verify detail fields."""
        sandbox_client = sandbox_lifecycle["sandbox_client"]
        sandbox_id = sandbox_lifecycle["sandbox_id"]
        detail = sandbox_client.get(sandbox_id)
        assert isinstance(detail, SandboxDetail)
        assert detail.sandbox_id == sandbox_id
        assert detail.image == sandbox_image()
        assert detail.cpu
        assert detail.memory
        assert detail.status == "Running"
        assert detail.ttl_seconds == 600
        assert detail.network_isolation == "egress"
        assert detail.ready_at
        assert detail.expires_at
        assert detail.from_warm_pool is False

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


class TestPyO3DataPlane:
    """Tests for direct PyO3 data-plane bindings backed by the Rust SDK."""

    def test_pyo3_exec(self, client, sandbox_lifecycle):
        sandbox = sandbox_lifecycle["sandbox"]
        result = client.sandbox_exec(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            ["echo", "hello-pyo3"],
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        assert result["exitCode"] == 0
        assert "hello-pyo3" in result["stdout"]

    def test_pyo3_exec_with_workdir(self, client, sandbox_lifecycle):
        sandbox = sandbox_lifecycle["sandbox"]
        result = client.sandbox_exec(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            ["pwd"],
            data_plane_url=sandbox_lifecycle["data_plane_url"],
            workdir="/tmp",
        )
        assert result["exitCode"] == 0
        assert result["stdout"].strip() == "/tmp"

    def test_pyo3_files_roundtrip(self, client, sandbox_lifecycle):
        sandbox = sandbox_lifecycle["sandbox"]
        client.sandbox_files_write(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-sdk-test.txt",
            "pyo3 roundtrip",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        read_result = client.sandbox_files_read(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-sdk-test.txt",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        assert read_result["content"] == "pyo3 roundtrip"

    def test_pyo3_files_list_and_delete(self, client, sandbox_lifecycle):
        sandbox = sandbox_lifecycle["sandbox"]
        client.sandbox_files_write(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-list-test.txt",
            "list-delete",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        listed = client.sandbox_files_list(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        assert any(f["name"] == "pyo3-list-test.txt" for f in listed["files"])

        client.sandbox_files_delete(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-list-test.txt",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        with pytest.raises(Exception):
            client.sandbox_files_read(
                sandbox.sandbox_id,
                sandbox.domain,
                sandbox.exec_agent_secret,
                "/tmp/pyo3-list-test.txt",
                data_plane_url=sandbox_lifecycle["data_plane_url"],
            )

    def test_pyo3_files_mkdir_and_stat(self, client, sandbox_lifecycle):
        sandbox = sandbox_lifecycle["sandbox"]
        client.sandbox_files_mkdir(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-dir-test",
            recursive=True,
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        stat_result = client.sandbox_files_stat(
            sandbox.sandbox_id,
            sandbox.domain,
            sandbox.exec_agent_secret,
            "/tmp/pyo3-dir-test",
            data_plane_url=sandbox_lifecycle["data_plane_url"],
        )
        assert stat_result["isDir"] is True


# ============================================================================
# URL helpers tests
# ============================================================================


class TestURLHelpers:
    """Tests for sandbox URL helper properties."""

    def test_data_plane_url(self, sandbox_lifecycle):
        """Verify data_plane_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.data_plane_url
        assert url == sandbox_lifecycle["data_plane_url"]

    def test_ws_url(self, sandbox_lifecycle):
        """Verify ws_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.ws_url
        assert url.startswith("ws://"), f"should start with ws://, got: {url}"
        assert url.endswith("/ws"), f"should end with /ws, got: {url}"
        assert str(sandbox_lifecycle["data_plane_url"]).removeprefix("http://") in url

    def test_exec_url(self, sandbox_lifecycle):
        """Verify exec_url returns correct URL."""
        sandbox = sandbox_lifecycle["sandbox"]
        url = sandbox.exec_url
        assert url.startswith("http://"), f"should start with http://, got: {url}"
        assert url.endswith("/exec"), f"should end with /exec, got: {url}"


# ============================================================================
# Secret rotation tests (run after data-plane tests)
# ============================================================================


class TestSecretRotation:
    def test_rotate_secret(self, sandbox_lifecycle):
        """Rotate the sandbox secret, verify old auth fails and new auth works."""
        sandbox_client = sandbox_lifecycle["sandbox_client"]
        sandbox = sandbox_lifecycle["sandbox"]
        sandbox_id = sandbox_lifecycle["sandbox_id"]
        old_secret = sandbox.exec_agent_secret
        old_uid = subprocess.check_output(
            [
                "kubectl",
                "get",
                "pod",
                f"sandbox-{sandbox_id}",
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={.metadata.uid}",
            ],
            text=True,
        ).strip()

        new_secret = sandbox_client.rotate_secret(sandbox_id)
        assert new_secret
        assert new_secret != old_secret

        pod_name = f"sandbox-{sandbox_id}"
        start = time.time()
        while True:
            try:
                current_uid = subprocess.check_output(
                    [
                        "kubectl",
                        "get",
                        "pod",
                        pod_name,
                        "-n",
                        NAMESPACE,
                        "-o",
                        "jsonpath={.metadata.uid}",
                    ],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                current_uid = ""
            if current_uid and current_uid != old_uid:
                break
            if time.time() - start > 90:
                pytest.fail("sandbox pod was not recreated after secret rotation")
            time.sleep(2)

        subprocess.run(
            [
                "kubectl",
                "wait",
                "--for=condition=Ready",
                f"pod/{pod_name}",
                "-n",
                NAMESPACE,
                "--timeout=90s",
            ],
            check=True,
            capture_output=True,
        )

        local_port = 30000 + (os.getpid() % 10000)
        pf_proc = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "-n",
                NAMESPACE,
                f"pod/{pod_name}",
                f"{local_port}:9999",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        try:
            rotated_url = f"http://localhost:{local_port}"
            stale_sandbox = Sandbox(
                sandbox_id=sandbox.sandbox_id,
                domain=sandbox.domain,
                status=sandbox.status,
                exec_agent_secret=old_secret,
            ).with_data_plane_url(rotated_url)

            with pytest.raises(Exception) as exc_info:
                stale_sandbox.exec(["echo", "stale-secret"])
            assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)

            rotated_sandbox = Sandbox(
                sandbox_id=sandbox.sandbox_id,
                domain=sandbox.domain,
                status=sandbox.status,
                exec_agent_secret=new_secret,
            ).with_data_plane_url(rotated_url)
            rotated = rotated_sandbox.exec(["echo", "secret-rotated"])
            assert rotated["exitCode"] == 0
            assert "secret-rotated" in rotated["stdout"]
        finally:
            pf_proc.terminate()
            try:
                pf_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pf_proc.kill()


# ============================================================================
# Deletion tests (run last)
# ============================================================================


class TestDeletion:
    """Tests for sandbox deletion. These use their own sandboxes."""

    def test_delete_sandbox(self, sandbox_client):
        """Delete sandbox, verify it's gone."""
        # Create a temporary sandbox for deletion test
        sandbox = sandbox_client.create(image=sandbox_image(), ttl_seconds=60)
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
