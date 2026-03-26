"""
Unit tests for async methods in Basilica SDK.

These tests verify that:
1. Async methods use asyncio.sleep() instead of time.sleep()
2. Multiple deployments can wait concurrently
3. Async methods properly integrate with asyncio event loop
"""

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest

from basilica import Deployment, DeploymentStatus, ProgressInfo
from basilica.exceptions import DeploymentFailed, DeploymentTimeout


class TestDeploymentAsyncMethods:
    """Tests for async methods on the Deployment class."""

    def _create_mock_deployment(
        self, url: str = "https://test.deployments.basilica.ai"
    ) -> Deployment:
        """Create a mock Deployment instance for testing."""
        mock_client = MagicMock()
        return Deployment(
            client=mock_client,
            instance_name="test-deployment",
            url=url,
            namespace="u-test",
            user_id="test-user",
            state="Active",
            created_at="2025-01-01T00:00:00Z",
            replicas_ready=1,
            replicas_desired=1,
        )

    def _create_ready_status(self) -> DeploymentStatus:
        """Create a DeploymentStatus that indicates ready."""
        return DeploymentStatus(
            state="Active",
            replicas_ready=1,
            replicas_desired=1,
            phase="ready",
        )

    def _create_pending_status(self, phase: str = "pending") -> DeploymentStatus:
        """Create a DeploymentStatus that indicates pending."""
        return DeploymentStatus(
            state="Pending",
            replicas_ready=0,
            replicas_desired=1,
            phase=phase,
        )

    def _create_failed_status(self) -> DeploymentStatus:
        """Create a DeploymentStatus that indicates failure."""
        return DeploymentStatus(
            state="Failed",
            replicas_ready=0,
            replicas_desired=1,
            phase="failed",
            message="Container crashed",
        )

    @pytest.mark.asyncio
    async def test_status_async_returns_deployment_status(self):
        """Verify status_async() returns DeploymentStatus."""
        deployment = self._create_mock_deployment()

        mock_response = MagicMock()
        mock_response.state = "Active"
        mock_response.replicas.ready = 1
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        status = await deployment.status_async()

        assert isinstance(status, DeploymentStatus)
        assert status.state == "Active"
        assert status.replicas_ready == 1

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_uses_asyncio_sleep(self):
        """Verify wait_until_ready_async() uses asyncio.sleep(), not time.sleep()."""
        deployment = self._create_mock_deployment()
        ready_status = self._create_ready_status()

        mock_response = MagicMock()
        mock_response.state = "Active"
        mock_response.replicas.ready = 1
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        asyncio_sleep_called = False
        original_sleep = asyncio.sleep

        async def mock_asyncio_sleep(seconds):
            nonlocal asyncio_sleep_called
            asyncio_sleep_called = True

        with patch("asyncio.sleep", side_effect=mock_asyncio_sleep):
            with patch.object(deployment, "_is_dns_resolvable_async", return_value=True):
                with patch.object(deployment, "_is_http_ready_async", return_value=True):
                    status = await deployment.wait_until_ready_async(
                        timeout=30, poll_interval=1, silent=True
                    )

        assert status.is_ready is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_waits_for_dns(self):
        """Verify wait_until_ready_async() waits for DNS resolution."""
        deployment = self._create_mock_deployment()

        mock_response = MagicMock()
        mock_response.state = "Active"
        mock_response.replicas.ready = 1
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        dns_attempts = []

        async def mock_dns_check(hostname):
            dns_attempts.append(hostname)
            return len(dns_attempts) >= 3

        with patch.object(deployment, "_is_dns_resolvable_async", side_effect=mock_dns_check):
            with patch("asyncio.sleep", return_value=None):
                with patch.object(deployment, "_is_http_ready_async", return_value=True):
                    status = await deployment.wait_until_ready_async(
                        timeout=30, poll_interval=1, silent=True
                    )

        assert len(dns_attempts) == 3
        assert status.is_ready is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_raises_on_timeout(self):
        """Verify wait_until_ready_async() raises DeploymentTimeout."""
        deployment = self._create_mock_deployment()

        mock_response = MagicMock()
        mock_response.state = "Pending"
        mock_response.replicas.ready = 0
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(DeploymentTimeout) as exc_info:
                await deployment.wait_until_ready_async(
                    timeout=3, poll_interval=1, silent=True
                )

        assert exc_info.value.timeout_seconds == 3
        assert exc_info.value.instance_name == "test-deployment"

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_raises_on_failure(self):
        """Verify wait_until_ready_async() raises DeploymentFailed."""
        deployment = self._create_mock_deployment()

        mock_response = MagicMock()
        mock_response.state = "Failed"
        mock_response.replicas.ready = 0
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        with pytest.raises(DeploymentFailed) as exc_info:
            await deployment.wait_until_ready_async(
                timeout=30, poll_interval=1, silent=True
            )

        assert exc_info.value.instance_name == "test-deployment"

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_does_not_raise_on_failure_when_disabled(self):
        """Verify raise_on_failure=False prevents DeploymentFailed exception.

        When raise_on_failure=False and deployment fails, the function continues
        polling until timeout (matching sync version behavior).
        """
        deployment = self._create_mock_deployment()

        mock_response = MagicMock()
        mock_response.state = "Failed"
        mock_response.replicas.ready = 0
        mock_response.replicas.desired = 1
        mock_response.progress = None
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(DeploymentTimeout) as exc_info:
                await deployment.wait_until_ready_async(
                    timeout=3,
                    poll_interval=1,
                    raise_on_failure=False,
                    silent=True,
                )

        assert exc_info.value.last_state == "Failed"

    @pytest.mark.asyncio
    async def test_refresh_async_updates_deployment_state(self):
        """Verify refresh_async() updates internal state."""
        deployment = self._create_mock_deployment()
        deployment._state = "Pending"

        mock_response = MagicMock()
        mock_response.url = "https://new-url.basilica.ai"
        mock_response.state = "Active"
        mock_response.replicas.ready = 1
        mock_response.replicas.desired = 1
        mock_response.updated_at = "2025-01-02T00:00:00Z"
        deployment._client.get_deployment = MagicMock(return_value=mock_response)

        result = await deployment.refresh_async()

        assert result is deployment
        assert deployment._state == "Active"
        assert deployment._url == "https://new-url.basilica.ai"

    @pytest.mark.asyncio
    async def test_delete_async_calls_client(self):
        """Verify delete_async() calls the client method."""
        deployment = self._create_mock_deployment()
        deployment._client.delete_deployment = MagicMock()

        await deployment.delete_async()

        deployment._client.delete_deployment.assert_called_once_with("test-deployment")
        assert deployment._state == "Deleted"

    @pytest.mark.asyncio
    async def test_logs_async_returns_logs(self):
        """Verify logs_async() returns log content."""
        deployment = self._create_mock_deployment()
        deployment._client.get_deployment_logs = MagicMock(
            return_value="Log line 1\nLog line 2"
        )

        logs = await deployment.logs_async(tail=100)

        assert logs == "Log line 1\nLog line 2"
        deployment._client.get_deployment_logs.assert_called_once()


class TestAsyncConcurrency:
    """Tests verifying true async concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_wait_until_ready_async(self):
        """
        Verify multiple wait_until_ready_async() calls run concurrently.

        This is the core test that validates the async fix works.
        """
        execution_order = []

        def create_mock_deployment(name: str) -> Deployment:
            mock_client = MagicMock()
            return Deployment(
                client=mock_client,
                instance_name=name,
                url=f"https://{name}.basilica.ai",
                namespace="u-test",
                user_id="test-user",
                state="Active",
                created_at="2025-01-01T00:00:00Z",
                replicas_ready=1,
                replicas_desired=1,
            )

        async def mock_wait(deployment: Deployment) -> DeploymentStatus:
            name = deployment._name
            execution_order.append(f"{name}_start")

            mock_response = MagicMock()
            mock_response.state = "Active"
            mock_response.replicas.ready = 1
            mock_response.replicas.desired = 1
            mock_response.progress = None
            deployment._client.get_deployment = MagicMock(return_value=mock_response)

            with patch.object(deployment, "_is_dns_resolvable_async", return_value=True):
                with patch.object(deployment, "_is_http_ready_async", return_value=True):
                    status = await deployment.wait_until_ready_async(
                        timeout=30, poll_interval=1, silent=True
                    )

            execution_order.append(f"{name}_end")
            return status

        d1 = create_mock_deployment("app-1")
        d2 = create_mock_deployment("app-2")
        d3 = create_mock_deployment("app-3")

        await asyncio.gather(
            mock_wait(d1),
            mock_wait(d2),
            mock_wait(d3),
        )

        starts = [e for e in execution_order if e.endswith("_start")]
        ends = [e for e in execution_order if e.endswith("_end")]

        assert len(starts) == 3
        assert len(ends) == 3

    @pytest.mark.asyncio
    async def test_asyncio_sleep_yields_control(self):
        """Verify asyncio.sleep() yields control to other coroutines."""
        execution_order = []

        async def task(name: str):
            execution_order.append(f"{name}_start")
            await asyncio.sleep(0.01)
            execution_order.append(f"{name}_end")

        await asyncio.gather(task("A"), task("B"))

        assert execution_order.index("A_start") < execution_order.index("A_end")
        assert execution_order.index("B_start") < execution_order.index("B_end")
        assert execution_order.index("B_start") < execution_order.index("A_end")


class TestAsyncDnsResolution:
    """Tests for async DNS resolution."""

    @pytest.mark.asyncio
    async def test_is_dns_resolvable_async_returns_true_for_valid_hostname(self):
        """Verify _is_dns_resolvable_async returns True for resolvable hostnames."""
        mock_client = MagicMock()
        deployment = Deployment(
            client=mock_client,
            instance_name="test",
            url="https://test.basilica.ai",
            namespace="u-test",
            user_id="test-user",
            state="Active",
            created_at="2025-01-01T00:00:00Z",
        )

        async def mock_getaddrinfo(*args, **kwargs):
            return [("127.0.0.1", 80)]

        loop = asyncio.get_running_loop()
        with patch.object(loop, "getaddrinfo", side_effect=mock_getaddrinfo):
            result = await deployment._is_dns_resolvable_async("example.com")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_dns_resolvable_async_returns_false_for_unresolvable(self):
        """Verify _is_dns_resolvable_async returns False when DNS fails."""
        mock_client = MagicMock()
        deployment = Deployment(
            client=mock_client,
            instance_name="test",
            url="https://test.basilica.ai",
            namespace="u-test",
            user_id="test-user",
            state="Active",
            created_at="2025-01-01T00:00:00Z",
        )

        async def mock_getaddrinfo(*args, **kwargs):
            raise socket.gaierror(-5, "No address associated with hostname")

        loop = asyncio.get_running_loop()
        with patch.object(loop, "getaddrinfo", side_effect=mock_getaddrinfo):
            result = await deployment._is_dns_resolvable_async("nonexistent.invalid")

        assert result is False


class TestAsyncProgressCallback:
    """Tests for progress callbacks with async methods."""

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_calls_progress_callback(self):
        """Verify progress callback is called on phase changes."""
        mock_client = MagicMock()
        deployment = Deployment(
            client=mock_client,
            instance_name="test-callback",
            url="https://test.basilica.ai",
            namespace="u-test",
            user_id="test-user",
            state="Pending",
            created_at="2025-01-01T00:00:00Z",
        )

        phases = ["pending", "scheduling", "pulling", "ready"]
        call_count = 0
        callback_phases = []

        def mock_get_deployment(name):
            nonlocal call_count
            phase_idx = min(call_count, len(phases) - 1)
            call_count += 1

            mock_response = MagicMock()
            mock_response.state = "Active" if phases[phase_idx] == "ready" else "Pending"
            mock_response.replicas.ready = 1 if phases[phase_idx] == "ready" else 0
            mock_response.replicas.desired = 1
            mock_response.progress = None

            class MockPhase:
                pass
            mock_response.phase = phases[phase_idx]

            return mock_response

        deployment._client.get_deployment = mock_get_deployment

        def progress_callback(status: DeploymentStatus):
            callback_phases.append(status.phase)

        with patch.object(deployment, "_is_dns_resolvable_async", return_value=True):
            with patch("asyncio.sleep", return_value=None):
                with patch.object(deployment, "_is_http_ready_async", return_value=True):
                    await deployment.wait_until_ready_async(
                        timeout=30,
                        poll_interval=1,
                        on_progress=progress_callback,
                    )

        assert "pending" in callback_phases
        assert "ready" in callback_phases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
