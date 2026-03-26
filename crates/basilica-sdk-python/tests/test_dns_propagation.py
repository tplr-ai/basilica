"""
Unit tests for DNS propagation fix in wait_until_ready()

These tests verify that wait_until_ready() waits for DNS resolution
before returning, fixing the race condition where the SDK would return
before the deployment URL was DNS-resolvable.
"""

import socket
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from basilica import Deployment, DeploymentStatus


class TestDnsResolutionInWaitUntilReady:
    """Tests for DNS resolution verification in wait_until_ready()"""

    def _create_mock_deployment(self, url: str = "https://test.deployments.basilica.ai") -> Deployment:
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

    def test_is_dns_resolvable_returns_true_for_valid_hostname(self):
        """Verify _is_dns_resolvable returns True for resolvable hostnames."""
        deployment = self._create_mock_deployment()

        with patch("socket.gethostbyname", return_value="1.2.3.4"):
            result = deployment._is_dns_resolvable("example.com")

        assert result is True

    def test_is_dns_resolvable_returns_false_for_unresolvable_hostname(self):
        """Verify _is_dns_resolvable returns False when DNS fails."""
        deployment = self._create_mock_deployment()

        with patch("socket.gethostbyname", side_effect=socket.gaierror):
            result = deployment._is_dns_resolvable("nonexistent.invalid")

        assert result is False

    def test_wait_until_ready_waits_for_dns_resolution(self):
        """
        Verify wait_until_ready() waits for DNS to resolve before returning.

        This is the core test for the DNS propagation fix.
        """
        deployment = self._create_mock_deployment()
        ready_status = self._create_ready_status()

        # Mock status() to always return ready
        deployment.status = MagicMock(return_value=ready_status)

        # Track DNS resolution attempts
        dns_attempts = []

        def mock_gethostbyname(hostname):
            dns_attempts.append(hostname)
            if len(dns_attempts) < 3:
                # First 2 attempts fail (DNS not propagated)
                raise socket.gaierror(-5, "No address associated with hostname")
            # Third attempt succeeds (DNS propagated)
            return "1.2.3.4"

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            with patch("time.sleep"):  # Don't actually sleep in tests
                with patch.object(deployment, "_is_http_ready", return_value=True):
                    status = deployment.wait_until_ready(
                        timeout=30, poll_interval=1, silent=True
                    )

        # Verify: DNS was checked multiple times before returning
        assert len(dns_attempts) == 3, f"Expected 3 DNS attempts, got {len(dns_attempts)}"
        assert status.is_ready is True

    def test_wait_until_ready_returns_immediately_when_dns_resolves(self):
        """Verify wait_until_ready() returns immediately when DNS resolves on first try."""
        deployment = self._create_mock_deployment()
        ready_status = self._create_ready_status()

        deployment.status = MagicMock(return_value=ready_status)

        dns_attempts = []

        def mock_gethostbyname(hostname):
            dns_attempts.append(hostname)
            return "1.2.3.4"  # DNS resolves immediately

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            with patch.object(deployment, "_is_http_ready", return_value=True):
                status = deployment.wait_until_ready(
                    timeout=30, poll_interval=1, silent=True
                )

        # Verify: DNS was checked only once
        assert len(dns_attempts) == 1
        assert status.is_ready is True

    def test_wait_until_ready_times_out_if_dns_never_resolves(self):
        """Verify wait_until_ready() times out if DNS never resolves."""
        deployment = self._create_mock_deployment()
        ready_status = self._create_ready_status()

        deployment.status = MagicMock(return_value=ready_status)

        def mock_gethostbyname(hostname):
            raise socket.gaierror(-5, "No address associated with hostname")

        from basilica.exceptions import DeploymentTimeout

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            with patch("time.sleep"):
                with patch.object(deployment, "_is_http_ready", return_value=True):
                    with pytest.raises(DeploymentTimeout):
                        deployment.wait_until_ready(
                            timeout=5, poll_interval=1, silent=True
                        )

    def test_wait_until_ready_skips_dns_check_if_no_url(self):
        """Verify wait_until_ready() skips DNS check if deployment has no URL."""
        deployment = self._create_mock_deployment(url="")
        ready_status = self._create_ready_status()

        deployment.status = MagicMock(return_value=ready_status)

        dns_check_called = False

        def mock_gethostbyname(hostname):
            nonlocal dns_check_called
            dns_check_called = True
            return "1.2.3.4"

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            status = deployment.wait_until_ready(
                timeout=30, poll_interval=1, silent=True
            )

        # Verify: DNS check was not called because URL is empty
        assert dns_check_called is False
        assert status.is_ready is True

    def test_wait_until_ready_handles_url_without_hostname(self):
        """Verify wait_until_ready() handles URLs that parse to no hostname."""
        deployment = self._create_mock_deployment(url="file:///local/path")
        ready_status = self._create_ready_status()

        deployment.status = MagicMock(return_value=ready_status)

        dns_check_called = False

        def mock_gethostbyname(hostname):
            nonlocal dns_check_called
            dns_check_called = True
            return "1.2.3.4"

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            status = deployment.wait_until_ready(
                timeout=30, poll_interval=1, silent=True
            )

        # Verify: DNS check was not called because hostname is None
        assert dns_check_called is False
        assert status.is_ready is True


class TestDnsResolutionIntegration:
    """Integration-style tests that verify the full flow."""

    def test_dns_check_uses_correct_hostname_from_url(self):
        """Verify the correct hostname is extracted from the deployment URL."""
        mock_client = MagicMock()
        deployment = Deployment(
            client=mock_client,
            instance_name="my-app",
            url="https://my-app.deployments.basilica.ai:443/api",
            namespace="u-test",
            user_id="test-user",
            state="Active",
            created_at="2025-01-01T00:00:00Z",
            replicas_ready=1,
            replicas_desired=1,
        )

        ready_status = DeploymentStatus(
            state="Active",
            replicas_ready=1,
            replicas_desired=1,
            phase="ready",
        )
        deployment.status = MagicMock(return_value=ready_status)

        checked_hostname = None

        def mock_gethostbyname(hostname):
            nonlocal checked_hostname
            checked_hostname = hostname
            return "1.2.3.4"

        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            with patch.object(deployment, "_is_http_ready", return_value=True):
                deployment.wait_until_ready(
                    timeout=30, poll_interval=1, silent=True
                )

        # Verify: Correct hostname was extracted (without port or path)
        assert checked_hostname == "my-app.deployments.basilica.ai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
