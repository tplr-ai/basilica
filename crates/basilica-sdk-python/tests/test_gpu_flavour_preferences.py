"""
Functional tests for GPU Flavour Preferences.

Tests the full SDK path: type construction -> API call -> response parsing.

Note: Tests that hit /secure-cloud/gpu-prices require a token with
secure-cloud permissions. They are skipped when the token lacks access.
"""
import pytest
from basilica import BasilicaClient, GpuPriceQuery


@pytest.fixture(scope="module")
def client():
    try:
        return BasilicaClient()
    except Exception as e:
        pytest.skip(f"No authentication available: {e}")


@pytest.fixture(scope="module")
def can_list_gpus(client):
    """Check if token has secure-cloud access."""
    try:
        client.list_secure_cloud_gpus()
        return True
    except PermissionError:
        return False


# -------------------------------------------------------------------------
# GPU Price Query - API calls
# -------------------------------------------------------------------------

class TestListSecureCloudGpus:
    def test_no_filter(self, client, can_list_gpus):
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        result = client.list_secure_cloud_gpus()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_interconnect_sxm(self, client, can_list_gpus):
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        result = client.list_secure_cloud_gpus(
            query=GpuPriceQuery(interconnect="SXM")
        )
        for gpu in result:
            assert gpu.interconnect in ("SXM4", "SXM5", "SXM6"), (
                f"SXM filter returned interconnect={gpu.interconnect}"
            )

    def test_exclude_spot(self, client, can_list_gpus):
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        result = client.list_secure_cloud_gpus(
            query=GpuPriceQuery(exclude_spot=True)
        )
        for gpu in result:
            assert not gpu.is_spot, f"exclude_spot returned spot offering {gpu.id}"

    def test_spot_only(self, client, can_list_gpus):
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        result = client.list_secure_cloud_gpus(
            query=GpuPriceQuery(spot_only=True)
        )
        for gpu in result:
            assert gpu.is_spot, f"spot_only returned non-spot offering {gpu.id}"

    def test_region_filter_reduces_results(self, client, can_list_gpus):
        """Verify region=US actually filters out non-US offerings."""
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        all_gpus = client.list_secure_cloud_gpus()
        us_gpus = client.list_secure_cloud_gpus(
            query=GpuPriceQuery(region="US")
        )
        assert len(us_gpus) < len(all_gpus), (
            f"region=US returned {len(us_gpus)} offerings, same as unfiltered {len(all_gpus)}"
        )

    def test_backward_compat_no_args(self, client, can_list_gpus):
        if not can_list_gpus:
            pytest.skip("token lacks secure-cloud permission")
        result = client.list_secure_cloud_gpus()
        assert isinstance(result, list)

    def test_query_sends_to_api(self, client):
        """Verify the SDK sends the request (even if 403, no crash)."""
        try:
            client.list_secure_cloud_gpus(
                query=GpuPriceQuery(interconnect="SXM", region="US", exclude_spot=True)
            )
        except PermissionError:
            pass  # 403 is expected if token lacks permission


# -------------------------------------------------------------------------
# Deployment with flavour fields - uses high-level create_deployment()
# -------------------------------------------------------------------------

class TestDeployWithFlavour:
    def test_create_deployment_with_flavour(self, client):
        """Create a real deployment with flavour fields, verify, delete."""
        resp = client.create_deployment(
            instance_name="pytest-flavour",
            image="hashicorp/http-echo",
            port=5678,
            gpu_count=1,
            gpu_models=["H100"],
            interconnect="SXM",
            ttl_seconds=120,
            public=True,
        )
        name = resp.instance_name
        try:
            assert resp.state in ("Active", "Pending", "active", "pending")
            assert resp.url is not None
            dep = client.get_deployment(name)
            assert dep.instance_name == name
        finally:
            client.delete_deployment(name)

    def test_create_deployment_no_flavour(self, client):
        """Backward compat: deployment without flavour fields still works."""
        resp = client.create_deployment(
            instance_name="pytest-no-flavour",
            image="hashicorp/http-echo",
            port=5678,
            gpu_count=1,
            gpu_models=["H100"],
            ttl_seconds=120,
            public=True,
        )
        name = resp.instance_name
        try:
            assert resp.state in ("Active", "Pending", "active", "pending")
        finally:
            client.delete_deployment(name)

    def test_create_deployment_exclude_spot(self, client):
        """spot=False maps to NotIn scheduling on the CRD."""
        resp = client.create_deployment(
            instance_name="pytest-no-spot",
            image="hashicorp/http-echo",
            port=5678,
            gpu_count=1,
            gpu_models=["H100"],
            spot=False,
            ttl_seconds=120,
            public=True,
        )
        name = resp.instance_name
        try:
            assert resp.state in ("Active", "Pending", "active", "pending")
        finally:
            client.delete_deployment(name)
