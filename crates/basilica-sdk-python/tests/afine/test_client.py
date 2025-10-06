"""
Unit tests for Client proxy and create() function.
"""

from unittest.mock import MagicMock, patch

import pytest
import httpx

from basilica.afine import Client
from basilica.afine.client import wait_for_health, generate_ssh_keypair


def test_client_context_manager():
    """Test that Client implements context manager protocol."""
    client = Client(
        rental_id="test-rental",
        base_url="http://localhost:8000",
        rental_secret="test-secret"
    )

    assert not client._closed

    with client as c:
        assert c is client
        assert not c._closed

    assert client._closed


def test_client_dynamic_method_proxy():
    """Test that Client creates dynamic method proxies."""
    client = Client(
        rental_id="test-rental",
        base_url="http://localhost:8000",
        rental_secret="test-secret"
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": [1, 1]}

    with patch.object(client._http_client, 'post', return_value=mock_response):
        result = client.reset()
        assert result == [1, 1]

        client._http_client.post.assert_called_once()
        call_args = client._http_client.post.call_args

        assert call_args[0][0] == "http://localhost:8000/reset"
        assert call_args[1]["headers"]["X-Basilica-Secret"] == "test-secret"
        assert call_args[1]["json"] == {"args": [], "kwargs": {}}

    client.close()


def test_client_authentication_error():
    """Test that Client raises RuntimeError on 401."""
    client = Client(
        rental_id="test-rental",
        base_url="http://localhost:8000",
        rental_secret="wrong-secret"
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=MagicMock(),
        response=mock_response
    )

    with patch.object(client._http_client, 'post', return_value=mock_response):
        with pytest.raises(RuntimeError, match="Authentication failed"):
            client.reset()

    client.close()


def test_client_method_not_found():
    """Test that Client raises AttributeError on 404."""
    client = Client(
        rental_id="test-rental",
        base_url="http://localhost:8000",
        rental_secret="test-secret"
    )

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found",
        request=MagicMock(),
        response=mock_response
    )

    with patch.object(client._http_client, 'post', return_value=mock_response):
        with pytest.raises(AttributeError, match="Method 'unknown_method' not found"):
            client.unknown_method()

    client.close()


def test_client_closed_state():
    """Test that Client raises error when used after close."""
    client = Client(
        rental_id="test-rental",
        base_url="http://localhost:8000",
        rental_secret="test-secret"
    )

    client.close()
    assert client._closed

    with pytest.raises(RuntimeError, match="Client is closed"):
        client.reset()


def test_wait_for_health_success():
    """Test successful health check polling."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}

    with patch("basilica.afine.client.httpx.get", return_value=mock_response):
        result = wait_for_health("http://localhost:8000", timeout=5.0, check_interval=0.1)
        assert result is True


def test_wait_for_health_timeout():
    """Test health check timeout."""
    with patch("basilica.afine.client.httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        result = wait_for_health("http://localhost:8000", timeout=0.5, check_interval=0.1)
        assert result is False


def test_generate_ssh_keypair():
    """Test SSH keypair generation."""
    public_key, private_key_path = generate_ssh_keypair()

    assert public_key.startswith("ssh-ed25519")
    assert private_key_path.endswith(".key")

    import os
    assert os.path.exists(private_key_path)

    stat = os.stat(private_key_path)
    assert oct(stat.st_mode)[-3:] == '600'
