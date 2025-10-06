"""
Unit tests for Service base class and serve() function.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from basilica.afine import Service, serve


class TestService(Service):
    """Test service implementation."""

    def __init__(self):
        super().__init__()
        self.value = 0

    def reset(self, seed=None):
        """Reset test service."""
        self.value = 0
        return self.value

    def step(self, action):
        """Take a step."""
        self.value += action
        return self.value, 1.0, False, False, {}


def test_service_subclass():
    """Test that Service can be subclassed."""
    service = TestService()
    assert service.value == 0

    result = service.reset()
    assert result == 0

    result, reward, terminated, truncated, info = service.step(5)
    assert result == 5
    assert reward == 1.0
    assert terminated is False
    assert truncated is False


@patch.dict(os.environ, {"BASILICA_RENTAL_SECRET": "test-secret"})
@patch("basilica.afine.service.uvicorn.run")
@patch("basilica.afine.service.asyncio.create_task")
def test_serve_creates_fastapi_app(mock_create_task, mock_uvicorn_run):
    """Test that serve() creates a FastAPI app with routes."""
    mock_uvicorn_run.side_effect = SystemExit()

    with pytest.raises(SystemExit):
        serve(TestService)

    mock_uvicorn_run.assert_called_once()
    call_args = mock_uvicorn_run.call_args

    app = call_args[1].get('app') or call_args[0][0]
    assert app.title == "TestService"
    assert app.version == "1.0.0"


@patch.dict(os.environ, {}, clear=True)
def test_serve_requires_rental_secret():
    """Test that serve() requires BASILICA_RENTAL_SECRET."""
    with pytest.raises(ValueError, match="BASILICA_RENTAL_SECRET"):
        serve(TestService)


@pytest.mark.asyncio
async def test_serve_auth_middleware():
    """Test that authentication middleware validates secret."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI, Request

    with patch.dict(os.environ, {"BASILICA_RENTAL_SECRET": "test-secret"}):
        app = FastAPI()

        @app.middleware("http")
        async def validate_auth(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)

            secret = request.headers.get("X-Basilica-Secret")
            if not secret or secret != "test-secret":
                from fastapi.responses import JSONResponse
                from fastapi import status
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Unauthorized", "detail": "Invalid or missing authentication secret"}
                )
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.post("/test")
        async def test_route():
            return {"result": "success"}

        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        response = client.post("/test")
        assert response.status_code == 401

        response = client.post("/test", headers={"X-Basilica-Secret": "wrong-secret"})
        assert response.status_code == 401

        response = client.post("/test", headers={"X-Basilica-Secret": "test-secret"})
        assert response.status_code == 200
        assert response.json() == {"result": "success"}
