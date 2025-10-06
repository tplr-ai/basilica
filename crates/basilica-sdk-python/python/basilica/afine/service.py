"""
Service base class and serve() function for Basilica AFINE SDK.
"""

import asyncio
import inspect
import os
import signal
import sys
from abc import ABC
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .models import ErrorResponse, HealthResponse, RPCRequest, RPCResponse

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class Service(ABC, Generic[ObsType, ActType]):
    """
    Base class for defining Basilica services.

    Users subclass this and define public methods which will be
    automatically exposed as HTTP RPC endpoints with authentication.

    Type parameters:
        ObsType: Type of observations returned by environment
        ActType: Type of actions accepted by environment
    """

    def __init__(self) -> None:
        """Initialize the service instance."""
        pass

    def reset(self, seed: Optional[int] = None) -> ObsType:
        """
        Reset the environment and return initial observation.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Initial observation
        """
        raise NotImplementedError

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            observation: Current observation
            reward: Reward from the action
            terminated: Whether episode terminated (goal reached/failed)
            truncated: Whether episode truncated (time limit)
            info: Additional information
        """
        raise NotImplementedError

    def render(self) -> Optional[Any]:
        """Render the environment (optional)."""
        pass

    def close(self) -> None:
        """Clean up resources (optional)."""
        pass


shutdown_event = asyncio.Event()


def serve(
    service_class: Type[Service],
    host: str = "0.0.0.0",
    port: int = 8000,
    state_persistence: Optional[Any] = None
) -> None:
    """
    Start an HTTP server exposing all public methods of the service with authentication.

    Args:
        service_class: The Service subclass to serve
        host: Host to bind to
        port: Port to bind to
        state_persistence: Optional state persistence handler

    Security:
        Requires BASILICA_RENTAL_SECRET environment variable for authentication.
        All requests must include X-Basilica-Secret header with matching value.
    """
    app = FastAPI(
        title=service_class.__name__,
        description=f"Basilica Service: {service_class.__name__}",
        version="1.0.0"
    )

    service_instance = service_class()

    rental_secret = os.environ.get("BASILICA_RENTAL_SECRET")
    if not rental_secret:
        raise ValueError(
            "BASILICA_RENTAL_SECRET environment variable not set. "
            "This should be automatically set by Basilica during deployment."
        )

    @app.middleware("http")
    async def validate_auth(request: Request, call_next):
        """Validate shared secret on every request except /health."""
        if request.url.path == "/health":
            return await call_next(request)

        secret = request.headers.get("X-Basilica-Secret")
        if not secret or secret != rental_secret:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ErrorResponse(
                    error="Unauthorized",
                    detail="Invalid or missing authentication secret"
                ).model_dump()
            )
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint for readiness probes."""
        return HealthResponse(
            status="healthy",
            service=service_class.__name__,
            version="1.0.0"
        )

    methods = inspect.getmembers(
        service_instance,
        predicate=lambda m: inspect.ismethod(m) and not m.__name__.startswith('_')
    )

    def create_handler(method_func, method_name: str):
        """
        Factory function to create route handler with proper closure capture.

        This fixes the Python late-binding closure bug where all handlers
        would call the last method in the loop.
        """
        async def handler(request: RPCRequest) -> RPCResponse:
            try:
                result = method_func(*request.args, **request.kwargs)

                if inspect.iscoroutine(result):
                    result = await result

                return RPCResponse(result=result)

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error executing {method_name}: {str(e)}"
                )

        return handler

    for method_name, method_func in methods:
        handler = create_handler(method_func, method_name)
        app.post(
            f"/{method_name}",
            response_model=RPCResponse,
            summary=f"Call {method_name}",
            description=f"Execute {method_name} method on {service_class.__name__}"
        )(handler)

    async def graceful_shutdown():
        """Handle graceful shutdown with state persistence."""
        await shutdown_event.wait()

        await asyncio.sleep(5)

        if state_persistence:
            try:
                state_persistence.save_state(service_instance)
            except Exception as e:
                print(f"Warning: Failed to save state during shutdown: {e}", file=sys.stderr)

        sys.exit(0)

    def signal_handler(signum, frame):
        """Trigger graceful shutdown on SIGTERM/SIGINT."""
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    asyncio.create_task(graceful_shutdown())

    if state_persistence:
        saved_state = state_persistence.load_state()
        if saved_state:
            service_instance.__dict__.update(saved_state)
            print("Restored state from previous session")

    uvicorn.run(app, host=host, port=port, log_level="info")
