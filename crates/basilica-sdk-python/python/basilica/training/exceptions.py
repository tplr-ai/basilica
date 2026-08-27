"""
Basilica Training SDK - Exception definitions.

This module contains all custom exceptions used by the training SDK.
"""


class TrainingError(Exception):
    """Base exception for training operations."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SessionNotFoundError(TrainingError):
    """Training session not found."""

    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found", status_code=404)
        self.session_id = session_id


class SessionNotReadyError(TrainingError):
    """Training session not in ready state."""

    def __init__(self, session_id: str, phase: str):
        super().__init__(
            f"Session {session_id} not ready (phase: {phase})", status_code=503
        )
        self.session_id = session_id
        self.phase = phase


class SessionTimeoutError(TrainingError):
    """Training session did not become ready in time."""

    def __init__(self, session_id: str, timeout: float):
        super().__init__(
            f"Session {session_id} not ready after {timeout}s", status_code=408
        )
        self.session_id = session_id
        self.timeout = timeout


class AuthenticationError(TrainingError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class RateLimitError(TrainingError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: float = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(TrainingError):
    """Request validation failed."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class CheckpointError(TrainingError):
    """Checkpoint operation failed."""

    pass


class ModelNotFoundError(TrainingError):
    """Model not found or not supported."""

    def __init__(self, model: str):
        super().__init__(f"Model {model} not found or not supported", status_code=404)
        self.model = model


class InsufficientResourcesError(TrainingError):
    """Insufficient GPU resources available."""

    def __init__(self, message: str = "Insufficient GPU resources"):
        super().__init__(message, status_code=503)


# === Export ===

__all__ = [
    "TrainingError",
    "SessionNotFoundError",
    "SessionNotReadyError",
    "SessionTimeoutError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "CheckpointError",
    "ModelNotFoundError",
    "InsufficientResourcesError",
]
