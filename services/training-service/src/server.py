"""FastAPI server for training service."""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import structlog
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .backend import (
    ForwardBackwardResult,
    ForwardResult,
    LoraConfiguration,
    OptimizerConfiguration,
    SampleResult,
    TrainingBackend,
)

logger = structlog.get_logger()

# Global backend instance
backend: Optional[TrainingBackend] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize backend on startup."""
    global backend

    backend = TrainingBackend(
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "/models"),
        checkpoint_dir=os.getenv("CHECKPOINT_DIR", "/checkpoints"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("training_service_started")
    yield
    logger.info("training_service_stopped")


app = FastAPI(
    title="Basilica Training Service",
    version="0.1.0",
    lifespan=lifespan,
)


# === Request/Response Models ===


class LoraConfigRequest(BaseModel):
    """LoRA configuration request."""

    rank: int = Field(default=32, ge=1, le=256)
    alpha: int = Field(default=64, ge=1, le=512)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = None


class OptimizerConfigRequest(BaseModel):
    """Optimizer configuration request."""

    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    beta1: float = Field(default=0.9, ge=0, le=1)
    beta2: float = Field(default=0.999, ge=0, le=1)
    epsilon: float = Field(default=1e-8, gt=0)
    grad_clip: Optional[float] = Field(default=1.0, gt=0)


class CreateSessionRequest(BaseModel):
    """Create session request."""

    session_id: str
    base_model: str
    lora_config: Optional[LoraConfigRequest] = None
    optimizer_config: Optional[OptimizerConfigRequest] = None
    seed: Optional[int] = None


class CreateSessionResponse(BaseModel):
    """Create session response."""

    session_id: str
    status: str = "created"


class ForwardRequest(BaseModel):
    """Forward-only request (no gradients)."""

    input_ids: List[List[int]]
    attention_mask: List[List[int]]


class ForwardResponse(BaseModel):
    """Forward-only response."""

    logprobs: List[List[float]]
    tokens_processed: int


class ForwardBackwardRequest(BaseModel):
    """Forward-backward request."""

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]
    loss_weights: Optional[List[List[float]]] = None


class ForwardBackwardResponse(BaseModel):
    """Forward-backward response."""

    loss: float
    logprobs: List[List[float]]
    tokens_processed: int


class ComputeLogprobsRequest(BaseModel):
    """Compute logprobs request."""

    token_ids: List[int]


class ComputeLogprobsResponse(BaseModel):
    """Compute logprobs response."""

    logprobs: List[Optional[float]]


class OptimStepResponse(BaseModel):
    """Optimizer step response."""

    step: int


class SampleRequest(BaseModel):
    """Sample request."""

    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0, le=2.0)
    top_p: float = Field(default=1.0, ge=0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    include_logprobs: bool = False


class SampleResponse(BaseModel):
    """Sample response."""

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str


class SaveStateRequest(BaseModel):
    """Save state request."""

    checkpoint_name: str
    include_optimizer: bool = True


class SaveStateResponse(BaseModel):
    """Save state response."""

    checkpoint_path: str


class LoadStateRequest(BaseModel):
    """Load state request."""

    checkpoint_path: str
    load_optimizer: bool = True


class SessionStatusResponse(BaseModel):
    """Session status response."""

    session_id: str
    base_model: str
    step_count: int
    tokens_processed: int
    lora_rank: int
    learning_rate: float


# === Health Check ===


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.get("/sessions")
async def list_sessions() -> Dict[str, List[str]]:
    """List all active sessions."""
    return {"sessions": backend.list_sessions()}


# === Session Management ===


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new training session."""
    try:
        # Build config objects
        lora_kwargs = {}
        if request.lora_config:
            lora_kwargs = request.lora_config.model_dump(exclude_none=True)
            if "target_modules" not in lora_kwargs or lora_kwargs["target_modules"] is None:
                lora_kwargs.pop("target_modules", None)

        optimizer_kwargs = {}
        if request.optimizer_config:
            optimizer_kwargs = request.optimizer_config.model_dump(exclude_none=True)

        lora_config = LoraConfiguration(**lora_kwargs)
        optimizer_config = OptimizerConfiguration(**optimizer_kwargs)

        backend.create_session(
            session_id=request.session_id,
            base_model=request.base_model,
            lora_config=lora_config,
            optimizer_config=optimizer_config,
            seed=request.seed,
        )

        return CreateSessionResponse(session_id=request.session_id)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("create_session_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str) -> SessionStatusResponse:
    """Get session status."""
    try:
        session_status = backend.get_session_status(session_id)
        return SessionStatusResponse(**session_status)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a training session."""
    try:
        backend.delete_session(session_id)
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# === Training Operations ===


@app.post("/sessions/{session_id}/forward", response_model=ForwardResponse)
async def forward(session_id: str, request: ForwardRequest) -> ForwardResponse:
    """Forward pass without gradient computation.

    Used for inference-only operations like computing logprobs.
    """
    try:
        input_ids = torch.tensor(request.input_ids)
        attention_mask = torch.tensor(request.attention_mask)

        result = backend.forward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return ForwardResponse(
            logprobs=result.logprobs,
            tokens_processed=result.tokens_processed,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("forward_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/sessions/{session_id}/compute_logprobs", response_model=ComputeLogprobsResponse)
async def compute_logprobs(
    session_id: str, request: ComputeLogprobsRequest
) -> ComputeLogprobsResponse:
    """Compute log probabilities for a token sequence.

    Returns logprob for each token given its prefix.
    First token returns None (no conditioning context).
    """
    try:
        logprobs = backend.compute_logprobs(
            session_id=session_id,
            token_ids=request.token_ids,
        )

        return ComputeLogprobsResponse(logprobs=logprobs)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("compute_logprobs_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/sessions/{session_id}/forward_backward", response_model=ForwardBackwardResponse)
async def forward_backward(
    session_id: str, request: ForwardBackwardRequest
) -> ForwardBackwardResponse:
    """Compute forward pass and gradients."""
    try:
        # Convert to tensors
        input_ids = torch.tensor(request.input_ids)
        attention_mask = torch.tensor(request.attention_mask)
        labels = torch.tensor(request.labels)
        loss_weights = torch.tensor(request.loss_weights) if request.loss_weights else None

        result = backend.forward_backward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_weights=loss_weights,
        )

        return ForwardBackwardResponse(
            loss=result.loss,
            logprobs=[result.logprobs],  # Wrap in list for batch dimension
            tokens_processed=result.tokens_processed,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("forward_backward_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/sessions/{session_id}/optim_step", response_model=OptimStepResponse)
async def optim_step(session_id: str) -> OptimStepResponse:
    """Apply gradients and update weights."""
    try:
        step = backend.optim_step(session_id)
        return OptimStepResponse(step=step)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# === Inference ===


@app.post("/sessions/{session_id}/sample", response_model=SampleResponse)
async def sample(session_id: str, request: SampleRequest) -> SampleResponse:
    """Generate text completion."""
    try:
        result = backend.sample(
            session_id=session_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            include_logprobs=request.include_logprobs,
        )

        return SampleResponse(
            text=result.text,
            token_ids=result.token_ids,
            logprobs=result.logprobs,
            finish_reason=result.finish_reason,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# === Checkpoints ===


@app.post("/sessions/{session_id}/save", response_model=SaveStateResponse)
async def save_state(session_id: str, request: SaveStateRequest) -> SaveStateResponse:
    """Save checkpoint."""
    try:
        path = backend.save_state(
            session_id=session_id,
            checkpoint_name=request.checkpoint_name,
            include_optimizer=request.include_optimizer,
        )
        return SaveStateResponse(checkpoint_path=path)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/sessions/{session_id}/load")
async def load_state(session_id: str, request: LoadStateRequest) -> Dict[str, str]:
    """Load checkpoint."""
    try:
        backend.load_state(
            session_id=session_id,
            checkpoint_path=request.checkpoint_path,
            load_optimizer=request.load_optimizer,
        )
        return {"status": "loaded"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
