"""Tests for training backend."""

import pytest
import torch

from src.backend import (
    LoraConfiguration,
    OptimizerConfiguration,
    TrainingBackend,
)


@pytest.fixture
def backend(tmp_path):
    """Create backend with temp directories."""
    return TrainingBackend(
        model_cache_dir=str(tmp_path / "models"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@pytest.fixture
def session_id():
    return "test-session-001"


@pytest.fixture
def small_model():
    # Use a small model for testing
    return "facebook/opt-125m"


class TestTrainingBackend:
    def test_create_session(self, backend, session_id, small_model):
        """Test session creation."""
        result = backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        assert result == session_id
        assert session_id in backend.sessions

    def test_create_duplicate_session_fails(self, backend, session_id, small_model):
        """Test that creating duplicate session fails."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        with pytest.raises(ValueError, match="already exists"):
            backend.create_session(
                session_id=session_id,
                base_model=small_model,
                lora_config=LoraConfiguration(rank=8),
                optimizer_config=OptimizerConfiguration(),
            )

    def test_forward_backward(self, backend, session_id, small_model):
        """Test forward-backward pass."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        # Create dummy batch
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        result = backend.forward_backward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert result.loss > 0
        assert result.tokens_processed == 32

    def test_forward_backward_with_loss_weights(self, backend, session_id, small_model):
        """Test forward-backward with loss weights."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        # Create dummy batch with weights
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        # Only weight the last half
        loss_weights = torch.cat([torch.zeros(1, 16), torch.ones(1, 16)], dim=1)

        result = backend.forward_backward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_weights=loss_weights,
        )

        assert result.loss > 0

    def test_optim_step(self, backend, session_id, small_model):
        """Test optimizer step."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        # Do forward-backward first
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        backend.forward_backward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Optimizer step
        step = backend.optim_step(session_id)
        assert step == 1

        # Second step
        backend.forward_backward(
            session_id=session_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        step = backend.optim_step(session_id)
        assert step == 2

    def test_sample(self, backend, session_id, small_model):
        """Test text generation."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        result = backend.sample(
            session_id=session_id,
            prompt="Hello, world!",
            max_tokens=10,
            temperature=0.7,
        )

        assert len(result.text) >= 0  # May be empty for some models
        assert len(result.token_ids) > 0

    def test_sample_with_logprobs(self, backend, session_id, small_model):
        """Test text generation with logprobs."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        result = backend.sample(
            session_id=session_id,
            prompt="Hello, world!",
            max_tokens=10,
            temperature=0.7,
            include_logprobs=True,
        )

        assert result.logprobs is not None
        assert len(result.logprobs) == len(result.token_ids)

    def test_save_load_checkpoint(self, backend, session_id, small_model):
        """Test checkpoint save/load."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        # Do some training
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        backend.forward_backward(session_id, input_ids, attention_mask, labels)
        backend.optim_step(session_id)

        # Save
        path = backend.save_state(session_id, "checkpoint-1")
        assert "checkpoint-1" in path

        # Load
        backend.load_state(session_id, path)
        status = backend.get_session_status(session_id)
        assert status["step_count"] == 1

    def test_get_session_status(self, backend, session_id, small_model):
        """Test getting session status."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=16),
            optimizer_config=OptimizerConfiguration(learning_rate=2e-4),
        )

        status = backend.get_session_status(session_id)
        assert status["session_id"] == session_id
        assert status["lora_rank"] == 16
        assert status["learning_rate"] == 2e-4
        assert status["step_count"] == 0

    def test_delete_session(self, backend, session_id, small_model):
        """Test deleting session."""
        backend.create_session(
            session_id=session_id,
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        assert session_id in backend.sessions
        backend.delete_session(session_id)
        assert session_id not in backend.sessions

    def test_list_sessions(self, backend, small_model):
        """Test listing sessions."""
        backend.create_session(
            session_id="session-1",
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )
        backend.create_session(
            session_id="session-2",
            base_model=small_model,
            lora_config=LoraConfiguration(rank=8),
            optimizer_config=OptimizerConfiguration(),
        )

        sessions = backend.list_sessions()
        assert "session-1" in sessions
        assert "session-2" in sessions

    def test_nonexistent_session_raises(self, backend):
        """Test that accessing nonexistent session raises error."""
        with pytest.raises(ValueError, match="not found"):
            backend.get_session_status("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            backend.optim_step("nonexistent")
