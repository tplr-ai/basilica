"""Unit tests for Basilica Training SDK clients."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basilica.training import (
    ServiceClient,
    TrainingClient,
    SamplingClient,
    RestClient,
    Datum,
    SamplingParams,
    ModelInput,
    ForwardBackwardResult,
    ForwardResult,
    APIFuture,
    GetServerCapabilitiesResponse,
)
from basilica.training.exceptions import (
    TrainingError,
    ValidationError,
)


class TestAPIFuture:
    """Tests for APIFuture wrapper."""

    def test_result_blocks(self):
        """Test that result() blocks until future completes."""
        future = Future()
        future.set_result(42)

        api_future = APIFuture(future, int)
        assert api_future.result() == 42

    def test_result_with_timeout(self):
        """Test result() with timeout."""
        future = Future()
        future.set_result("hello")

        api_future = APIFuture(future, str)
        assert api_future.result(timeout=1.0) == "hello"

    @pytest.mark.asyncio
    async def test_result_async(self):
        """Test async result."""
        future = Future()
        future.set_result({"key": "value"})

        api_future = APIFuture(future, dict)
        result = await api_future.result_async()
        assert result == {"key": "value"}


class TestDatum:
    """Tests for Datum data class."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        datum = Datum(input_ids=[1, 2, 3])
        d = datum.to_dict()
        assert d["input_ids"] == [1, 2, 3]
        assert "labels" not in d or d["labels"] is None

    def test_to_dict_with_labels(self):
        """Test to_dict with labels."""
        datum = Datum(input_ids=[1, 2, 3], labels=[4, 5, 6])
        d = datum.to_dict()
        assert d["input_ids"] == [1, 2, 3]
        assert d["labels"] == [4, 5, 6]

    def test_to_dict_with_loss_weights(self):
        """Test to_dict with loss weights."""
        datum = Datum(input_ids=[1, 2, 3], loss_weights=[0.0, 1.0, 1.0])
        d = datum.to_dict()
        assert d["input_ids"] == [1, 2, 3]
        assert d["loss_weights"] == [0.0, 1.0, 1.0]


class TestModelInput:
    """Tests for ModelInput class."""

    def test_from_ints(self):
        """Test creating ModelInput from token IDs."""
        model_input = ModelInput.from_ints([1, 2, 3, 4, 5])
        assert model_input.token_ids == [1, 2, 3, 4, 5]

    def test_from_string_with_tokenizer(self):
        """Test creating ModelInput from string with tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [10, 20, 30]

        model_input = ModelInput.from_string("hello world", mock_tokenizer)
        assert model_input.token_ids == [10, 20, 30]
        mock_tokenizer.encode.assert_called_once_with("hello world")


class TestSamplingParams:
    """Tests for SamplingParams class."""

    def test_defaults(self):
        """Test default sampling parameters."""
        params = SamplingParams()
        assert params.max_tokens == 256  # Default is 256
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == 0  # Default is 0 (disabled)

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )
        assert params.max_tokens == 50
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40

    def test_to_dict(self):
        """Test conversion to dict."""
        params = SamplingParams(max_tokens=20, temperature=0.5)
        d = params.to_dict()
        assert d["max_tokens"] == 20
        assert d["temperature"] == 0.5


class TestServiceClient:
    """Tests for ServiceClient."""

    def test_init_requires_api_key(self):
        """Test that ServiceClient requires API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any environment variables
            os.environ.pop("BASILICA_API_KEY", None)
            os.environ.pop("BASILICA_API_TOKEN", None)

            with pytest.raises(ValueError, match="API key required"):
                ServiceClient()

    def test_init_with_api_key(self):
        """Test ServiceClient initialization with API key."""
        client = ServiceClient(api_key="test_key", endpoint="http://test.local")
        assert client.api_key == "test_key"
        assert client.endpoint == "http://test.local"
        client.close()

    def test_create_lora_training_client_validates_target_modules(self):
        """Test that at least one target module must be enabled."""
        with patch("httpx.Client"):
            client = ServiceClient(api_key="test_key")

            with pytest.raises(ValidationError, match="At least one of"):
                client.create_lora_training_client(
                    "test-model",
                    train_mlp=False,
                    train_attn=False,
                    train_unembed=False,
                )
            client.close()

    def test_create_rest_client(self):
        """Test creating RestClient."""
        with patch("httpx.Client"):
            client = ServiceClient(api_key="test_key")
            rest = client.create_rest_client()
            assert isinstance(rest, RestClient)
            rest.close()
            client.close()

    def test_create_sampling_client_requires_model(self):
        """Test that SamplingClient requires model_path or base_model."""
        with patch("httpx.Client"):
            client = ServiceClient(api_key="test_key")

            with pytest.raises(ValueError, match="Either model_path or base_model"):
                client.create_sampling_client()

            client.close()


class TestTrainingClient:
    """Tests for TrainingClient."""

    def test_forward_backward_builds_request(self):
        """Test forward_backward request building."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "loss": 2.5,
            "tokens_processed": 10,
        }
        mock_client.post.return_value = mock_response

        training = TrainingClient(
            client=mock_client,
            session_id="test-session",
            internal_id="train-test",
            base_model="test-model",
        )

        future = training.forward_backward([Datum(input_ids=[1, 2, 3])])
        result = future.result()

        assert result.loss == 2.5
        assert result.tokens_processed == 10
        mock_client.post.assert_called()

    def test_optim_step(self):
        """Test optimizer step."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {"step": 5}
        mock_client.post.return_value = mock_response

        training = TrainingClient(
            client=mock_client,
            session_id="test-session",
            internal_id="train-test",
            base_model="test-model",
        )

        future = training.optim_step()
        step = future.result()

        assert step == 5


class TestRestClient:
    """Tests for RestClient."""

    def test_list_training_runs(self):
        """Test listing training runs."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "runs": [
                {"runId": "run-1", "baseModel": "model-1"},
                {"runId": "run-2", "baseModel": "model-2"},
            ]
        }
        mock_client.get.return_value = mock_response

        rest = RestClient(client=mock_client)
        runs = rest.list_training_runs().result()

        assert len(runs) == 2
        assert runs[0]["runId"] == "run-1"
        rest.close()

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "checkpoints": [
                {"checkpointId": "cp-1", "step": 100},
            ]
        }
        mock_client.get.return_value = mock_response

        rest = RestClient(client=mock_client)
        checkpoints = rest.list_checkpoints(run_id="run-1").result()

        assert len(checkpoints) == 1
        assert checkpoints[0]["checkpointId"] == "cp-1"
        rest.close()

    def test_get_checkpoint_archive_url(self):
        """Test getting checkpoint download URL."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {"url": "https://example.com/checkpoint.tar"}
        mock_client.get.return_value = mock_response

        rest = RestClient(client=mock_client)
        url = rest.get_checkpoint_archive_url("cp-123").result()

        assert url == "https://example.com/checkpoint.tar"
        rest.close()


class TestGetServerCapabilitiesResponse:
    """Tests for GetServerCapabilitiesResponse."""

    def test_from_dict(self):
        """Test creating response from dict (snake_case)."""
        data = {
            "models": ["model-1", "model-2"],
            "max_batch_tokens": 4096,
            "max_sequence_length": 2048,
        }

        # Test that the response can be created from dict
        response = GetServerCapabilitiesResponse(**data)
        assert response.models == ["model-1", "model-2"]
        assert response.max_batch_tokens == 4096
        assert response.max_sequence_length == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
