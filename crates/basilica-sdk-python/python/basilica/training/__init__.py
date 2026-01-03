"""
Basilica Training SDK

Fine-tune LLMs with LoRA on Basilica's GPU cloud.

Quick Start:
    >>> from basilica.training import ServiceClient, Datum
    >>>
    >>> client = ServiceClient()
    >>> with client.create_lora_training_client(
    ...     "meta-llama/Llama-3.1-8B-Instruct",
    ...     rank=32,
    ...     train_mlp=True,
    ...     train_attn=True,
    ... ) as training:
    ...     result = training.forward_backward([Datum(input_ids=[1, 2, 3])]).result()
    ...     training.optim_step().result()
    ...     print(training.sample("Hello!"))

Loss Functions:
    >>> # Standard cross-entropy (default)
    >>> result = training.forward_backward(data).result()
    >>>
    >>> # Importance sampling (policy gradient)
    >>> result = training.forward_backward(data, loss_fn="importance_sampling").result()
    >>>
    >>> # PPO with clipping
    >>> result = training.forward_backward(data, loss_fn="ppo").result()
    >>>
    >>> # DPO (Direct Preference Optimization)
    >>> result = training.forward_backward(data, loss_fn="dpo").result()

Authentication:
    export BASILICA_API_KEY="basilica_..."
"""

__version__ = "0.2.0"

# Import from modules
from .types import (
    Datum,
    ModelInput,
    SamplingParams,
    SampleResponse,
    ForwardBackwardResult,
    ForwardResult,
    GetServerCapabilitiesResponse,
    APIFuture,
)

from .exceptions import (
    TrainingError,
    SessionNotFoundError,
    SessionNotReadyError,
    SessionTimeoutError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    CheckpointError,
    ModelNotFoundError,
    InsufficientResourcesError,
)

from .service_client import ServiceClient
from .training_client import TrainingClient
from .sampling_client import SamplingClient
from .rest_client import RestClient

# Export all public symbols
__all__ = [
    # Version
    "__version__",
    # Main clients
    "ServiceClient",
    "TrainingClient",
    "SamplingClient",
    "RestClient",
    # Types
    "Datum",
    "ModelInput",
    "SamplingParams",
    "SampleResponse",
    "ForwardBackwardResult",
    "ForwardResult",
    "GetServerCapabilitiesResponse",
    "APIFuture",
    # Exceptions
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
