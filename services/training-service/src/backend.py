"""Core training backend using HuggingFace + PEFT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = structlog.get_logger()


@dataclass
class LoraConfiguration:
    """LoRA adapter configuration."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class OptimizerConfiguration:
    """Optimizer configuration."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    grad_clip: Optional[float] = 1.0


@dataclass
class SessionState:
    """State for a training session."""

    session_id: str
    base_model_name: str
    model: PeftModel
    optimizer: torch.optim.AdamW
    lora_config: LoraConfiguration
    optimizer_config: OptimizerConfiguration
    step_count: int = 0
    tokens_processed: int = 0


@dataclass
class ForwardBackwardResult:
    """Result of forward-backward pass."""

    loss: float
    logprobs: List[float]
    tokens_processed: int


@dataclass
class SampleResult:
    """Result of text generation."""

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"


class TrainingBackend:
    """Single-node training backend using HuggingFace + PEFT."""

    def __init__(
        self,
        model_cache_dir: str = "/models",
        checkpoint_dir: str = "/checkpoints",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_cache_dir = Path(model_cache_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Resolve actual device - avoid MPS for bfloat16 issues
        if device == "cuda" and not torch.cuda.is_available():
            # On Mac, use CPU to avoid MPS bfloat16 issues
            self.device = "cpu"
            self.dtype = torch.float32  # bfloat16 not well supported on CPU
        else:
            self.device = device
            self.dtype = dtype

        self.sessions: Dict[str, SessionState] = {}
        self.base_models: Dict[str, PreTrainedModel] = {}
        self.tokenizers: Dict[str, Any] = {}

        # Ensure directories exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "training_backend_initialized",
            device=device,
            dtype=str(dtype),
            cache_dir=str(model_cache_dir),
        )

    def _load_base_model(self, model_name: str) -> PreTrainedModel:
        """Load or retrieve cached base model."""
        if model_name not in self.base_models:
            logger.info("loading_base_model", model=model_name)

            # Use device_map="auto" only for CUDA, otherwise load to specific device
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map="auto",
                    cache_dir=self.model_cache_dir,
                    trust_remote_code=True,
                )
            else:
                # For CPU, load directly without device_map to avoid MPS issues
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    cache_dir=self.model_cache_dir,
                    trust_remote_code=True,
                ).to(self.device)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                trust_remote_code=True,
            )

            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.base_models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            logger.info("base_model_loaded", model=model_name)

        return self.base_models[model_name]

    def _get_tokenizer(self, model_name: str) -> Any:
        """Get tokenizer for a model."""
        if model_name not in self.tokenizers:
            self._load_base_model(model_name)
        return self.tokenizers[model_name]

    def create_session(
        self,
        session_id: str,
        base_model: str,
        lora_config: LoraConfiguration,
        optimizer_config: OptimizerConfiguration,
        seed: Optional[int] = None,
    ) -> str:
        """Create a new training session with LoRA adapter."""

        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")

        if seed is not None:
            torch.manual_seed(seed)

        logger.info(
            "creating_session",
            session_id=session_id,
            base_model=base_model,
            lora_rank=lora_config.rank,
        )

        # Load base model
        base = self._load_base_model(base_model)

        # Create PEFT config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias="none",
        )

        # Wrap with LoRA
        model = get_peft_model(base, peft_config)
        model.print_trainable_parameters()

        # Create optimizer (only for LoRA parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.epsilon,
        )

        # Store session
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            base_model_name=base_model,
            model=model,
            optimizer=optimizer,
            lora_config=lora_config,
            optimizer_config=optimizer_config,
        )

        logger.info("session_created", session_id=session_id)
        return session_id

    def forward_backward(
        self,
        session_id: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> ForwardBackwardResult:
        """Compute forward pass and gradients."""

        session = self._get_session(session_id)
        model = session.model
        model.train()

        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        if loss_weights is not None:
            loss_weights = loss_weights.to(self.device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Compute loss (with optional weighting)
        if loss_weights is not None:
            # Custom weighted cross-entropy
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            token_losses = token_losses.view(shift_labels.size())
            loss = (token_losses * shift_weights).sum() / shift_weights.sum().clamp(min=1)
        else:
            loss = outputs.loss

        # Backward pass
        loss.backward()

        # Compute logprobs for return
        with torch.no_grad():
            logprobs = self._compute_logprobs(outputs.logits, labels)

        tokens_processed = int(attention_mask.sum().item())
        session.tokens_processed += tokens_processed

        logger.debug(
            "forward_backward_complete",
            session_id=session_id,
            loss=loss.item(),
            tokens=tokens_processed,
        )

        return ForwardBackwardResult(
            loss=loss.item(),
            logprobs=logprobs.cpu().tolist(),
            tokens_processed=tokens_processed,
        )

    def optim_step(self, session_id: str) -> int:
        """Apply gradients and update weights."""

        session = self._get_session(session_id)

        # Gradient clipping
        if session.optimizer_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                session.model.parameters(),
                session.optimizer_config.grad_clip,
            )

        # Optimizer step
        session.optimizer.step()
        session.optimizer.zero_grad()
        session.step_count += 1

        logger.debug(
            "optim_step_complete",
            session_id=session_id,
            step=session.step_count,
        )

        return session.step_count

    def sample(
        self,
        session_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        include_logprobs: bool = False,
    ) -> SampleResult:
        """Generate text completion."""

        session = self._get_session(session_id)
        model = session.model
        model.eval()

        # Get tokenizer for the model
        tokenizer = self._get_tokenizer(session.base_model_name)

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=include_logprobs,
                return_dict_in_generate=True,
            )

        # Decode
        generated_ids = outputs.sequences[0, prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs if requested
        logprobs = None
        if include_logprobs and outputs.scores:
            logprobs = []
            for i, scores in enumerate(outputs.scores):
                if i < len(generated_ids):
                    log_probs = torch.nn.functional.log_softmax(scores[0], dim=-1)
                    token_id = generated_ids[i].item()
                    logprobs.append(log_probs[token_id].item())

        # Determine finish reason
        finish_reason = "length"
        if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
            finish_reason = "stop"

        return SampleResult(
            text=text,
            token_ids=generated_ids.cpu().tolist(),
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def save_state(
        self,
        session_id: str,
        checkpoint_name: str,
        include_optimizer: bool = True,
    ) -> str:
        """Save LoRA weights and optimizer state."""

        session = self._get_session(session_id)

        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / session_id / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights (safetensors)
        session.model.save_pretrained(checkpoint_path)

        # Save optimizer state
        if include_optimizer:
            torch.save(
                {
                    "optimizer_state_dict": session.optimizer.state_dict(),
                    "step_count": session.step_count,
                    "tokens_processed": session.tokens_processed,
                    "lora_config": {
                        "rank": session.lora_config.rank,
                        "alpha": session.lora_config.alpha,
                        "dropout": session.lora_config.dropout,
                        "target_modules": session.lora_config.target_modules,
                    },
                    "optimizer_config": {
                        "learning_rate": session.optimizer_config.learning_rate,
                        "weight_decay": session.optimizer_config.weight_decay,
                        "beta1": session.optimizer_config.beta1,
                        "beta2": session.optimizer_config.beta2,
                        "epsilon": session.optimizer_config.epsilon,
                        "grad_clip": session.optimizer_config.grad_clip,
                    },
                },
                checkpoint_path / "training_state.pt",
            )

        logger.info(
            "checkpoint_saved",
            session_id=session_id,
            checkpoint=checkpoint_name,
            path=str(checkpoint_path),
        )

        return str(checkpoint_path)

    def load_state(
        self,
        session_id: str,
        checkpoint_path: str,
        load_optimizer: bool = True,
    ) -> None:
        """Load LoRA weights and optimizer state."""

        session = self._get_session(session_id)
        checkpoint_path = Path(checkpoint_path)

        # Load adapter weights
        session.model = PeftModel.from_pretrained(
            session.model.base_model,
            checkpoint_path,
        )

        # Load optimizer state
        if load_optimizer and (checkpoint_path / "training_state.pt").exists():
            state = torch.load(checkpoint_path / "training_state.pt", weights_only=False)
            session.optimizer.load_state_dict(state["optimizer_state_dict"])
            session.step_count = state["step_count"]
            session.tokens_processed = state["tokens_processed"]

        logger.info(
            "checkpoint_loaded",
            session_id=session_id,
            path=str(checkpoint_path),
        )

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status."""
        session = self._get_session(session_id)
        return {
            "session_id": session_id,
            "base_model": session.base_model_name,
            "step_count": session.step_count,
            "tokens_processed": session.tokens_processed,
            "lora_rank": session.lora_config.rank,
            "learning_rate": session.optimizer_config.learning_rate,
        }

    def delete_session(self, session_id: str) -> None:
        """Delete a training session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("session_deleted", session_id=session_id)

    def list_sessions(self) -> List[str]:
        """List all active sessions."""
        return list(self.sessions.keys())

    def _get_session(self, session_id: str) -> SessionState:
        """Get session or raise error."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_logprobs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Flatten for return
        return token_logprobs.flatten()
