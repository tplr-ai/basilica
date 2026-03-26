"""
Model size estimation and GPU requirement calculation.

Estimates GPU memory requirements based on model name patterns.
Uses heuristics based on parameter count extracted from model names.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

# Default GPU memory for available GPUs (16GB for RTX A4000)
DEFAULT_GPU_MEMORY_GB = 16

# Regex pattern for extracting parameter count from model names
# Matches patterns like "7b", "70b", "0.5b", "1.5b", "7B", etc.
PARAM_PATTERN = re.compile(r"(\d+\.?\d*)b", re.IGNORECASE)


@dataclass
class GpuRequirements:
    """GPU requirements for a model."""

    gpu_count: int
    memory_gb: int
    recommended_gpu: str


def estimate_gpu_requirements(model: str) -> GpuRequirements:
    """
    Estimate GPU requirements based on model name.

    Uses heuristics based on parameter count extracted from model name.
    Falls back to model family detection if parameter count cannot be extracted.

    Args:
        model: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b")

    Returns:
        GpuRequirements with gpu_count, memory_gb, and recommended_gpu
    """
    model_lower = model.lower()

    # Extract parameter count from model name
    params = extract_param_count(model_lower)

    if params is not None:
        memory_gb = estimate_memory_from_params(params)
    else:
        memory_gb = estimate_from_model_family(model_lower)

    gpu_count = calculate_gpu_count(memory_gb, DEFAULT_GPU_MEMORY_GB)
    recommended_gpu = recommend_gpu(memory_gb)

    return GpuRequirements(
        gpu_count=gpu_count,
        memory_gb=memory_gb,
        recommended_gpu=recommended_gpu,
    )


def extract_param_count(model: str) -> Optional[float]:
    """
    Extract billion parameter count from model name.

    Patterns: "7b", "70b", "0.5b", "1.5b", "7B", etc.

    Args:
        model: Lowercase model name

    Returns:
        Parameter count in billions, or None if not found
    """
    match = PARAM_PATTERN.search(model)
    if match:
        return float(match.group(1))
    return None


def estimate_memory_from_params(params_billions: float) -> int:
    """
    Estimate VRAM needed in GB from parameter count.

    Rule of thumb: ~2GB per billion parameters for FP16.
    Add 20% overhead for KV cache and runtime.

    Args:
        params_billions: Model size in billions of parameters

    Returns:
        Estimated VRAM in GB, rounded to nearest 8GB
    """
    # Base memory: 2GB per billion params * 1.2 overhead
    base = params_billions * 2.0 * 1.2
    # Round up to nearest 8GB
    return ((int(base) + 7) // 8) * 8


def estimate_from_model_family(model: str) -> int:
    """
    Estimate memory from model family when parameter count is not available.

    Args:
        model: Lowercase model name

    Returns:
        Estimated VRAM in GB
    """
    # Check for known large models
    if any(
        x in model
        for x in ["llama-2-70b", "llama-70b", "mixtral", "qwen-72b"]
    ):
        return 160  # ~70B params

    if any(
        x in model
        for x in ["llama-2-13b", "llama-13b", "codellama-34b"]
    ):
        return 32  # ~13-34B params

    if any(
        x in model
        for x in ["llama-2-7b", "llama-7b", "mistral-7b", "qwen-7b"]
    ):
        return 16  # ~7B params

    if any(
        x in model
        for x in ["phi-2", "gemma-2b", "tinyllama", "qwen3-0.6b", "qwen2.5-0.5b"]
    ):
        return 8  # Small models

    # Default: assume medium-sized model
    return 16


def calculate_gpu_count(required_memory_gb: int, gpu_memory_gb: int = 16) -> int:
    """
    Calculate number of GPUs needed.

    Args:
        required_memory_gb: Required VRAM in GB
        gpu_memory_gb: Available memory per GPU (default: 16GB)

    Returns:
        Number of GPUs needed (clamped to 1-8)
    """
    count = math.ceil(required_memory_gb / gpu_memory_gb)
    return min(max(count, 1), 8)  # Clamp 1-8


def recommend_gpu(memory_gb: int) -> str:
    """
    Recommend GPU based on memory requirements.

    Args:
        memory_gb: Required VRAM in GB

    Returns:
        Recommended GPU model name
    """
    if memory_gb <= 16:
        return "NVIDIA-RTX-A4000"
    elif memory_gb <= 40:
        return "A100-40GB"
    elif memory_gb <= 80:
        return "A100-80GB"
    else:
        return "H100"
