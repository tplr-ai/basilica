"""
Deployment templates for common ML inference frameworks.

This module provides pre-configured deployment helpers for:
- vLLM: OpenAI-compatible LLM inference server
- SGLang: Fast LLM inference with RadixAttention
"""

from .model_size import GpuRequirements, estimate_gpu_requirements

__all__ = [
    "GpuRequirements",
    "estimate_gpu_requirements",
]
