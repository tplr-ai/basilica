# Dockerfile for vLLM with Kimi-K2.5 support
# Builds from vLLM nightly wheel per Moonshot deployment guidance:
# https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/docs/deploy_guidance.md

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Install vLLM from nightly wheels (required for KimiK25ForConditionalGeneration)
# Per Moonshot guidance: uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
RUN uv pip install --no-cache-dir \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN uv pip install --no-cache-dir \
    vllm \
    --extra-index-url https://wheels.vllm.ai/nightly

# Install additional dependencies for Kimi models
RUN uv pip install --no-cache-dir \
    tiktoken \
    transformers>=4.57.1

# Set working directory
WORKDIR /app

# Expose vLLM API port
EXPOSE 8000

# Default entrypoint for vLLM serve
ENTRYPOINT ["vllm"]
CMD ["--help"]
