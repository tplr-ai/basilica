# Use NVIDIA CUDA image for GPU support, or Python slim for CPU-only
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e "."

# Copy source code
COPY src/ src/
COPY tests/ tests/

# Create directories for models and checkpoints
RUN mkdir -p /models /checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/models
ENV CHECKPOINT_DIR=/checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run service
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
