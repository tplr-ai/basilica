# Executor Dockerfile with GPU support
# Multi-stage build for optimal image size

# Build stage
FROM rust:latest AS builder

# Install rustfmt which is required by crabtensor build
RUN rustup component add rustfmt

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    xxd \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/basilica

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY src/ ./src/

# Build executor and gpu-attestor - use BUILD_MODE arg to control debug/release
ARG BUILD_MODE=release
RUN if [ "$BUILD_MODE" = "debug" ]; then \
        cargo build -p executor; \
    else \
        cargo build --release -p executor; \
    fi

# Runtime stage - based on NVIDIA CUDA image for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    dmidecode \
    pciutils \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for container management
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /var/lib/basilica /config /var/run/sshd /data && \
    chmod 755 /data

# Copy binary from builder - handle both debug and release paths
ARG BUILD_MODE=release
COPY --from=builder /usr/src/basilica/target/${BUILD_MODE}/executor /usr/local/bin/executor
COPY --from=builder /usr/src/basilica/target/${BUILD_MODE}/gpu-attestor /usr/local/bin/gpu-attestor

# Set working directory
WORKDIR /var/lib/basilica

# Configure SSH for validator access
RUN echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# Expose executor gRPC port and SSH port
EXPOSE 50051 22

# Health check - using grpc_health_probe for gRPC services
# For now, we'll disable it since the executor is running fine
# HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
#     CMD grpc_health_probe -addr=:50051 || exit 1

# Default environment variables
ENV RUST_LOG=info
ENV BASILCA_CONFIG_FILE=/config/executor.toml
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Script to start services
RUN echo '#!/bin/bash\n\
# Start SSH service\n\
service ssh start\n\
\n\
# Start executor\n\
exec /usr/local/bin/executor "$@"\n\
' > /usr/local/bin/start-executor.sh && chmod +x /usr/local/bin/start-executor.sh

# Run as root (required for system monitoring and Docker socket access)
USER root

# Use the startup script
ENTRYPOINT ["/usr/local/bin/start-executor.sh"]