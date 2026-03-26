# syntax=docker/dockerfile:1
FROM rust:1.88-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    libfuse3-dev \
    fuse3 \
    && rm -rf /var/lib/apt/lists/*

# Install rustfmt for code formatting
RUN rustup component add rustfmt

WORKDIR /workspace

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Fetch dependencies with cache mount
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=basilica-cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=basilica-cargo-git \
    cargo fetch

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""
ARG BITTENSOR_NETWORK="finney"
ARG METADATA_CHAIN_ENDPOINT=""

# Set environment variables for the build - default to finney network if not specified
ENV BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-finney}
ENV METADATA_CHAIN_ENDPOINT=${METADATA_CHAIN_ENDPOINT}

# Set environment variables to reduce build size and use less disk space
ENV CARGO_TARGET_DIR=/tmp/target
ENV CARGO_INCREMENTAL=0
ENV RUST_BACKTRACE=1
# Add retry configuration for network operations during build
ENV CARGO_NET_RETRY=10
ENV CARGO_HTTP_TIMEOUT=120

# Build the validator directly (workspace should resolve dependencies)
# Set explicit endpoint if network is specified but endpoint is not
# Use cache mounts for cargo registry and target to speed up rebuilds
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=basilica-cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=basilica-cargo-git \
    --mount=type=cache,target=/tmp/target,id=basilica-cargo-target \
    if [ -n "$BITTENSOR_NETWORK" ] && [ -z "$METADATA_CHAIN_ENDPOINT" ]; then \
        case "$BITTENSOR_NETWORK" in \
            test) export METADATA_CHAIN_ENDPOINT="wss://test.finney.opentensor.ai:443" ;; \
            local) export METADATA_CHAIN_ENDPOINT="ws://subtensor:9944" ;; \
            finney|*) export METADATA_CHAIN_ENDPOINT="wss://entrypoint-finney.opentensor.ai:443" ;; \
        esac; \
    fi && \
    if [ "$BUILD_MODE" = "release" ]; then \
        if [ -n "$FEATURES" ]; then \
            cargo build --release -p basilica-validator --bin basilica-validator --features "$FEATURES"; \
        else \
            cargo build --release -p basilica-validator --bin basilica-validator; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p basilica-validator --bin basilica-validator --features "$FEATURES"; \
        else \
            cargo build -p basilica-validator --bin basilica-validator; \
        fi; \
    fi && \
    # Copy binary out of cache mount to a regular layer
    cp /tmp/target/${BUILD_MODE}/basilica-validator /workspace/basilica-validator

# Runtime image
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    libsqlite3-0 \
    openssh-client \
    libfuse3-3 \
    fuse3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /workspace/basilica-validator /usr/local/bin/basilica-validator

# Create basilica bin directory
RUN mkdir -p /opt/basilica/bin

# Copy veritas binaries if they exist in build context
COPY bin/executor-binary /opt/basilica/bin/executor-binary
COPY bin/validator-binary /opt/basilica/bin/validator-binary

# Make binaries executable
RUN chmod +x /usr/local/bin/basilica-validator && \
    chmod +x /opt/basilica/bin/executor-binary 2>/dev/null || true && \
    chmod +x /opt/basilica/bin/validator-binary 2>/dev/null || true

# Create data directory for SQLite database
RUN mkdir -p /var/lib/basilica/basilica-validator && \
    chmod 755 /var/lib/basilica/basilica-validator

EXPOSE 8080 9090 50052

# Default command
ENTRYPOINT ["/usr/local/bin/basilica-validator"]
