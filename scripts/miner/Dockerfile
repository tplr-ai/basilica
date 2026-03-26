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

# Set environment variables for the build - default to test network if not specified
ENV BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-test}
ENV METADATA_CHAIN_ENDPOINT=${METADATA_CHAIN_ENDPOINT}

# Set environment variables to reduce build size and use less disk space
ENV CARGO_TARGET_DIR=/tmp/target
ENV CARGO_INCREMENTAL=0
ENV RUST_BACKTRACE=1

# Build the miner directly (workspace should resolve dependencies)
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
            cargo build --release -p basilica-miner --features "$FEATURES"; \
        else \
            cargo build --release -p basilica-miner; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p basilica-miner --features "$FEATURES"; \
        else \
            cargo build -p basilica-miner; \
        fi; \
    fi && \
    # Copy binary out of cache mount to a regular layer
    cp /tmp/target/${BUILD_MODE}/basilica-miner /workspace/basilica-miner

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    libsqlite3-0 \
    openssh-client \
    procps \
    iproute2 \
    net-tools \
    libfuse3-3 \
    fuse3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /workspace/basilica-miner /usr/local/bin/basilica-miner

# Make binary executable
RUN chmod +x /usr/local/bin/basilica-miner

# Create required directories
RUN mkdir -p /var/lib/basilica/miner/data && \
    mkdir -p /var/lib/basilica/miner/.ssh && \
    mkdir -p /var/lib/basilica/miner/.bittensor && \
    chmod 755 /var/lib/basilica/miner && \
    chmod 755 /var/lib/basilica/miner/.ssh && \
    chmod 755 /var/lib/basilica/miner/.bittensor

# Set HOME environment variable for the basilica user
ENV HOME=/var/lib/basilica/miner

# Set working directory
WORKDIR /var/lib/basilica/miner

# Expose ports
# 8080: Miner gRPC server for validator communications
# 8091: Bittensor axon server port
# 9090: Prometheus metrics endpoint
EXPOSE 8080 8091 9090

# Default command
ENTRYPOINT ["/usr/local/bin/basilica-miner"]
