FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Install Rust and system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.87.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Install rustfmt for code formatting
RUN rustup component add rustfmt

WORKDIR /workspace

# Copy workspace files for dependency resolution
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Pre-build dependencies to cache them
RUN cargo fetch

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""
ARG BITTENSOR_NETWORK="test"
ARG METADATA_CHAIN_ENDPOINT=""

# Set environment variables for the build - default to test network if not specified
ENV BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-test}
ENV METADATA_CHAIN_ENDPOINT=${METADATA_CHAIN_ENDPOINT}

# Set environment variables to reduce build size and use less disk space
ENV CARGO_TARGET_DIR=/tmp/target
ENV CARGO_INCREMENTAL=0
ENV RUST_BACKTRACE=1

# Build the executor directly (workspace should resolve dependencies)
# Set explicit endpoint if network is specified but endpoint is not
RUN if [ -n "$BITTENSOR_NETWORK" ] && [ -z "$METADATA_CHAIN_ENDPOINT" ]; then \
        case "$BITTENSOR_NETWORK" in \
            test) export METADATA_CHAIN_ENDPOINT="wss://test.finney.opentensor.ai:443" ;; \
            local) export METADATA_CHAIN_ENDPOINT="ws://subtensor:9944" ;; \
            finney|*) export METADATA_CHAIN_ENDPOINT="wss://entrypoint-finney.opentensor.ai:443" ;; \
        esac; \
    fi && \
    if [ "$BUILD_MODE" = "release" ]; then \
        if [ -n "$FEATURES" ]; then \
            cargo build --release -p executor --features "$FEATURES"; \
        else \
            cargo build --release -p executor; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p executor --features "$FEATURES"; \
        else \
            cargo build -p executor; \
        fi; \
    fi && \
    # Clean up intermediate build artifacts to save space
    rm -rf /tmp/target/debug/deps /tmp/target/debug/incremental /tmp/target/release/deps /tmp/target/release/incremental && \
    find /tmp/target -name "*.rlib" -delete 2>/dev/null || true

# Runtime image
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for NVML library (required for GPU detection)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so || true

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /tmp/target/${BUILD_MODE}/executor /usr/local/bin/executor

# Make binary executable
RUN chmod +x /usr/local/bin/executor

EXPOSE 8080 8081

# Default command
ENTRYPOINT ["/usr/local/bin/executor"]
