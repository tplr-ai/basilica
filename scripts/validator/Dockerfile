FROM rust:1.87.0-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

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

# Build the validator directly (workspace should resolve dependencies)
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
            cargo build --release -p validator --features "$FEATURES"; \
        else \
            cargo build --release -p validator; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p validator --features "$FEATURES"; \
        else \
            cargo build -p validator; \
        fi; \
    fi && \
    # Clean up intermediate build artifacts to save space - updated cleanup logic
    rm -rf /tmp/target/debug/deps /tmp/target/debug/incremental /tmp/target/release/deps /tmp/target/release/incremental 2>/dev/null || true && \
    find /tmp/target -name "*.rlib" -delete 2>/dev/null || true

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    libsqlite3-0 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /tmp/target/${BUILD_MODE}/validator /usr/local/bin/validator

# Make binary executable
RUN chmod +x /usr/local/bin/validator

# Create data directory for SQLite database
RUN mkdir -p /var/lib/basilica/validator && \
    chmod 755 /var/lib/basilica/validator

EXPOSE 8080 9090

# Default command
ENTRYPOINT ["/usr/local/bin/validator"]