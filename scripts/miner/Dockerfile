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

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""
ARG BITTENSOR_NETWORK="test"
ARG METADATA_CHAIN_ENDPOINT=""
ARG VALIDATOR_PUBLIC_KEY=""

# Set environment variables for the build - default to test network if not specified
ENV BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-test}
ENV METADATA_CHAIN_ENDPOINT=${METADATA_CHAIN_ENDPOINT}
ENV VALIDATOR_PUBLIC_KEY=${VALIDATOR_PUBLIC_KEY}

# Build the miner directly (workspace should resolve dependencies)
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
            cargo build --release -p miner --features "$FEATURES"; \
        else \
            cargo build --release -p miner; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p miner --features "$FEATURES"; \
        else \
            cargo build -p miner; \
        fi; \
    fi

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /workspace/target/${BUILD_MODE}/miner /usr/local/bin/miner

# Make binary executable
RUN chmod +x /usr/local/bin/miner

# Create data directory for SQLite database and configuration
RUN mkdir -p /var/lib/basilica/miner && \
    chmod 755 /var/lib/basilica/miner

# Create user for running the service
RUN useradd --system --home /var/lib/basilica/miner --shell /bin/false basilica

# Set ownership of data directory
RUN chown -R basilica:basilica /var/lib/basilica/miner

# Switch to non-root user
USER basilica

# Set working directory
WORKDIR /var/lib/basilica/miner

# Expose ports
# 8092: Miner gRPC server for validator communications
# 8091: Bittensor axon server port
# 9092: Prometheus metrics endpoint
EXPOSE 8092 8091 9092

# Default command
ENTRYPOINT ["/usr/local/bin/miner"]