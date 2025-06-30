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
COPY src/ ./src/
COPY crates/ ./crates/

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""
ARG BITTENSOR_NETWORK=""
ARG METADATA_CHAIN_ENDPOINT=""

# Set environment variables for the build
ENV BITTENSOR_NETWORK=${BITTENSOR_NETWORK}
ENV METADATA_CHAIN_ENDPOINT=${METADATA_CHAIN_ENDPOINT}

# Build the miner directly (workspace should resolve dependencies)
RUN if [ "$BUILD_MODE" = "release" ]; then \
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