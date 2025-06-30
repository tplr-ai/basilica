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

# Build the validator directly (workspace should resolve dependencies)
RUN if [ "$BUILD_MODE" = "release" ]; then \
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
    fi

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
COPY --from=builder /workspace/target/${BUILD_MODE}/validator /usr/local/bin/validator

# Make binary executable
RUN chmod +x /usr/local/bin/validator

# Create data directory for SQLite database
RUN mkdir -p /var/lib/basilica/validator && \
    chmod 755 /var/lib/basilica/validator

EXPOSE 8080 9090

# Default command
ENTRYPOINT ["/usr/local/bin/validator"]