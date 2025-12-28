# Build stage
FROM rust:1.88-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    python3-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install rustfmt (required by some dependencies)
RUN rustup component add rustfmt

# Set working directory
WORKDIR /workspace

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Fetch dependencies
RUN cargo fetch

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""

# Set build environment for network if needed
ARG BITTENSOR_NETWORK=finney
ENV BITTENSOR_NETWORK=$BITTENSOR_NETWORK

# Auth0 configuration build args
ARG BASILICA_AUTH0_CLIENT_ID
ARG BASILICA_AUTH0_AUDIENCE
ARG BASILICA_AUTH0_ISSUER
ARG BASILICA_AUTH0_DOMAIN
ENV BASILICA_AUTH0_CLIENT_ID=$BASILICA_AUTH0_CLIENT_ID
ENV BASILICA_AUTH0_AUDIENCE=$BASILICA_AUTH0_AUDIENCE
ENV BASILICA_AUTH0_ISSUER=$BASILICA_AUTH0_ISSUER
ENV BASILICA_AUTH0_DOMAIN=$BASILICA_AUTH0_DOMAIN

# Set space-saving environment variables
ENV CARGO_TARGET_DIR=/tmp/target
ENV CARGO_INCREMENTAL=0
ENV RUST_BACKTRACE=1

# Build the basilica-cli binary
RUN if [ "$BUILD_MODE" = "release" ]; then \
        if [ -n "$FEATURES" ]; then \
            cargo build --locked --release -p basilica-cli --features "$FEATURES"; \
        else \
            cargo build --locked --release -p basilica-cli; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build --locked -p basilica-cli --features "$FEATURES"; \
        else \
            cargo build --locked -p basilica-cli; \
        fi; \
    fi && \
    # Clean up intermediate build artifacts to save space
    rm -rf /tmp/target/debug/deps /tmp/target/debug/incremental /tmp/target/release/deps /tmp/target/release/incremental && \
    find /tmp/target -name "*.rlib" -delete 2>/dev/null || true

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /tmp/target/${BUILD_MODE}/basilica /usr/local/bin/basilica

# Make binary executable
RUN chmod +x /usr/local/bin/basilica

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash basilica

# Create config directory
RUN mkdir -p /home/basilica/.basilica && chown basilica:basilica /home/basilica/.basilica

# Switch to non-root user
USER basilica

# Set working directory to user home
WORKDIR /home/basilica

# Default command
ENTRYPOINT ["/usr/local/bin/basilica"]
