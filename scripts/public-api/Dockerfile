# Build stage
FROM rust:1.87-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install rustfmt (required by crabtensor build)
RUN rustup component add rustfmt

# Set working directory
WORKDIR /workspace

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY crates/ ./crates/

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""

# Build the public-api binary
RUN if [ "$BUILD_MODE" = "release" ]; then \
        if [ -n "$FEATURES" ]; then \
            cargo build --release -p public-api --features "$FEATURES"; \
        else \
            cargo build --release -p public-api; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build -p public-api --features "$FEATURES"; \
        else \
            cargo build -p public-api; \
        fi; \
    fi

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
COPY --from=builder /workspace/target/${BUILD_MODE}/public-api /usr/local/bin/public-api

# Make binary executable
RUN chmod +x /usr/local/bin/public-api

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash basilica

# Create config directory
RUN mkdir -p /etc/basilica && chown basilica:basilica /etc/basilica

# Switch to non-root user
USER basilica

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["/usr/local/bin/public-api"]
