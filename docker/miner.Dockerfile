# Miner Dockerfile
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

# Build miner - use BUILD_MODE arg to control debug/release
ARG BUILD_MODE=release
RUN if [ "$BUILD_MODE" = "debug" ]; then \
        cargo build -p miner; \
    else \
        cargo build --release -p miner; \
    fi

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    sqlite3 \
    curl \
    wget \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install grpc-health-probe
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.25 && \
    wget -qO/usr/local/bin/grpc-health-probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /usr/local/bin/grpc-health-probe

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash basilica

# Create necessary directories
RUN mkdir -p /var/lib/basilica /config /data && \
    chown -R basilica:basilica /var/lib/basilica /config /data

# Copy binary from builder - handle both debug and release paths
ARG BUILD_MODE=release
COPY --from=builder /usr/src/basilica/target/${BUILD_MODE}/miner /usr/local/bin/miner

# Set working directory
WORKDIR /var/lib/basilica

# Create a startup script to ensure permissions
RUN echo '#!/bin/bash\n\
if [ ! -w /data ]; then \n\
    echo "Warning: /data is not writable, database operations may fail" \n\
fi \n\
# Fix SSH permissions if needed\n\
if [ -d /root/.ssh ]; then \n\
    chmod 700 /root/.ssh 2>/dev/null || true \n\
    chmod 600 /root/.ssh/* 2>/dev/null || true \n\
fi \n\
exec /usr/local/bin/miner "$@"' > /usr/local/bin/start-miner.sh && \
    chmod +x /usr/local/bin/start-miner.sh

# Keep running as root for SSH access
# USER basilica

# Expose miner API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default environment variables
ENV RUST_LOG=info
ENV BASILCA_CONFIG_FILE=/config/miner.toml

# Run miner through startup script
ENTRYPOINT ["/usr/local/bin/start-miner.sh"]