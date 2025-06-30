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

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY crates/ ./crates/

# Build arguments
ARG BUILD_MODE=release
ARG FEATURES=""

# Build the executor directly (workspace should resolve dependencies)
RUN if [ "$BUILD_MODE" = "release" ]; then \
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
    fi

# Runtime image
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /workspace/target/${BUILD_MODE}/executor /usr/local/bin/executor

# Make binary executable
RUN chmod +x /usr/local/bin/executor

EXPOSE 8080 8081

# Default command
ENTRYPOINT ["/usr/local/bin/executor"]
