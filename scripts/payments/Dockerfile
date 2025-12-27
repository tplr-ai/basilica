# Build stage
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies and rustfmt
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt

WORKDIR /app

# Copy workspace files but only needed crates (plus operator to satisfy workspace)
COPY Cargo.toml Cargo.lock ./
COPY crates/basilica-payments ./crates/basilica-payments
COPY crates/basilica-common ./crates/basilica-common
COPY crates/basilica-protocol ./crates/basilica-protocol
COPY crates/bittensor ./crates/bittensor
COPY crates/basilica-operator ./crates/basilica-operator

# Remove unrelated workspace members from Cargo.toml to avoid copying full repo
RUN sed -i '/crates\/basilica-executor/d' Cargo.toml && \
    sed -i '/crates\/basilica-miner/d' Cargo.toml && \
    sed -i '/crates\/basilica-validator/d' Cargo.toml && \
    sed -i '/crates\/basilica-billing/d' Cargo.toml && \
    sed -i '/crates\/basilica-api/d' Cargo.toml && \
    sed -i '/crates\/basilica-cli/d' Cargo.toml && \
    sed -i '/crates\/basilica-tui/d' Cargo.toml && \
    sed -i '/crates\/basilica-sdk/d' Cargo.toml && \
    sed -i '/crates\/basilica-sdk-python/d' Cargo.toml && \
    sed -i '/crates\/collateral/d' Cargo.toml && \
    sed -i '/crates\/basilica-storage/d' Cargo.toml && \
    sed -i '/crates\/basilica-aggregator/d' Cargo.toml && \
    sed -i '/crates\/basilica-autoscaler/d' Cargo.toml && \
    sed -i '/crates\/integration-tests/d' Cargo.toml

# Build the payments service
RUN cargo build --release -p basilica-payments

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder to standard location
COPY --from=builder /app/target/release/basilica-payments /usr/local/bin/basilica-payments

# Create non-root user
RUN useradd -m -u 1000 payments

USER payments

# Expose ports
EXPOSE 50061 8082 9092

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/basilica-payments"]
