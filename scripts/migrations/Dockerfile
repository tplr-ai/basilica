# AEAD Key Migration Tool Dockerfile
# Builds a minimal container for running the migration in ECS Fargate

FROM rust:1.80-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy migration source
COPY Cargo.toml migrate_aead_key.rs ./

# Build release binary
RUN cargo build --release --bin migrate-aead-key

# Runtime stage - minimal image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 1000 -s /bin/false migration

WORKDIR /app

COPY --from=builder /build/target/release/migrate-aead-key /app/migrate-aead-key

RUN chown migration:migration /app/migrate-aead-key

USER migration

ENTRYPOINT ["/app/migrate-aead-key"]
