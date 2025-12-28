# Build stage
FROM rust:1.88-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    libfuse3-dev \
    fuse3 \
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

# Set space-saving environment variables
ENV CARGO_TARGET_DIR=/tmp/target
ENV CARGO_INCREMENTAL=0
ENV RUST_BACKTRACE=1
# Limit cargo parallelism to avoid "too many open files" errors
ENV CARGO_BUILD_JOBS=20

# Build the basilica-api binary
RUN if [ "$BUILD_MODE" = "release" ]; then \
        if [ -n "$FEATURES" ]; then \
            cargo build --locked --release -p basilica-api --features "$FEATURES"; \
        else \
            cargo build --locked --release -p basilica-api; \
        fi; \
    else \
        if [ -n "$FEATURES" ]; then \
            cargo build --locked -p basilica-api --features "$FEATURES"; \
        else \
            cargo build --locked -p basilica-api; \
        fi; \
    fi && \
    # Clean up intermediate build artifacts to save space
    rm -rf /tmp/target/debug/deps /tmp/target/debug/incremental /tmp/target/release/deps /tmp/target/release/incremental && \
    find /tmp/target -name "*.rlib" -delete 2>/dev/null || true

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies including Python for btcli
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    wget \
    libfuse3-3 \
    fuse3 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
ARG BUILD_MODE=release
COPY --from=builder /tmp/target/${BUILD_MODE}/basilica-api /usr/local/bin/basilica-api

# Make binary executable
RUN chmod +x /usr/local/bin/basilica-api

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash basilica

RUN mkdir -p /etc/basilica && chown basilica:basilica /etc/basilica
RUN mkdir -p /home/basilica/.bittensor/wallets && chown -R basilica:basilica /home/basilica/.bittensor

# Install bittensor for programmatic wallet creation
RUN pip3 install --no-cache-dir bittensor --break-system-packages

# Switch to basilica user to generate wallet
USER basilica

# Generate read-only Bittensor wallet with no password for metagraph queries
# This wallet is only used for read-only operations (querying metagraph)
# It does NOT hold funds and does NOT sign transactions
RUN python3 -c "\
import bittensor as bt; \
wallet = bt.Wallet(name='default', hotkey='default'); \
wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False); \
print('Wallet created successfully')"

# Verify wallet was created
RUN test -f /home/basilica/.bittensor/wallets/default/hotkeys/default || \
    (echo "ERROR: Wallet generation failed" && exit 1)

# Expose port
EXPOSE 8000


# Default command
ENTRYPOINT ["/usr/local/bin/basilica-api"]
