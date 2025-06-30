FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Install Rust
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.87.0
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup component add rustfmt

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY src/ ./src/

ARG VALIDATOR_PUBLIC_KEY
ENV VALIDATOR_PUBLIC_KEY=${VALIDATOR_PUBLIC_KEY}

ARG FEATURES=""
ENV FEATURES=${FEATURES}

RUN if [ -z "$VALIDATOR_PUBLIC_KEY" ]; then \
        echo "Error: VALIDATOR_PUBLIC_KEY build argument is required"; \
        exit 1; \
    fi

RUN echo "$VALIDATOR_PUBLIC_KEY" | grep -qE '^[0-9a-fA-F]{66}$' || \
    (echo "Error: VALIDATOR_PUBLIC_KEY must be 66 hexadecimal characters" && exit 1)

RUN if [ -n "$FEATURES" ]; then \
        echo "Building with features: $FEATURES"; \
        cargo build --release -p gpu-attestor --features "$FEATURES"; \
    else \
        echo "Building without additional features"; \
        cargo build --release -p gpu-attestor; \
    fi

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false attestor

COPY --from=builder /workspace/target/release/gpu-attestor /usr/local/bin/gpu-attestor

RUN chmod +x /usr/local/bin/gpu-attestor
RUN mkdir -p /attestation && chown attestor:attestor /attestation

USER attestor
WORKDIR /attestation

ENTRYPOINT ["/usr/local/bin/gpu-attestor"]
CMD ["--output", "/attestation"]
