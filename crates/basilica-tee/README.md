# basilica-tee

TEE (Trusted Execution Environment) support for Basilica, providing Intel TDX quote verification and NVIDIA GPU Confidential Computing attestation.

## Overview

This crate enables validators to verify that executor nodes are running in secure, hardware-backed environments:

- **Intel TDX**: Verify VMs are running in Trust Domain Extensions with attestable measurements
- **NVIDIA GPU CC**: Verify H100/H200 GPUs are in Confidential Computing mode

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `server` | ✅ | Axum-based HTTP attestation server |
| `nvml` | ❌ | NVIDIA NVML bindings for GPU device info |
| `remote-attestation` | ❌ | Remote attestation service integration |

## Modules

- **`tdx`** - Intel TDX quote parsing, generation, and verification
- **`gpu`** - NVIDIA GPU device info and CC attestation
- **`bootstrap`** - Remote TEE setup for executor nodes via SSH
- **`server`** - HTTP attestation server endpoints
- **`config`** - Configuration types
- **`types`** - Shared data types

## Usage

### TDX Quote Verification

```rust
use basilica_tee::tdx::{TdxQuoteV4, TdxQuoteVerifier};
use basilica_tee::types::ExpectedMeasurements;

// Parse a TDX quote
let quote = TdxQuoteV4::parse(&quote_bytes)?;

// Verify measurements
let expected = ExpectedMeasurements {
    mrtd: Some([0xAA; 48]),  // Expected MRTD value
    ..Default::default()
};

let verifier = TdxQuoteVerifier::new(expected);
let result = verifier.verify(&quote_bytes, Some(b"nonce"))?;

if result.quote_valid && result.mrtd_matches {
    println!("TDX verification passed");
}
```

### GPU CC Verification

```rust
use basilica_tee::gpu::{GpuEvidenceParser, GpuDeviceProvider};

// Parse GPU attestation evidence
let evidence = GpuEvidenceParser::parse(&evidence_json)?;

// Verify with nonce
let result = GpuEvidenceParser::verify(&evidence[0], Some("nonce"))?;

if result.cc_mode_enabled && result.nonce_verified {
    println!("GPU CC verification passed");
}
```

### Remote TEE Bootstrap

Validators can automatically set up TEE on executor nodes:

```rust
use basilica_tee::bootstrap::{TeeBootstrap, tdx_commands, gpu_commands};

let bootstrap = TeeBootstrap::default_config();

// Get commands to run via SSH
let detect_cmds = bootstrap.tdx_detect_commands();
let setup_cmds = bootstrap.tdx_setup_commands();

// Run commands on executor via SSH...
// Parse results...
let result = bootstrap.build_result(tdx_outputs, gpu_outputs);

if result.success {
    println!("TEE setup complete: {}", result.summary);
}
```

## Configuration

```toml
[tee]
enabled = true
require_tee = false

[tee.tdx]
expected_mrtd = "aa...aa"  # 96 hex chars
expected_rtmr0 = "bb...bb"

[tee.gpu]
require_cc_mode = true
allowed_models = ["H100 PCIe", "H100 SXM", "H200"]
```

## Requirements

### TDX

- Intel 4th Gen Xeon (Sapphire Rapids) or newer
- Linux kernel 6.2+ with TDX guest support
- Intel TDX DCAP SDK (`libtdx-attest`, `tdx-qgs`)

### GPU CC

- NVIDIA H100 or H200 GPU
- NVIDIA driver 535+
- Platform with CC support (Azure DCesv5, specific bare metal)

## Tools Used

All tools are open source:

| Tool | Source | Purpose |
|------|--------|---------|
| `tdx_attest` | Intel TDX DCAP SDK | TDX quote generation |
| `libtdx-attest` | Intel TDX DCAP SDK | TDX attestation library |
| `nvidia-smi` | NVIDIA Driver | GPU info and CC mode check |
| `nv-attestation-tool` | NVIDIA Attestation SDK | GPU attestation (optional) |

## Testing

```bash
# Run all tests
cargo test -p basilica-tee

# Run with NVML support (requires NVIDIA GPU)
cargo test -p basilica-tee --features nvml
```

## License

See the main Basilica repository for license information.

