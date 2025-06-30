<p align="center">⛪</p>

# <p align="center">Basilica</p>

<p align="center">
  <em>Sacred Compute</em>
</p>

---

<p align="center">
  <a href="docs/miner.md">Miner</a> • 
  <a href="docs/validator.md">Validator</a> • 
  <a href="docs/architecture.md">Architecture</a>
</p>

## Overview

Basilica creates a trustless marketplace for GPU compute by:

- **Hardware Attestation**: Cryptographically verifiable GPU capabilities using P256 ECDSA signatures
- **Remote Validation**: SSH-based verification of computational tasks and hardware specifications
- **Bittensor Integration**: Native participation in Bittensor's consensus mechanism
- **Fleet Management**: Efficient orchestration of distributed GPU resources

## Key Components

- **Miner**: Manages GPU executor machines and serves compute requests
- **Validator**: Verifies hardware capabilities and scores miner performance
- **Executor**: GPU machine agent handling secure task execution
- **GPU-Attestor**: Generates cryptographic proofs of hardware capabilities

## Network Information

- **Mainnet**: Bittensor Finney, Subnet 39
- **Testnet**: Bittensor Test Network, Subnet 387
- **Chain Endpoint**: `wss://entrypoint-finney.opentensor.ai:443` (mainnet)

## License

Copyright (c) 2025 Basilica Contributors (tplr.ai)

Licensed under MIT
