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

- **Hardware Verification**: Binary validation system for secure GPU verification and profiling
- **Remote Validation**: SSH-based verification of computational tasks and hardware specifications
- **Bittensor Integration**: Native participation in Bittensor's consensus mechanism with weight allocation
- **Fleet Management**: Efficient orchestration of distributed GPU resources with assignment management
- **Public API Gateway**: Smart HTTP gateway providing load-balanced access to the validator network

## Key Components

- **Validator**: Verifies hardware capabilities, maintains GPU profiles, and scores miner performance
- **Miner**: Manages GPU executor fleets, handles assignments, and serves compute requests via Axon
- **Executor**: GPU machine agent with container management, system monitoring, and secure task execution
- **Public API**: HTTP gateway with authentication, caching, rate limiting, and request aggregation
- **Common**: Shared utilities including crypto, SSH management, storage, and configuration
- **Protocol**: gRPC/protobuf definitions for inter-component communication
- **Bittensor**: Network integration for registration, discovery, and weight management

## Network Information

- **Mainnet**: Bittensor Finney, Subnet 39
- **Testnet**: Bittensor Test Network, Subnet 387
- **Chain Endpoint**: `wss://entrypoint-finney.opentensor.ai:443` (mainnet)

## License

Copyright (c) 2025 Basilica Contributors (tplr.ai)

Licensed under MIT
