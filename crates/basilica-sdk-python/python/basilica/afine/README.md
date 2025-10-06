# Basilica AFINE SDK

**Agent Framework for Interactive Network Environments**

A Python SDK for defining, deploying, and interacting with containerized multi-turn reinforcement learning environments on the Basilica protocol.

## Features

- **Declarative Environment Definition**: Subclass `Service` to define RL environments
- **Automatic RPC Generation**: FastAPI-based HTTP-RPC endpoints from Python methods
- **Secure Communication**: Shared secret authentication between client and service
- **Remote GPU Execution**: Automatic deployment to Basilica protocol GPU nodes
- **Gymnasium Compatibility**: Standard `reset()`, `step()`, `render()`, `close()` interface
- **Docker Hub Integration**: Publish and pull environments from Docker Hub
- **Stateful Session Management**: Persistent state across container lifecycles
- **Type Safety**: Pydantic models for request/response validation
- **Resource Management**: Proper cleanup with context manager protocol

## Installation

```bash
pip install basilica[afine]
```

## Quick Start

### Define a Service

```python
# service.py
import basilica.afine as bs

class MathEnv(bs.Service[tuple[int, int], int]):
    def reset(self, seed=None) -> tuple[int, int]:
        self.x, self.y = 1, 1
        return (self.x, self.y)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, bool, dict]:
        self.x += action
        return (self.x, self.y), 1.0, False, False, {}

if __name__ == "__main__":
    bs.serve(MathEnv)
```

### Deploy and Use

```python
import basilica.afine as bs

# Deploy to Basilica network and get a client
with bs.create("./mathenv") as client:
    obs = client.reset()
    print(f"Initial state: {obs}")

    obs, reward, terminated, truncated, info = client.step(3)
    print(f"After step: {obs}, reward: {reward}")
```

## Authentication

The SDK uses shared secret authentication for secure container-to-client communication:

1. **Server Side**: Service requires `BASILICA_RENTAL_SECRET` environment variable
2. **Client Side**: Client sends secret via `X-Basilica-Secret` header on every RPC call
3. **Automatic**: The `create()` function generates and injects the secret automatically

## State Persistence

State is automatically saved during graceful shutdown and restored on startup:

```python
class StatefulEnv(bs.Service):
    def __init__(self):
        self.episode_count = 0

    def reset(self, seed=None):
        self.episode_count += 1  # Persisted across container restarts
        return self.episode_count
```

### Handling Unpicklable Objects

For objects that can't be serialized (GPU tensors, file handles, etc.):

```python
class GPUEnv(bs.Service):
    def __init__(self):
        self.model = None

    def reset(self, seed=None):
        if self.model is None:
            import torch
            self.model = torch.nn.Linear(10, 10).cuda()
        return torch.randn(10).tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None  # Don't serialize GPU model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Model will be recreated on first reset()
```

## Examples

See `examples/afine/` for complete examples:

- **mathenv**: Simple arithmetic environment
- **satenv**: SAT-style question environment
- **cartpole**: Gymnasium CartPole integration

## API Reference

### Core Classes

#### `Service[ObsType, ActType]`

Abstract base class for defining environments.

**Methods:**
- `reset(seed=None) -> ObsType`: Reset environment
- `step(action: ActType) -> Tuple[ObsType, float, bool, bool, dict]`: Take a step
- `render() -> Optional[Any]`: Render environment (optional)
- `close() -> None`: Clean up resources (optional)

#### `serve(service_class, host="0.0.0.0", port=8000)`

Start an HTTP server exposing service methods.

**Parameters:**
- `service_class`: Service subclass to serve
- `host`: Host to bind to
- `port`: Port to bind to

**Environment Variables:**
- `BASILICA_RENTAL_SECRET`: Required for authentication

#### `create(image_or_path, **kwargs) -> Client`

Deploy and connect to a remote service.

**Parameters:**
- `image_or_path`: Docker Hub image or local directory path
- `api_key`: Basilica API key (or set `BASILICA_API_KEY` env var)
- `gpu_requirements`: GPU requirements dict
- `node_id`: Specific node ID (optional)
- `environment`: Environment variables dict
- `timeout`: Container readiness timeout (seconds)

**Returns:** Client proxy with context manager support

#### `Client`

Dynamic proxy for remote service instances.

**Methods:**
- `close()`: Close HTTP client and release resources
- `kill()`: Terminate rental and stop container
- `logs(follow=False, tail=None)`: Stream container logs
- `status()`: Get rental status

**Usage:** Always use as context manager for automatic cleanup

## Architecture

The SDK follows SOLID principles with clear separation of concerns:

- **Service**: Define environment methods only
- **Server Runtime**: HTTP server and route registration
- **Client Proxy**: RPC proxy and lifecycle management
- **DockerManager**: Container operations
- **BasilicaAPIClient**: API communication
- **StatePersistence**: State save/restore

## Security

- **Shared Secret Authentication**: 32-byte URL-safe tokens
- **Environment Isolation**: Containers run in isolated Docker networks
- **Secure by Default**: Authentication required, not optional
- **No Secret Logging**: Secrets never exposed in logs or responses

## Troubleshooting

### Container won't start

```python
# Check logs
client = bs.create("user/image:tag")
client.logs(tail=100)
```

### Authentication errors

Ensure `BASILICA_RENTAL_SECRET` is set in the container environment. The `create()` function does this automatically.

### State persistence fails

Check for unpicklable objects and implement `__getstate__`/`__setstate__` methods.

## License

See LICENSE file in repository root.
