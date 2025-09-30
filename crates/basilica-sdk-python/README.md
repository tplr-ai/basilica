# Basilica Python SDK

Python bindings for the Basilica GPU rental network SDK.

## Installation

### From Source

```bash
# Using uv (recommended)
uv pip install -e crates/basilica-sdk-python

# Or using pip with maturin
pip install maturin
cd crates/basilica-sdk-python
maturin develop
```

### From PyPI (when published)

```bash
pip install basilica
```

## Quick Start

The SDK provides a simple, intuitive interface for managing GPU rentals. With automatic environment variable detection and sensible defaults, you can get started with minimal configuration.

For complete working examples, see the `examples/` directory:
- `quickstart.py` - Minimal example to get started quickly
- `start_rental.py` - Full rental workflow with SSH setup
- `list_executors.py` - Finding available GPU resources
- `health_check.py` - API health monitoring
- `ssh_utils.py` - SSH credential handling examples

## Features

### 🚀 Auto-Configuration
- Automatically detects environment variables:
  - `BASILICA_API_URL` - API endpoint URL (default: `https://api.basilica.ai`)
  - `BASILICA_API_TOKEN` - Your API token for authentication
- SSH keys auto-detected from `~/.ssh/basilica_ed25519.pub` by default
- Sensible defaults for all parameters

### 🔐 Enhanced SSH Handling
- Built-in SSH utilities for credential parsing and command generation
- Automatic SSH key detection and validation
- Formatted SSH connection instructions with error handling

### 🎯 Simplified API
The SDK provides both minimal and customizable approaches to starting rentals. You can use all defaults for quick starts or specify exactly what you need. See the examples directory for detailed usage patterns and API documentation.

## Examples

All code examples are available in the `examples/` directory. These provide complete, runnable demonstrations of the SDK's capabilities:

### Available Examples

- **`quickstart.py`** - Get started with minimal code, demonstrating the simplest way to rent a GPU
- **`start_rental.py`** - Complete rental workflow including custom configuration, SSH setup, and resource management
- **`list_executors.py`** - Query and filter available GPU executors based on your requirements
- **`health_check.py`** - Monitor API health and availability
- **`ssh_utils.py`** - Work with SSH credentials, including parsing, formatting, and connection management

Each example is fully documented and can be run directly after installing the SDK. They demonstrate best practices and common patterns for working with the Basilica GPU rental network.

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/basilica/basilica.git
cd basilica/crates/basilica-sdk-python

# Create virtual environment
uv venv
source .venv/bin/activate

# Build and install
uv pip install -e .

# Run tests
pytest tests/
```

### Environment Variables

The SDK automatically detects these environment variables:

- `BASILICA_API_URL`: API endpoint (default: `https://api.basilica.ai`)
- `BASILICA_API_TOKEN`: Your API token for authentication

### Authentication

To use the SDK, you need to create an API token via the Basilica CLI:

```bash
# Create a new API token
basilica tokens create

# The command will output the token - set it as an environment variable
export BASILICA_API_TOKEN="basilica_..."

# Or pass it directly to the client
client = BasilicaClient(api_key="basilica_...")
```

### SSH Key Configuration

By default, SDK looks for keys at `~/.ssh/basilica_ed25519.pub`.

To set up SSH keys for Basilica:
```bash
# Generate a Basilica-specific ED25519 SSH key
ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519

# The public key will be auto-detected by the SDK
ls ~/.ssh/basilica_ed25519.pub
```

## License

MIT OR Apache-2.0