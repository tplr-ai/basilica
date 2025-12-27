# Basilica TUI

A terminal user interface for the Basilica GPU compute marketplace. Provides a unified interface for both end-users renting GPUs and miners managing their fleets.

## Features

### User Mode (Default)
- **Dashboard** - Overview of active rentals, spending, and quick actions
- **Rentals** - Manage GPU rentals with SSH, exec, copy, restart, and terminate
- **Marketplace** - Browse and provision available GPUs with filtering
- **Deployments** - Deploy and scale applications (vLLM, SGLang templates)
- **Billing** - View balance, deposit address, and transaction history
- **Settings** - Authentication, API tokens, and SSH key management

### Miner Mode (`--miner`)
- **Fleet** - Monitor node health and GPU utilization across your fleet
- **Validators** - Track validator assignments and discovery status
- **Nodes** - Manage individual nodes
- **Earnings** - Revenue charts and payment history
- **Logs** - Aggregated logs from all nodes

## Installation

```bash
# Build from source
cargo build -p basilica-tui --release

# The binary will be at ./target/release/basilica-tui
```

## Usage

```bash
# Launch TUI (shows startup screen to choose mode)
basilica-tui

# Skip startup and go directly to miner mode
basilica-tui --miner

# Dev mode with mock data (no API connection required)
basilica-tui --dev

# Connect to local API (for testing)
BASILICA_API_URL=http://localhost:8000 basilica-tui

# Custom config file
basilica-tui --config /path/to/config.toml

# Verbose logging
basilica-tui -vvv
```

## Startup Screen

When launching without flags, the TUI shows a welcome screen where you can choose:
- **User Mode** - Rent GPUs, deploy applications, manage billing
- **Miner Mode** - Manage fleet, track earnings, monitor validators

Use arrow keys or `j`/`k` to select, `Enter` to confirm, or press `u`/`m` for quick selection.

## Keybindings

### Startup Screen
| Key | Action |
|-----|--------|
| `↑` / `↓` / `←` / `→` | Navigate selection |
| `Tab` | Toggle selection |
| `Enter` / `Space` | Confirm selection |
| `u` / `1` | Quick select User mode |
| `m` / `2` | Quick select Miner mode |
| `q` / `Esc` | Quit |

### Global
| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `?` | Toggle help |
| `m` | Switch between User/Miner mode |
| `Tab` / `Shift+Tab` | Navigate screens |
| `1-6` | Jump to screen by number |
| `j` / `↓` | Select next item |
| `k` / `↑` | Select previous item |
| `r` | Refresh data |

### Rentals Screen
| Key | Action |
|-----|--------|
| `s` | SSH into rental |
| `e` | Execute command |
| `c` | Copy files (SCP) |
| `r` | Restart container |
| `d` | Terminate rental |
| `l` | Toggle logs |
| `f` | Toggle filters |
| `h` | Toggle history |

### Marketplace Screen
| Key | Action |
|-----|--------|
| `Enter` | Provision selected GPU |
| `f` / `/` | Toggle filter panel |
| `s` | Sort by price |
| `c` | Clear filters |

### Deployments Screen
| Key | Action |
|-----|--------|
| `n` | New deployment |
| `v` | Quick deploy vLLM |
| `g` | Quick deploy SGLang |
| `d` | Delete deployment |
| `s` | Scale deployment |
| `l` | Toggle logs |

### Settings Screen
| Key | Action |
|-----|--------|
| `Tab` | Switch section |
| `l` | Login (Auth section) |
| `o` | Logout (Auth section) |
| `a` | Add token/SSH key |
| `d` | Delete token/SSH key |

## Configuration

The TUI looks for configuration in:
1. Path specified via `--config`
2. `~/.config/basilica/tui.toml`
3. `~/.basilica/tui.toml`

**Environment Variables:**
- `BASILICA_API_URL` - Override API endpoint (e.g., `http://localhost:8000`)

Example configuration:

```toml
[theme]
name = "default"

[refresh]
balance = 30      # seconds
rentals = 10      # seconds
metrics = 5       # seconds

[api]
url = "https://api.basilica.cloud"
```

## Authentication

The TUI uses the same authentication as the CLI. You can:

1. **Login via TUI**: Go to Settings → Auth and press `l` to login
2. **Login via CLI first**: Run `basilica login` before starting the TUI
3. **Use dev mode**: Run with `--dev` to use mock data without authentication

Authentication tokens are stored in `~/.local/share/basilica/auth.json`.

## Architecture

```
src/
├── app/
│   ├── mod.rs       # Module exports
│   └── core.rs      # Main App state machine
├── ui/
│   ├── components/  # Reusable UI components (header, footer, dialog)
│   ├── screens/     # Screen implementations
│   └── widgets/     # Custom widgets (tables, gauges, sparklines)
├── data/
│   ├── user.rs      # User mode data fetching
│   ├── miner.rs     # Miner mode data fetching
│   └── streams.rs   # Real-time log streaming
├── events/          # Input and tick event handling
├── config.rs        # Configuration management
├── actions.rs       # External actions (SSH, SCP, clipboard)
└── main.rs          # Entry point
```

## Dependencies

- **ratatui** - Terminal UI framework
- **crossterm** - Cross-platform terminal handling
- **tokio** - Async runtime
- **basilica-sdk** - API client and auth

## Development

```bash
# Run in dev mode with mock data
cargo run -p basilica-tui -- --dev

# Run with verbose logging
cargo run -p basilica-tui -- --dev -vvv

# Run unit tests
cargo test -p basilica-tui
```

## Integration Testing

Test the TUI against a real API:

### Option 1: Local API (Docker)

```bash
# Start API in dev mode (no Bittensor required)
cd scripts/tui-test
./start.sh

# Run TUI against local API
BASILICA_API_URL=http://localhost:8000 cargo run -p basilica-tui

# Stop services
docker compose down
```

### Option 2: TUI Dev Mode

```bash
# Mock data, no API connection needed
cargo run -p basilica-tui -- --dev
```

### Running Integration Tests

```bash
# Start local API first
cd scripts/tui-test && docker compose up -d

# Run integration tests
cargo test -p basilica-tui --test integration

# Tests auto-skip if API is unavailable
```

### Test Coverage

| Test Type | Requires API | Description |
|-----------|--------------|-------------|
| Unit tests | No | Config, dialog, actions |
| Integration tests | Yes* | API endpoints |

*Tests skip gracefully if API is not running

## License

MIT OR Apache-2.0

