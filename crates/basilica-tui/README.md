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
# Start in user mode (default)
basilica-tui

# Start in miner mode
basilica-tui --miner

# Dev mode with mock data (no API connection required)
basilica-tui --dev

# Custom config file
basilica-tui --config /path/to/config.toml

# Verbose logging
basilica-tui -vvv
```

## Keybindings

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

# Run tests
cargo test -p basilica-tui
```

## License

MIT OR Apache-2.0

