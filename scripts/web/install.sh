#!/bin/bash
set -e

# Basilica CLI Installation Script
# Usage: curl -sSL https://basilica.ai/install.sh | bash

BINARY_NAME="basilica"
GITHUB_REPO="one-covenant/basilica"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--repo owner/repo]"
            exit 1
            ;;
    esac
done

TEMP_DIR=$(mktemp -d)
TEMP_BINARY="$TEMP_DIR/$BINARY_NAME"

# Determine install directory
BASILICA_DIR="${BASILICA_DIR:-$HOME/.basilica}"
INSTALL_DIR="$BASILICA_DIR/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Show ASCII art
show_logo() {
    echo -e "${CYAN}"
    cat << 'EOF'
 /$$                           /$$ /$$ /$$
| $$                          |__/| $$|__/
| $$$$$$$   /$$$$$$   /$$$$$$$ /$$| $$ /$$  /$$$$$$$  /$$$$$$
| $$__  $$ |____  $$ /$$_____/| $$| $$| $$ /$$_____/ |____  $$
| $$  \ $$  /$$$$$$$|  $$$$$$ | $$| $$| $$| $$        /$$$$$$$
| $$  | $$ /$$__  $$ \____  $$| $$| $$| $$| $$       /$$__  $$
| $$$$$$$/|  $$$$$$$ /$$$$$$$/| $$| $$| $$|  $$$$$$$|  $$$$$$$
|_______/  \_______/|_______/ |__/|__/|__/ \_______/ \_______/

EOF
    echo -e "${NC}"
}

# Print colored output
print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_step() {
    echo -e "${BLUE}→${NC} $1"
}

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Check and setup installation directory
setup_install_dir() {
    mkdir -p "$INSTALL_DIR"
}

# Remove binaries left by the old install layout (pre-~/.basilica era).
# Old locations: ~/.local/bin or /usr/local/bin.
# Does NOT modify shell profiles — the new env file appended by setup_shells()
# naturally overrides the old PATH/completion lines.
migrate_from_old_install() {
    for old_dir in "$HOME/.local/bin" "/usr/local/bin"; do
        if [ -f "$old_dir/basilica" ]; then
            print_step "Removing old binary from $old_dir..."
            rm -f "$old_dir/basilica" 2>/dev/null || true
            rm -f "$old_dir/bs" 2>/dev/null || true
            print_info "Cleaned up old install location"
        fi
    done
}

# Detect all installed shells on the system
detect_installed_shells() {
    local shells=""
    if command -v bash >/dev/null 2>&1; then
        shells="bash"
    fi
    if command -v zsh >/dev/null 2>&1; then
        shells="${shells:+$shells }zsh"
    fi
    if command -v fish >/dev/null 2>&1; then
        shells="${shells:+$shells }fish"
    fi
    echo "$shells"
}

# Get profile file for a given shell type
get_profile_for_shell() {
    local shell_type="$1"

    case "$shell_type" in
        zsh)
            echo "$HOME/.zshrc"
            ;;
        fish)
            echo "$HOME/.config/fish/config.fish"
            ;;
        bash|*)
            if [ -f "$HOME/.bashrc" ]; then
                echo "$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                echo "$HOME/.bash_profile"
            else
                echo "$HOME/.profile"
            fi
            ;;
    esac
}

# Create env files in BASILICA_DIR for shell setup
# These are self-contained files that handle PATH and completions,
# following the pattern used by cargo (~/.cargo/env) and deno (~/.deno/env).
create_env_files() {
    # Create ~/.basilica/env (sh-compatible, sourced by bash & zsh)
    cat > "$BASILICA_DIR/env" << 'ENVEOF'
#!/bin/sh
# basilica shell setup (sourced by bash and zsh)

# Add to PATH if not already present (cargo-style dedup)
case ":${PATH}:" in
    *:"$HOME/.basilica/bin":*)
        ;;
    *)
        export PATH="$HOME/.basilica/bin:$PATH"
        ;;
esac

# basilica completions
if [ -n "$BASH_VERSION" ]; then
    eval "$(COMPLETE=bash basilica)" 2>/dev/null
    eval "$(complete -p basilica 2>/dev/null | sed 's/ basilica$/ bs/')" 2>/dev/null
elif [ -n "$ZSH_VERSION" ]; then
    eval "$(COMPLETE=zsh basilica)" 2>/dev/null
    compdef bs=basilica 2>/dev/null
fi
ENVEOF

    # Create ~/.basilica/env.fish
    cat > "$BASILICA_DIR/env.fish" << 'FISHEOF'
# basilica shell setup (sourced by fish)

# Add to PATH if not already present
if not contains "$HOME/.basilica/bin" $PATH
    set -x PATH "$HOME/.basilica/bin" $PATH
end

# basilica completions
COMPLETE=fish basilica | source 2>/dev/null
complete -c bs -w basilica 2>/dev/null
FISHEOF

    print_info "Created shell env files in $BASILICA_DIR"
}

# Add a source line to a shell's profile file
add_source_to_profile() {
    local shell_type="$1"
    local profile_file="$2"
    local source_line

    # shellcheck disable=SC2016 # $HOME must stay literal in profile files
    if [ "$shell_type" = "fish" ]; then
        source_line='source "$HOME/.basilica/env.fish"'
    else
        source_line='. "$HOME/.basilica/env"'
    fi

    # Ensure profile directory exists (important for fish where ~/.config/fish/ may not exist)
    mkdir -p "$(dirname "$profile_file")" 2>/dev/null || true

    # Check if source line is already present
    if [ -f "$profile_file" ] && grep -qF ".basilica/env" "$profile_file" 2>/dev/null; then
        return 0
    fi

    if echo "$source_line" >> "$profile_file" 2>/dev/null; then
        print_info "Added basilica env to $profile_file"
    else
        print_warning "Could not update $profile_file automatically"
        print_info "Please add this to your $profile_file: $source_line"
    fi
}

# Detect architecture
detect_arch() {
    local arch
    arch=$(uname -m)
    case $arch in
        x86_64)
            echo "amd64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            print_info "Supported architectures: x86_64, aarch64"
            exit 1
            ;;
    esac
}

# Detect operating system
detect_os() {
    local os
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    case $os in
        linux)
            echo "linux"
            ;;
        darwin)
            echo "darwin"
            ;;
        *)
            print_error "Unsupported OS: $os"
            print_info "Supported operating systems: Linux, macOS"
            exit 1
            ;;
    esac
}

# Get Rust target triple for the current platform
get_rust_target() {
    local os
    local arch
    os=$(detect_os)
    arch=$(detect_arch)

    case "${os}-${arch}" in
        linux-amd64)
            echo "x86_64-unknown-linux-musl"
            ;;
        linux-arm64)
            echo "aarch64-unknown-linux-musl"
            ;;
        darwin-amd64)
            echo "x86_64-apple-darwin"
            ;;
        darwin-arm64)
            echo "aarch64-apple-darwin"
            ;;
        *)
            print_error "Unsupported platform: ${os}-${arch}"
            exit 1
            ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get latest basilica-cli release tag from GitHub
get_latest_cli_release() {
    local releases_json

    print_step "Fetching latest release information from GitHub..." >&2

    # Fetch releases from GitHub API
    if command_exists curl; then
        releases_json=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/releases" 2>/dev/null)
    elif command_exists wget; then
        releases_json=$(wget -qO- "https://api.github.com/repos/$GITHUB_REPO/releases" 2>/dev/null)
    else
        print_error "Neither curl nor wget found" >&2
        return 1
    fi

    # Check if API call was successful
    if [ -z "$releases_json" ]; then
        print_error "Failed to fetch releases from GitHub" >&2
        print_info "Please check your internet connection or try again later" >&2
        return 1
    fi

    # Check for rate limiting
    if echo "$releases_json" | grep -q "API rate limit exceeded"; then
        print_error "GitHub API rate limit exceeded" >&2
        print_info "Please try again later or download manually from:" >&2
        print_info "  https://github.com/$GITHUB_REPO/releases" >&2
        return 1
    fi

    # Parse JSON to find latest non-prerelease basilica-cli-v* tag
    # Pipeline explanation:
    # 1. grep -E '"tag_name"|"prerelease"' - Extract only tag_name and prerelease lines
    # 2. grep -B1 '"prerelease": false' - Find non-prerelease entries and include 1 line before (the tag_name)
    # 3. grep 'tag_name' - Filter to only tag_name lines from the previous output
    # 4. grep 'basilica-cli-v' - Keep only tags starting with basilica-cli-v
    # 5. cut -d '"' -f 4 - Extract the tag value between quotes
    # 6. sort -V -r - Sort by version number (descending)
    # 7. head -1 - Take the highest version
    local latest_tag
    latest_tag=$(echo "$releases_json" | \
        grep -E '"tag_name"|"prerelease"' | \
        grep -B1 '"prerelease": false' | \
        grep 'tag_name' | \
        grep 'basilica-cli-v' | \
        cut -d '"' -f 4 | \
        sort -V -r | \
        head -1)

    if [ -z "$latest_tag" ]; then
        print_error "No stable basilica-cli releases found" >&2
        print_info "Please check https://github.com/$GITHUB_REPO/releases" >&2
        return 1
    fi

    echo "$latest_tag"
    return 0
}

# Install dependencies
check_dependencies() {
    if ! command_exists curl && ! command_exists wget; then
        print_error "Please install curl or wget first"
        exit 1
    fi
}

# Download binary
download_binary() {
    local arch
    local os
    local target
    local latest_tag

    # Get latest release tag first (this will print "Fetching latest release information...")
    latest_tag=$(get_latest_cli_release 2>/dev/null || true)
    if [ -z "$latest_tag" ]; then
        print_error "Unable to fetch latest version (rate limited). Try again in a few minutes."
        exit 1
    fi

    # Extract version number for display
    local version
    version="${latest_tag#basilica-cli-v}"
    print_info "Found latest version: v$version"

    # Detect platform
    arch=$(detect_arch)
    os=$(detect_os)
    target=$(get_rust_target)
    local archive_name="basilica-${version}-${target}.tar.gz"
    local download_url="https://github.com/${GITHUB_REPO}/releases/download/${latest_tag}/${archive_name}"
    local temp_archive="$TEMP_DIR/archive.tar.gz"

    print_step "Checking availability for ${os}-${arch} (${target})..."

    # Check if the binary exists on GitHub first
    local http_status
    if command_exists curl; then
        http_status=$(curl -o /dev/null -s -w "%{http_code}" -I -L "$download_url" 2>/dev/null)
    elif command_exists wget; then
        http_status=$(wget --spider -S "$download_url" 2>&1 | grep "HTTP/" | awk '{print $2}' | tail -1)
    else
        http_status="000"
    fi

    if [ "$http_status" = "404" ]; then
        print_error "Archive not found for your platform: ${target}"
        print_info "This combination may not be supported in release $latest_tag"
        print_info "Check available archives at: https://github.com/$GITHUB_REPO/releases/tag/$latest_tag"
        exit 1
    elif [ "$http_status" = "403" ]; then
        print_error "Access denied to archive (HTTP 403)"
        print_info "The release may be private or access may be restricted"
        print_info "URL attempted: $download_url"
        exit 1
    elif [ "$http_status" != "200" ] && [ "$http_status" != "302" ] && [ "$http_status" != "301" ]; then
        print_warning "Unexpected response from GitHub (HTTP $http_status)"
        print_info "Attempting download anyway..."
    fi

    print_step "Downloading Basilica CLI v$version..."

    if command_exists curl; then
        if ! curl -fsSL -L "$download_url" -o "$temp_archive" 2>/dev/null; then
            local curl_exit_code=$?
            if [ $curl_exit_code -eq 22 ]; then
                print_error "HTTP error from GitHub (likely 403 or 404)"
                print_info "The archive may not be available for ${target} in release $latest_tag"
                exit 1
            else
                print_error "Download failed"
                print_info "URL attempted: $download_url"
                print_info "Please check your network connection and try again"
                exit 1
            fi
        fi
    elif command_exists wget; then
        if ! wget -q "$download_url" -O "$temp_archive" 2>/dev/null; then
            print_error "Download failed"
            print_info "URL attempted: $download_url"
            print_info "Please check your network connection and try again"
            exit 1
        fi
    fi

    if [ ! -f "$temp_archive" ] || [ ! -s "$temp_archive" ]; then
        print_error "Download failed - archive is missing or empty"
        print_info "URL attempted: $download_url"
        print_info "Please verify the archive is available for your platform at:"
        print_info "  https://github.com/$GITHUB_REPO/releases/tag/$latest_tag"
        exit 1
    fi

    print_step "Extracting binary from archive..."

    # Extract the binary from the tarball
    if ! tar -xzf "$temp_archive" -C "$TEMP_DIR" 2>/dev/null; then
        print_error "Failed to extract archive"
        print_info "The downloaded archive may be corrupted"
        exit 1
    fi

    # Verify the extracted binary exists
    if [ ! -f "$TEMP_BINARY" ]; then
        print_error "Binary not found in archive"
        print_info "Expected to find 'basilica' in the archive"
        exit 1
    fi
}

# Verify binary
verify_binary() {
    chmod +x "$TEMP_BINARY"

    if ! "$TEMP_BINARY" --help >/dev/null 2>&1; then
        print_error "Binary verification failed"
        exit 1
    fi
}

# Check if binary already exists and prompt user
check_existing_installation() {
    # Find existing binary: check new location first, then old locations
    local existing_binary=""
    for candidate in "$INSTALL_DIR/$BINARY_NAME" "$HOME/.local/bin/$BINARY_NAME" "/usr/local/bin/$BINARY_NAME"; do
        if [ -f "$candidate" ]; then
            existing_binary="$candidate"
            break
        fi
    done

    if [ -n "$existing_binary" ]; then
        echo
        print_warning "Basilica CLI is already installed at $existing_binary"

        # Try to get current version
        local current_version
        local current_version_clean
        if current_version=$("$existing_binary" --version 2>/dev/null | head -n1); then
            # Extract just the version number (e.g., "basilica 0.1.0" -> "0.1.0")
            current_version_clean="${current_version#"${current_version%%[0-9]*}"}"
            current_version_clean="${current_version_clean%%[!0-9.]*}"
        else
            current_version_clean="unknown"
        fi

        # Try to fetch latest version
        local latest_tag
        local latest_version_clean
        print_step "Checking for latest version..."

        # Suppress the "Fetching latest release information..." message from get_latest_cli_release
        latest_tag=$(get_latest_cli_release 2>/dev/null || true)

        if [ -n "$latest_tag" ]; then
            # Extract version from tag (e.g., "basilica-cli-v0.2.0" -> "0.2.0")
            latest_version_clean="${latest_tag#basilica-cli-v}"
        else
            latest_version_clean="unable to fetch"
        fi

        # Display version comparison
        if [ "$latest_version_clean" != "unable to fetch" ]; then
            echo
            if [ "$current_version_clean" != "unknown" ]; then
                print_info "Current version: v$current_version_clean"
            else
                print_info "Current version: unable to determine"
            fi

            print_info "Latest version:  v$latest_version_clean"

            # Check if versions match
            if [ "$current_version_clean" = "$latest_version_clean" ]; then
                print_info "You already have the latest version!"
                exit 0
            elif [ "$current_version_clean" != "unknown" ]; then
                print_warning "Update available!"
            fi
        else
            echo
            print_warning "Unable to check for updates (rate limited). Try again in a few minutes."
            exit 0
        fi

        echo
        # Check if we're in a pipe (common when using curl | bash)
        if [ ! -t 0 ]; then
            print_info "Running in non-interactive mode, proceeding with replacement..."
            print_info "To cancel, press Ctrl+C within 3 seconds..."
            sleep 3
            return 0
        fi

        # Prompt for update
        printf "Do you want to update? [y/N]: "

        if read -r response < /dev/tty 2>/dev/null; then
            case "$response" in
                [yY][eE][sS]|[yY])
                    print_info "Proceeding with installation..."
                    return 0
                    ;;
                *)
                    print_info "Installation cancelled."
                    exit 0
                    ;;
            esac
        else
            # Fallback if /dev/tty is not available
            print_info "Cannot read user input, proceeding with replacement..."
            return 0
        fi
    fi
}

# Clean up old backups
cleanup_old_backups() {
    # Clean up old binary backups silently
    for backup in "$INSTALL_DIR/$BINARY_NAME.backup."*; do
        if [ -f "$backup" ]; then
            rm -f "$backup" 2>/dev/null
        fi
    done

    # Clean up old config backups silently
    local config_dir="$HOME/.config/basilica"
    for config_backup in "$config_dir/config.toml.bak."*; do
        if [ -f "$config_backup" ]; then
            rm -f "$config_backup" 2>/dev/null
        fi
    done
}

# Install binary
install_binary() {
    print_step "Installing to $INSTALL_DIR..."

    # Directly overwrite existing binary
    mv -f "$TEMP_BINARY" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"

    # Create 'bs' alias symlink (relative, so it survives directory moves)
    print_step "Creating 'bs' alias..."
    if (cd "$INSTALL_DIR" && ln -sf "$BINARY_NAME" "bs") 2>/dev/null; then
        print_info "'bs' alias created successfully"
    else
        print_warning "Failed to create 'bs' alias"
    fi
}

# Setup shells: create env files and add source lines to profiles
setup_shells() {
    create_env_files

    local installed_shells
    installed_shells="$(detect_installed_shells)"

    print_step "Configuring shells: $installed_shells"

    for shell in $installed_shells; do
        local profile
        profile="$(get_profile_for_shell "$shell")"
        add_source_to_profile "$shell" "$profile"
    done
}

# Show completion message
show_completion() {
    local installed_shells
    installed_shells="$(detect_installed_shells)"

    echo
    print_info "Basilica CLI installed successfully!"
    print_info "You can use 'basilica' or the shorter 'bs' alias"
    echo

    print_info "PATH and completions configured for: $installed_shells"
    print_info "Please restart your terminal or run:"
    echo -e "  ${CYAN}source \"\$HOME/.basilica/env\"${NC}      (bash/zsh)"
    echo -e "  ${CYAN}source \"\$HOME/.basilica/env.fish\"${NC}  (fish)"
    echo

    echo "Get started:"
    echo "  basilica login                    # Login to Basilica"
    echo "  basilica ls                       # List available GPUs"
    echo "  basilica up                       # Start a GPU rental"
    echo "  basilica exec <uid> \"python train.py\"  # Run your code"
    echo "  basilica down <uid>               # Terminate rental"
    echo
    echo "Use TAB to autocomplete commands and options!"
    echo
}

# Main installation flow
main() {
    show_logo
    echo "Welcome to the Basilica CLI installer!"
    echo

    setup_install_dir
    migrate_from_old_install
    check_existing_installation
    check_dependencies
    cleanup_old_backups
    download_binary
    verify_binary
    install_binary
    setup_shells
    show_completion
}

# Run main function
main "$@"
