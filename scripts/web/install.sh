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

# Determine install directory based on permissions
if [ "$EUID" -eq 0 ] || [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
    SYSTEM_INSTALL=true
else
    INSTALL_DIR="$HOME/.local/bin"
    SYSTEM_INSTALL=false
fi

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

# Detect user's shell type
detect_shell_type() {
    # Prefer the currently running shell over SHELL environment variable
    local shell_path
    local shell_name

    # Try to detect the current running shell first
    shell_path="$(ps -p $$ -o comm= 2>/dev/null || echo "")"

    # If ps command fails or returns empty, fall back to SHELL env var
    if [ -z "$shell_path" ]; then
        shell_path="${SHELL:-/bin/bash}"
    fi

    # Extract just the shell name
    shell_name="$(basename "$shell_path")"

    case "$shell_name" in
        bash) echo "bash" ;;
        zsh) echo "zsh" ;;
        fish) echo "fish" ;;
        sh) echo "bash" ;;  # Treat sh as bash for completion purposes
        *) echo "bash" ;;   # Default to bash if unknown
    esac
}

# Detect user's shell profile file
detect_shell_profile() {
    local shell_type
    shell_type="$(detect_shell_type)"

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

# Add directory to PATH in shell profile
add_to_path() {
    local dir="$1"
    local profile_file
    profile_file=$(detect_shell_profile)

    # Check if PATH export already exists
    if [ -f "$profile_file" ] && grep -q "export PATH.*$dir" "$profile_file" 2>/dev/null; then
        return 0
    fi

    # Check if directory is already in current PATH
    if echo "$PATH" | grep -q "$dir"; then
        return 0
    fi

    # Add PATH export to profile
    if echo "export PATH=\"$dir:\$PATH\"" >> "$profile_file" 2>/dev/null; then
        print_info "Added $dir to PATH"
    else
        print_warning "Could not add to PATH automatically"
        print_info "Please add this to your shell profile: export PATH=\"$dir:\$PATH\""
    fi
    return 0
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
    # 5. head -1 - Take the first match (GitHub API returns releases in newest-first order)
    # 6. cut -d '"' -f 4 - Extract the tag value between quotes
    local latest_tag
    latest_tag=$(echo "$releases_json" | \
        grep -E '"tag_name"|"prerelease"' | \
        grep -B1 '"prerelease": false' | \
        grep 'tag_name' | \
        grep 'basilica-cli-v' | \
        head -1 | \
        cut -d '"' -f 4)

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
    local latest_tag

    # Get latest release tag first (this will print "Fetching latest release information...")
    latest_tag=$(get_latest_cli_release 2>/dev/null || true)
    if [ -z "$latest_tag" ]; then
        print_error "Unable to fetch latest version (rate limited). Try again in a few minutes."
        exit 1
    fi

    # Extract version number for display
    local version
    version=$(echo "$latest_tag" | sed 's/basilica-cli-v//')
    print_info "Found latest version: v$version"

    # Detect platform
    arch=$(detect_arch)
    os=$(detect_os)
    local binary_name="basilica-${os}-${arch}"
    local download_url="https://github.com/${GITHUB_REPO}/releases/download/${latest_tag}/${binary_name}"

    print_step "Checking availability for ${os}-${arch}..."

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
        print_error "Binary not found for your platform: ${os}-${arch}"
        print_info "This combination may not be supported in release $latest_tag"
        print_info "Check available binaries at: https://github.com/$GITHUB_REPO/releases/tag/$latest_tag"
        exit 1
    elif [ "$http_status" = "403" ]; then
        print_error "Access denied to binary (HTTP 403)"
        print_info "The release may be private or access may be restricted"
        print_info "URL attempted: $download_url"
        exit 1
    elif [ "$http_status" != "200" ] && [ "$http_status" != "302" ] && [ "$http_status" != "301" ]; then
        print_warning "Unexpected response from GitHub (HTTP $http_status)"
        print_info "Attempting download anyway..."
    fi

    print_step "Downloading Basilica CLI v$version..."

    if command_exists curl; then
        if ! curl -fsSL -L "$download_url" -o "$TEMP_BINARY" 2>/dev/null; then
            local curl_exit_code=$?
            if [ $curl_exit_code -eq 22 ]; then
                print_error "HTTP error from GitHub (likely 403 or 404)"
                print_info "The binary may not be available for ${os}-${arch} in release $latest_tag"
                exit 1
            else
                print_error "Download failed"
                print_info "URL attempted: $download_url"
                print_info "Please check your network connection and try again"
                exit 1
            fi
        fi
    elif command_exists wget; then
        if ! wget -q "$download_url" -O "$TEMP_BINARY" 2>/dev/null; then
            print_error "Download failed"
            print_info "URL attempted: $download_url"
            print_info "Please check your network connection and try again"
            exit 1
        fi
    fi

    if [ ! -f "$TEMP_BINARY" ] || [ ! -s "$TEMP_BINARY" ]; then
        print_error "Download failed - file is missing or empty"
        print_info "URL attempted: $download_url"
        print_info "Please verify the binary is available for your platform at:"
        print_info "  https://github.com/$GITHUB_REPO/releases/tag/$latest_tag"
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
    if [ -f "$INSTALL_DIR/$BINARY_NAME" ]; then
        echo
        print_warning "Basilica CLI is already installed at $INSTALL_DIR/$BINARY_NAME"

        # Try to get current version
        local current_version
        local current_version_clean
        if current_version=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null | head -n1); then
            # Extract just the version number (e.g., "basilica 0.1.0" -> "0.1.0")
            current_version_clean=$(echo "$current_version" | sed 's/^[^0-9]*\([0-9.]*\).*/\1/')
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
            latest_version_clean=$(echo "$latest_tag" | sed 's/basilica-cli-v//')
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
}

# Setup shell completions
setup_shell_completions() {
    # Separate declarations from assignments to avoid SC2155
    local shell_type
    local profile_file
    local profile_dir
    shell_type="$(detect_shell_type)"
    profile_file="$(detect_shell_profile)"
    local completion_marker="# Basilica CLI completions"

    print_step "Setting up shell completions for $shell_type..."

    # Check if completions are already configured
    if [ -f "$profile_file" ] && grep -q "$completion_marker" "$profile_file" 2>/dev/null; then
        print_info "Shell completions already configured"
        return 0
    fi

    # Ensure profile file and its directory exist
    profile_dir="$(dirname "$profile_file")"
    if [ ! -d "$profile_dir" ]; then
        mkdir -p "$profile_dir" 2>/dev/null || true
    fi

    # Touch the profile file to ensure it exists
    if [ ! -f "$profile_file" ]; then
        touch "$profile_file" 2>/dev/null || true
    fi

    # Add completion based on shell type
    local completion_cmd=""
    case "$shell_type" in
        bash)
            completion_cmd='eval "$(COMPLETE=bash basilica)"'
            ;;
        zsh)
            completion_cmd='eval "$(COMPLETE=zsh basilica)"'
            ;;
        fish)
            completion_cmd='COMPLETE=fish basilica | source'
            ;;
        *)
            print_warning "Unknown shell type: $shell_type"
            print_info "Please add shell completions manually for your shell"
            return 1
            ;;
    esac

    # Add completion to profile using direct conditional (fixes SC2320)
    if {
        echo ""
        echo "$completion_marker"
        echo "$completion_cmd"
    } >> "$profile_file" 2>/dev/null; then
        print_info "Added shell completions to $profile_file"
        return 0
    else
        print_warning "Could not add completions automatically"
        print_info "Please add this to your $profile_file:"
        echo "  $completion_cmd"
        return 1
    fi
}

# Setup PATH if needed
setup_path() {
    if [ "$SYSTEM_INSTALL" = false ] && ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        if add_to_path "$HOME/.local/bin"; then
            print_info "PATH updated in $(detect_shell_profile)"
        else
            print_warning "Manually add ~/.local/bin to your PATH:"
            echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
        echo
    fi
}

# Show completion message
show_completion() {
    local profile_file="$(detect_shell_profile)"

    echo
    print_info "Basilica CLI installed successfully!"
    echo

    # Inform about shell completions
    print_info "Shell completions have been configured for tab support"
    print_info "Please restart your terminal or run:"
    echo -e "  ${CYAN}source $profile_file${NC}"
    echo

    # Show manual completion setup instructions
    print_info "For other shells, add the appropriate completion command to your shell config:"
    echo "  Bash:  eval \"\$(COMPLETE=bash basilica)\""
    echo "  Zsh:   eval \"\$(COMPLETE=zsh basilica)\""
    echo "  Fish:  COMPLETE=fish basilica | source"
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
    check_existing_installation
    check_dependencies
    cleanup_old_backups
    download_binary
    verify_binary
    install_binary
    setup_path
    setup_shell_completions
    show_completion
}

# Run main function
main "$@"
