#!/usr/bin/env bash

set -euo pipefail

# Configuration
readonly DOWNLOAD_SIZE_MB=50
readonly UPLOAD_SIZE_MB=15
readonly UPLOAD_TIMEOUT=30
readonly CONNECT_TIMEOUT=10
readonly MAX_TIME=60
readonly DOMAIN="speed.cloudflare.com"
readonly DOWNLOAD_URL="https://${DOMAIN}/__down"
readonly UPLOAD_URL="https://${DOMAIN}/__up"

# Cloudflare IP ranges URL
readonly CLOUDFLARE_IPS_URL="https://www.cloudflare.com/ips-v4"

# Fallback IP ranges if fetch fails (minimal set for bootstrap)
readonly -a CLOUDFLARE_IPV4_FALLBACK=(
    "104.16.0.0/13"
    "104.24.0.0/14"
    "172.64.0.0/13"
    "162.158.0.0/15"
)

# Dynamic array populated at runtime
CLOUDFLARE_IPV4_RANGES=()

# Fetch Cloudflare IP ranges dynamically
fetch_cloudflare_ranges() {
    local ranges
    ranges=$(curl -sf --max-time 5 --connect-timeout 3 "$CLOUDFLARE_IPS_URL" 2>/dev/null) || return 1

    # Validate response contains CIDR ranges
    if [[ -z "$ranges" ]] || ! echo "$ranges" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/[0-9]+$'; then
        return 1
    fi

    echo "$ranges"
}

# Initialize Cloudflare IP ranges (fetch or fallback)
init_cloudflare_ranges() {
    local ranges
    if ranges=$(fetch_cloudflare_ranges); then
        mapfile -t CLOUDFLARE_IPV4_RANGES <<< "$ranges"
        echo "Loaded ${#CLOUDFLARE_IPV4_RANGES[@]} Cloudflare IP ranges" >&2
    else
        CLOUDFLARE_IPV4_RANGES=("${CLOUDFLARE_IPV4_FALLBACK[@]}")
        echo "Using fallback Cloudflare IP ranges" >&2
    fi
}

# Check for required commands
check_dependencies() {
    local missing_deps=()

    for cmd in curl awk date dd getent; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies: ${missing_deps[*]}" >&2
        return 1
    fi
}

# Convert IP to integer for CIDR matching
ip_to_int() {
    IFS='.' read -r o1 o2 o3 o4 <<< "$1"
    echo $(( (o1 << 24) + (o2 << 16) + (o3 << 8) + o4 ))
}

# Check if IP is within a CIDR range
ip_in_cidr() {
    local ip="$1"
    local cidr="$2"
    local network="${cidr%/*}"
    local prefix="${cidr#*/}"

    local ip_int network_int mask
    ip_int=$(ip_to_int "$ip")
    network_int=$(ip_to_int "$network")
    mask=$(( (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF ))

    (( (ip_int & mask) == (network_int & mask) ))
}

# Validate IP is not private, localhost, or reserved
validate_public_ip() {
    local ip="$1"

    # Basic IPv4 format validation
    if ! [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Invalid IP format: $ip" >&2
        return 1
    fi

    local o1 o2 o3 o4
    IFS='.' read -r o1 o2 o3 o4 <<< "$ip"

    # Validate octet ranges
    for octet in "$o1" "$o2" "$o3" "$o4"; do
        if (( octet < 0 || octet > 255 )); then
            echo "Error: Invalid octet in IP: $ip" >&2
            return 1
        fi
    done

    # Block loopback (127.0.0.0/8)
    if (( o1 == 127 )); then
        echo "Error: Blocked loopback IP: $ip" >&2
        return 1
    fi

    # Block RFC1918 private networks
    if (( o1 == 10 )); then
        echo "Error: Blocked private IP (10.0.0.0/8): $ip" >&2
        return 1
    fi
    if (( o1 == 172 && o2 >= 16 && o2 <= 31 )); then
        echo "Error: Blocked private IP (172.16.0.0/12): $ip" >&2
        return 1
    fi
    if (( o1 == 192 && o2 == 168 )); then
        echo "Error: Blocked private IP (192.168.0.0/16): $ip" >&2
        return 1
    fi

    # Block link-local (169.254.0.0/16)
    if (( o1 == 169 && o2 == 254 )); then
        echo "Error: Blocked link-local IP: $ip" >&2
        return 1
    fi

    # Block CGNAT (100.64.0.0/10)
    if (( o1 == 100 && o2 >= 64 && o2 <= 127 )); then
        echo "Error: Blocked CGNAT IP: $ip" >&2
        return 1
    fi

    # Block current network (0.0.0.0/8)
    if (( o1 == 0 )); then
        echo "Error: Blocked current network IP: $ip" >&2
        return 1
    fi

    # Block multicast (224.0.0.0/4)
    if (( o1 >= 224 && o1 <= 239 )); then
        echo "Error: Blocked multicast IP: $ip" >&2
        return 1
    fi

    # Block reserved (240.0.0.0/4)
    if (( o1 >= 240 )); then
        echo "Error: Blocked reserved IP: $ip" >&2
        return 1
    fi

    # Block broadcast
    if (( o1 == 255 && o2 == 255 && o3 == 255 && o4 == 255 )); then
        echo "Error: Blocked broadcast IP: $ip" >&2
        return 1
    fi

    return 0
}

# Validate IP belongs to Cloudflare
validate_cloudflare_ip() {
    local ip="$1"

    for range in "${CLOUDFLARE_IPV4_RANGES[@]}"; do
        if ip_in_cidr "$ip" "$range"; then
            return 0
        fi
    done

    echo "Error: IP $ip is not in Cloudflare's published IP ranges" >&2
    return 1
}

# Resolve domain using DNS-over-HTTPS for security
resolve_domain_securely() {
    local domain="$1"
    local resolved_ip=""

    # Primary: Use Cloudflare DoH
    resolved_ip=$(curl -sf \
        --max-time 5 \
        --connect-timeout 3 \
        -H "accept: application/dns-json" \
        "https://1.1.1.1/dns-query?name=${domain}&type=A" 2>/dev/null | \
        grep -oP '"data"\s*:\s*"\K[0-9.]+' | head -1) || true

    # Fallback: Use system resolver if DoH fails
    if [[ -z "$resolved_ip" ]]; then
        resolved_ip=$(getent hosts "$domain" 2>/dev/null | awk '{print $1; exit}') || true
    fi

    if [[ -z "$resolved_ip" ]]; then
        echo "Error: Failed to resolve $domain" >&2
        return 1
    fi

    echo "$resolved_ip"
}

# Validate domain resolution and return verified IP
validate_domain() {
    local domain="$1"

    echo "Resolving $domain securely..." >&2

    local resolved_ip
    resolved_ip=$(resolve_domain_securely "$domain") || return 1

    echo "Resolved to: $resolved_ip" >&2

    # Validate IP is public (not private/localhost/reserved)
    validate_public_ip "$resolved_ip" || return 1

    # Validate IP belongs to Cloudflare
    validate_cloudflare_ip "$resolved_ip" || return 1

    echo "Validated: IP $resolved_ip is in Cloudflare range" >&2
    echo "$resolved_ip"
}

# Secure curl wrapper with TLS hardening
secure_curl() {
    local resolved_ip="$1"
    shift

    curl \
        --fail \
        --silent \
        --tlsv1.2 \
        --tls-max 1.3 \
        --proto '=https' \
        --connect-timeout "$CONNECT_TIMEOUT" \
        --resolve "${DOMAIN}:443:${resolved_ip}" \
        --ciphers 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:!aNULL:!MD5:!DSS:!RC4:!3DES' \
        "$@"
}

# Get current time with high precision
get_time() {
    if [[ -r /proc/uptime ]]; then
        awk '{print $1}' /proc/uptime
    else
        date +%s.%N 2>/dev/null || date +%s
    fi
}

# Verify response came from Cloudflare
verify_cloudflare_response() {
    local headers_file="$1"

    if ! grep -qi "cf-ray:" "$headers_file" 2>/dev/null; then
        echo "Warning: Response missing CF-RAY header - may not be from Cloudflare" >&2
        return 1
    fi

    return 0
}

# Calculate download speed
download_test() {
    local resolved_ip="$1"
    local bytes=$((DOWNLOAD_SIZE_MB * 1024 * 1024))
    local url="${DOWNLOAD_URL}?bytes=${bytes}"

    echo "Running download test (${DOWNLOAD_SIZE_MB}MB)..." >&2

    local headers_file
    headers_file=$(mktemp)

    local start_time end_time
    start_time=$(get_time)

    local received_bytes
    received_bytes=$(secure_curl "$resolved_ip" \
        --max-time "$MAX_TIME" \
        -D "$headers_file" \
        "$url" | wc -c) || {
        rm -f "$headers_file"
        echo "Error: Download failed" >&2
        return 1
    }

    end_time=$(get_time)

    # Verify response is from Cloudflare
    verify_cloudflare_response "$headers_file" || true
    rm -f "$headers_file"

    if [[ "$received_bytes" -ne "$bytes" ]]; then
        echo "Error: Expected $bytes bytes, received $received_bytes" >&2
        return 1
    fi

    local elapsed speed_mbps
    elapsed=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")
    speed_mbps=$(awk "BEGIN {printf \"%.2f\", ($bytes * 8) / ($elapsed * 1000000)}")

    echo "$speed_mbps"
}

# Calculate upload speed
upload_test() {
    local resolved_ip="$1"
    local bytes=$((UPLOAD_SIZE_MB * 1024 * 1024))

    echo "Running upload test (${UPLOAD_SIZE_MB}MB)..." >&2

    # Use private temp directory for security
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" RETURN

    local temp_file="$temp_dir/upload_data"

    # Generate random data
    if [[ -r /dev/urandom ]]; then
        dd if=/dev/urandom of="$temp_file" bs=1M count="$UPLOAD_SIZE_MB" 2>/dev/null
    else
        dd if=/dev/zero of="$temp_file" bs=1M count="$UPLOAD_SIZE_MB" 2>/dev/null
    fi

    local start_time end_time
    start_time=$(get_time)

    if secure_curl "$resolved_ip" \
        --max-time "$UPLOAD_TIMEOUT" \
        -X POST \
        -H "Content-Type: application/octet-stream" \
        --data-binary "@$temp_file" \
        "$UPLOAD_URL" > /dev/null; then

        end_time=$(get_time)

        local elapsed speed_mbps
        elapsed=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")
        speed_mbps=$(awk "BEGIN {printf \"%.2f\", ($bytes * 8) / ($elapsed * 1000000)}")
        echo "$speed_mbps"
    else
        echo "Upload test failed or timed out" >&2
        echo "0.0"
    fi
}

# Main execution
main() {
    # Check dependencies first
    if ! check_dependencies; then
        exit 1
    fi

    # Fetch Cloudflare IP ranges
    init_cloudflare_ranges

    # Validate domain and get verified IP
    local resolved_ip
    resolved_ip=$(validate_domain "$DOMAIN") || {
        echo "Error: Domain validation failed - aborting for security" >&2
        exit 1
    }

    # Run tests with validated IP
    local download_speed upload_speed

    download_speed=$(download_test "$resolved_ip") || download_speed="0.0"
    upload_speed=$(upload_test "$resolved_ip") || upload_speed="0.0"

    # Output results in a consistent format
    echo "Download: ${download_speed} Mbps"
    echo "Upload: ${upload_speed} Mbps"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
