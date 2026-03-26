"""
Rental utilities for Basilica SDK examples

Provides functions for SSH credential handling, key management, and rental lifecycle operations.
"""

from typing import Optional, Tuple
import os
import re
import time
from pathlib import Path


def parse_ssh_credentials(credentials: str) -> Tuple[str, str, int]:
    """
    Parse SSH credentials string in various formats.

    Supported formats:
    - 'user@host:port' (e.g., 'root@84.200.81.243:32776')
    - 'user@host' (defaults to port 22)
    - 'ssh user@host' (strips "ssh" prefix, defaults to port 22)
    - 'ssh user@host -p port' (strips "ssh", extracts port)

    Args:
        credentials: SSH credentials string

    Returns:
        Tuple of (user, host, port)

    Raises:
        ValueError: If credentials format is invalid
    """
    if not credentials:
        raise ValueError("SSH credentials cannot be empty")

    try:
        # Strip "ssh " prefix if present
        creds = credentials.strip()
        if creds.startswith("ssh "):
            creds = creds[4:].strip()

        # Check for -p port flag (e.g., "user@host -p 2222")
        port = 22  # default
        port_match = re.search(r'-p\s+(\d+)', creds)
        if port_match:
            port = int(port_match.group(1))
            # Remove the -p flag from the string
            creds = re.sub(r'\s+-p\s+\d+', '', creds).strip()

        # Split user@host:port or user@host
        if '@' not in creds:
            raise ValueError(f"Invalid SSH credentials format (missing @): {credentials}")

        user_part, host_port = creds.split('@', 1)

        # Check if port is specified with :
        if ':' in host_port:
            host, port_str = host_port.rsplit(':', 1)
            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    raise ValueError(f"Invalid port number: {port}")
            except ValueError:
                raise ValueError(f"Invalid port number: {port_str}")
        else:
            # No port specified, use default or -p flag value
            host = host_port

        return user_part, host, port

    except Exception as e:
        raise ValueError(f"Failed to parse SSH credentials '{credentials}': {e}")


def format_ssh_command(
    credentials: str,
    ssh_key_path: Optional[str] = None
) -> str:
    """
    Generate a complete SSH command from credentials string.
    
    Args:
        credentials: SSH credentials string (e.g., 'root@84.200.81.243:32776')
        ssh_key_path: Optional path to SSH private key. 
                      Defaults to ~/.ssh/basilica_ed25519
        
    Returns:
        Complete SSH command string
        
    Raises:
        ValueError: If credentials format is invalid
    """
    user, host, port = parse_ssh_credentials(credentials)
    
    # Use default key path if not provided (keep ~ unexpanded for display)
    if ssh_key_path is None:
        display_key_path = "~/.ssh/basilica_ed25519"
    else:
        display_key_path = ssh_key_path
    
    # Build SSH command
    return f"ssh -i {display_key_path} {user}@{host} -p {port}"


def print_ssh_instructions(
    credentials: Optional[str],
    rental_id: str,
    ssh_key_path: Optional[str] = None
) -> None:
    """
    Print formatted SSH connection instructions.
    
    Args:
        credentials: SSH credentials string or None
        rental_id: Rental ID for context
        ssh_key_path: Optional path to SSH private key
    """
    if not credentials:
        print(f"No SSH access available for rental {rental_id}")
        print("(SSH not yet provisioned)")
        return
    
    try:
        ssh_command = format_ssh_command(credentials, ssh_key_path)
        print(f"\nSSH Connection Instructions for rental {rental_id}:")
        print(f"  Command: {ssh_command}")
        
        # Parse for additional details
        user, host, port = parse_ssh_credentials(credentials)
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  User: {user}")
        
        # Check if key exists (expand for checking, but display with ~)
        key_path_expanded = os.path.expanduser(ssh_key_path or "~/.ssh/basilica_ed25519")
        display_path = ssh_key_path or "~/.ssh/basilica_ed25519"
        if not os.path.exists(key_path_expanded):
            print(f"\n  Warning: SSH key not found at {display_path}")
            print(f"     Please ensure your SSH key is properly configured")
            
    except ValueError as e:
        print(f"Error parsing SSH credentials: {e}")
        print(f"Raw credentials: {credentials}")


def find_private_key_for_public_key(registered_public_key: str) -> Optional[str]:
    """
    Find the local private key that matches the registered public key.

    Searches ~/.ssh/*.pub files for a matching key (comparing key type and key data,
    ignoring the optional comment field).

    Args:
        registered_public_key: The registered SSH public key content
            Format: "key-type key-data [optional-comment]"

    Returns:
        Path to the matching private key, or None if not found
    """
    # Parse registered key - format: "key-type key-data [comment]"
    parts = registered_public_key.strip().split()
    if len(parts) < 2:
        return None
    registered_type, registered_data = parts[0], parts[1]

    # Search ~/.ssh/*.pub files
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.exists():
        return None

    for pub_file in ssh_dir.glob("*.pub"):
        try:
            content = pub_file.read_text().strip()
            file_parts = content.split()
            if len(file_parts) >= 2:
                # Compare key type and key data (ignore comment)
                if file_parts[0] == registered_type and file_parts[1] == registered_data:
                    # Found match - return private key path (without .pub)
                    private_key = pub_file.with_suffix("")
                    if private_key.exists():
                        return str(private_key)
        except (IOError, OSError):
            continue

    return None


def wait_for_rental_ready(client, rental_id: str, rental_type: str = "cpu", timeout: int = 300):
    """
    Wait for a rental to become ready with SSH access.

    Args:
        client: BasilicaClient instance
        rental_id: The rental ID to wait for
        rental_type: Type of rental - "cpu", "gpu", or "secure_cloud" (default: "cpu")
        timeout: Maximum time to wait in seconds (default: 5 minutes)

    Returns:
        The rental info dict when ready (with SSH command available)

    Raises:
        TimeoutError: If rental doesn't become ready within timeout
        RuntimeError: If rental fails
        ValueError: If rental_type is not recognized
    """
    # Map rental type to the appropriate list method
    list_methods = {
        "cpu": "list_cpu_rentals",
        "gpu": "list_secure_cloud_rentals",
        "secure_cloud": "list_secure_cloud_rentals",
    }

    if rental_type not in list_methods:
        raise ValueError(f"Unknown rental_type: {rental_type}. Must be one of: {', '.join(list_methods.keys())}")

    list_method_name = list_methods[rental_type]
    list_method = getattr(client, list_method_name)

    print(f"\nWaiting for rental {rental_id} to be ready...")
    start_time = time.time()
    last_status = None
    waiting_for_ssh = False

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Rental did not become ready within {timeout} seconds")

        # Find the rental in the list
        rentals_response = list_method()
        rental = None
        for r in rentals_response.rentals:
            if r.rental_id == rental_id:
                rental = r
                break

        if rental is None:
            raise RuntimeError(f"Rental {rental_id} not found")

        status = rental.status.lower()

        # Print status updates
        if status != last_status:
            print(f"  Status: {rental.status} (elapsed: {int(elapsed)}s)")
            last_status = status

        # Check terminal states
        if status == "running":
            # Also wait for SSH command to be available
            if rental.ssh_command:
                return rental
            elif not waiting_for_ssh:
                print(f"  Waiting for SSH access... (elapsed: {int(elapsed)}s)")
                waiting_for_ssh = True
        elif status in ("failed", "error", "terminated", "stopped"):
            raise RuntimeError(f"Rental failed with status: {rental.status}")

        # Wait before next poll
        time.sleep(3)