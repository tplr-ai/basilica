#!/usr/bin/env python3
"""
Initialize local Subtensor devnet using bittensor Python library.
Creates wallets, subnet, and registers hotkeys.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import bittensor as bt
    from bittensor_wallet import Wallet
except ImportError:
    print("❌ bittensor not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "bittensor"])
    import bittensor as bt
    from bittensor_wallet import Wallet


# Configuration from environment or defaults
CHAIN_ENDPOINT = os.getenv("CHAIN_ENDPOINT", "ws://127.0.0.1:19944")
NETUID = int(os.getenv("NETUID", "2"))
WALLET_PATH = Path(os.getenv("WALLET_PATH", str(Path.home() / ".bittensor" / "wallets")))

# Set SSL cert for WSS endpoints
SCRIPT_DIR = Path(__file__).parent.resolve()
if CHAIN_ENDPOINT.startswith("wss://"):
    ca_cert = SCRIPT_DIR / "tls" / "ca.crt"
    if ca_cert.exists():
        os.environ["SSL_CERT_FILE"] = str(ca_cert)
    else:
        # Try to find it in the environment
        if "SSL_CERT_FILE" not in os.environ:
            os.environ["SSL_CERT_FILE"] = str(ca_cert)  # Set it anyway, will fail gracefully if missing

# Colors
class C:
    R = '\033[0;31m'
    G = '\033[0;32m'
    Y = '\033[1;33m'
    NC = '\033[0m'


def log(msg: str, color: str = C.NC):
    """Print colored log message."""
    print(f"{color}{msg}{C.NC}", flush=True)


def wait_for_chain(subtensor: bt.Subtensor, max_attempts: int = 30) -> bool:
    """Wait for chain to be ready."""
    log("[init] Verifying chain connectivity via bittensor SDK...", C.Y)
    for i in range(max_attempts):
        try:
            # Try to get block number to verify connection
            _ = subtensor.get_current_block()
            log("[init] bittensor SDK connectivity OK", C.G)
            return True
        except Exception as e:
            if i < max_attempts - 1:
                log(f"[init] Chain not ready yet (attempt {i+1}/{max_attempts}). Waiting...", C.Y)
                time.sleep(2)
            else:
                log(f"[init] Failed to connect: {e}", C.R)
    return False


def disable_evm_whitelist(subtensor: bt.Subtensor) -> bool:
    """Disable EVM deployment whitelist via sudo, matching mainnet behavior."""
    log("[init] Disabling EVM deployment whitelist (matching mainnet)...", C.Y)
    try:
        from substrateinterface import Keypair

        substrate = subtensor.substrate
        alice = Keypair.create_from_uri("//Alice")

        inner_call = substrate.compose_call(
            call_module="EVM",
            call_function="disable_whitelist",
            call_params={"disabled": True},
        )
        sudo_call = substrate.compose_call(
            call_module="Sudo",
            call_function="sudo",
            call_params={"call": inner_call},
        )
        extrinsic = substrate.create_signed_extrinsic(call=sudo_call, keypair=alice)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        if receipt.is_success:
            log("[init] EVM deployment whitelist disabled", C.G)
            return True
        else:
            log(f"[init] WARNING: EVM whitelist disable failed: {receipt.error_message}", C.Y)
            return False
    except Exception as e:
        log(f"[init] WARNING: Could not disable EVM whitelist: {e}", C.Y)
        return False


def create_alice_from_seed(wallet_path: Path) -> Optional[Wallet]:
    """Create Alice wallet from known seed."""
    wallet = Wallet(name="Alice", hotkey="default", path=str(wallet_path))

    coldkey_path = wallet_path / "Alice" / "coldkey"
    if coldkey_path.exists():
        log("[init] Alice wallet already exists")
    else:
        log("[init] Creating Alice wallet (coldkey) from seed")
        # Alice's well-known seed for substrate dev chains
        seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
        try:
            wallet.regenerate_coldkey(seed=seed, use_password=False, overwrite=False, suppress=True)
        except Exception as e:
            log(f"[init] Alice coldkey may already exist: {e}")

    return wallet


def ensure_hotkey(wallet_name: str, hotkey_name: str, wallet_path: Path) -> Wallet:
    """Ensure hotkey exists for wallet."""
    wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
    hotkey_path = wallet_path / wallet_name / "hotkeys" / hotkey_name

    if not hotkey_path.exists():
        log(f"[init] Creating hotkey {hotkey_name}")
        wallet.create_new_hotkey(use_password=False, overwrite=False, suppress=True)

    return wallet


def wait_for_subnet_ready(subtensor: bt.Subtensor, netuid: int, max_attempts: int = 30) -> bool:
    """Wait for subnet to be queryable and ready for registrations."""
    log(f"[init] Waiting for subnet {netuid} to be ready for registrations...")

    for attempt in range(max_attempts):
        try:
            # Try to get metagraph - this will fail if subnet isn't ready
            metagraph = subtensor.metagraph(netuid=netuid)
            log(f"[init] ✅ Subnet {netuid} is ready (neurons: {len(metagraph.neurons)})", C.G)
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(2)  # Wait 2 seconds between attempts
            else:
                log(f"[init] WARNING: Subnet {netuid} not ready after {max_attempts} attempts: {e}", C.Y)
                return False

    return False


def create_subnet_with_retry(subtensor: bt.Subtensor, wallet: Wallet, max_attempts: int = 10) -> bool:
    """Create subnet with retries."""
    log("[init] Creating subnet (may already exist)")

    for attempt in range(max_attempts):
        try:
            success = subtensor.register_subnet(
                wallet=wallet,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )
            if success:
                log(f"[init] ✅ Registered subnetwork with netuid: {NETUID}", C.G)
                return True
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower() or "SubNetworkExists" in error_msg:
                log(f"[init] Subnet {NETUID} already exists")
                return True

            if attempt < max_attempts - 1:
                log(f"[init] subnet create failed (attempt {attempt+1} of {max_attempts}). Waiting before retry...")
                time.sleep(6)

    log("[init] subnet create may have failed but continuing...", C.Y)
    return True  # Continue anyway


def register_hotkey_with_retry(
    subtensor: bt.Subtensor,
    wallet: Wallet,
    netuid: int,
    hotkey_name: str,
    max_attempts: int = 5
) -> bool:
    """Register hotkey with retries."""
    for attempt in range(max_attempts):
        try:
            success = subtensor.burned_register(
                wallet=wallet,
                netuid=netuid,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )

            if success:
                log(f"[init] Registered {hotkey_name}", C.G)
                return True

        except Exception as e:
            error_msg = str(e)

            # Check if already registered
            if any(x in error_msg.lower() for x in ["already registered", "AlreadyRegistered"]):
                log(f"[init] {hotkey_name} already registered")
                return True

            # Check for specific errors
            if "ancient birth block" in error_msg.lower():
                log(f"[init] Got 'ancient birth block' error for {hotkey_name}, waiting longer...")
                time.sleep(10)
            else:
                if attempt < max_attempts - 1:
                    log(f"[init] Registration failed for {hotkey_name}: {error_msg}, retrying...")
                    time.sleep(5)

    log(f"[init] Could not register {hotkey_name} after {max_attempts} attempts", C.Y)
    return True  # Continue anyway


def start_emission_with_retry(subtensor: bt.Subtensor, wallet: Wallet, netuid: int, max_attempts: int = 10) -> bool:
    """Start subnet emission with retries."""
    log("[init] Starting emission (may already be started)")

    for attempt in range(max_attempts):
        try:
            # Note: start_call returns (success, message) tuple
            result = subtensor.start_call(
                wallet=wallet,
                netuid=netuid,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )

            # Handle both tuple and boolean returns
            success = result[0] if isinstance(result, tuple) else result

            if success:
                log("[init] Subnet emission schedule started", C.G)
                return True

        except Exception as e:
            error_msg = str(e)
            if "already started" in error_msg.lower() or "AlreadyStarted" in error_msg:
                log("[init] Subnet emission already started")
                return True

            if attempt < max_attempts - 1:
                log(f"[init] subnet start failed (attempt {attempt+1} of {max_attempts}). Waiting before retry...")
                time.sleep(6)

    log("[init] subnet start may have failed but continuing...", C.Y)
    return True  # Continue anyway


def main():
    """Main initialization function."""
    log(f"[init] Using CHAIN_ENDPOINT={CHAIN_ENDPOINT} NETUID={NETUID} WALLET_PATH={WALLET_PATH}")
    if "SSL_CERT_FILE" in os.environ:
        log(f"[init] SSL_CERT_FILE={os.environ['SSL_CERT_FILE']}")

    # Create wallet directory
    WALLET_PATH.mkdir(parents=True, exist_ok=True)
    (WALLET_PATH / "Alice" / "hotkeys").mkdir(parents=True, exist_ok=True)

    # Connect to subtensor
    log("[init] Connecting to local subtensor...")
    subtensor = bt.Subtensor(network=CHAIN_ENDPOINT)

    # Wait for chain to be ready
    if not wait_for_chain(subtensor):
        log("[init] ERROR: Could not connect to chain", C.R)
        return 1

    log(f"[init] bittensor version: {bt.__version__}")

    # Disable EVM deployment whitelist (matches mainnet behavior)
    disable_evm_whitelist(subtensor)

    # Create Alice wallet from seed
    alice_default = create_alice_from_seed(WALLET_PATH)
    if not alice_default:
        log("[init] ERROR: Failed to create Alice wallet", C.R)
        return 1

    # Create hotkeys
    ensure_hotkey("Alice", "default", WALLET_PATH)
    alice_m1 = ensure_hotkey("Alice", "M1", WALLET_PATH)
    alice_m2 = ensure_hotkey("Alice", "M2", WALLET_PATH)

    # Create subnet
    if not create_subnet_with_retry(subtensor, alice_default):
        log("[init] WARNING: Subnet creation may have failed", C.Y)

    # Wait for subnet to be ready for registrations
    if not wait_for_subnet_ready(subtensor, NETUID):
        log("[init] ERROR: Subnet not ready, cannot proceed with registrations", C.R)
        return 1

    # Register hotkeys (validator + miners)
    log("[init] Registering validator hotkey (default)")
    register_hotkey_with_retry(subtensor, alice_default, NETUID, "default (validator)")

    log("[init] Registering miner hotkeys M1/M2")
    register_hotkey_with_retry(subtensor, alice_m1, NETUID, "M1")
    register_hotkey_with_retry(subtensor, alice_m2, NETUID, "M2")

    # Start emission
    start_emission_with_retry(subtensor, alice_default, NETUID)

    log(f"[init] Complete. Validator hotkey (Alice/default): 5DJBmrfyRqe6eUUHLaWSho3Wgr5i8gDTWKxxWEmXvFhHvWTM", C.G)
    log(f"[init] Coldkey (Alice): 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", C.G)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n[init] Interrupted by user", C.R)
        sys.exit(1)
    except Exception as e:
        log(f"\n[init] ERROR: {e}", C.R)
        import traceback
        traceback.print_exc()
        sys.exit(1)
