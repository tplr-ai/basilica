#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "bittensor>=9.0.0",
#   "substrate-interface>=1.7.0",
#   "click>=8.1.0",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""
Basilica Bittensor Network Monitor

Monitor extrinsics and transactions on a Bittensor substrate blockchain,
specifically focusing on miner and validator operations.

Usage:
    ./monitoring.py                              # Monitor local network
    ./monitoring.py --network test               # Monitor testnet
    ./monitoring.py --network finney --netuid 1  # Monitor mainnet subnet 1
    ./monitoring.py --network ws://custom:9944   # Custom node URL

The tool monitors and displays:
- Miner registrations (serve_axon)
- Validator weight updates (set_weights)
- Other SubtensorModule transactions from known miners/validators

Example output:
    2025-07-02 14:32:01 Block 12345 Extrinsic 2
      Type: miner
      Call: SubtensorModule.serve_axon
      Signer: 5GrwvaEF5ZtKh...
      Axon: 192.168.1.100:8091
      Netuid: 1

Press Ctrl+C to stop monitoring.
"""

import sys
import time
from datetime import datetime
from typing import Set

import bittensor
import click
from substrateinterface import SubstrateInterface


class Monitor:
    def __init__(self, network: str = "local", netuid: int = 1, show_failures: bool = True):
        self.network = network
        self.netuid = netuid
        self.show_failures = show_failures
        self.substrate = None
        self.subtensor = None
        self.known_miners: Set[str] = set()
        self.known_validators: Set[str] = set()
        self.block_count = 0
        self.extrinsic_count = 0
        self.failed_count = 0

    def connect(self):
        """Connect to the Bittensor network."""
        if self.network == "local":
            url = "ws://127.0.0.1:9944"
        elif self.network == "test":
            url = "wss://test.finney.opentensor.ai:443"
        elif self.network == "finney":
            url = "wss://entrypoint-finney.opentensor.ai:443"
        else:
            url = self.network

        print(f"Connecting to {url}...")

        try:
            self.substrate = SubstrateInterface(url=url)
            self.subtensor = bittensor.subtensor(network=self.network)
            print(f"Connected to {url}")
            self.load_metagraph()
        except Exception as e:
            print(f"Failed to connect: {e}")
            sys.exit(1)

    def load_metagraph(self):
        """Load metagraph to identify miners and validators."""
        try:
            metagraph = self.subtensor.metagraph(self.netuid)

            for neuron in metagraph.neurons:
                if neuron.hotkey:
                    if hasattr(neuron, "validator_permit") and neuron.validator_permit:
                        self.known_validators.add(neuron.hotkey)
                    else:
                        self.known_miners.add(neuron.hotkey)

            print(
                f"Loaded {len(self.known_miners)} miners, {len(self.known_validators)} validators"
            )
        except Exception as e:
            print(f"Warning: Failed to load metagraph: {e}")

    def categorize_address(self, address: str) -> str:
        """Categorize an address as miner, validator, or unknown."""
        if address in self.known_validators:
            return "validator"
        elif address in self.known_miners:
            return "miner"
        else:
            return "unknown"

    def process_extrinsic(self, extrinsic_data, block_number: int, idx: int):
        """Process a single extrinsic."""
        try:
            # Extract the actual data
            extrinsic = extrinsic_data.value if hasattr(extrinsic_data, "value") else extrinsic_data

            # Skip unsigned extrinsics (inherents like timestamp)
            if not extrinsic.get("signature"):
                return

            # Extract call information
            call = extrinsic.get("call")
            if not call:
                return

            module = call.get("call_module", "unknown")
            function = call.get("call_function", "unknown")

            # Skip non-SubtensorModule calls unless they're system errors
            if module != "SubtensorModule" and module != "System":
                return

            # Get signer address
            signer = extrinsic.get("address", "unknown")
            if hasattr(signer, "value"):
                signer = signer.value

            category = self.categorize_address(str(signer))

            # Extract call arguments
            args_dict = {}
            if "call_args" in call:
                for arg in call["call_args"]:
                    args_dict[arg["name"]] = arg["value"]

            # Log the extrinsic
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{timestamp} Block {block_number} Extrinsic {idx}")
            print(f"  Type: {category}")
            print(f"  Call: {module}.{function}")
            print(f"  Signer: {signer}")

            # Show raw parameters for all calls
            if args_dict:
                print("  Parameters:")
                for key, value in args_dict.items():
                    print(f"    {key}: {value}")

            # Log specific formatted details for important calls
            if function == "serve_axon" and args_dict:
                ip = args_dict.get("ip", "unknown")
                port = args_dict.get("port", "unknown")
                netuid = args_dict.get("netuid", "unknown")
                print(f"  Summary: Registering on subnet {netuid} at {ip}:{port}")

            elif function == "set_weights" and args_dict:
                dests = args_dict.get("dests", [])
                values = args_dict.get("values", [])
                version = args_dict.get("version_key", "unknown")
                print(f"  Summary: Setting {len(dests)} weights, version {version}")

                # Show all weights
                if dests and values:
                    print("  Weight details:")
                    weights = sorted(
                        zip(dests, values, strict=False), key=lambda x: x[1], reverse=True
                    )
                    for uid, weight in weights:
                        print(f"    UID {uid}: {weight}")

            self.extrinsic_count += 1

        except Exception as e:
            print(f"Error processing extrinsic: {e}")

    def process_events(self, block_hash, block_number: int):
        """Process events for a block to find transaction results."""
        if not self.show_failures:
            return

        try:
            # Get events for this block
            events = self.substrate.get_events(block_hash)

            if not events:
                return

            for event in events:
                event_data = event.value if hasattr(event, "value") else event

                # Check for ExtrinsicSuccess or ExtrinsicFailed events
                if event_data.get("module_id") == "System":
                    event_id = event_data.get("event_id")

                    if event_id == "ExtrinsicFailed":
                        self.failed_count += 1

                        # Extract failure information
                        attributes = event_data.get("attributes", {})
                        error_info = attributes.get("dispatch_error", {})
                        extrinsic_idx = attributes.get("extrinsic_index", "unknown")

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n{timestamp} Block {block_number} - FAILED TRANSACTION")
                        print(f"  Extrinsic Index: {extrinsic_idx}")

                        # Try to decode the error
                        if isinstance(error_info, dict):
                            if "Module" in error_info:
                                module_info = error_info["Module"]
                                if isinstance(module_info, dict):
                                    print(f"  Module: {module_info.get('index', 'unknown')}")
                                    print(f"  Error: {module_info.get('error', 'unknown')}")
                            elif "BadOrigin" in error_info:
                                print("  Error: BadOrigin - insufficient permissions")
                            elif "CannotLookup" in error_info:
                                print("  Error: CannotLookup - failed to lookup account")
                            elif "ConsumerRemaining" in error_info:
                                print("  Error: ConsumerRemaining - account has active consumers")
                            elif "NoProviders" in error_info:
                                print("  Error: NoProviders - account has no providers")
                            elif "TooManyConsumers" in error_info:
                                print("  Error: TooManyConsumers - too many consumers")
                            elif "Token" in error_info:
                                token_error = error_info["Token"]
                                print(f"  Error: Token error - {token_error}")
                            elif "Arithmetic" in error_info:
                                arith_error = error_info["Arithmetic"]
                                print(f"  Error: Arithmetic error - {arith_error}")
                            else:
                                print(f"  Error Info: {error_info}")
                        else:
                            print(f"  Raw Error: {error_info}")

        except Exception as e:
            # Don't interrupt monitoring for event processing errors
            if self.block_count < 5:  # Only show errors in first few blocks
                print(f"Debug: Event processing error: {e}")

    def monitor(self):
        """Monitor blocks for extrinsics and events."""
        print("Starting monitoring...")
        if self.show_failures:
            print("Monitoring both successful and failed transactions")
        else:
            print("Monitoring successful transactions only")
        print("Press Ctrl+C to stop\n")

        last_block_hash = None

        try:
            while True:
                try:
                    # Get latest block
                    block_hash = self.substrate.get_block_hash()

                    if block_hash != last_block_hash:
                        last_block_hash = block_hash
                        block = self.substrate.get_block(block_hash=block_hash)

                        if block:
                            block_number = block["header"]["number"]
                            self.block_count += 1

                            # Progress indicator every 10 blocks
                            if self.block_count % 10 == 0:
                                status = f"[Block {block_number}] Monitoring... ({self.extrinsic_count} extrinsics"
                                if self.show_failures:
                                    status += f", {self.failed_count} failed"
                                status += ")"
                                print(f"\r{status}", end="", flush=True)

                            # Process events first to catch failures
                            if self.show_failures:
                                self.process_events(block_hash, block_number)

                            # Process extrinsics
                            if "extrinsics" in block:
                                for idx, extrinsic in enumerate(block["extrinsics"]):
                                    self.process_extrinsic(extrinsic, block_number, idx)

                    time.sleep(0.5)

                except Exception as e:
                    print(f"\nError in monitoring loop: {e}")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            print(f"Processed {self.block_count} blocks")
            print(f"Found {self.extrinsic_count} SubtensorModule extrinsics")
            if self.show_failures:
                print(f"Failed transactions detected: {self.failed_count}")


@click.command()
@click.option(
    "--network", default="local", help="Network to connect to (local, test, finney, or custom URL)"
)
@click.option("--netuid", default=1, type=int, help="Network UID to monitor")
@click.option("--no-failures", is_flag=True, help="Don't show failed transactions")
def main(network: str, netuid: int, no_failures: bool):
    """Monitor Bittensor blockchain for miner and validator extrinsics."""
    print("Basilica Network Monitor")
    print(f"Network: {network}, Netuid: {netuid}")
    print("-" * 50)

    monitor = Monitor(network=network, netuid=netuid, show_failures=not no_failures)
    monitor.connect()
    monitor.monitor()


if __name__ == "__main__":
    main()
