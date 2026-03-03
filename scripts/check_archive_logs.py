#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "rich"]
# ///
"""
Check whether the Bittensor archive node serves EVM eth_getLogs for blocks older than 2000.
Tests both the lite node and archive node for comparison.
"""

import sys
import requests
from rich.console import Console
from rich.table import Table

console = Console()

ENDPOINTS = {
    "lite (current)": "https://lite.chain.opentensor.ai",
    "archive": "https://archive.chain.opentensor.ai",
}

# How far back to test (well past the ~300 block pruning limit on lite nodes)
TEST_OFFSETS = [100, 500, 2000, 5000, 10000]


def rpc(url: str, method: str, params: list) -> dict:
    resp = requests.post(
        url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)


def check_endpoint(name: str, url: str) -> None:
    console.rule(f"[bold cyan]{name}[/] — {url}")

    # 1. Get current block
    try:
        result = rpc(url, "eth_blockNumber", [])
    except Exception as e:
        console.print(f"[red]Cannot connect:[/] {e}")
        return

    if "error" in result:
        console.print(f"[red]eth_blockNumber error:[/] {result['error']}")
        return

    current_block = hex_to_int(result["result"])
    console.print(f"Current block: [bold]{current_block:,}[/]")

    table = Table("Blocks back", "Block range", "Logs returned", "Error")
    table.show_header = True

    for offset in TEST_OFFSETS:
        from_block = max(0, current_block - offset - 100)
        to_block = max(0, current_block - offset)

        if from_block >= current_block:
            continue

        try:
            result = rpc(
                url,
                "eth_getLogs",
                [
                    {
                        "fromBlock": hex(from_block),
                        "toBlock": hex(to_block),
                        # No address filter — we just want to know if the node has the data
                    }
                ],
            )
        except Exception as e:
            table.add_row(f"-{offset}", f"{from_block}–{to_block}", "—", str(e))
            continue

        if "error" in result:
            err = result["error"]
            table.add_row(
                f"-{offset}",
                f"{from_block:,}–{to_block:,}",
                "[red]error[/]",
                f"{err.get('code')}: {err.get('message', '')}",
            )
        else:
            logs = result.get("result", [])
            table.add_row(
                f"-{offset}",
                f"{from_block:,}–{to_block:,}",
                f"[green]{len(logs)}[/]",
                "",
            )

    console.print(table)


def main() -> None:
    console.print("\n[bold]Bittensor EVM archive node log availability check[/]\n")

    for name, url in ENDPOINTS.items():
        check_endpoint(name, url)
        console.print()

    console.print(
        "[dim]Note: '0 logs' is fine — it means the node responded with an empty result "
        "(no matching events in that range). An 'error' means the node cannot serve that range.[/dim]\n"
    )


if __name__ == "__main__":
    main()
