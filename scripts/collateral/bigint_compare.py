#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Big-integer comparison and formatting for EVM balance values.

Bash overflows at 2^63, which is less than 10 TAO in wei (10e18).

Usage:
    python3 bigint_compare.py gt  <a> <b>       # exit 0 if a > b
    python3 bigint_compare.py gte <a> <b>       # exit 0 if a >= b
    python3 bigint_compare.py eq  <a> <b>       # exit 0 if a == b
    python3 bigint_compare.py lt  <a> <b>       # exit 0 if a < b
    python3 bigint_compare.py lte <a> <b>       # exit 0 if a <= b
    python3 bigint_compare.py fmt_tao   <wei>   # print human-readable TAO value
    python3 bigint_compare.py fmt_alpha <rao>   # print human-readable alpha value

Handles hex (0x...) and decimal inputs.
Exit codes: 0 = true, 1 = false, 2 = usage error.
"""
import sys


def parse_int(s: str) -> int:
    s = s.strip()
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    return int(s)


def fmt_tao(wei: int) -> str:
    whole = wei // 10**18
    frac = wei % 10**18
    return f"{whole}.{frac:018d} TAO ({wei} wei)"


def fmt_alpha(rao: int) -> str:
    whole = rao // 10**9
    frac = rao % 10**9
    return f"{whole}.{frac:09d} alpha ({rao} rao)"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    cmd = sys.argv[1]

    if cmd == "fmt_tao":
        print(fmt_tao(parse_int(sys.argv[2])))
        return 0

    if cmd == "fmt_alpha":
        print(fmt_alpha(parse_int(sys.argv[2])))
        return 0

    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2

    a = parse_int(sys.argv[2])
    b = parse_int(sys.argv[3])

    ops = {"gt": a > b, "gte": a >= b, "eq": a == b, "lt": a < b, "lte": a <= b}

    if cmd not in ops:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        return 2

    return 0 if ops[cmd] else 1


if __name__ == "__main__":
    sys.exit(main())
