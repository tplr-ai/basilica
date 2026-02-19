#!/usr/bin/env python3
"""
Script to update ABI and bytecode from compiled Solidity contract.

Usage:
    python update_abi.py
"""

import json
import re
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
COMPILED_JSON = SCRIPT_DIR / "out/CollateralUpgradeable.sol/CollateralUpgradeable.json"

ABI_OUTPUT = SCRIPT_DIR / "src/CollateralUpgradableABI.json"
LIB_RS = SCRIPT_DIR / "src/lib.rs"


def read_compiled_contract():
    """Read the compiled contract JSON."""
    print(f"Reading compiled contract from: {COMPILED_JSON}")
    with open(COMPILED_JSON, 'r') as f:
        return json.load(f)

def update_abi_file(abi):
    """Update the ABI JSON file."""
    print(f"Writing ABI to: {ABI_OUTPUT}")
    with open(ABI_OUTPUT, 'w') as f:
        json.dump(abi, f, indent=2)
    print(f"✓ Updated {ABI_OUTPUT}")

def update_lib_rs_bytecode(bytecode_obj):
    """Update the bytecode in lib.rs sol! macro."""
    print(f"Updating bytecode in: {LIB_RS}")
    
    # Read the current lib.rs
    with open(LIB_RS, 'r') as f:
        content = f.read()
    
    # Find the sol! macro block and replace bytecode
    # Pattern: bytecode = "0x..."
    pattern = r'(bytecode\s*=\s*)"0x[a-fA-F0-9]+"'
    replacement = rf'\1"{bytecode_obj}"'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        print("⚠ Warning: Could not find bytecode pattern in lib.rs")
        return False
    
    # Write back
    with open(LIB_RS, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated bytecode in {LIB_RS} ({count} occurrence(s))")
    print(f"  Bytecode size: {len(bytecode_obj)} characters ({len(bytecode_obj)//2} bytes)")
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("Updating ABI and Bytecode from Compiled Contract")
    print("=" * 60)
    
    # Read compiled contract
    contract_data = read_compiled_contract()

    # Extract ABI
    abi = contract_data.get('abi')
    if not abi:
        print("❌ Error: No 'abi' field found in compiled contract")
        return 1

    # Extract bytecode
    bytecode_obj = contract_data.get('bytecode', {}).get('object')
    if not bytecode_obj:
        print("❌ Error: No bytecode.object found in compiled contract")
        return 1

    # Ensure bytecode starts with 0x
    if not bytecode_obj.startswith('0x'):
        bytecode_obj = '0x' + bytecode_obj

    # Update files
    update_abi_file(abi)
    update_lib_rs_bytecode(bytecode_obj)

    print("=" * 60)
    print("✅ Update completed successfully!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
