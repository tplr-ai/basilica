#!/usr/bin/env python3
"""Check subnet registration status on local chain"""

import asyncio
from substrateinterface import SubstrateInterface

async def check_subnet():
    # Connect to local subtensor
    substrate = SubstrateInterface(
        url="ws://localhost:9944"
    )
    
    print("Connected to local subtensor")
    
    # Check if subnet 1 exists
    try:
        # Get subnet info
        result = substrate.query(
            module='SubtensorModule',
            storage_function='NetworksAdded',
            params=[1]
        )
        
        if result.value:
            print(f"Subnet 1 exists: {result.value}")
        else:
            print("Subnet 1 does not exist")
            
        # Try to get neurons on subnet 1
        print("\nChecking for registered neurons on subnet 1...")
        
        # Get UIDs
        uids = substrate.query(
            module='SubtensorModule', 
            storage_function='Uids',
            params=[1]
        )
        
        if uids:
            print(f"Found UIDs: {uids.value}")
            
            # Get neuron info for first few UIDs
            for uid in range(min(5, len(uids.value) if uids.value else 0)):
                neuron = substrate.query(
                    module='SubtensorModule',
                    storage_function='Neurons', 
                    params=[1, uid]
                )
                if neuron and neuron.value:
                    print(f"\nNeuron UID {uid}:")
                    print(f"  Hotkey: {neuron.value.get('hotkey', 'N/A')}")
                    print(f"  Coldkey: {neuron.value.get('coldkey', 'N/A')}")
                    print(f"  Active: {neuron.value.get('active', 'N/A')}")
        else:
            print("No neurons found on subnet 1")
            
    except Exception as e:
        print(f"Error: {e}")
    
    substrate.close()

if __name__ == "__main__":
    asyncio.run(check_subnet())