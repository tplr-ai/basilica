#!/usr/bin/env python3
import subprocess
import sys

try:
    # Run nvidia-smi to check memory usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 4:
                name, total, used, free = parts
                print(f"GPU: {name}")
                print(f"  Total: {total} MB")
                print(f"  Used: {used} MB")
                print(f"  Free: {free} MB")
                print()
    else:
        print("Failed to run nvidia-smi")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)