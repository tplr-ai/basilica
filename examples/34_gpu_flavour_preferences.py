#!/usr/bin/env python3
"""
Query GPU offerings with flavour filters (interconnect, region, spot).

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 34_gpu_flavour_preferences.py
"""
from basilica import BasilicaClient, GpuPriceQuery

client = BasilicaClient()

# SXM interconnect in the US
gpus = client.list_secure_cloud_gpus(
    query=GpuPriceQuery(interconnect="SXM", region="US")
)
print(f"SXM in US: {len(gpus)}")
for g in gpus[:3]:
    print(f"  {g.gpu_type} x{g.gpu_count}  {g.interconnect}  {g.region}  ${g.hourly_rate}/hr")

# Non-spot only
gpus = client.list_secure_cloud_gpus(query=GpuPriceQuery(exclude_spot=True))
print(f"\nNon-spot: {len(gpus)}")

# Spot only
gpus = client.list_secure_cloud_gpus(query=GpuPriceQuery(spot_only=True))
print(f"Spot: {len(gpus)}")
