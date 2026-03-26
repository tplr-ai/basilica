#!/usr/bin/env python3
"""
GPU deployment - PyTorch with CUDA.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 04_gpu.py
"""
from basilica import BasilicaClient

client = BasilicaClient()

deployment = client.deploy(
    name="gpu-test",
    source="""
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(info).encode())

HTTPServer(('', 8000), Handler).serve_forever()
""",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    port=8000,
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="8Gi",
    ttl_seconds=600,
    timeout=300,
)

print(f"GPU info at: {deployment.url}")
