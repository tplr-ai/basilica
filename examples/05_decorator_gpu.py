#!/usr/bin/env python3
"""
Decorator-based GPU deployment with PyTorch.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 05_decorator_gpu.py
"""
import basilica


@basilica.deployment(
    name="decorator-gpu",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    port=8000,
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="8Gi",
    ttl_seconds=600,
    timeout=300,
)
def serve():
    import json
    import torch
    from http.server import HTTPServer, BaseHTTPRequestHandler

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

        def log_message(self, *a):
            pass

    HTTPServer(('', 8000), Handler).serve_forever()


deployment = serve()
print(f"GPU info: {deployment.url}")
