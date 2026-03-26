#!/usr/bin/env python3
"""
Deploy containers with GPU flavour preferences (interconnect, geo, spot).

Shows all deployment methods: create_deployment, deploy, deploy_vllm, deploy_sglang.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 35_deploy_with_flavour.py
"""
from basilica import BasilicaClient

client = BasilicaClient()

# --- create_deployment with flavour (returns immediately) ---
print("create_deployment with flavour...")
resp = client.create_deployment(
    instance_name="flavour-echo",
    image="hashicorp/http-echo",
    port=5678,
    gpu_count=1,
    gpu_models=["H100"],
    interconnect="SXM",
    ttl_seconds=120,
)
print(f"  {resp.instance_name}  state={resp.state}  url={resp.url}")
client.delete_deployment(resp.instance_name)
print("  deleted")

# --- deploy with flavour (waits until ready) ---
print("\ndeploy with flavour...")
deployment = client.deploy(
    name="flavour-hello",
    source="""
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'hello from flavour deploy')
HTTPServer(('', 8000), H).serve_forever()
""",
    port=8000,
    ttl_seconds=120,
)
print(f"  {deployment.url}")
deployment.delete()
print("  deleted")

# --- deploy_vllm with flavour ---
# Uncomment to run (waits for GPU scheduling + model download):
#
# deployment = client.deploy_vllm(
#     model="Qwen/Qwen3-0.6B",
#     interconnect="SXM",
#     ttl_seconds=600,
# )
# print(f"vLLM: {deployment.url}/v1/chat/completions")
# deployment.delete()

# --- deploy_sglang with flavour ---
# Uncomment to run:
#
# deployment = client.deploy_sglang(
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     interconnect="SXM",
#     ttl_seconds=600,
# )
# print(f"SGLang: {deployment.url}/v1/chat/completions")
# deployment.delete()
