#!/usr/bin/env python3
"""
Get Deployment by Name - Reconnect to a deployment by the name you chose.

When you deploy, Basilica returns a stable UUID as the deployment's
`instance_name`. The deployment name you passed to `client.deploy(name=...)`
is preserved separately as `friendly_name`. This example shows how to look
a deployment back up by its name, so scripts can reconnect across restarts
without copy-pasting UUIDs from `basilica deploy ls`.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 36_get_by_name.py
"""
from basilica import BasilicaClient

DEPLOYMENT_NAME = "lookup-demo"

client = BasilicaClient()

# Step 1: Deploy with a name we'll remember.
deployment = client.deploy(
    name=DEPLOYMENT_NAME,
    source="""
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello from Basilica!')

HTTPServer(('', 8000), Handler).serve_forever()
""",
    port=8000,
    ttl_seconds=600,
)

print(f"Deployed:")
print(f"  name:        {deployment.friendly_name}")
print(f"  instance id: {deployment.name}  (the stable UUID)")
print(f"  url:         {deployment.url}")

# Step 2: Forget the UUID. Look the deployment back up by its name.
# This is what a follow-up script would do — no UUID needed.
found = client.get_by_name(DEPLOYMENT_NAME)

assert found.name == deployment.name, "should resolve to the same deployment"
print(f"\nLooked up by name '{DEPLOYMENT_NAME}':")
print(f"  url: {found.url}")

# Step 3: Operate on the looked-up deployment normally.
found.delete()
print("\nDeleted.")
