#!/usr/bin/env python3
"""
Async Concurrent Deployments - Deploy 50 apps in parallel.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 21_async_concurrent.py
"""

import asyncio
import random
import time
import urllib.request

from basilica import BasilicaClient

NUM_APPS = 50


def app():
    import os
    from http.server import HTTPServer, BaseHTTPRequestHandler

    app_id = os.environ.get("APP_ID", "00")

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"app-{app_id} hello from basilica".encode())

        def log_message(self, *a):
            pass

    HTTPServer(("", 8000), H).serve_forever()


async def deploy(client: BasilicaClient, id: int) -> dict:
    start = time.monotonic()
    try:
        d = await client.deploy_async(
            name=f"async-{id:02d}",
            source=app,
            env={"APP_ID": f"{id:02d}"},
            port=8000,
            ttl_seconds=180,
            timeout=180,
        )
        elapsed = time.monotonic() - start
        print(f"  [{id:02d}] ready in {elapsed:.1f}s")
        return {"url": d.url, "elapsed": elapsed, "deployment": d}
    except Exception as e:
        print(f"  [{id:02d}] FAILED: {e}")
        return {"error": str(e)}


async def main():
    print(f"\nDeploying {NUM_APPS} apps in parallel...\n")

    client = BasilicaClient()
    start = time.monotonic()

    results = await asyncio.gather(*[deploy(client, i) for i in range(1, NUM_APPS + 1)])

    total = time.monotonic() - start
    ok = [r for r in results if "url" in r]
    sum_time = sum(r["elapsed"] for r in ok)

    print(f"\n  {len(ok)}/{NUM_APPS} successful in {total:.1f}s")
    print(f"  Sequential would take {sum_time:.0f}s")
    print(f"  Speedup: {sum_time/total:.1f}x\n")

    print("Verifying 5 random apps...")
    for r in random.sample(ok, min(5, len(ok))):
        try:
            body = urllib.request.urlopen(r["url"], timeout=10).read().decode()
            print(f"  {r['url']} -> {body}")
        except Exception as e:
            print(f"  {r['url']} -> ERROR: {e}")

    print(f"\nCleaning up {len(ok)} deployments...")
    await asyncio.gather(*[r["deployment"].delete_async() for r in ok], return_exceptions=True)
    print("Done!\n")


if __name__ == "__main__":
    asyncio.run(main())
